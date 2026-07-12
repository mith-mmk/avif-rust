use crate::DecoderError;

const EC_MIN_PROB: u32 = 4;
const EC_PROB_SHIFT: u32 = 6;

#[derive(Debug, Clone)]
pub struct EntropyDecoder<'a> {
    data: &'a [u8],
    bit_offset: usize,
    symbol_range: u32,
    symbol_dif: u32,
    symbol_max_bits: i32,
    refill_offset: usize,
    refill_count: i32,
    disable_cdf_update: bool,
}

impl<'a> EntropyDecoder<'a> {
    pub fn new(data: &'a [u8], disable_cdf_update: bool) -> Result<Self, DecoderError> {
        if data.len() < 2 {
            return Err(DecoderError::NotEnoughData(
                "AV1 entropy tile payload is too short".to_string(),
            ));
        }
        let mut decoder = Self {
            data,
            bit_offset: 15,
            symbol_range: 0x8000,
            symbol_dif: 0x7fff_ffff,
            symbol_max_bits: data.len() as i32 * 8 - 15,
            refill_offset: 0,
            refill_count: -15,
            disable_cdf_update,
        };
        decoder.refill();
        Ok(decoder)
    }

    pub fn read_bool(&mut self) -> Result<u8, DecoderError> {
        let mut cdf = [1 << 14, 1 << 15, 0];
        self.read_symbol(&mut cdf).map(|value| value as u8)
    }

    pub fn read_literal(&mut self, bits: usize) -> Result<u32, DecoderError> {
        let mut value = 0u32;
        for _ in 0..bits {
            value = (value << 1) | u32::from(self.read_raw_bit()?);
        }
        Ok(value)
    }

    pub fn read_uniform(&mut self, n: usize) -> Result<usize, DecoderError> {
        if n <= 1 {
            return Ok(0);
        }
        let bits = usize::BITS as usize - (n - 1).leading_zeros() as usize;
        let threshold = (1usize << bits) - n;
        let mut value = 0usize;
        for bit_index in 0..bits - 1 {
            let bit = usize::from(self.read_raw_bit()?);
            value = (value << 1) | bit;
            if std::env::var_os("AVIF_TRACE_WML2_MODES").is_some() && n == 5 {
                eprintln!(
                    "Rust uniform bit={bit_index} value={value} state={:?}",
                    self.trace_state()
                );
            }
        }
        if value < threshold {
            Ok(value)
        } else {
            Ok((value << 1) - threshold + self.read_literal(1)? as usize)
        }
    }

    pub fn read_symbol(&mut self, cdf: &mut [u16]) -> Result<usize, DecoderError> {
        if cdf.len() < 3 {
            return Err(DecoderError::InvalidParam(
                "AV1 CDF must contain at least two symbols and a count".to_string(),
            ));
        }
        let symbol_count = cdf.len() - 1;
        if cdf[symbol_count - 1] != 1 << 15 {
            return Err(DecoderError::Bitstream(
                "AV1 CDF terminal value is invalid".to_string(),
            ));
        }

        let coded = self.symbol_dif >> 16;
        let mut cur = self.symbol_range;
        let mut symbol = 0usize;
        let mut prev;
        loop {
            prev = cur;
            let f = (1u32 << 15) - u32::from(cdf[symbol]);
            cur = ((self.symbol_range >> 8) * (f >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT);
            cur += EC_MIN_PROB * (symbol_count - symbol - 1) as u32;
            if coded >= cur {
                break;
            }
            symbol += 1;
            if symbol >= symbol_count {
                return Err(DecoderError::Bitstream(
                    "AV1 entropy symbol search exceeded CDF".to_string(),
                ));
            }
        }

        self.symbol_range = prev - cur;
        self.symbol_dif = self.symbol_dif.wrapping_sub(cur << 16);
        self.renormalize()?;
        if !self.disable_cdf_update {
            update_cdf(cdf, symbol);
        }
        Ok(symbol)
    }

    // AOM's literal bits use a binary probability of 128, which maps to the
    // Q15 value 16384.
    fn read_raw_bit(&mut self) -> Result<u8, DecoderError> {
        let f = 16_384u32;
        let split = (((self.symbol_range >> 8) * (f >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT))
            + EC_MIN_PROB;
        let bit = if self.symbol_dif >> 16 >= split {
            self.symbol_range -= split;
            self.symbol_dif = self.symbol_dif.wrapping_sub(split << 16);
            0
        } else {
            self.symbol_range = split;
            1
        };
        self.renormalize()?;
        Ok(bit)
    }

    pub fn exit(&mut self) -> Result<usize, DecoderError> {
        if self.symbol_max_bits < -14 {
            return Err(DecoderError::Bitstream(
                "AV1 entropy decoder exited after too many padding bits".to_string(),
            ));
        }
        let logical_offset = self.bit_offset.saturating_sub(14);
        let (trailing_bit_position, padding_end_position) = if self.symbol_max_bits > 0 {
            let trailing_distance = usize::try_from(15.min(self.symbol_max_bits + 15)).unwrap();
            let trailing = logical_offset
                .checked_sub(trailing_distance)
                .ok_or_else(|| {
                    DecoderError::Bitstream("AV1 entropy trailing one bit is missing".to_string())
                })?;
            (trailing, logical_offset + self.symbol_max_bits as usize)
        } else {
            let end = self.data.len() * 8;
            let trailing = (0..end)
                .rev()
                .find(|position| bit_at(self.data, *position).unwrap_or(0) == 1)
                .ok_or_else(|| {
                    DecoderError::Bitstream("AV1 entropy trailing one bit is missing".to_string())
                })?;
            (trailing, end)
        };
        if bit_at(self.data, trailing_bit_position)? != 1 {
            return Err(DecoderError::Bitstream(
                "AV1 entropy trailing one bit is missing".to_string(),
            ));
        }
        for bit_position in trailing_bit_position + 1..padding_end_position {
            if bit_at(self.data, bit_position)? != 0 {
                return Err(DecoderError::Bitstream(
                    "AV1 entropy trailing zero bit is not zero".to_string(),
                ));
            }
        }
        Ok(padding_end_position.div_ceil(8))
    }

    pub fn bit_position(&self) -> usize {
        self.bit_offset
    }

    pub(crate) fn trace_state(&self) -> (u32, u32, usize) {
        (
            self.symbol_range,
            self.symbol_dif >> 16,
            self.bit_offset.saturating_sub(14),
        )
    }

    fn renormalize(&mut self) -> Result<(), DecoderError> {
        let bits = 15 - floor_log2(self.symbol_range);
        self.symbol_range <<= bits;
        self.symbol_dif = self.symbol_dif.wrapping_add(1).wrapping_shl(bits) - 1;
        self.refill_count -= bits as i32;
        self.bit_offset += bits as usize;
        self.symbol_max_bits -= bits as i32;
        if self.refill_count < 0 {
            self.refill();
        }
        Ok(())
    }

    fn refill(&mut self) {
        let mut shift = 23 - (self.refill_count + 15);
        while shift >= 0 && self.refill_offset < self.data.len() {
            self.symbol_dif ^= u32::from(self.data[self.refill_offset]) << shift;
            self.refill_offset += 1;
            self.refill_count += 8;
            shift -= 8;
        }
        if self.refill_offset >= self.data.len() {
            self.refill_count = 0x4000;
        }
    }
}

fn update_cdf(cdf: &mut [u16], symbol: usize) {
    let symbol_count = cdf.len() - 1;
    let count = cdf[symbol_count];
    let rate = 3
        + u16::from(count > 15)
        + u16::from(count > 31)
        + floor_log2(symbol_count as u32).min(2) as u16;
    // CDF tables are stored in the normal cumulative form, while AOM's
    // reader updates inverse CDF values.  Convert that update direction
    // directly: entries before the decoded symbol move toward zero and the
    // remaining cumulative entries move toward CDF_PROB_TOP.
    for (index, entry) in cdf.iter_mut().take(symbol_count - 1).enumerate() {
        if index < symbol {
            *entry -= *entry >> rate;
        } else {
            *entry += ((1 << 15) - *entry) >> rate;
        }
    }
    if cdf[symbol_count] < 32 {
        cdf[symbol_count] += 1;
    }
}

fn floor_log2(value: u32) -> u32 {
    31 - value.leading_zeros()
}

fn bit_at(data: &[u8], bit_offset: usize) -> Result<u8, DecoderError> {
    let byte = *data.get(bit_offset / 8).ok_or_else(|| {
        DecoderError::NotEnoughData("AV1 entropy bitstream is truncated".to_string())
    })?;
    Ok((byte >> (7 - (bit_offset % 8))) & 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initializes_from_tile_payload_bits() {
        let decoder = EntropyDecoder::new(&[0b1010_1010, 0b0101_0101], false).unwrap();

        assert_eq!(decoder.bit_position(), 15);
    }

    #[test]
    fn rejects_short_tile_payload() {
        let err = EntropyDecoder::new(&[0], false).unwrap_err();
        assert!(err.to_string().contains("too short"));
    }

    #[test]
    fn exit_rejects_missing_trailing_one_bit() {
        let mut decoder = EntropyDecoder::new(&[0, 0], false).unwrap();
        let err = decoder.exit().unwrap_err();

        assert!(
            matches!(err, DecoderError::Bitstream(message) if message.contains("trailing one bit"))
        );
    }

    #[test]
    fn read_symbol_rejects_malformed_cdf_terminal() {
        let mut decoder = EntropyDecoder::new(&[0xff, 0x80], false).unwrap();
        let mut cdf = [1 << 14, (1 << 15) - 1, 0];
        let err = decoder.read_symbol(&mut cdf).unwrap_err();

        assert!(
            matches!(err, DecoderError::Bitstream(message) if message.contains("terminal value"))
        );
    }

    #[test]
    fn updates_cdf_count_when_enabled() {
        let mut cdf = [1 << 14, 1 << 15, 0];
        update_cdf(&mut cdf, 0);

        assert_eq!(cdf[2], 1);
    }
}
