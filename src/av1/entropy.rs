use crate::DecoderError;

const EC_MIN_PROB: u32 = 4;
const EC_PROB_SHIFT: u32 = 6;

#[derive(Debug, Clone)]
pub struct EntropyDecoder<'a> {
    data: &'a [u8],
    bit_offset: usize,
    symbol_range: u32,
    symbol_value: u32,
    symbol_max_bits: i32,
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
            bit_offset: 0,
            symbol_range: 0x8000,
            symbol_value: 0,
            symbol_max_bits: data.len() as i32 * 8 - 15,
            disable_cdf_update,
        };
        let num_bits = (data.len() * 8).min(15);
        let buf = decoder.read_bits_raw(num_bits, "entropy_init")?;
        let padded_buf = buf << (15 - num_bits);
        decoder.symbol_value = ((1 << 15) - 1) ^ padded_buf;
        Ok(decoder)
    }

    pub fn read_bool(&mut self) -> Result<u8, DecoderError> {
        let mut cdf = [1 << 14, 1 << 15, 0];
        self.read_symbol(&mut cdf).map(|value| value as u8)
    }

    pub fn read_literal(&mut self, bits: usize) -> Result<u32, DecoderError> {
        let mut value = 0u32;
        for _ in 0..bits {
            value = (value << 1) | u32::from(self.read_bool()?);
        }
        Ok(value)
    }

    pub fn read_uniform(&mut self, n: usize) -> Result<usize, DecoderError> {
        if n <= 1 {
            return Ok(0);
        }
        let bits = usize::BITS as usize - (n - 1).leading_zeros() as usize;
        let threshold = (1usize << bits) - n;
        let value = self.read_literal(bits - 1)? as usize;
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

        let mut cur = self.symbol_range;
        let mut symbol = 0usize;
        let mut prev;
        loop {
            prev = cur;
            let f = (1u32 << 15) - u32::from(cdf[symbol]);
            cur = ((self.symbol_range >> 8) * (f >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT);
            cur += EC_MIN_PROB * (symbol_count - symbol - 1) as u32;
            if self.symbol_value >= cur {
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
        self.symbol_value -= cur;
        self.renormalize()?;
        if !self.disable_cdf_update {
            update_cdf(cdf, symbol);
        }
        Ok(symbol)
    }

    pub fn exit(&mut self) -> Result<usize, DecoderError> {
        if self.symbol_max_bits < -14 {
            return Err(DecoderError::Bitstream(
                "AV1 entropy decoder exited after too many padding bits".to_string(),
            ));
        }
        let trailing_bit_position =
            self.bit_offset - usize::try_from(15.min(self.symbol_max_bits + 15)).unwrap();
        if self.symbol_max_bits > 0 {
            self.bit_offset += self.symbol_max_bits as usize;
        }
        let padding_end_position = self.bit_offset;
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

    fn renormalize(&mut self) -> Result<(), DecoderError> {
        let bits = 15 - floor_log2(self.symbol_range);
        self.symbol_range <<= bits;
        let num_bits = bits.min(self.symbol_max_bits.max(0) as u32);
        let new_data = self.read_bits_raw(num_bits as usize, "entropy_renormalize")?;
        let padded_data = new_data << (bits - num_bits);
        self.symbol_value = padded_data ^ (((self.symbol_value + 1) << bits) - 1);
        self.symbol_max_bits -= bits as i32;
        Ok(())
    }

    fn read_bits_raw(&mut self, bits: usize, name: &str) -> Result<u32, DecoderError> {
        if bits > 32 {
            return Err(DecoderError::InvalidParam(format!(
                "{name} requests more than 32 bits"
            )));
        }
        let mut value = 0u32;
        for _ in 0..bits {
            value = (value << 1) | u32::from(bit_at(self.data, self.bit_offset)?);
            self.bit_offset += 1;
        }
        Ok(value)
    }
}

fn update_cdf(cdf: &mut [u16], symbol: usize) {
    let symbol_count = cdf.len() - 1;
    let count = cdf[symbol_count];
    let rate = 3
        + u16::from(count > 15)
        + u16::from(count > 31)
        + floor_log2(symbol_count as u32).min(2) as u16;
    let mut tmp = 0u16;
    for (index, entry) in cdf.iter_mut().take(symbol_count - 1).enumerate() {
        if index == symbol {
            tmp = 1 << 15;
        }
        if tmp < *entry {
            *entry -= (*entry - tmp) >> rate;
        } else {
            *entry += (tmp - *entry) >> rate;
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
    fn updates_cdf_count_when_enabled() {
        let mut cdf = [1 << 14, 1 << 15, 0];
        update_cdf(&mut cdf, 0);

        assert_eq!(cdf[2], 1);
    }
}
