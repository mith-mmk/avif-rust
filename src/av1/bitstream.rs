use crate::DecoderError;

pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    bit_offset: usize,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            bit_offset: 0,
        }
    }

    pub(crate) fn new_at(data: &'a [u8], bit_offset: usize) -> Result<Self, DecoderError> {
        if bit_offset > data.len() * 8 {
            return Err(DecoderError::NotEnoughData(
                "bit reader start offset is outside data".to_string(),
            ));
        }
        Ok(Self { data, bit_offset })
    }

    pub(crate) fn read_bool(&mut self, name: &str) -> Result<bool, DecoderError> {
        Ok(self.read_bits(1, name)? != 0)
    }

    pub(crate) fn read_bits(&mut self, count: usize, name: &str) -> Result<u32, DecoderError> {
        if count > 32 {
            return Err(DecoderError::InvalidParam(format!(
                "{name} requests more than 32 bits"
            )));
        }

        let mut value = 0u32;
        for _ in 0..count {
            let byte_index = self.bit_offset / 8;
            let bit_index = 7 - (self.bit_offset % 8);
            let byte = *self
                .data
                .get(byte_index)
                .ok_or_else(|| DecoderError::NotEnoughData(format!("{name} is truncated")))?;
            self.bit_offset += 1;
            value = (value << 1) | u32::from((byte >> bit_index) & 1);
        }
        Ok(value)
    }

    pub(crate) fn read_uvlc(&mut self, name: &str) -> Result<u32, DecoderError> {
        let mut leading_zeroes = 0usize;
        while !self.read_bool(name)? {
            leading_zeroes += 1;
            if leading_zeroes >= 32 {
                return Err(DecoderError::Bitstream(format!("{name} uvlc is too large")));
            }
        }
        if leading_zeroes == 0 {
            return Ok(0);
        }

        let suffix = self.read_bits(leading_zeroes, name)?;
        Ok(((1u32 << leading_zeroes) - 1) + suffix)
    }

    pub(crate) fn read_ns(&mut self, n: u32, name: &str) -> Result<u32, DecoderError> {
        if n == 0 {
            return Err(DecoderError::InvalidParam(format!(
                "{name} ns range must be non-zero"
            )));
        }
        let width = 32 - n.leading_zeros();
        let m = (1u32 << width) - n;
        let value = self.read_bits(width as usize - 1, name)?;
        if value < m {
            Ok(value)
        } else {
            let extra_bit = self.read_bits(1, name)?;
            Ok((value << 1) - m + extra_bit)
        }
    }

    pub(crate) fn bit_position(&self) -> usize {
        self.bit_offset
    }

    pub(crate) fn byte_position_ceil(&self) -> usize {
        self.bit_offset.div_ceil(8)
    }

    pub(crate) fn byte_align_zero(&mut self, name: &str) -> Result<(), DecoderError> {
        while !self.bit_offset.is_multiple_of(8) {
            if self.read_bool(name)? {
                return Err(DecoderError::Bitstream(format!(
                    "{name} byte alignment bit is not zero"
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposes_bit_and_byte_positions() {
        let mut reader = BitReader::new(&[0b1010_0000, 0xff]);

        assert_eq!(reader.read_bits(3, "prefix").unwrap(), 0b101);
        assert_eq!(reader.bit_position(), 3);
        assert_eq!(reader.byte_position_ceil(), 1);
    }

    #[test]
    fn reads_ns_values() {
        let mut reader = BitReader::new(&[0b1110_0000]);

        assert_eq!(reader.read_ns(8, "ns").unwrap(), 7);
        assert_eq!(reader.bit_position(), 3);
    }

    #[test]
    fn byte_align_zero_rejects_one_padding_bits() {
        let mut reader = BitReader::new(&[0b1011_0000]);
        assert_eq!(reader.read_bits(3, "prefix").unwrap(), 0b101);
        let err = reader.byte_align_zero("align").unwrap_err();
        assert!(err.to_string().contains("alignment bit"));
    }
}
