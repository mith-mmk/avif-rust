use super::{TileDecoder, palette::inv_recenter_finite_nonneg};
use crate::DecoderError;
use crate::av1::sequence::SequenceHeader;

impl<'a> TileDecoder<'a> {
    pub(super) fn read_restoration_units(
        &mut self,
        sequence: &SequenceHeader,
        x: usize,
        y: usize,
    ) -> Result<(), DecoderError> {
        if !self.restoration.uses_lr {
            return Ok(());
        }
        let superblock_size = if sequence.use_128x128_superblock {
            128
        } else {
            64
        };
        let unit_size = superblock_size << self.restoration.unit_shift;
        if x % unit_size != 0 || y % unit_size != 0 {
            return Ok(());
        }

        let planes = if sequence.color_config.monochrome {
            1
        } else {
            3
        };
        for plane in 0..planes {
            let restoration_type = match self.restoration.lr_type[plane] {
                0 => continue,
                1 => usize::from(self.reader.read_symbol(self.cdf.wiener_restore_cdf_mut())? != 0),
                2 => {
                    let enabled = self
                        .reader
                        .read_symbol(self.cdf.sgrproj_restore_cdf_mut())?
                        != 0;
                    if enabled { 2 } else { 0 }
                }
                3 => self
                    .reader
                    .read_symbol(self.cdf.switchable_restore_cdf_mut())?,
                value => {
                    return Err(DecoderError::Bitstream(format!(
                        "AV1 restoration type {value} is invalid"
                    )));
                }
            };
            match restoration_type {
                0 => {}
                1 => self.read_wiener_filter(plane)?,
                2 => self.read_sgrproj_filter(plane)?,
                value => {
                    return Err(DecoderError::Bitstream(format!(
                        "AV1 switchable restoration symbol {value} is invalid"
                    )));
                }
            }
        }
        Ok(())
    }

    fn read_wiener_filter(&mut self, plane: usize) -> Result<(), DecoderError> {
        const BITS: [usize; 3] = [4, 5, 6];
        const SUBEXP_K: [usize; 3] = [1, 2, 3];
        const MIN: [i16; 3] = [-5, -23, -17];
        let first_tap = usize::from(plane > 0);
        for direction in 0..2 {
            for tap in first_tap..3 {
                let n = 1usize << BITS[tap];
                let reference = usize::try_from(self.wiener_refs[plane][direction][tap] - MIN[tap])
                    .map_err(|_| {
                        DecoderError::Bitstream("AV1 Wiener reference is invalid".to_string())
                    })?;
                let value = self.read_primitive_refsubexpfin(n, SUBEXP_K[tap], reference)?;
                self.wiener_refs[plane][direction][tap] = i16::try_from(value).map_err(|_| {
                    DecoderError::Bitstream("AV1 Wiener tap exceeds i16".to_string())
                })? + MIN[tap];
            }
        }
        Ok(())
    }

    fn read_sgrproj_filter(&mut self, plane: usize) -> Result<(), DecoderError> {
        const MIN: [i16; 2] = [-96, -32];
        const MAX: [i16; 2] = [31, 95];
        let index = self.reader.read_literal(4)? as usize;
        let read_value = |decoder: &mut Self, coefficient: usize| -> Result<i16, DecoderError> {
            let reference =
                usize::try_from(decoder.sgrproj_refs[plane][coefficient] - MIN[coefficient])
                    .map_err(|_| {
                        DecoderError::Bitstream("AV1 SGRPROJ reference is invalid".to_string())
                    })?;
            let value = decoder.read_primitive_refsubexpfin(128, 4, reference)?;
            Ok(i16::try_from(value).map_err(|_| {
                DecoderError::Bitstream("AV1 SGRPROJ coefficient exceeds i16".to_string())
            })? + MIN[coefficient])
        };
        if (10..=13).contains(&index) {
            self.sgrproj_refs[plane][0] = 0;
            self.sgrproj_refs[plane][1] = read_value(self, 1)?;
        } else if index >= 14 {
            self.sgrproj_refs[plane][0] = read_value(self, 0)?;
            self.sgrproj_refs[plane][1] = (128 - self.sgrproj_refs[plane][0]).clamp(MIN[1], MAX[1]);
        } else {
            self.sgrproj_refs[plane][0] = read_value(self, 0)?;
            self.sgrproj_refs[plane][1] = read_value(self, 1)?;
        }
        Ok(())
    }

    fn read_primitive_refsubexpfin(
        &mut self,
        n: usize,
        k: usize,
        reference: usize,
    ) -> Result<usize, DecoderError> {
        let value = self.read_primitive_subexpfin(n, k)?;
        Ok(inv_recenter_finite_nonneg(n, reference, value))
    }

    fn read_primitive_subexpfin(&mut self, n: usize, k: usize) -> Result<usize, DecoderError> {
        let mut index = 0usize;
        let mut mk = 0usize;
        loop {
            let bits = if index == 0 { k } else { k + index - 1 };
            let step = 1usize << bits;
            if n <= mk + 3 * step {
                return self.read_primitive_quniform(n - mk).map(|value| value + mk);
            }
            if self.reader.read_literal(1)? == 0 {
                return self
                    .reader
                    .read_literal(bits)
                    .map(|value| value as usize + mk);
            }
            index += 1;
            mk += step;
        }
    }

    fn read_primitive_quniform(&mut self, n: usize) -> Result<usize, DecoderError> {
        if n <= 1 {
            return Ok(0);
        }
        let bits = usize::BITS as usize - n.leading_zeros() as usize;
        let threshold = (1usize << bits) - n;
        let value = self.reader.read_literal(bits - 1)? as usize;
        if value < threshold {
            Ok(value)
        } else {
            Ok((value << 1) - threshold + self.reader.read_literal(1)? as usize)
        }
    }
}
