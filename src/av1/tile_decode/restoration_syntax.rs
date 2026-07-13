use super::{TileDecoder, palette::inv_recenter_finite_nonneg, post_filter_state::RestorationUnit};
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
        let planes = if sequence.color_config.monochrome {
            1
        } else {
            3
        };
        for plane in 0..planes {
            if self.restoration.lr_type[plane] == 0 {
                continue;
            }
            let subsampling_x = usize::from(plane > 0 && sequence.color_config.subsampling_x);
            let subsampling_y = usize::from(plane > 0 && sequence.color_config.subsampling_y);
            let plane_width = (self.mi_cols * 4).div_ceil(1 << subsampling_x);
            let plane_height = (self.mi_rows * 4).div_ceil(1 << subsampling_y);
            let plane_x = x >> subsampling_x;
            let plane_y = y >> subsampling_y;
            let plane_sb_width = superblock_size >> subsampling_x;
            let plane_sb_height = superblock_size >> subsampling_y;
            let plane_unit_size = unit_size;
            let Some((cols, rows)) = restoration_unit_ranges(
                plane_x,
                plane_y,
                plane_sb_width,
                plane_sb_height,
                plane_unit_size,
                plane_width,
                plane_height,
            ) else {
                continue;
            };
            for row in rows {
                for col in cols.clone() {
                    let mut sgrproj_index = None;
                    let restoration_type = match self.restoration.lr_type[plane] {
                        1 => usize::from(
                            self.reader.read_symbol(self.cdf.wiener_restore_cdf_mut())? != 0,
                        ),
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
                        2 => {
                            sgrproj_index = Some(self.read_sgrproj_filter(plane)?);
                        }
                        value => {
                            return Err(DecoderError::Bitstream(format!(
                                "AV1 switchable restoration symbol {value} is invalid"
                            )));
                        }
                    }
                    self.restoration_units.push(RestorationUnit {
                        x: col * plane_unit_size,
                        y: row * plane_unit_size,
                        plane,
                        restoration_type: restoration_type as u8,
                        wiener: (restoration_type == 1).then_some(self.wiener_refs[plane]),
                        sgrproj: (restoration_type == 2).then_some(self.sgrproj_refs[plane]),
                        sgrproj_index,
                    });
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

    fn read_sgrproj_filter(&mut self, plane: usize) -> Result<u8, DecoderError> {
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
        Ok(index as u8)
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

fn restoration_unit_ranges(
    x: usize,
    y: usize,
    superblock_width: usize,
    superblock_height: usize,
    unit_size: usize,
    plane_width: usize,
    plane_height: usize,
) -> Option<(std::ops::Range<usize>, std::ops::Range<usize>)> {
    let horizontal_units = ((plane_width + unit_size / 2) / unit_size).max(1);
    let vertical_units = ((plane_height + unit_size / 2) / unit_size).max(1);
    let col_start = x.div_ceil(unit_size);
    let row_start = y.div_ceil(unit_size);
    let col_end = (x + superblock_width)
        .div_ceil(unit_size)
        .min(horizontal_units);
    let row_end = (y + superblock_height)
        .div_ceil(unit_size)
        .min(vertical_units);
    (col_start < col_end && row_start < row_end).then_some((col_start..col_end, row_start..row_end))
}

#[cfg(test)]
mod tests {
    use super::restoration_unit_ranges;

    #[test]
    fn restoration_units_merge_a_small_right_edge_remainder() {
        assert_eq!(
            restoration_unit_ranges(768, 0, 128, 128, 128, 900, 900),
            Some((6..7, 0..1))
        );
        assert_eq!(
            restoration_unit_ranges(896, 0, 128, 128, 128, 900, 900),
            None
        );
    }
}
