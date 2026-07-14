use super::sequence::ChromaSamplePosition;
use crate::DecoderError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InitialPresentationDelay {
    pub minus_one: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Av1CodecConfiguration {
    pub version: u8,
    pub seq_profile: u8,
    pub seq_level_idx_0: u8,
    pub seq_tier_0: bool,
    pub high_bitdepth: bool,
    pub twelve_bit: bool,
    pub monochrome: bool,
    pub chroma_subsampling_x: bool,
    pub chroma_subsampling_y: bool,
    pub chroma_sample_position: ChromaSamplePosition,
    pub initial_presentation_delay: Option<InitialPresentationDelay>,
}

impl Av1CodecConfiguration {
    pub fn bit_depth(&self) -> u8 {
        if self.twelve_bit {
            12
        } else if self.high_bitdepth {
            10
        } else {
            8
        }
    }
}

pub fn parse_av1_config(data: &[u8]) -> Result<Av1CodecConfiguration, DecoderError> {
    if data.len() < 4 {
        return Err(DecoderError::NotEnoughData(
            "av1C payload is too short".to_string(),
        ));
    }
    if data[0] & 0x80 == 0 {
        return Err(DecoderError::Bitstream(
            "av1C marker bit is not set".to_string(),
        ));
    }
    let version = data[0] & 0x7f;
    if version != 1 {
        return Err(DecoderError::Unsupported(format!(
            "av1C version {version} is not supported"
        )));
    }

    let seq_profile = data[1] >> 5;
    let seq_level_idx_0 = data[1] & 0x1f;
    let seq_tier_0 = data[2] & 0x80 != 0;
    let high_bitdepth = data[2] & 0x40 != 0;
    let twelve_bit = data[2] & 0x20 != 0;
    let monochrome = data[2] & 0x10 != 0;
    // Profile 2 8-bit streams imply 4:2:2; av1C leaves these two bits zero.
    let (chroma_subsampling_x, chroma_subsampling_y) = if seq_profile == 2 && !high_bitdepth {
        (true, false)
    } else {
        (data[2] & 0x08 != 0, data[2] & 0x04 != 0)
    };
    let chroma_sample_position = ChromaSamplePosition::from_bits(u32::from(data[2] & 0x03));
    if data[3] & 0xe0 != 0 {
        return Err(DecoderError::Bitstream(
            "av1C reserved bits are not zero".to_string(),
        ));
    }
    let initial_presentation_delay = if data[3] & 0x10 != 0 {
        Some(InitialPresentationDelay {
            minus_one: data[3] & 0x0f,
        })
    } else {
        None
    };

    Ok(Av1CodecConfiguration {
        version,
        seq_profile,
        seq_level_idx_0,
        seq_tier_0,
        high_bitdepth,
        twelve_bit,
        monochrome,
        chroma_subsampling_x,
        chroma_subsampling_y,
        chroma_sample_position,
        initial_presentation_delay,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_missing_av1_config_marker() {
        let err = parse_av1_config(&[0x01, 0, 0, 0]).unwrap_err();
        assert!(err.to_string().contains("marker bit"));
    }
}
