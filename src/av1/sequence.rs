use super::bitstream::BitReader;
use crate::DecoderError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorRange {
    Studio,
    Full,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaSamplePosition {
    Unknown,
    Vertical,
    Colocated,
    Reserved,
}

impl ChromaSamplePosition {
    pub(crate) fn from_bits(value: u32) -> Self {
        match value {
            0 => Self::Unknown,
            1 => Self::Vertical,
            2 => Self::Colocated,
            _ => Self::Reserved,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorDescription {
    pub color_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorConfig {
    pub high_bitdepth: bool,
    pub twelve_bit: bool,
    pub bit_depth: u8,
    pub monochrome: bool,
    pub color_description: Option<ColorDescription>,
    pub color_range: ColorRange,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
    pub chroma_sample_position: Option<ChromaSamplePosition>,
    pub separate_uv_delta_q: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceHeader {
    pub seq_profile: u8,
    pub still_picture: bool,
    pub reduced_still_picture_header: bool,
    pub seq_level_idx_0: u8,
    pub frame_width_bits: u8,
    pub frame_height_bits: u8,
    pub max_frame_width: u32,
    pub max_frame_height: u32,
    pub frame_id_numbers_present: bool,
    pub use_128x128_superblock: bool,
    pub enable_filter_intra: bool,
    pub enable_intra_edge_filter: bool,
    pub enable_order_hint: bool,
    pub enable_warped_motion: bool,
    pub order_hint_bits: u8,
    pub seq_force_screen_content_tools: u8,
    pub seq_force_integer_mv: u8,
    pub enable_ref_frame_mvs: bool,
    pub enable_superres: bool,
    pub enable_cdef: bool,
    pub enable_restoration: bool,
    pub color_config: ColorConfig,
    pub film_grain_params_present: bool,
}

pub fn parse_sequence_header(data: &[u8]) -> Result<SequenceHeader, DecoderError> {
    let mut reader = BitReader::new(data);
    let seq_profile = reader.read_bits(3, "seq_profile")? as u8;
    if seq_profile > 2 {
        return Err(DecoderError::Bitstream(format!(
            "seq_profile {seq_profile} is reserved"
        )));
    }

    let still_picture = reader.read_bool("still_picture")?;
    let reduced_still_picture_header = reader.read_bool("reduced_still_picture_header")?;
    let seq_level_idx_0 = if reduced_still_picture_header {
        reader.read_bits(5, "seq_level_idx[0]")? as u8
    } else {
        parse_operating_points(&mut reader)?
    };

    let frame_width_bits = reader.read_bits(4, "frame_width_bits_minus_1")? as u8 + 1;
    let frame_height_bits = reader.read_bits(4, "frame_height_bits_minus_1")? as u8 + 1;
    let max_frame_width =
        reader.read_bits(frame_width_bits as usize, "max_frame_width_minus_1")? + 1;
    let max_frame_height =
        reader.read_bits(frame_height_bits as usize, "max_frame_height_minus_1")? + 1;

    let frame_id_numbers_present =
        !reduced_still_picture_header && reader.read_bool("frame_id_numbers_present_flag")?;
    if frame_id_numbers_present {
        let _delta_frame_id_length_minus_2 =
            reader.read_bits(4, "delta_frame_id_length_minus_2")?;
        let _additional_frame_id_length_minus_1 =
            reader.read_bits(3, "additional_frame_id_length_minus_1")?;
    }

    let use_128x128_superblock = reader.read_bool("use_128x128_superblock")?;
    let enable_filter_intra = reader.read_bool("enable_filter_intra")?;
    let enable_intra_edge_filter = reader.read_bool("enable_intra_edge_filter")?;

    let inter_tools = if reduced_still_picture_header {
        InterTools::reduced_still()
    } else {
        parse_inter_tools(&mut reader)?
    };

    let enable_superres = reader.read_bool("enable_superres")?;
    let enable_cdef = reader.read_bool("enable_cdef")?;
    let enable_restoration = reader.read_bool("enable_restoration")?;
    let color_config = parse_color_config(&mut reader, seq_profile)?;
    let film_grain_params_present = reader.read_bool("film_grain_params_present")?;

    Ok(SequenceHeader {
        seq_profile,
        still_picture,
        reduced_still_picture_header,
        seq_level_idx_0,
        frame_width_bits,
        frame_height_bits,
        max_frame_width,
        max_frame_height,
        frame_id_numbers_present,
        use_128x128_superblock,
        enable_filter_intra,
        enable_intra_edge_filter,
        enable_order_hint: inter_tools.enable_order_hint,
        enable_warped_motion: inter_tools.enable_warped_motion,
        order_hint_bits: inter_tools.order_hint_bits,
        seq_force_screen_content_tools: inter_tools.seq_force_screen_content_tools,
        seq_force_integer_mv: inter_tools.seq_force_integer_mv,
        enable_ref_frame_mvs: inter_tools.enable_ref_frame_mvs,
        enable_superres,
        enable_cdef,
        enable_restoration,
        color_config,
        film_grain_params_present,
    })
}

fn parse_operating_points(reader: &mut BitReader<'_>) -> Result<u8, DecoderError> {
    let timing_info_present_flag = reader.read_bool("timing_info_present_flag")?;
    let mut buffer_delay_length = 0usize;
    let decoder_model_info_present_flag = if timing_info_present_flag {
        let _num_units_in_display_tick = reader.read_bits(32, "num_units_in_display_tick")?;
        let _time_scale = reader.read_bits(32, "time_scale")?;
        if reader.read_bool("equal_picture_interval")? {
            let _num_ticks_per_picture_minus_1 =
                reader.read_uvlc("num_ticks_per_picture_minus_1")?;
        }
        let decoder_model_info_present_flag =
            reader.read_bool("decoder_model_info_present_flag")?;
        if decoder_model_info_present_flag {
            let _buffer_delay_length_minus_1 =
                reader.read_bits(5, "buffer_delay_length_minus_1")?;
            buffer_delay_length = _buffer_delay_length_minus_1 as usize + 1;
            let _num_units_in_decoding_tick = reader.read_bits(32, "num_units_in_decoding_tick")?;
            let _buffer_removal_time_length_minus_1 =
                reader.read_bits(5, "buffer_removal_time_length_minus_1")?;
            let _frame_presentation_time_length_minus_1 =
                reader.read_bits(5, "frame_presentation_time_length_minus_1")?;
        }
        decoder_model_info_present_flag
    } else {
        false
    };
    let initial_display_delay_present_flag =
        reader.read_bool("initial_display_delay_present_flag")?;
    let operating_points_cnt_minus_1 = reader.read_bits(5, "operating_points_cnt_minus_1")?;
    let mut seq_level_idx_0 = 0u8;
    for index in 0..=operating_points_cnt_minus_1 {
        let _operating_point_idc = reader.read_bits(12, "operating_point_idc")?;
        let seq_level_idx = reader.read_bits(5, "seq_level_idx")? as u8;
        if index == 0 {
            seq_level_idx_0 = seq_level_idx;
        }
        if seq_level_idx > 7 {
            let _seq_tier = reader.read_bool("seq_tier")?;
        }
        if decoder_model_info_present_flag
            && reader.read_bool("decoder_model_present_for_this_op")?
        {
            let _decoder_buffer_delay =
                reader.read_bits(buffer_delay_length, "decoder_buffer_delay")?;
            let _encoder_buffer_delay =
                reader.read_bits(buffer_delay_length, "encoder_buffer_delay")?;
            let _low_delay_mode_flag = reader.read_bool("low_delay_mode_flag")?;
        }
        if initial_display_delay_present_flag
            && reader.read_bool("initial_display_delay_present_for_this_op")?
        {
            let _initial_display_delay_minus_1 =
                reader.read_bits(4, "initial_display_delay_minus_1")?;
        }
    }
    Ok(seq_level_idx_0)
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct InterTools {
    enable_order_hint: bool,
    enable_warped_motion: bool,
    enable_ref_frame_mvs: bool,
    order_hint_bits: u8,
    seq_force_screen_content_tools: u8,
    seq_force_integer_mv: u8,
}

impl InterTools {
    fn reduced_still() -> Self {
        Self {
            seq_force_screen_content_tools: 2,
            seq_force_integer_mv: 2,
            ..Self::default()
        }
    }
}

fn parse_inter_tools(reader: &mut BitReader<'_>) -> Result<InterTools, DecoderError> {
    let _enable_interintra_compound = reader.read_bool("enable_interintra_compound")?;
    let _enable_masked_compound = reader.read_bool("enable_masked_compound")?;
    let enable_warped_motion = reader.read_bool("enable_warped_motion")?;
    let _enable_dual_filter = reader.read_bool("enable_dual_filter")?;
    let enable_order_hint = reader.read_bool("enable_order_hint")?;
    let enable_ref_frame_mvs;
    if enable_order_hint {
        let _enable_jnt_comp = reader.read_bool("enable_jnt_comp")?;
        enable_ref_frame_mvs = reader.read_bool("enable_ref_frame_mvs")?;
    } else {
        enable_ref_frame_mvs = false;
    }

    let seq_choose_screen_content_tools = reader.read_bool("seq_choose_screen_content_tools")?;
    let seq_force_screen_content_tools = if seq_choose_screen_content_tools {
        2
    } else {
        reader.read_bits(1, "seq_force_screen_content_tools")? as u8
    };
    let mut seq_force_integer_mv = 2;
    if seq_force_screen_content_tools > 0 {
        let seq_choose_integer_mv = reader.read_bool("seq_choose_integer_mv")?;
        if !seq_choose_integer_mv {
            seq_force_integer_mv = reader.read_bits(1, "seq_force_integer_mv")? as u8;
        }
    }
    let order_hint_bits = if enable_order_hint {
        reader.read_bits(3, "order_hint_bits_minus_1")? as u8 + 1
    } else {
        0
    };
    Ok(InterTools {
        enable_order_hint,
        enable_warped_motion,
        enable_ref_frame_mvs,
        order_hint_bits,
        seq_force_screen_content_tools,
        seq_force_integer_mv,
    })
}

fn parse_color_config(
    reader: &mut BitReader<'_>,
    seq_profile: u8,
) -> Result<ColorConfig, DecoderError> {
    let high_bitdepth = reader.read_bool("high_bitdepth")?;
    let twelve_bit = if seq_profile == 2 && high_bitdepth {
        reader.read_bool("twelve_bit")?
    } else {
        false
    };
    let bit_depth = if seq_profile == 2 && high_bitdepth {
        if twelve_bit { 12 } else { 10 }
    } else if high_bitdepth {
        10
    } else {
        8
    };

    let monochrome = if seq_profile == 1 {
        false
    } else {
        reader.read_bool("mono_chrome")?
    };

    let color_description = if reader.read_bool("color_description_present_flag")? {
        Some(ColorDescription {
            color_primaries: reader.read_bits(8, "color_primaries")? as u8,
            transfer_characteristics: reader.read_bits(8, "transfer_characteristics")? as u8,
            matrix_coefficients: reader.read_bits(8, "matrix_coefficients")? as u8,
        })
    } else {
        None
    };

    if monochrome {
        return Ok(ColorConfig {
            high_bitdepth,
            twelve_bit,
            bit_depth,
            monochrome,
            color_description,
            color_range: read_color_range(reader)?,
            subsampling_x: true,
            subsampling_y: true,
            chroma_sample_position: None,
            separate_uv_delta_q: false,
        });
    }

    if color_description.is_some_and(|description| {
        description.color_primaries == 1
            && description.transfer_characteristics == 13
            && description.matrix_coefficients == 0
    }) {
        return Ok(ColorConfig {
            high_bitdepth,
            twelve_bit,
            bit_depth,
            monochrome,
            color_description,
            color_range: ColorRange::Full,
            subsampling_x: false,
            subsampling_y: false,
            chroma_sample_position: None,
            separate_uv_delta_q: reader.read_bool("separate_uv_delta_q")?,
        });
    }

    let color_range = read_color_range(reader)?;
    let (subsampling_x, subsampling_y) = match seq_profile {
        0 => (true, true),
        1 => (false, false),
        2 => {
            if bit_depth == 12 {
                let subsampling_x = reader.read_bool("subsampling_x")?;
                let subsampling_y = if subsampling_x {
                    reader.read_bool("subsampling_y")?
                } else {
                    false
                };
                (subsampling_x, subsampling_y)
            } else {
                (true, false)
            }
        }
        _ => {
            return Err(DecoderError::Bitstream(format!(
                "seq_profile {seq_profile} is reserved"
            )));
        }
    };
    let chroma_sample_position = if subsampling_x && subsampling_y {
        Some(ChromaSamplePosition::from_bits(
            reader.read_bits(2, "chroma_sample_position")?,
        ))
    } else {
        None
    };
    let separate_uv_delta_q = reader.read_bool("separate_uv_delta_q")?;

    Ok(ColorConfig {
        high_bitdepth,
        twelve_bit,
        bit_depth,
        monochrome,
        color_description,
        color_range,
        subsampling_x,
        subsampling_y,
        chroma_sample_position,
        separate_uv_delta_q,
    })
}

fn read_color_range(reader: &mut BitReader<'_>) -> Result<ColorRange, DecoderError> {
    if reader.read_bool("color_range")? {
        Ok(ColorRange::Full)
    } else {
        Ok(ColorRange::Studio)
    }
}

#[cfg(test)]
mod tests {}
