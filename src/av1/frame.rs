use super::bitstream::BitReader;
use super::sequence::SequenceHeader;
use super::tile::{TileInfo, parse_tile_info};
use crate::DecoderError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    Key,
    Inter,
    IntraOnly,
    Switch,
}

impl FrameType {
    fn from_bits(value: u32) -> Result<Self, DecoderError> {
        match value {
            0 => Ok(Self::Key),
            1 => Ok(Self::Inter),
            2 => Ok(Self::IntraOnly),
            3 => Ok(Self::Switch),
            _ => Err(DecoderError::Bitstream(format!(
                "frame_type {value} is reserved"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameHeader {
    pub frame_type: FrameType,
    pub show_existing_frame: bool,
    pub show_frame: bool,
    pub showable_frame: bool,
    pub error_resilient_mode: bool,
    pub disable_cdf_update: bool,
    pub allow_screen_content_tools: bool,
    pub force_integer_mv: u8,
    pub frame_size_override_flag: bool,
    pub order_hint: u32,
    pub primary_ref_frame: u8,
    pub refresh_frame_flags: u8,
    pub frame_width: u32,
    pub frame_height: u32,
    pub upscaled_width: u32,
    pub render_width: u32,
    pub render_height: u32,
    pub allow_intrabc: bool,
    pub disable_frame_end_update_cdf: bool,
    pub tile_info: TileInfo,
    pub base_q_idx: u8,
    pub quantization: QuantizationParams,
    pub delta_q: DeltaQParams,
    pub delta_lf: DeltaLfParams,
    pub loop_filter: LoopFilterParams,
    pub cdef: CdefParams,
    pub restoration: RestorationParams,
    pub tx_mode: TxMode,
    pub reduced_tx_set: bool,
    pub uncompressed_header_bits: usize,
    pub payload_after_header_offset: usize,
}

pub fn parse_frame_header(
    data: &[u8],
    sequence: &SequenceHeader,
) -> Result<FrameHeader, DecoderError> {
    let mut reader = BitReader::new(data);

    if sequence.reduced_still_picture_header {
        let frame_size = parse_frame_size(&mut reader, sequence, false)?;
        let render_size = parse_render_size(&mut reader, frame_size.width, frame_size.height)?;
        let tile_info =
            parse_tile_info(&mut reader, sequence, frame_size.width, frame_size.height)?;
        let trailing = parse_frame_header_trailing_params(
            &mut reader,
            sequence,
            false,
            frame_type_is_intra(FrameType::Key),
        )?;
        return Ok(FrameHeader {
            frame_type: FrameType::Key,
            show_existing_frame: false,
            show_frame: true,
            showable_frame: false,
            error_resilient_mode: true,
            disable_cdf_update: false,
            allow_screen_content_tools: false,
            force_integer_mv: 2,
            frame_size_override_flag: false,
            order_hint: 0,
            primary_ref_frame: 7,
            refresh_frame_flags: 0xff,
            frame_width: frame_size.width,
            frame_height: frame_size.height,
            upscaled_width: frame_size.upscaled_width,
            render_width: render_size.width,
            render_height: render_size.height,
            allow_intrabc: false,
            disable_frame_end_update_cdf: false,
            tile_info,
            base_q_idx: trailing.quantization.base_q_idx,
            quantization: trailing.quantization,
            delta_q: trailing.delta_q,
            delta_lf: trailing.delta_lf,
            loop_filter: trailing.loop_filter,
            cdef: trailing.cdef,
            restoration: trailing.restoration,
            tx_mode: trailing.tx_mode,
            reduced_tx_set: trailing.reduced_tx_set,
            uncompressed_header_bits: reader.bit_position(),
            payload_after_header_offset: reader.byte_position_ceil(),
        });
    }

    let show_existing_frame = reader.read_bool("show_existing_frame")?;
    if show_existing_frame {
        return Err(DecoderError::Unsupported(
            "show_existing_frame AV1 frames are not supported yet".to_string(),
        ));
    }

    let frame_type = FrameType::from_bits(reader.read_bits(2, "frame_type")?)?;
    let show_frame = reader.read_bool("show_frame")?;
    let showable_frame = if show_frame {
        frame_type != FrameType::Key
    } else {
        reader.read_bool("showable_frame")?
    };
    let error_resilient_mode =
        if frame_type == FrameType::Switch || (frame_type == FrameType::Key && show_frame) {
            true
        } else {
            reader.read_bool("error_resilient_mode")?
        };
    let disable_cdf_update = reader.read_bool("disable_cdf_update")?;
    let allow_screen_content_tools = read_allow_screen_content_tools(&mut reader, sequence)?;
    let force_integer_mv =
        read_force_integer_mv(&mut reader, sequence, allow_screen_content_tools)?;
    let frame_size_override_flag = if frame_type == FrameType::Switch {
        true
    } else {
        reader.read_bool("frame_size_override_flag")?
    };
    let order_hint = if sequence.enable_order_hint {
        reader.read_bits(sequence.order_hint_bits as usize, "order_hint")?
    } else {
        0
    };
    let primary_ref_frame = if error_resilient_mode || frame_type == FrameType::IntraOnly {
        7
    } else {
        reader.read_bits(3, "primary_ref_frame")? as u8
    };
    let refresh_frame_flags = if frame_type == FrameType::Key && show_frame {
        0xff
    } else {
        reader.read_bits(8, "refresh_frame_flags")? as u8
    };

    if !matches!(frame_type, FrameType::Key | FrameType::IntraOnly) {
        return Err(DecoderError::Unsupported(format!(
            "{frame_type:?} AV1 frames are not supported yet"
        )));
    }

    let frame_size = parse_frame_size(&mut reader, sequence, frame_size_override_flag)?;
    let render_size = parse_render_size(&mut reader, frame_size.width, frame_size.height)?;
    let allow_intrabc =
        if allow_screen_content_tools && frame_size.upscaled_width == frame_size.width {
            reader.read_bool("allow_intrabc")?
        } else {
            false
        };
    let disable_frame_end_update_cdf = reader.read_bool("disable_frame_end_update_cdf")?;
    let tile_info = parse_tile_info(&mut reader, sequence, frame_size.width, frame_size.height)?;
    let trailing = parse_frame_header_trailing_params(
        &mut reader,
        sequence,
        allow_intrabc,
        frame_type_is_intra(frame_type),
    )?;

    Ok(FrameHeader {
        frame_type,
        show_existing_frame,
        show_frame,
        showable_frame,
        error_resilient_mode,
        disable_cdf_update,
        allow_screen_content_tools,
        force_integer_mv,
        frame_size_override_flag,
        order_hint,
        primary_ref_frame,
        refresh_frame_flags,
        frame_width: frame_size.width,
        frame_height: frame_size.height,
        upscaled_width: frame_size.upscaled_width,
        render_width: render_size.width,
        render_height: render_size.height,
        allow_intrabc,
        disable_frame_end_update_cdf,
        tile_info,
        base_q_idx: trailing.quantization.base_q_idx,
        quantization: trailing.quantization,
        delta_q: trailing.delta_q,
        delta_lf: trailing.delta_lf,
        loop_filter: trailing.loop_filter,
        cdef: trailing.cdef,
        restoration: trailing.restoration,
        tx_mode: trailing.tx_mode,
        reduced_tx_set: trailing.reduced_tx_set,
        uncompressed_header_bits: reader.bit_position(),
        payload_after_header_offset: reader.byte_position_ceil(),
    })
}

#[derive(Debug, Clone, Copy)]
struct FrameSize {
    width: u32,
    height: u32,
    upscaled_width: u32,
}

#[derive(Debug, Clone, Copy)]
struct RenderSize {
    width: u32,
    height: u32,
}

fn read_allow_screen_content_tools(
    reader: &mut BitReader<'_>,
    sequence: &SequenceHeader,
) -> Result<bool, DecoderError> {
    match sequence.seq_force_screen_content_tools {
        2 => reader.read_bool("allow_screen_content_tools"),
        1 => Ok(true),
        _ => Ok(false),
    }
}

fn read_force_integer_mv(
    reader: &mut BitReader<'_>,
    sequence: &SequenceHeader,
    allow_screen_content_tools: bool,
) -> Result<u8, DecoderError> {
    if !allow_screen_content_tools {
        return Ok(2);
    }
    match sequence.seq_force_integer_mv {
        2 => Ok(reader.read_bool("force_integer_mv")? as u8),
        value => Ok(value),
    }
}

fn parse_frame_size(
    reader: &mut BitReader<'_>,
    sequence: &SequenceHeader,
    frame_size_override_flag: bool,
) -> Result<FrameSize, DecoderError> {
    let upscaled_width = if frame_size_override_flag {
        reader.read_bits(sequence.frame_width_bits as usize, "frame_width_minus_1")? + 1
    } else {
        sequence.max_frame_width
    };
    let height = if frame_size_override_flag {
        reader.read_bits(sequence.frame_height_bits as usize, "frame_height_minus_1")? + 1
    } else {
        sequence.max_frame_height
    };

    let superres_denom = if sequence.enable_superres && reader.read_bool("use_superres")? {
        reader.read_bits(3, "coded_denom")? + 9
    } else {
        8
    };
    let width = if superres_denom == 8 {
        upscaled_width
    } else {
        (upscaled_width * 8 + (superres_denom / 2)) / superres_denom
    };

    Ok(FrameSize {
        width,
        height,
        upscaled_width,
    })
}

fn parse_render_size(
    reader: &mut BitReader<'_>,
    frame_width: u32,
    frame_height: u32,
) -> Result<RenderSize, DecoderError> {
    if reader.read_bool("render_and_frame_size_different")? {
        Ok(RenderSize {
            width: reader.read_bits(16, "render_width_minus_1")? + 1,
            height: reader.read_bits(16, "render_height_minus_1")? + 1,
        })
    } else {
        Ok(RenderSize {
            width: frame_width,
            height: frame_height,
        })
    }
}

fn parse_frame_header_trailing_params(
    reader: &mut BitReader<'_>,
    sequence: &SequenceHeader,
    allow_intrabc: bool,
    frame_is_intra: bool,
) -> Result<FrameHeaderTrailingParams, DecoderError> {
    let quantization = parse_quantization_params(reader, sequence)?;
    parse_segmentation_params(reader)?;
    let delta_q = parse_delta_q_params(reader, quantization.base_q_idx)?;
    let delta_lf = parse_delta_lf_params(reader, delta_q.present, allow_intrabc)?;
    let coded_lossless = quantization.coded_lossless();
    let loop_filter = parse_loop_filter_params(reader, sequence, coded_lossless, allow_intrabc)?;
    let cdef = parse_cdef_params(reader, sequence, coded_lossless, allow_intrabc)?;
    let restoration = parse_lr_params(reader, sequence, coded_lossless, allow_intrabc)?;
    let tx_mode = parse_tx_mode(reader, coded_lossless)?;
    let reduced_tx_set = reader.read_bool("reduced_tx_set")?;
    if !frame_is_intra {
        return Err(DecoderError::Unsupported(
            "inter frame header trailing parameters are not supported yet".to_string(),
        ));
    }
    Ok(FrameHeaderTrailingParams {
        quantization,
        delta_q,
        delta_lf,
        loop_filter,
        cdef,
        restoration,
        tx_mode,
        reduced_tx_set,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameHeaderTrailingParams {
    pub quantization: QuantizationParams,
    pub delta_q: DeltaQParams,
    pub delta_lf: DeltaLfParams,
    pub loop_filter: LoopFilterParams,
    pub cdef: CdefParams,
    pub restoration: RestorationParams,
    pub tx_mode: TxMode,
    pub reduced_tx_set: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizationParams {
    pub base_q_idx: u8,
    pub delta_q_y_dc: i8,
    pub delta_q_u_dc: i8,
    pub delta_q_u_ac: i8,
    pub delta_q_v_dc: i8,
    pub delta_q_v_ac: i8,
}

impl QuantizationParams {
    fn coded_lossless(&self) -> bool {
        self.base_q_idx == 0
            && self.delta_q_y_dc == 0
            && self.delta_q_u_dc == 0
            && self.delta_q_u_ac == 0
            && self.delta_q_v_dc == 0
            && self.delta_q_v_ac == 0
    }
}

fn parse_quantization_params(
    reader: &mut BitReader<'_>,
    sequence: &SequenceHeader,
) -> Result<QuantizationParams, DecoderError> {
    let base_q_idx = reader.read_bits(8, "base_q_idx")? as u8;
    let delta_q_y_dc = read_delta_q(reader, "delta_q_y_dc")?;
    let mut delta_q_u_dc = 0;
    let mut delta_q_u_ac = 0;
    let mut delta_q_v_dc = 0;
    let mut delta_q_v_ac = 0;

    if !sequence.color_config.monochrome {
        let diff_uv_delta = if sequence.color_config.separate_uv_delta_q {
            reader.read_bool("diff_uv_delta")?
        } else {
            false
        };
        delta_q_u_dc = read_delta_q(reader, "delta_q_u_dc")?;
        delta_q_u_ac = read_delta_q(reader, "delta_q_u_ac")?;
        if diff_uv_delta {
            delta_q_v_dc = read_delta_q(reader, "delta_q_v_dc")?;
            delta_q_v_ac = read_delta_q(reader, "delta_q_v_ac")?;
        } else {
            delta_q_v_dc = delta_q_u_dc;
            delta_q_v_ac = delta_q_u_ac;
        }
    }

    if reader.read_bool("using_qmatrix")? {
        let _qm_y = reader.read_bits(4, "qm_y")?;
        let _qm_u = reader.read_bits(4, "qm_u")?;
        if !sequence.color_config.separate_uv_delta_q {
            let _qm_v = reader.read_bits(4, "qm_v")?;
        }
    }

    Ok(QuantizationParams {
        base_q_idx,
        delta_q_y_dc,
        delta_q_u_dc,
        delta_q_u_ac,
        delta_q_v_dc,
        delta_q_v_ac,
    })
}

fn read_delta_q(reader: &mut BitReader<'_>, name: &str) -> Result<i8, DecoderError> {
    if reader.read_bool(name)? {
        let raw = reader.read_bits(7, name)? as u8;
        Ok(if raw & 0x40 != 0 {
            (raw as i16 - 128) as i8
        } else {
            raw as i8
        })
    } else {
        Ok(0)
    }
}

fn parse_segmentation_params(reader: &mut BitReader<'_>) -> Result<(), DecoderError> {
    if reader.read_bool("segmentation_enabled")? {
        return Err(DecoderError::Unsupported(
            "AV1 segmentation parameters are not supported yet".to_string(),
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeltaQParams {
    pub present: bool,
    pub res: u8,
}

fn parse_delta_q_params(
    reader: &mut BitReader<'_>,
    base_q_idx: u8,
) -> Result<DeltaQParams, DecoderError> {
    let present = base_q_idx > 0 && reader.read_bool("delta_q_present")?;
    let res = if present {
        reader.read_bits(2, "delta_q_res")? as u8
    } else {
        0
    };
    Ok(DeltaQParams { present, res })
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeltaLfParams {
    pub present: bool,
    pub res: u8,
    pub multi: bool,
}

fn parse_delta_lf_params(
    reader: &mut BitReader<'_>,
    delta_q_present: bool,
    allow_intrabc: bool,
) -> Result<DeltaLfParams, DecoderError> {
    let present = delta_q_present && !allow_intrabc && reader.read_bool("delta_lf_present")?;
    if !present {
        return Ok(DeltaLfParams::default());
    }
    Ok(DeltaLfParams {
        present,
        res: reader.read_bits(2, "delta_lf_res")? as u8,
        multi: reader.read_bool("delta_lf_multi")?,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopFilterParams {
    pub levels: [u8; 4],
    pub sharpness: u8,
    pub delta_enabled: bool,
    pub delta_update: bool,
}

impl Default for LoopFilterParams {
    fn default() -> Self {
        Self {
            levels: [0; 4],
            sharpness: 0,
            delta_enabled: false,
            delta_update: false,
        }
    }
}

fn parse_loop_filter_params(
    reader: &mut BitReader<'_>,
    sequence: &SequenceHeader,
    coded_lossless: bool,
    allow_intrabc: bool,
) -> Result<LoopFilterParams, DecoderError> {
    if coded_lossless || allow_intrabc {
        return Ok(LoopFilterParams::default());
    }
    let mut levels = [0u8; 4];
    levels[0] = reader.read_bits(6, "loop_filter_level[0]")? as u8;
    levels[1] = reader.read_bits(6, "loop_filter_level[1]")? as u8;
    if !sequence.color_config.monochrome {
        levels[2] = reader.read_bits(6, "loop_filter_level[2]")? as u8;
        levels[3] = reader.read_bits(6, "loop_filter_level[3]")? as u8;
    }
    let sharpness = reader.read_bits(3, "loop_filter_sharpness")? as u8;
    let delta_enabled = reader.read_bool("loop_filter_delta_enabled")?;
    let delta_update = delta_enabled && reader.read_bool("loop_filter_delta_update")?;
    if delta_update {
        return Err(DecoderError::Unsupported(
            "AV1 loop filter delta updates are not supported yet".to_string(),
        ));
    }
    Ok(LoopFilterParams {
        levels,
        sharpness,
        delta_enabled,
        delta_update,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CdefStrength {
    pub y_pri: u8,
    pub y_sec: u8,
    pub uv_pri: u8,
    pub uv_sec: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CdefParams {
    pub enabled: bool,
    pub damping: u8,
    pub bits: u8,
    pub strengths: [CdefStrength; 8],
}

impl Default for CdefParams {
    fn default() -> Self {
        Self {
            enabled: false,
            damping: 0,
            bits: 0,
            strengths: [CdefStrength {
                y_pri: 0,
                y_sec: 0,
                uv_pri: 0,
                uv_sec: 0,
            }; 8],
        }
    }
}

fn parse_cdef_params(
    reader: &mut BitReader<'_>,
    sequence: &SequenceHeader,
    coded_lossless: bool,
    allow_intrabc: bool,
) -> Result<CdefParams, DecoderError> {
    if !sequence.enable_cdef || coded_lossless || allow_intrabc {
        return Ok(CdefParams::default());
    }
    let damping = reader.read_bits(2, "cdef_damping_minus_3")? as u8 + 3;
    let cdef_bits = reader.read_bits(2, "cdef_bits")?;
    let mut strengths = CdefParams::default().strengths;
    for strength in strengths.iter_mut().take(1usize << cdef_bits) {
        strength.y_pri = reader.read_bits(4, "cdef_y_pri_strength")? as u8;
        strength.y_sec = reader.read_bits(2, "cdef_y_sec_strength")? as u8;
        if !sequence.color_config.monochrome {
            strength.uv_pri = reader.read_bits(4, "cdef_uv_pri_strength")? as u8;
            strength.uv_sec = reader.read_bits(2, "cdef_uv_sec_strength")? as u8;
        }
    }
    Ok(CdefParams {
        enabled: true,
        damping,
        bits: cdef_bits as u8,
        strengths,
    })
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RestorationParams {
    pub uses_lr: bool,
    pub lr_type: [u8; 3],
    pub unit_shift: u8,
}

fn parse_lr_params(
    reader: &mut BitReader<'_>,
    sequence: &SequenceHeader,
    coded_lossless: bool,
    allow_intrabc: bool,
) -> Result<RestorationParams, DecoderError> {
    if !sequence.enable_restoration || coded_lossless || allow_intrabc {
        return Ok(RestorationParams::default());
    }
    let planes = if sequence.color_config.monochrome {
        1
    } else {
        3
    };
    let mut uses_lr = false;
    let mut lr_type = [0u8; 3];
    for plane in 0..planes {
        let plane_lr_type = match reader.read_bits(2, "lr_type")? {
            0 => 0,
            1 => 3,
            2 => 1,
            3 => 2,
            _ => unreachable!("two-bit restoration type"),
        };
        lr_type[plane] = plane_lr_type;
        uses_lr |= plane_lr_type != 0;
    }
    let mut unit_shift = 0;
    if uses_lr {
        unit_shift = reader.read_bool("lr_unit_shift")? as u8;
        if !sequence.use_128x128_superblock && unit_shift == 1 {
            unit_shift += reader.read_bool("lr_unit_extra_shift")? as u8;
        }
    }
    Ok(RestorationParams {
        uses_lr,
        lr_type,
        unit_shift,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxMode {
    Only4x4,
    Largest,
    Select,
}

fn parse_tx_mode(reader: &mut BitReader<'_>, coded_lossless: bool) -> Result<TxMode, DecoderError> {
    if coded_lossless {
        return Ok(TxMode::Only4x4);
    }
    if reader.read_bool("tx_mode")? {
        Ok(TxMode::Select)
    } else {
        Ok(TxMode::Largest)
    }
}

fn frame_type_is_intra(frame_type: FrameType) -> bool {
    matches!(frame_type, FrameType::Key | FrameType::IntraOnly)
}
