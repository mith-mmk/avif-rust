use super::frame::{FrameHeader, TxMode};
use super::sequence::SequenceHeader;
use super::tile_group::TileGroup;
use crate::DecoderError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaneLayout {
    pub plane: u8,
    pub width: usize,
    pub height: usize,
    pub subsampling_x: u8,
    pub subsampling_y: u8,
    pub sample_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileDecodePlan {
    pub tile_id: u32,
    pub tile_col: u32,
    pub tile_row: u32,
    pub sb_col_start: u32,
    pub sb_col_end: u32,
    pub sb_row_start: u32,
    pub sb_row_end: u32,
    pub mi_col_start: u32,
    pub mi_col_end: u32,
    pub mi_row_start: u32,
    pub mi_row_end: u32,
    pub pixel_x: usize,
    pub pixel_y: usize,
    pub pixel_width: usize,
    pub pixel_height: usize,
    pub payload_offset: usize,
    pub payload_len: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameDecodePlan {
    pub width: usize,
    pub height: usize,
    pub render_width: usize,
    pub render_height: usize,
    pub bit_depth: u8,
    pub base_q_idx: u8,
    pub tx_mode: TxMode,
    pub superblock_size: usize,
    pub superblock_cols: u32,
    pub superblock_rows: u32,
    pub uses_cdef: bool,
    pub uses_restoration: bool,
    pub planes: Vec<PlaneLayout>,
    pub tiles: Vec<TileDecodePlan>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaneBuffer {
    pub layout: PlaneLayout,
    pub samples: Vec<u16>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameBuffers {
    pub width: usize,
    pub height: usize,
    pub planes: Vec<PlaneBuffer>,
}

pub fn build_still_decode_plan(
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    tile_group: &TileGroup,
) -> Result<FrameDecodePlan, DecoderError> {
    if frame.upscaled_width != frame.frame_width {
        return Err(DecoderError::Unsupported(
            "AV1 superres upscaling is not supported yet".to_string(),
        ));
    }
    if frame.frame_width == 0 || frame.frame_height == 0 {
        return Err(DecoderError::Bitstream(
            "AV1 frame dimensions must be non-zero".to_string(),
        ));
    }

    let width = usize::try_from(frame.frame_width)
        .map_err(|_| DecoderError::InvalidParam("AV1 frame width is too large".to_string()))?;
    let height = usize::try_from(frame.frame_height)
        .map_err(|_| DecoderError::InvalidParam("AV1 frame height is too large".to_string()))?;
    let superblock_size: usize = if sequence.use_128x128_superblock {
        128
    } else {
        64
    };
    let superblock_mi: usize = superblock_size / 4;
    let mi_cols = round_shift_usize(width, 2);
    let mi_rows = round_shift_usize(height, 2);
    let superblock_cols = round_shift_usize(mi_cols, superblock_mi.trailing_zeros() as u8) as u32;
    let superblock_rows = round_shift_usize(mi_rows, superblock_mi.trailing_zeros() as u8) as u32;
    let planes = build_plane_layouts(sequence, width, height)?;
    let tiles = build_tile_plans(frame, tile_group, width, height, superblock_mi as u32)?;

    Ok(FrameDecodePlan {
        width,
        height,
        render_width: usize::try_from(frame.render_width)
            .map_err(|_| DecoderError::InvalidParam("AV1 render width is too large".to_string()))?,
        render_height: usize::try_from(frame.render_height).map_err(|_| {
            DecoderError::InvalidParam("AV1 render height is too large".to_string())
        })?,
        bit_depth: sequence.color_config.bit_depth,
        base_q_idx: frame.base_q_idx,
        tx_mode: frame.tx_mode,
        superblock_size,
        superblock_cols,
        superblock_rows,
        uses_cdef: frame.cdef.enabled,
        uses_restoration: frame.restoration.uses_lr,
        planes,
        tiles,
    })
}

pub fn alloc_frame_buffers(plan: &FrameDecodePlan) -> Result<FrameBuffers, DecoderError> {
    if plan.bit_depth > 16 {
        return Err(DecoderError::Unsupported(format!(
            "AV1 {}-bit output buffers are not supported",
            plan.bit_depth
        )));
    }
    let mut planes = Vec::with_capacity(plan.planes.len());
    for layout in &plan.planes {
        planes.push(PlaneBuffer {
            layout: *layout,
            samples: vec![0; layout.sample_count],
        });
    }
    Ok(FrameBuffers {
        width: plan.width,
        height: plan.height,
        planes,
    })
}

fn build_plane_layouts(
    sequence: &SequenceHeader,
    width: usize,
    height: usize,
) -> Result<Vec<PlaneLayout>, DecoderError> {
    let plane_count = if sequence.color_config.monochrome {
        1
    } else {
        3
    };
    let mut planes = Vec::with_capacity(plane_count);
    for plane in 0..plane_count {
        let subsampling_x = if plane == 0 {
            0
        } else {
            sequence.color_config.subsampling_x as u8
        };
        let subsampling_y = if plane == 0 {
            0
        } else {
            sequence.color_config.subsampling_y as u8
        };
        let plane_width = round_shift_usize(width, subsampling_x);
        let plane_height = round_shift_usize(height, subsampling_y);
        let sample_count = plane_width.checked_mul(plane_height).ok_or_else(|| {
            DecoderError::InvalidParam("AV1 plane sample count overflow".to_string())
        })?;
        planes.push(PlaneLayout {
            plane: plane as u8,
            width: plane_width,
            height: plane_height,
            subsampling_x,
            subsampling_y,
            sample_count,
        });
    }
    Ok(planes)
}

fn build_tile_plans(
    frame: &FrameHeader,
    tile_group: &TileGroup,
    width: usize,
    height: usize,
    superblock_mi: u32,
) -> Result<Vec<TileDecodePlan>, DecoderError> {
    let tile_cols = frame.tile_info.tile_cols;
    if tile_cols == 0 || frame.tile_info.tile_rows == 0 {
        return Err(DecoderError::Bitstream(
            "AV1 tile grid must be non-zero".to_string(),
        ));
    }

    let mut plans = Vec::with_capacity(tile_group.tiles.len());
    for payload in &tile_group.tiles {
        let tile_col = payload.tile_id % tile_cols;
        let tile_row = payload.tile_id / tile_cols;
        let mi_col_start = *frame
            .tile_info
            .mi_col_starts
            .get(tile_col as usize)
            .ok_or_else(|| DecoderError::Bitstream("AV1 tile column is invalid".to_string()))?;
        let mi_col_end = *frame
            .tile_info
            .mi_col_starts
            .get(tile_col as usize + 1)
            .ok_or_else(|| DecoderError::Bitstream("AV1 tile column end is invalid".to_string()))?;
        let mi_row_start = *frame
            .tile_info
            .mi_row_starts
            .get(tile_row as usize)
            .ok_or_else(|| DecoderError::Bitstream("AV1 tile row is invalid".to_string()))?;
        let mi_row_end = *frame
            .tile_info
            .mi_row_starts
            .get(tile_row as usize + 1)
            .ok_or_else(|| DecoderError::Bitstream("AV1 tile row end is invalid".to_string()))?;
        let pixel_x = ((mi_col_start as usize) << 2).min(width);
        let pixel_y = ((mi_row_start as usize) << 2).min(height);
        let pixel_end_x = ((mi_col_end as usize) << 2).min(width);
        let pixel_end_y = ((mi_row_end as usize) << 2).min(height);
        plans.push(TileDecodePlan {
            tile_id: payload.tile_id,
            tile_col,
            tile_row,
            sb_col_start: mi_col_start / superblock_mi,
            sb_col_end: round_shift_u32(mi_col_end, superblock_mi.trailing_zeros() as u8),
            sb_row_start: mi_row_start / superblock_mi,
            sb_row_end: round_shift_u32(mi_row_end, superblock_mi.trailing_zeros() as u8),
            mi_col_start,
            mi_col_end,
            mi_row_start,
            mi_row_end,
            pixel_x,
            pixel_y,
            pixel_width: pixel_end_x.saturating_sub(pixel_x),
            pixel_height: pixel_end_y.saturating_sub(pixel_y),
            payload_offset: payload.offset,
            payload_len: payload.len,
        });
    }
    Ok(plans)
}

fn round_shift_usize(value: usize, shift: u8) -> usize {
    (value + ((1usize << shift) - 1)) >> shift
}

fn round_shift_u32(value: u32, shift: u8) -> u32 {
    (value + ((1u32 << shift) - 1)) >> shift
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av1::{parse_frame_header, parse_sequence_header, parse_tile_group};
    use crate::container::parse_avif;
    use crate::obu::{ObuType, find_obu_payload};

    #[test]
    fn builds_sample_still_decode_plan() {
        let data = std::fs::read(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .join("samples")
                .join("WML2Viewer.avif"),
        )
        .expect("sample AVIF should exist");
        let info = parse_avif(&data).unwrap();
        let sequence_payload =
            find_obu_payload(&info.primary_item_payload, ObuType::SequenceHeader)
                .unwrap()
                .expect("sequence header OBU should exist");
        let sequence = parse_sequence_header(sequence_payload).unwrap();
        let frame_payload = find_obu_payload(&info.primary_item_payload, ObuType::Frame)
            .unwrap()
            .expect("frame OBU should exist");
        let frame = parse_frame_header(frame_payload, &sequence).unwrap();
        let tile_group = parse_tile_group(
            frame_payload,
            frame.uncompressed_header_bits,
            &frame.tile_info,
        )
        .unwrap();

        let plan = build_still_decode_plan(&sequence, &frame, &tile_group).unwrap();

        assert_eq!(plan.width, 900);
        assert_eq!(plan.height, 900);
        assert_eq!(plan.bit_depth, 8);
        assert_eq!(plan.superblock_size, 128);
        assert_eq!(plan.superblock_cols, 8);
        assert_eq!(plan.superblock_rows, 8);
        assert_eq!(plan.planes.len(), 3);
        assert!(plan.planes.iter().all(|plane| plane.width == 900));
        assert!(plan.planes.iter().all(|plane| plane.height == 900));
        assert_eq!(plan.tiles.len(), 1);
        assert_eq!(plan.tiles[0].pixel_x, 0);
        assert_eq!(plan.tiles[0].pixel_y, 0);
        assert_eq!(plan.tiles[0].pixel_width, 900);
        assert_eq!(plan.tiles[0].pixel_height, 900);
        assert_eq!(plan.tiles[0].sb_col_start, 0);
        assert_eq!(plan.tiles[0].sb_col_end, 8);
        assert_eq!(plan.tiles[0].sb_row_start, 0);
        assert_eq!(plan.tiles[0].sb_row_end, 8);
        assert!(plan.tiles[0].payload_len > 0);
    }

    #[test]
    fn allocates_sample_frame_buffers_from_plan() {
        let data = std::fs::read(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .join("samples")
                .join("WML2Viewer.avif"),
        )
        .expect("sample AVIF should exist");
        let info = parse_avif(&data).unwrap();
        let sequence_payload =
            find_obu_payload(&info.primary_item_payload, ObuType::SequenceHeader)
                .unwrap()
                .expect("sequence header OBU should exist");
        let sequence = parse_sequence_header(sequence_payload).unwrap();
        let frame_payload = find_obu_payload(&info.primary_item_payload, ObuType::Frame)
            .unwrap()
            .expect("frame OBU should exist");
        let frame = parse_frame_header(frame_payload, &sequence).unwrap();
        let tile_group = parse_tile_group(
            frame_payload,
            frame.uncompressed_header_bits,
            &frame.tile_info,
        )
        .unwrap();
        let plan = build_still_decode_plan(&sequence, &frame, &tile_group).unwrap();

        let buffers = alloc_frame_buffers(&plan).unwrap();

        assert_eq!(buffers.planes.len(), 3);
        assert_eq!(buffers.planes[0].samples.len(), 900 * 900);
        assert_eq!(buffers.planes[1].samples.len(), 900 * 900);
        assert_eq!(buffers.planes[2].samples.len(), 900 * 900);
    }
}
