use super::cdf::CdfContext;
use super::entropy::EntropyDecoder;
use super::frame::{FrameHeader, RestorationParams};
use super::syntax::{BlockSize, TxSize, TxType};
use super::transform::{TransformBlock, coefficient_scan};
use crate::DecoderError;

mod block_syntax;
mod coefficient;
mod coefficient_context;
mod context_grid;
mod context_state;
mod decode_flow;
mod diagnostic;
mod palette;
mod partition_syntax;
mod public_api;
mod reconstruction;
mod residual_decode;
mod residual_preview;
mod residual_probe;
mod residual_state;
mod restoration_syntax;
mod syntax_helpers;
mod tx_type_syntax;

#[cfg(test)]
#[allow(unused_imports)]
use coefficient_context::{
    BR_CDF_SIZE, COEFF_BR_CDF_ROUNDS, COEFF_CONTEXT_BITS, COEFFICIENT_LEVEL_MASK,
    MAX_BASE_BR_RANGE, NUM_BASE_LEVELS, clamp_coefficient_level, coeff_base_context_1d,
    coeff_base_context_2d, coeff_base_eob_context, coeff_base_non_zero_count, coeff_br_context_1d,
    coeff_br_context_2d, eob_base_from_pt, eob_multisize, eob_tx_class_context, first_signed_coeff,
};
use coefficient_context::{
    TxbContext, coefficient_entropy_context, set_txb_entropy_context, txb_context,
};
pub use diagnostic::{
    BlockModeProbe, DecodedBlockPrefix, DecodedLumaBlock, DecodedTransform, PartitionProbe,
    ResidualProbe, TileEntropyState,
};
use diagnostic::{CoeffBaseProbe, CoeffBaseRead, CoeffBrProbe, CoeffSignRead};
pub use public_api::{
    decode_first_luma_block, decode_first_luma_transform, decode_luma_root_block_prefix,
    decode_luma_root_blocks, prepare_tile_entropy, probe_first_block_residuals,
    probe_tile_block_modes, probe_tile_partitions,
};

#[derive(Debug, Clone, PartialEq, Eq)]
struct PlaneEntropyContexts {
    above: Vec<u8>,
    left: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PalettePlaneInfo {
    colors: Vec<u16>,
    color_map: Vec<u8>,
    map_width: usize,
    map_height: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PaletteBlockInfo {
    y: Option<PalettePlaneInfo>,
    uv: Option<PalettePlaneInfo>,
}

pub struct TileDecoder<'a> {
    reader: EntropyDecoder<'a>,
    cdf: CdfContext,
    mi_cols: usize,
    mi_rows: usize,
    y_mode_grid: Vec<Option<usize>>,
    y_palette_size_grid: Vec<Option<usize>>,
    uv_palette_size_grid: Vec<Option<usize>>,
    y_palette_colors_grid: Vec<Option<Vec<u16>>>,
    u_palette_colors_grid: Vec<Option<Vec<u16>>>,
    y_smooth_grid: Vec<Option<bool>>,
    uv_smooth_grid: Vec<Option<bool>>,
    skip_grid: Vec<Option<bool>>,
    above_partition_context: Vec<u8>,
    left_partition_context: Vec<u8>,
    cdef_transmitted: [bool; 4],
    above_txfm_context: Vec<usize>,
    left_txfm_context: Vec<usize>,
    plane_entropy_contexts: [PlaneEntropyContexts; 3],
    restoration: RestorationParams,
    wiener_refs: [[[i16; 3]; 2]; 3],
    sgrproj_refs: [[i16; 2]; 3],
}

impl<'a> TileDecoder<'a> {
    pub fn new(payload: &'a [u8], frame: &FrameHeader) -> Result<Self, DecoderError> {
        let mi_cols = (usize::try_from(frame.frame_width)
            .map_err(|_| DecoderError::InvalidParam("AV1 frame width is too large".to_string()))?
            + 3)
            >> 2;
        let mi_rows = (usize::try_from(frame.frame_height).map_err(|_| {
            DecoderError::InvalidParam("AV1 frame height is too large".to_string())
        })? + 3)
            >> 2;
        Ok(Self {
            reader: EntropyDecoder::new(payload, frame.disable_cdf_update)?,
            cdf: CdfContext::new(frame.base_q_idx),
            mi_cols,
            mi_rows,
            y_mode_grid: vec![None; mi_cols * mi_rows],
            y_palette_size_grid: vec![None; mi_cols * mi_rows],
            uv_palette_size_grid: vec![None; mi_cols * mi_rows],
            y_palette_colors_grid: vec![None; mi_cols * mi_rows],
            u_palette_colors_grid: vec![None; mi_cols * mi_rows],
            y_smooth_grid: vec![None; mi_cols * mi_rows],
            uv_smooth_grid: vec![None; mi_cols * mi_rows],
            skip_grid: vec![None; mi_cols * mi_rows],
            above_partition_context: vec![0; mi_cols],
            left_partition_context: vec![0; mi_rows],
            cdef_transmitted: [false; 4],
            above_txfm_context: vec![0; mi_cols],
            left_txfm_context: vec![0; mi_rows],
            plane_entropy_contexts: std::array::from_fn(|_| PlaneEntropyContexts {
                above: vec![0; mi_cols],
                left: vec![0; mi_rows],
            }),
            restoration: frame.restoration,
            wiener_refs: [[[3, -7, 15]; 2]; 3],
            sgrproj_refs: [[-32, 31]; 3],
        })
    }

    pub(super) fn txb_context(
        &self,
        block_size: BlockSize,
        transform: TransformBlock,
    ) -> TxbContext {
        let contexts = &self.plane_entropy_contexts[transform.plane];
        txb_context(block_size, transform, &contexts.above, &contexts.left)
    }

    pub(super) fn set_txb_entropy_context(&mut self, transform: TransformBlock, value: u8) {
        let contexts = &mut self.plane_entropy_contexts[transform.plane];
        set_txb_entropy_context(transform, value, &mut contexts.above, &mut contexts.left);
    }
}

#[cfg(test)]
#[path = "tests/tile_decode_coeff.rs"]
mod coeff_tests;

#[cfg(test)]
#[path = "tile_decode/tests/tile_decode_residual.rs"]
mod residual_tests;

#[cfg(test)]
#[path = "tile_decode/tests/tile_decode_block_syntax.rs"]
mod block_syntax_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av1::{
        alloc_frame_buffers, build_still_decode_plan, parse_frame_header, parse_sequence_header,
        parse_tile_group,
    };
    use crate::container::parse_avif;
    use crate::obu::{ObuType, find_obu_payload};

    #[test]
    fn prepares_sample_tile_entropy_state() {
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

        let states = prepare_tile_entropy(frame_payload, &tile_group, &frame).unwrap();

        assert_eq!(states.len(), 1);
        assert_eq!(states[0].tile_id, 0);
        assert_eq!(states[0].entropy_start_bits, 15);
        assert!(states[0].payload_len > 0);
    }

    #[test]
    fn decodes_sample_first_luma_transform_into_frame_buffer() {
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
        let mut buffers = alloc_frame_buffers(&plan).unwrap();
        buffers.planes[0].samples.fill(u16::MAX);

        let residual = decode_first_luma_transform(
            frame_payload,
            &tile_group,
            &sequence,
            &frame,
            &plan,
            &mut buffers,
        )
        .unwrap();

        assert_eq!(residual.tile_id, 0);
        assert!(residual.first_tx_size.is_some());
        if residual.first_non_zero_transform.is_none() {
            assert_eq!(residual.zero_transform_count, residual.transform_count);
        } else {
            assert!(
                buffers.planes[0]
                    .samples
                    .iter()
                    .any(|sample| *sample != u16::MAX)
            );
        }
    }

    #[test]
    fn decodes_sample_first_luma_block_transforms_into_frame_buffer() {
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
        let mut buffers = alloc_frame_buffers(&plan).unwrap();
        for plane in &mut buffers.planes {
            plane.samples.fill(u16::MAX);
        }

        let decoded = decode_first_luma_block(
            frame_payload,
            &tile_group,
            &sequence,
            &frame,
            &plan,
            &mut buffers,
        )
        .unwrap();

        assert!(
            decoded
                .iter()
                .all(|transform| transform.transform.plane == 0)
        );
        assert!(
            buffers
                .planes
                .iter()
                .all(|plane| { plane.samples.iter().any(|sample| *sample != u16::MAX) })
        );
    }

    #[test]
    fn decodes_sample_luma_root_block_prefix_with_split_children() {
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
        let mut buffers = alloc_frame_buffers(&plan).unwrap();

        let prefix = decode_luma_root_block_prefix(
            frame_payload,
            &tile_group,
            &sequence,
            &frame,
            &plan,
            &mut buffers,
            8,
        )
        .unwrap();
        let blocks = prefix.blocks;

        assert_eq!(blocks.len(), 8);
        assert_eq!((blocks[0].x, blocks[0].y), (0, 0));
        assert_eq!((blocks[1].x, blocks[1].y), (64, 0));
        assert!(blocks.iter().any(|block| !block.transforms.is_empty()));
        assert!(buffers.planes[1].samples.iter().any(|sample| *sample != 0));
        assert!(buffers.planes[2].samples.iter().any(|sample| *sample != 0));
        assert_eq!(prefix.next_unsupported, None);
    }

    #[test]
    fn decodes_sample_prefix_through_palette_blocks() {
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
        let mut buffers = alloc_frame_buffers(&plan).unwrap();

        let prefix = decode_luma_root_block_prefix(
            frame_payload,
            &tile_group,
            &sequence,
            &frame,
            &plan,
            &mut buffers,
            4096,
        )
        .unwrap();

        assert_eq!(prefix.blocks.len(), 2037);
        assert_eq!(prefix.next_unsupported, None);
        assert!(
            buffers
                .planes
                .iter()
                .all(|plane| { plane.samples.iter().any(|sample| *sample != 0) })
        );
    }

    #[test]
    fn coeff_base_context_2d_matches_square_offset_rules() {
        let mut quant = vec![0; super::super::syntax::TxSize::Tx32x32.sample_count()];

        assert_eq!(
            coeff_base_context_2d(super::super::syntax::TxSize::Tx32x32, 0, &quant).unwrap(),
            (0, 0)
        );

        quant[2] = 3;
        assert_eq!(
            coeff_base_context_2d(super::super::syntax::TxSize::Tx32x32, 1, &quant).unwrap(),
            (3, 3)
        );

        assert_eq!(
            coeff_base_context_2d(super::super::syntax::TxSize::Tx32x32, 4 * 32 + 4, &quant)
                .unwrap(),
            (21, 0)
        );
    }

    #[test]
    fn txb_context_uses_neighbor_levels_and_dc_signs() {
        let transform = TransformBlock {
            plane: 0,
            x: 4,
            y: 4,
            tx_size: TxSize::Tx4x4,
        };
        let mut above = vec![0; 8];
        let mut left = vec![0; 8];

        assert_eq!(
            txb_context(BlockSize::Block8x8, transform, &above, &left),
            TxbContext {
                skip: 1,
                dc_sign: 0
            }
        );

        above[1] = 4 | (2 << COEFF_CONTEXT_BITS);
        left[1] = 2 | (1 << COEFF_CONTEXT_BITS);
        assert_eq!(
            txb_context(BlockSize::Block8x8, transform, &above, &left),
            TxbContext {
                skip: 5,
                dc_sign: 0
            }
        );

        left[1] = 0;
        assert_eq!(
            txb_context(BlockSize::Block8x8, transform, &above, &left),
            TxbContext {
                skip: 3,
                dc_sign: 2
            }
        );
    }

    #[test]
    fn chroma_txb_skip_context_uses_non_zero_neighbors_and_block_area() {
        let transform = TransformBlock {
            plane: 1,
            x: 0,
            y: 0,
            tx_size: TxSize::Tx4x4,
        };
        assert_eq!(
            txb_context(BlockSize::Block4x4, transform, &[1], &[0]).skip,
            8
        );
        assert_eq!(
            txb_context(BlockSize::Block8x8, transform, &[1], &[1]).skip,
            12
        );
    }

    #[test]
    fn coefficient_entropy_context_caps_level_and_encodes_dc_sign() {
        assert_eq!(coefficient_entropy_context(&[0, 0]), 0);
        assert_eq!(coefficient_entropy_context(&[2, 3]), 5 | 16);
        assert_eq!(coefficient_entropy_context(&[-10, 4]), 7 | 8);

        let transform = TransformBlock {
            plane: 0,
            x: 4,
            y: 8,
            tx_size: TxSize::Tx8x8,
        };
        let mut above = vec![0; 8];
        let mut left = vec![0; 8];
        set_txb_entropy_context(transform, 23, &mut above, &mut left);
        assert_eq!(&above[1..3], &[23, 23]);
        assert_eq!(&left[2..4], &[23, 23]);
    }

    #[test]
    fn eob_context_distinguishes_2d_and_directional_transforms() {
        assert_eq!(eob_tx_class_context(TxType::DctDct), 0);
        assert_eq!(eob_tx_class_context(TxType::Identity), 0);
        assert_eq!(eob_tx_class_context(TxType::VerticalDct), 1);
        assert_eq!(eob_tx_class_context(TxType::HorizontalDct), 1);
    }

    #[test]
    fn coeff_br_context_2d_matches_square_tx_rules() {
        let mut quant = vec![0; super::super::syntax::TxSize::Tx32x32.sample_count()];

        assert_eq!(
            coeff_br_context_2d(super::super::syntax::TxSize::Tx32x32, 0, &quant).unwrap(),
            0
        );

        quant[1] = 3;
        assert_eq!(
            coeff_br_context_2d(super::super::syntax::TxSize::Tx32x32, 0, &quant).unwrap(),
            2
        );

        assert_eq!(
            coeff_br_context_2d(super::super::syntax::TxSize::Tx32x32, 32 + 1, &quant).unwrap(),
            7
        );

        assert_eq!(
            coeff_br_context_2d(super::super::syntax::TxSize::Tx32x32, 4 * 32 + 4, &quant).unwrap(),
            14
        );
    }

    #[test]
    fn directional_coefficient_contexts_follow_aom_1d_axes() {
        let tx_size = super::super::syntax::TxSize::Tx8x8;
        let mut quant = vec![0; tx_size.sample_count()];
        quant[2] = 3;
        quant[16] = 2;

        assert_eq!(
            coeff_base_context_1d(tx_size, TxType::VerticalDct, 1, &quant).unwrap(),
            (33, 3)
        );
        assert_eq!(
            coeff_base_context_1d(tx_size, TxType::HorizontalDct, 0, &quant).unwrap(),
            (0, 2)
        );
        assert_eq!(
            coeff_br_context_1d(tx_size, TxType::VerticalDct, 0, &quant).unwrap(),
            2
        );
        assert_eq!(
            coeff_br_context_1d(tx_size, TxType::HorizontalDct, 8, &quant).unwrap(),
            15
        );
    }

    #[test]
    fn coefficient_level_is_clamped_to_av1_twenty_bit_range() {
        assert_eq!(clamp_coefficient_level(0), 0);
        assert_eq!(clamp_coefficient_level(COEFFICIENT_LEVEL_MASK), 0x0f_ffff);
        assert_eq!(clamp_coefficient_level(1 << 20), 0);
        assert_eq!(clamp_coefficient_level((1 << 20) + 7), 7);
    }
}
