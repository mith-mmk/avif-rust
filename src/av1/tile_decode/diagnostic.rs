use super::PaletteBlockInfo;
use crate::av1::syntax::{BlockSize, Partition, PredictionMode, TxSize, TxType, UvPredictionMode};
use crate::av1::transform::TransformBlock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileEntropyState {
    pub tile_id: u32,
    pub payload_offset: usize,
    pub payload_len: usize,
    pub entropy_start_bits: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PartitionProbe {
    pub tile_id: u32,
    pub block_size: BlockSize,
    pub context: usize,
    pub symbol: usize,
    pub partition: Partition,
    pub bit_position_after: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockModeProbe {
    pub tile_id: u32,
    pub block_size: BlockSize,
    pub qindex: u8,
    pub skip_context: usize,
    pub skip_symbol: usize,
    pub skip: bool,
    pub cdef_idx: Option<u32>,
    pub y_above_context: usize,
    pub y_left_context: usize,
    pub y_mode_symbol: usize,
    pub y_mode: PredictionMode,
    pub angle_delta_y: Option<i8>,
    pub y_smooth_neighbour: bool,
    pub filter_intra_mode: Option<usize>,
    pub uv_mode_symbol: Option<usize>,
    pub uv_mode: Option<UvPredictionMode>,
    pub angle_delta_uv: Option<i8>,
    pub uv_smooth_neighbour: bool,
    pub palette: PaletteBlockInfo,
    pub tx_size_context: Option<usize>,
    pub tx_size_symbol: Option<usize>,
    pub tx_size: TxSize,
    pub bit_position_after: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidualProbe {
    pub tile_id: u32,
    pub block_size: BlockSize,
    pub skipped: bool,
    pub transform_count: usize,
    pub zero_transform_count: usize,
    pub first_tx_size: Option<TxSize>,
    pub first_non_zero_transform_index: Option<usize>,
    pub first_non_zero_transform: Option<TransformBlock>,
    pub first_non_zero_tx_size: Option<TxSize>,
    pub tx_type_read: bool,
    pub tx_type_set: Option<usize>,
    pub tx_type_symbol: Option<usize>,
    pub tx_type: Option<TxType>,
    pub txb_skip_context: Option<usize>,
    pub all_zero_symbol: Option<usize>,
    pub first_transform_all_zero: bool,
    pub eob_multisize: Option<usize>,
    pub eob_pt_symbol: Option<usize>,
    pub eob_pt: Option<usize>,
    pub eob_base: Option<usize>,
    pub eob_extra_context: Option<usize>,
    pub eob_extra_symbol: Option<usize>,
    pub eob_extra_literal_bits: Option<usize>,
    pub eob: Option<usize>,
    pub coeff_base_eob_context: Option<usize>,
    pub coeff_base_eob_symbol: Option<usize>,
    pub coeff_base_eob_level: Option<usize>,
    pub regular_coeff_base_count: Option<usize>,
    pub regular_coeff_base_decoded_count: Option<usize>,
    pub coeff_base_non_zero_count: Option<usize>,
    pub coeff_base_range_count: Option<usize>,
    pub coeff_br_decoded_count: Option<usize>,
    pub first_coeff_br_scan_index: Option<usize>,
    pub first_coeff_br_position: Option<usize>,
    pub first_coeff_br_context: Option<usize>,
    pub first_coeff_br_symbol: Option<usize>,
    pub first_coeff_br_level: Option<usize>,
    pub sign_decoded_count: Option<usize>,
    pub dc_sign_context: Option<usize>,
    pub dc_sign_symbol: Option<usize>,
    pub first_ac_sign_scan_index: Option<usize>,
    pub first_ac_sign_bit: Option<usize>,
    pub golomb_decoded_count: Option<usize>,
    pub first_golomb_scan_index: Option<usize>,
    pub first_golomb_value: Option<usize>,
    pub signed_coeff_non_zero_count: Option<usize>,
    pub first_signed_coeff_scan_index: Option<usize>,
    pub first_signed_coeff_position: Option<usize>,
    pub first_signed_coeff_value: Option<i32>,
    pub dequant_non_zero_count: Option<usize>,
    pub first_dequant_coeff_position: Option<usize>,
    pub first_dequant_coeff_value: Option<i32>,
    pub residual_preview_tx_type: Option<TxType>,
    pub residual_preview_sample_count: Option<usize>,
    pub first_residual_preview_sample: Option<i32>,
    pub first_coeff_base_scan_index: Option<usize>,
    pub first_coeff_base_position: Option<usize>,
    pub first_coeff_base_context: Option<usize>,
    pub first_coeff_base_reference_magnitude: Option<usize>,
    pub first_coeff_base_symbol: Option<usize>,
    pub first_coeff_base_level: Option<usize>,
    pub first_quantized_coefficients: Option<Vec<i32>>,
    pub bit_position_after: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedTransform {
    pub transform: TransformBlock,
    pub tx_type: TxType,
    pub coefficients: Vec<i32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedLumaBlock {
    pub x: usize,
    pub y: usize,
    pub block_size: BlockSize,
    pub palette: PaletteBlockInfo,
    pub transforms: Vec<DecodedTransform>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedBlockPrefix {
    pub blocks: Vec<DecodedLumaBlock>,
    pub next_unsupported: Option<crate::DecoderError>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CoeffBaseProbe {
    pub(super) remaining_count: usize,
    pub(super) decoded_count: usize,
    pub(super) scan_index: Option<usize>,
    pub(super) position: Option<usize>,
    pub(super) context: Option<usize>,
    pub(super) reference_magnitude: Option<usize>,
    pub(super) symbol: Option<usize>,
    pub(super) level: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct TxTypeProbe {
    pub(super) read: bool,
    pub(super) set: Option<usize>,
    pub(super) symbol: Option<usize>,
    pub(super) tx_type: TxType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CoeffBaseRead {
    pub(super) probe: CoeffBaseProbe,
    pub(super) base_levels: Vec<i32>,
    pub(super) non_zero_count: usize,
    pub(super) base_range_count: usize,
    pub(super) coeff_br_symbol_count: usize,
    pub(super) first_coeff_br: Option<CoeffBrProbe>,
    pub(super) signs: CoeffSignRead,
    pub(super) signed_non_zero_count: usize,
    pub(super) first_signed_coeff: Option<SignedCoeffProbe>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CoeffBrProbe {
    pub(super) scan_index: usize,
    pub(super) position: usize,
    pub(super) context: usize,
    pub(super) symbol: usize,
    pub(super) level_after_symbol: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CoeffSignRead {
    pub(super) sign_count: usize,
    pub(super) dc_sign_context: Option<usize>,
    pub(super) dc_sign_symbol: Option<usize>,
    pub(super) first_ac_sign_scan_index: Option<usize>,
    pub(super) first_ac_sign_bit: Option<usize>,
    pub(super) golomb_count: usize,
    pub(super) first_golomb_scan_index: Option<usize>,
    pub(super) first_golomb_value: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SignedCoeffProbe {
    pub(super) scan_index: usize,
    pub(super) position: usize,
    pub(super) value: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ResidualPreview {
    pub(super) tx_type: TxType,
    pub(super) dequant_non_zero_count: usize,
    pub(super) first_dequant_coeff: Option<DequantCoeffProbe>,
    pub(super) residual_sample_count: usize,
    pub(super) first_residual_sample: Option<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DequantCoeffProbe {
    pub(super) position: usize,
    pub(super) value: i32,
}
