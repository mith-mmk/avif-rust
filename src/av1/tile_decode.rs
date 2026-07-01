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
#[path = "tile_decode/tests/tile_decode_reconstruction.rs"]
mod reconstruction_tests;
