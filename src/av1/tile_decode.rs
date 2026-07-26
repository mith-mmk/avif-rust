use super::cdf::CdfContext;
use super::decode::FrameBuffers;
use super::entropy::EntropyDecoder;
use super::frame::{FrameHeader, InterpolationFilter, RestorationParams};
use super::sequence::SequenceHeader;
use super::syntax::{BlockSize, TxSize, TxType, mi_dimension};
use super::transform::TransformBlock;
use crate::DecoderError;
use std::sync::Arc;

mod block_syntax;
mod coefficient;
mod coefficient_context;
mod context_grid;
mod context_state;
mod decode_flow;
mod diagnostic;
mod palette;
mod partition_syntax;
mod post_filter_state;
#[cfg(test)]
pub(crate) use post_filter_state::RestorationUnit;
#[cfg(test)]
pub(crate) use post_filter_state::wiener_filter_unit;
pub(crate) use post_filter_state::{
    BlockFilterState, PostFilterState, cdef_adjust_primary_strength, cdef_chroma_direction,
    cdef_filter_block_region_with_edge_mode_into_bit_depth_visible_scaled,
    cdef_find_direction_with_variance_visible, deblock_filter_edge_with_visible_bounds,
    sgrproj_filter_unit_into_with_scratch_bit_depth_visible,
    wiener_filter_unit_into_with_scratch_bit_depth_visible,
};
mod public_api;
mod reconstruction;
mod reconstruction_coverage;
mod residual_decode;
mod residual_preview;
mod residual_probe;
mod residual_state;
mod restoration_syntax;
mod syntax_helpers;
mod tx_type_syntax;
mod warped_filter;

use coefficient::CoefficientScanCache;
use coefficient_context::{
    TxbContext, coefficient_entropy_context, set_txb_entropy_context, txb_context,
};
pub use diagnostic::{
    BlockModeProbe, CompoundMask, DecodedBlockPrefix, DecodedLumaBlock, DecodedTransform,
    InterIntraMode, LocalWarpSample, MotionMode, PartitionProbe, ResidualProbe, TileEntropyState,
};
use diagnostic::{CoeffBaseProbe, CoeffBaseRead, CoeffBrProbe, CoeffSignRead};
#[cfg(test)]
pub(crate) use public_api::decode_luma_root_block_prefix_with_post_filter_state_and_entropy;
#[cfg(test)]
pub(crate) use public_api::decode_luma_root_block_prefix_with_post_filter_state_and_entropy_options;
pub(crate) use public_api::decode_luma_root_block_prefix_with_post_filter_state_and_entropy_options_with_references_and_cdf_and_motion;
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

impl PalettePlaneInfo {
    pub fn colors(&self) -> &[u16] {
        &self.colors
    }

    pub fn color_map(&self) -> &[u8] {
        &self.color_map
    }

    pub fn map_width(&self) -> usize {
        self.map_width
    }

    pub fn map_height(&self) -> usize {
        self.map_height
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PaletteBlockInfo {
    y: Option<PalettePlaneInfo>,
    uv: Option<PalettePlaneInfo>,
}

/// Motion vectors and reference slots retained for AV1 temporal MV
/// prediction. Entries are stored at 4x4-MI granularity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MotionField {
    pub(crate) mi_cols: usize,
    pub(crate) mi_rows: usize,
    pub(crate) reference_frames: Vec<Option<u8>>,
    pub(crate) motion_vectors: Vec<Option<(i32, i32)>>,
}

impl MotionField {
    pub(crate) fn empty(mi_cols: usize, mi_rows: usize) -> Self {
        let count = mi_cols.saturating_mul(mi_rows);
        Self {
            mi_cols,
            mi_rows,
            reference_frames: vec![None; count],
            motion_vectors: vec![None; count],
        }
    }

    pub(crate) fn merge(&mut self, tile: Self) {
        if self.mi_cols != tile.mi_cols || self.mi_rows != tile.mi_rows {
            return;
        }
        for (destination, source) in self.reference_frames.iter_mut().zip(tile.reference_frames) {
            if source.is_some() {
                *destination = source;
            }
        }
        for (destination, source) in self.motion_vectors.iter_mut().zip(tile.motion_vectors) {
            if source.is_some() {
                *destination = source;
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CflParams {
    alpha_u_q3: i8,
    alpha_v_q3: i8,
}

impl PaletteBlockInfo {
    pub fn y(&self) -> Option<&PalettePlaneInfo> {
        self.y.as_ref()
    }

    pub fn uv(&self) -> Option<&PalettePlaneInfo> {
        self.uv.as_ref()
    }

    pub fn has_palette(&self) -> bool {
        self.y.is_some() || self.uv.is_some()
    }

    pub fn has_non_empty_color_map(&self) -> bool {
        self.y
            .as_ref()
            .is_some_and(|palette| !palette.color_map.is_empty())
            || self
                .uv
                .as_ref()
                .is_some_and(|palette| !palette.color_map.is_empty())
    }
}

pub struct TileDecoder<'a> {
    reader: EntropyDecoder<'a>,
    cdf: CdfContext,
    mi_cols: usize,
    mi_rows: usize,
    tile_mi_col_start: usize,
    tile_mi_row_start: usize,
    y_mode_grid: Vec<Option<usize>>,
    is_inter_grid: Vec<Option<bool>>,
    reference_frame_grid: Vec<Option<u8>>,
    reference_frame_type_grid: Vec<Option<u8>>,
    inter_new_mv_grid: Vec<Option<bool>>,
    motion_vector_grid: Vec<Option<(i32, i32)>>,
    interpolation_filter_grid: Vec<Option<(InterpolationFilter, InterpolationFilter)>>,
    motion_block_size_grid: Vec<Option<BlockSize>>,
    reference_frame_secondary_grid: Vec<Option<u8>>,
    reference_frame_secondary_type_grid: Vec<Option<u8>>,
    motion_vector_secondary_grid: Vec<Option<(i32, i32)>>,
    intra_bc_mv_grid: Vec<Option<(i32, i32)>>,
    y_palette_size_grid: Vec<Option<usize>>,
    uv_palette_size_grid: Vec<Option<usize>>,
    y_palette_colors_grid: Vec<Option<Vec<u16>>>,
    u_palette_colors_grid: Vec<Option<Vec<u16>>>,
    y_smooth_grid: Vec<Option<bool>>,
    uv_smooth_grid: Vec<Option<bool>>,
    skip_grid: Vec<Option<bool>>,
    skip_mode_grid: Vec<Option<bool>>,
    segmentation_map: Vec<u8>,
    above_partition_context: Vec<u8>,
    left_partition_context: Vec<u8>,
    cdef_transmitted: [bool; 4],
    above_txfm_context: Vec<usize>,
    left_txfm_context: Vec<usize>,
    reconstructed_mi_grid: [Vec<bool>; 3],
    current_cfl: Option<CflParams>,
    plane_entropy_contexts: [PlaneEntropyContexts; 3],
    plane_entropy_contexts_configured: bool,
    plane_subsampling_x: [usize; 3],
    plane_subsampling_y: [usize; 3],
    restoration: RestorationParams,
    wiener_refs: [[[i16; 3]; 2]; 3],
    sgrproj_refs: [[i16; 2]; 3],
    cdef_units: Vec<post_filter_state::CdefUnit>,
    cdef_blocks: Vec<post_filter_state::CdefBlockIndex>,
    transform_boundaries: Vec<post_filter_state::TransformBoundary>,
    restoration_units: Vec<post_filter_state::RestorationUnit>,
    block_filter_states: Vec<post_filter_state::BlockFilterState>,
    coefficient_scratch: Vec<i32>,
    dequant_scratch: [i32; 64 * 64],
    residual_scratch: [i32; 64 * 64],
    prediction_scratch: [u16; 64 * 64],
    inter_intra_scratch: [u16; 64 * 64],
    reconstruction_scratch: [u16; 64 * 64],
    coefficient_scan_cache: CoefficientScanCache,
    current_qindex: u8,
    current_delta_lf: [i8; 4],
    reference_buffers: [Option<Arc<FrameBuffers>>; 8],
    temporal_motion_field: Option<Arc<MotionField>>,
}

pub(super) fn is_chroma_reference(
    sequence: &SequenceHeader,
    block_size: BlockSize,
    x: usize,
    y: usize,
) -> bool {
    if sequence.color_config.monochrome {
        return false;
    }
    let mi_col = x / 4;
    let mi_row = y / 4;
    let block_mi_width = block_size.width() / 4;
    let block_mi_height = block_size.height() / 4;
    (!mi_row.is_multiple_of(2)
        || block_mi_height.is_multiple_of(2)
        || !sequence.color_config.subsampling_y)
        && (!mi_col.is_multiple_of(2)
            || block_mi_width.is_multiple_of(2)
            || !sequence.color_config.subsampling_x)
}

impl<'a> TileDecoder<'a> {
    pub fn new(payload: &'a [u8], frame: &FrameHeader) -> Result<Self, DecoderError> {
        Self::new_with_references(payload, frame, std::array::from_fn(|_| None))
    }

    pub fn new_with_references(
        payload: &'a [u8],
        frame: &FrameHeader,
        reference_buffers: [Option<Arc<FrameBuffers>>; 8],
    ) -> Result<Self, DecoderError> {
        Self::new_with_references_and_cdf(payload, frame, reference_buffers, None)
    }

    pub fn new_with_references_and_cdf(
        payload: &'a [u8],
        frame: &FrameHeader,
        reference_buffers: [Option<Arc<FrameBuffers>>; 8],
        initial_cdf: Option<CdfContext>,
    ) -> Result<Self, DecoderError> {
        let mi_cols = usize::try_from(mi_dimension(frame.frame_width))
            .map_err(|_| DecoderError::InvalidParam("AV1 frame width is too large".to_string()))?;
        let mi_rows = usize::try_from(mi_dimension(frame.frame_height))
            .map_err(|_| DecoderError::InvalidParam("AV1 frame height is too large".to_string()))?;
        let mi_count = mi_cols.checked_mul(mi_rows).ok_or_else(|| {
            DecoderError::InvalidParam("AV1 frame dimensions are too large".to_string())
        })?;
        // Post-filter metadata is recorded once per decoded block/unit. Reserve
        // the usual frame-scale capacity up front so large tiled frames do not
        // repeatedly grow these vectors while reconstruction is in progress;
        // caps keep very large resource-limit-sized frames from over-reserving.
        let block_filter_capacity = mi_count.div_ceil(4).min(32_768);
        let cdef_capacity = mi_count.div_ceil(256).min(4_096);
        Ok(Self {
            reader: EntropyDecoder::new(payload, frame.disable_cdf_update)?,
            cdf: initial_cdf.unwrap_or_else(|| CdfContext::new(frame.base_q_idx)),
            mi_cols,
            mi_rows,
            tile_mi_col_start: 0,
            tile_mi_row_start: 0,
            y_mode_grid: vec![None; mi_count],
            is_inter_grid: vec![None; mi_count],
            reference_frame_grid: vec![None; mi_count],
            reference_frame_type_grid: vec![None; mi_count],
            inter_new_mv_grid: vec![None; mi_count],
            motion_vector_grid: vec![None; mi_count],
            interpolation_filter_grid: vec![None; mi_count],
            motion_block_size_grid: vec![None; mi_count],
            reference_frame_secondary_grid: vec![None; mi_count],
            reference_frame_secondary_type_grid: vec![None; mi_count],
            motion_vector_secondary_grid: vec![None; mi_count],
            intra_bc_mv_grid: vec![None; mi_count],
            y_palette_size_grid: vec![None; mi_count],
            uv_palette_size_grid: vec![None; mi_count],
            y_palette_colors_grid: vec![None; mi_count],
            u_palette_colors_grid: vec![None; mi_count],
            y_smooth_grid: vec![None; mi_count],
            uv_smooth_grid: vec![None; mi_count],
            skip_grid: vec![None; mi_count],
            skip_mode_grid: vec![None; mi_count],
            segmentation_map: vec![0; mi_count],
            above_partition_context: vec![0; mi_cols],
            left_partition_context: vec![0; mi_rows],
            cdef_transmitted: [false; 4],
            above_txfm_context: vec![0; mi_cols],
            left_txfm_context: vec![64; mi_rows],
            reconstructed_mi_grid: std::array::from_fn(|_| vec![false; mi_count]),
            current_cfl: None,
            plane_entropy_contexts: std::array::from_fn(|_| PlaneEntropyContexts {
                above: vec![0; mi_cols],
                left: vec![0; mi_rows],
            }),
            plane_entropy_contexts_configured: false,
            plane_subsampling_x: [0; 3],
            plane_subsampling_y: [0; 3],
            restoration: frame.restoration,
            // Chroma Wiener restoration uses the reduced 5-tap window, so
            // its outer coefficient is implicit zero and is not signaled.
            wiener_refs: [[[3, -7, 15]; 2], [[0, -7, 15]; 2], [[0, -7, 15]; 2]],
            sgrproj_refs: [[-32, 31]; 3],
            cdef_units: Vec::with_capacity(cdef_capacity),
            cdef_blocks: Vec::with_capacity(cdef_capacity),
            transform_boundaries: Vec::with_capacity(block_filter_capacity),
            restoration_units: Vec::new(),
            block_filter_states: Vec::with_capacity(block_filter_capacity),
            coefficient_scratch: Vec::with_capacity(TxSize::Tx64x64.sample_count()),
            dequant_scratch: [0; 64 * 64],
            residual_scratch: [0; 64 * 64],
            prediction_scratch: [0; 64 * 64],
            inter_intra_scratch: [0; 64 * 64],
            reconstruction_scratch: [0; 64 * 64],
            coefficient_scan_cache: CoefficientScanCache::new(),
            current_qindex: frame.segmentation.effective_qindex(frame.base_q_idx),
            current_delta_lf: [0; 4],
            reference_buffers,
            temporal_motion_field: None,
        })
    }

    pub(crate) fn cdf_snapshot(&self) -> CdfContext {
        self.cdf.clone()
    }

    pub(super) fn reference_buffer(&self, slot: u8) -> Result<Arc<FrameBuffers>, DecoderError> {
        self.reference_buffers
            .get(usize::from(slot))
            .and_then(Option::as_ref)
            .map(Arc::clone)
            .ok_or_else(|| {
                DecoderError::Unsupported(format!("AV1 inter reference slot {slot} is unavailable"))
            })
    }

    pub(super) fn set_tile_bounds(&mut self, tile: &crate::av1::TileDecodePlan) {
        self.tile_mi_col_start = tile.mi_col_start as usize;
        self.tile_mi_row_start = tile.mi_row_start as usize;
    }

    pub(super) fn set_temporal_motion_field(&mut self, field: Option<Arc<MotionField>>) {
        self.temporal_motion_field = field;
    }

    pub(super) fn motion_field(&self) -> MotionField {
        MotionField {
            mi_cols: self.mi_cols,
            mi_rows: self.mi_rows,
            reference_frames: self.reference_frame_grid.clone(),
            motion_vectors: self.motion_vector_grid.clone(),
        }
    }

    pub(super) fn read_segmentation_id(
        &mut self,
        frame: &FrameHeader,
        block_size: BlockSize,
        x: usize,
        y: usize,
        skip: bool,
    ) -> Result<u8, DecoderError> {
        let max_segment = frame.segmentation.last_active_segment;
        if !frame.segmentation.enabled || !frame.segmentation.update_map {
            return Ok(0);
        }
        let mi_x = x / 4;
        let mi_y = y / 4;
        let have_left = mi_x > self.tile_mi_col_start;
        let have_top = mi_y > self.tile_mi_row_start;
        let left = have_left.then(|| self.segmentation_map[mi_y * self.mi_cols + mi_x - 1]);
        let top = have_top.then(|| self.segmentation_map[(mi_y - 1) * self.mi_cols + mi_x]);
        let above_left = (have_left && have_top)
            .then(|| self.segmentation_map[(mi_y - 1) * self.mi_cols + mi_x - 1]);
        let (predicted, context) = match (left, top, above_left) {
            (Some(left), Some(top), Some(above_left)) => {
                let context = if left == top && top == above_left {
                    2
                } else if left == top || top == above_left || left == above_left {
                    1
                } else {
                    0
                };
                (if top == above_left { top } else { left }, context)
            }
            (Some(left), _, _) => (left, 0),
            (_, Some(top), _) => (top, 0),
            _ => (0, 0),
        };
        let max_count = usize::from(max_segment) + 1;
        let segment_id = if skip {
            predicted
        } else {
            // The syntax always codes the segment ID with the full
            // MAX_SEGMENTS CDF. `last_active_segment` limits the valid result
            // after inverse deinterleaving; it does not shorten the entropy
            // alphabet or move the terminal CDF boundary.
            let diff = self.reader.read_symbol(self.cdf.seg_id_cdf_mut(context))? as u8;
            neg_deinterleave(diff, predicted, max_count as u8)
        };
        let segment_id = if usize::from(segment_id) < max_count {
            segment_id
        } else {
            0
        };
        self.set_segmentation_id(block_size, x, y, segment_id);
        self.current_qindex = frame
            .segmentation
            .effective_qindex_for_segment(frame.base_q_idx, segment_id);
        Ok(segment_id)
    }

    pub(super) fn read_skip_mode(
        &mut self,
        frame: &FrameHeader,
        block_size: BlockSize,
        x: usize,
        y: usize,
        segment_id: u8,
    ) -> Result<bool, DecoderError> {
        let allowed = frame.skip_mode_present
            && block_size.width() >= 8
            && block_size.height() >= 8
            && !frame.segmentation.segment_skip[usize::from(segment_id)];
        let skip_mode = if allowed {
            let context = self.skip_mode_context(x, y);
            self.reader
                .read_symbol(self.cdf.skip_mode_cdf_mut(context))?
                != 0
        } else {
            false
        };
        self.set_skip_mode(block_size, x, y, skip_mode);
        Ok(skip_mode)
    }

    fn skip_mode_context(&self, x: usize, y: usize) -> usize {
        let mi_col = x / 4;
        let mi_row = y / 4;
        let above = (mi_row > self.tile_mi_row_start)
            .then(|| self.skip_mode_grid[(mi_row - 1) * self.mi_cols + mi_col])
            .flatten()
            .unwrap_or(false);
        let left = (mi_col > self.tile_mi_col_start)
            .then(|| self.skip_mode_grid[mi_row * self.mi_cols + mi_col - 1])
            .flatten()
            .unwrap_or(false);
        usize::from(above) + usize::from(left)
    }

    fn set_skip_mode(&mut self, block_size: BlockSize, x: usize, y: usize, value: bool) {
        context_grid::fill_mi_grid(
            &mut self.skip_mode_grid,
            self.mi_cols,
            self.mi_rows,
            x,
            y,
            block_size,
            value,
        );
    }

    fn set_segmentation_id(&mut self, block_size: BlockSize, x: usize, y: usize, segment_id: u8) {
        let start_x = x / 4;
        let start_y = y / 4;
        let end_x = (start_x + block_size.width() / 4).min(self.mi_cols);
        let end_y = (start_y + block_size.height() / 4).min(self.mi_rows);
        for row in start_y..end_y {
            let start = row * self.mi_cols + start_x;
            self.segmentation_map[start..row * self.mi_cols + end_x].fill(segment_id);
        }
    }

    pub(super) fn txb_context(
        &self,
        block_size: BlockSize,
        transform: TransformBlock,
    ) -> TxbContext {
        let contexts = &self.plane_entropy_contexts[transform.plane];
        txb_context(block_size, transform, &contexts.above, &contexts.left)
    }

    pub(super) fn configure_plane_entropy_contexts(&mut self, sequence: &SequenceHeader) {
        if self.plane_entropy_contexts_configured {
            return;
        }
        for plane in 0..3 {
            let subsampling_x = usize::from(plane > 0 && sequence.color_config.subsampling_x);
            let subsampling_y = usize::from(plane > 0 && sequence.color_config.subsampling_y);
            self.plane_subsampling_x[plane] = subsampling_x;
            self.plane_subsampling_y[plane] = subsampling_y;
            self.plane_entropy_contexts[plane].above =
                vec![0; self.mi_cols.div_ceil(1usize << subsampling_x)];
            self.plane_entropy_contexts[plane].left =
                vec![0; self.mi_rows.div_ceil(1usize << subsampling_y)];
        }
        self.plane_entropy_contexts_configured = true;
    }

    pub(super) fn set_txb_entropy_context(&mut self, transform: TransformBlock, value: u8) {
        let contexts = &mut self.plane_entropy_contexts[transform.plane];
        set_txb_entropy_context(transform, value, &mut contexts.above, &mut contexts.left);
    }

    pub(super) fn finish_entropy(&mut self) -> Result<usize, DecoderError> {
        self.reader.exit()
    }
}

fn neg_deinterleave(diff: u8, predicted: u8, max: u8) -> u8 {
    if predicted == 0 {
        diff
    } else if predicted + 1 >= max {
        max.wrapping_sub(diff + 1)
    } else if 2 * predicted < max {
        if diff <= 2 * predicted {
            if diff & 1 != 0 {
                predicted + (diff + 1) / 2
            } else {
                predicted - diff / 2
            }
        } else {
            diff
        }
    } else if diff <= 2 * (max - predicted - 1) {
        if diff & 1 != 0 {
            predicted + (diff + 1) / 2
        } else {
            predicted - diff / 2
        }
    } else {
        max.wrapping_sub(diff + 1)
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

#[cfg(test)]
#[path = "tile_decode/tests/tile_decode_palette_diagnostic.rs"]
mod palette_diagnostic_tests;

#[cfg(test)]
#[path = "tile_decode/tests/tile_decode_context_grid.rs"]
mod context_grid_tests;

#[cfg(test)]
#[path = "tile_decode/tests/tile_decode_tx_type_syntax.rs"]
mod tx_type_syntax_tests;

#[cfg(test)]
#[path = "tile_decode/tests/tile_decode_reconstruction_coverage.rs"]
mod reconstruction_coverage_tests;

#[cfg(test)]
#[path = "tile_decode/tests/tile_decode_partition.rs"]
mod partition_tests;
