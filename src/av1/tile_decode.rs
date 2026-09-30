use super::cdf::CdfContext;
use super::decode::FrameBuffers;
use super::entropy::EntropyDecoder;
use super::frame::{FrameHeader, InterpolationFilter, RestorationParams};
use super::sequence::SequenceHeader;
use super::syntax::{BlockSize, TxSize, TxType, mi_dimension};
use super::transform::TransformBlock;
use crate::DecoderError;
use crate::allocation::{
    AllocationClass, AllocationTicket, admit_fresh_with, capacity_bytes, fresh_replacement,
};
use crate::container::DecodeBudget;
use std::cell::Cell;
use std::sync::Arc;

mod block_syntax;
mod coefficient;
mod coefficient_context;
mod context_grid;
mod context_state;
mod decode_flow;
mod diagnostic;
mod palette;
#[cfg(test)]
pub(crate) use palette::{
    observed_palette_allocation_labels, reset_observed_palette_allocation_labels,
};
mod partition_syntax;
mod post_filter_state;
#[cfg(test)]
pub(crate) use post_filter_state::RestorationUnit;
pub(crate) use post_filter_state::TransformBoundary;
#[cfg(test)]
pub(crate) use post_filter_state::wiener_filter_unit;
pub(crate) use post_filter_state::{
    BlockFilterState, PostFilterState, cdef_adjust_primary_strength, cdef_chroma_direction,
    cdef_filter_block_region_with_edge_mode_into_bit_depth_visible_scaled,
    cdef_find_direction_with_variance_visible, deblock_filter_edge_with_visible_bounds,
    sgrproj_filter_unit_into_with_fixed_scratch_bit_depth_visible,
    sgrproj_filter_unit_into_with_scratch_bit_depth_visible,
    wiener_filter_unit_into_with_fixed_scratch_bit_depth_visible,
    wiener_filter_unit_into_with_scratch_bit_depth_visible,
};
#[cfg(test)]
pub(crate) use post_filter_state::{CdefBlockIndex, CdefUnit};
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
pub(crate) use public_api::decode_luma_root_block_prefix_with_post_filter_state_and_entropy_options_with_references_and_cdf_and_motion_and_budget;
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

/// The five fixed-size work planes used while reconstructing a tile.
#[derive(Debug)]
struct TileScratch {
    dequant: Vec<i32>,
    residual: Vec<i32>,
    prediction: Vec<u16>,
    inter_intra: Vec<u16>,
    reconstruction: Vec<u16>,
}

#[derive(Debug)]
struct TileScratchFailure {
    error: DecoderError,
    scratch: TileScratch,
    tickets: [Option<AllocationTicket>; 5],
}

/// The checked allocation plan for the state owned by one strict AV1 tile
/// decoder.  This is deliberately a value-only plan: no `Vec` is created
/// while dimensions and products are being checked.  The actual vectors are
/// admitted only after `requested_bytes` has passed the native live budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TileDecoderMemoryPlan {
    pub(crate) mi_cols: usize,
    pub(crate) mi_rows: usize,
    pub(crate) tile_mi_cols: usize,
    pub(crate) tile_mi_rows: usize,
    pub(crate) mi_count: usize,
    pub(crate) block_filter_capacity: usize,
    pub(crate) cdef_capacity: usize,
    pub(crate) restoration_capacity: usize,
    pub(crate) coefficient_samples: usize,
    pub(crate) fixed_scratch_samples: usize,
    pub(crate) storage_bytes: usize,
    pub(crate) dynamic_bytes: usize,
    pub(crate) requested_bytes: usize,
}

impl TileDecoderMemoryPlan {
    pub(crate) fn for_frame(
        frame: &FrameHeader,
        tile: Option<&crate::av1::TileDecodePlan>,
    ) -> Result<Self, DecoderError> {
        let frame_mi_cols = usize::try_from(mi_dimension(frame.frame_width))
            .map_err(|_| DecoderError::InvalidParam("AV1 frame width is too large".to_string()))?;
        let frame_mi_rows = usize::try_from(mi_dimension(frame.frame_height))
            .map_err(|_| DecoderError::InvalidParam("AV1 frame height is too large".to_string()))?;
        let (tile_mi_cols, tile_mi_rows) = if let Some(tile) = tile {
            let tile_mi_cols = usize::try_from(tile.mi_col_end.saturating_sub(tile.mi_col_start))
                .map_err(|_| {
                DecoderError::InvalidParam("AV1 tile columns are too large".to_string())
            })?;
            let tile_mi_rows = usize::try_from(tile.mi_row_end.saturating_sub(tile.mi_row_start))
                .map_err(|_| {
                DecoderError::InvalidParam("AV1 tile rows are too large".to_string())
            })?;
            if tile.mi_col_end > u32::try_from(frame_mi_cols).unwrap_or(u32::MAX)
                || tile.mi_row_end > u32::try_from(frame_mi_rows).unwrap_or(u32::MAX)
            {
                return Err(DecoderError::InvalidParam(
                    "AV1 tile bounds exceed frame context grid".to_string(),
                ));
            }
            (tile_mi_cols, tile_mi_rows)
        } else {
            (frame_mi_cols, frame_mi_rows)
        };
        if tile_mi_cols == 0 || tile_mi_rows == 0 {
            return Err(DecoderError::InvalidParam(
                "AV1 tile dimensions must be non-zero".to_string(),
            ));
        }
        // Context grids are indexed in frame coordinates; tile bounds only
        // constrain traversal.  Allocating a tile-local grid would make a
        // non-zero tile origin index the wrong row or panic at the edge.
        let mi_cols = frame_mi_cols;
        let mi_rows = frame_mi_rows;
        let mi_count = mi_cols.checked_mul(mi_rows).ok_or_else(|| {
            DecoderError::InvalidParam("AV1 tile context grid size overflows".to_string())
        })?;
        let block_filter_capacity = mi_count.checked_add(3).ok_or_else(|| {
            DecoderError::InvalidParam("AV1 block filter capacity overflows".to_string())
        })? / 4;
        let block_filter_capacity = block_filter_capacity.min(32_768);
        let cdef_capacity = mi_count
            .checked_add(255)
            .ok_or_else(|| DecoderError::InvalidParam("AV1 CDEF capacity overflows".to_string()))?
            / 256;
        let cdef_capacity = cdef_capacity.min(4_096);
        let tile_superblocks = block_filter_capacity.checked_add(3).ok_or_else(|| {
            DecoderError::InvalidParam("AV1 restoration capacity overflows".to_string())
        })? / 4;
        let restoration_capacity = tile_superblocks.checked_mul(3).ok_or_else(|| {
            DecoderError::InvalidParam("AV1 restoration capacity overflows".to_string())
        })?;
        let coefficient_samples = TxSize::Tx64x64.sample_count();
        let fixed_scratch_samples = TILE_SCRATCH_SAMPLES;
        let storage_bytes = Self::storage_bytes(
            mi_cols,
            mi_rows,
            mi_count,
            block_filter_capacity,
            cdef_capacity,
            restoration_capacity,
        )?;
        let dynamic_bytes = Self::dynamic_bytes(mi_count)?;
        let coefficient_bytes =
            Self::bytes::<i32>(coefficient_samples, "tile coefficient scratch")?;
        let fixed_i32_bytes = Self::bytes::<i32>(
            fixed_scratch_samples.checked_mul(2).ok_or_else(|| {
                DecoderError::InvalidParam("AV1 fixed scratch size overflows".to_string())
            })?,
            "tile fixed scratch",
        )?;
        let fixed_u16_bytes = Self::bytes::<u16>(
            fixed_scratch_samples.checked_mul(3).ok_or_else(|| {
                DecoderError::InvalidParam("AV1 fixed scratch size overflows".to_string())
            })?,
            "tile fixed scratch",
        )?;
        let requested_bytes = Self::sum_bytes([
            Ok(storage_bytes),
            Ok(dynamic_bytes),
            Ok(coefficient_bytes),
            Ok(fixed_i32_bytes),
            Ok(fixed_u16_bytes),
        ])?;
        Ok(Self {
            mi_cols,
            mi_rows,
            tile_mi_cols,
            tile_mi_rows,
            mi_count,
            block_filter_capacity,
            cdef_capacity,
            restoration_capacity,
            coefficient_samples,
            fixed_scratch_samples,
            storage_bytes,
            dynamic_bytes,
            requested_bytes,
        })
    }

    fn bytes<T>(count: usize, label: &str) -> Result<usize, DecoderError> {
        capacity_bytes::<T>(count, label)
    }

    fn sum_bytes<I>(items: I) -> Result<usize, DecoderError>
    where
        I: IntoIterator<Item = Result<usize, DecoderError>>,
    {
        items.into_iter().try_fold(0usize, |total, item| {
            total.checked_add(item?).ok_or_else(|| {
                DecoderError::InvalidParam("AV1 tile memory plan overflows".to_string())
            })
        })
    }

    fn storage_bytes(
        mi_cols: usize,
        mi_rows: usize,
        mi_count: usize,
        block_filter_capacity: usize,
        cdef_capacity: usize,
        restoration_capacity: usize,
    ) -> Result<usize, DecoderError> {
        let option_usize = Self::bytes::<Option<usize>>(mi_count, "tile mode grid")?;
        let option_bool = Self::bytes::<Option<bool>>(mi_count, "tile boolean grid")?;
        let option_u8 = Self::bytes::<Option<u8>>(mi_count, "tile reference grid")?;
        let option_mv = Self::bytes::<Option<(i32, i32)>>(mi_count, "tile motion grid")?;
        let option_filter = Self::bytes::<Option<(InterpolationFilter, InterpolationFilter)>>(
            mi_count,
            "tile interpolation grid",
        )?;
        let option_block = Self::bytes::<Option<BlockSize>>(mi_count, "tile block grid")?;
        let option_palette = Self::bytes::<Option<Vec<u16>>>(mi_count, "tile palette grid")?;
        let option_compound = Self::bytes::<Option<u8>>(mi_count, "tile compound grid")?;
        let plain_u8 = Self::bytes::<u8>(mi_count, "tile segmentation map")?;
        let plain_usize = Self::bytes::<usize>(mi_cols, "tile transform context")?;
        let plain_usize_rows = Self::bytes::<usize>(mi_rows, "tile transform context")?;
        let plain_bool = Self::bytes::<bool>(mi_count, "tile reconstruction grid")?;
        let plain_tx = Self::bytes::<TxType>(mi_count, "tile transform type grid")?;
        let context_u8 = Self::bytes::<u8>(mi_cols, "tile entropy context")?;
        let context_u8_rows = Self::bytes::<u8>(mi_rows, "tile entropy context")?;
        let cdef = Self::bytes::<post_filter_state::CdefUnit>(cdef_capacity, "tile CDEF units")?;
        let cdef_block =
            Self::bytes::<post_filter_state::CdefBlockIndex>(cdef_capacity, "tile CDEF blocks")?;
        let boundary = Self::bytes::<post_filter_state::TransformBoundary>(
            block_filter_capacity,
            "tile transform boundaries",
        )?;
        let restoration = Self::bytes::<post_filter_state::RestorationUnit>(
            restoration_capacity,
            "tile restoration units",
        )?;
        let block_filter = Self::bytes::<post_filter_state::BlockFilterState>(
            block_filter_capacity,
            "tile block filter states",
        )?;
        // There are 23 MI grids in TileDecoder.  Keep this list explicit so a
        // new grid cannot silently escape the plan during review.
        Self::sum_bytes([
            Ok(option_usize), // y_mode
            Ok(option_bool),  // is_inter
            Ok(option_u8),    // reference_frame
            Ok(option_u8),    // reference_frame_type
            Ok(option_bool),  // inter_new_mv
            Ok(option_mv),    // motion_vector
            Ok(option_filter),
            Ok(option_block),
            Ok(option_bool),  // interintra
            Ok(option_u8),    // reference_frame_secondary
            Ok(option_u8),    // reference_frame_secondary_type
            Ok(option_mv),    // motion_vector_secondary
            Ok(option_mv),    // intra_bc_mv
            Ok(option_usize), // y_palette_size
            Ok(option_usize), // uv_palette_size
            Ok(option_palette),
            Ok(option_palette),
            Ok(option_bool), // y_smooth
            Ok(option_bool), // uv_smooth
            Ok(option_bool), // skip
            Ok(option_bool), // skip_mode
            Ok(option_compound),
            Ok(option_compound),
            Ok(plain_u8),
            Ok(Self::bytes::<u8>(mi_cols, "tile partition context")?),
            Ok(Self::bytes::<u8>(mi_rows, "tile partition context")?),
            Ok(plain_usize),
            Ok(plain_usize_rows),
            Ok(plain_bool.checked_mul(3).ok_or_else(|| {
                DecoderError::InvalidParam("tile reconstruction grids overflow".to_string())
            })?),
            Ok(plain_tx),
            Ok(context_u8.checked_mul(3).ok_or_else(|| {
                DecoderError::InvalidParam("tile entropy contexts overflow".to_string())
            })?),
            Ok(context_u8_rows.checked_mul(3).ok_or_else(|| {
                DecoderError::InvalidParam("tile entropy contexts overflow".to_string())
            })?),
            Ok(cdef),
            Ok(cdef_block),
            Ok(boundary),
            Ok(restoration),
            Ok(block_filter),
        ])
    }

    fn dynamic_bytes(mi_count: usize) -> Result<usize, DecoderError> {
        // Palette colors are retained in one Y and one U MI grid.  The V
        // values are kept in the block-local palette and are not duplicated
        // into a second grid.  The transient allowance covers the bounded
        // palette decoder vectors and the two 64x64 color maps; it is a
        // reservation, not an eager materialization.
        let palette_grid = capacity_bytes::<u16>(
            mi_count
                .checked_mul(2)
                .and_then(|count| count.checked_mul(palette::PALETTE_MAX_SIZE))
                .ok_or_else(|| {
                    DecoderError::InvalidParam("AV1 palette storage size overflows".to_string())
                })?,
            "tile nested palette colors",
        )?;
        let scan_cache = coefficient::CoefficientScanCache::strict_outer_bytes()?
            .checked_add(coefficient::CoefficientScanCache::strict_lazy_bytes()?)
            .ok_or_else(|| {
                DecoderError::InvalidParam("AV1 coefficient cache size overflows".to_string())
            })?;
        let transient = palette::palette_transient_bytes()?;
        palette_grid
            .checked_add(scan_cache)
            .and_then(|bytes| bytes.checked_add(transient))
            .ok_or_else(|| {
                DecoderError::InvalidParam("AV1 dynamic tile storage overflows".to_string())
            })
    }

    pub(crate) fn storage_bytes_checked(&self) -> usize {
        self.storage_bytes
    }
}

/// Vectors allocated from `TileDecoderMemoryPlan`.  Keeping this bundle
/// separate from `TileDecoder` makes the failure path transactional: a
/// partially populated bundle is dropped before the aggregate ticket is
/// charged, and only then is it moved into the decoder.
#[derive(Debug)]
struct TileDecoderStorage {
    y_mode_grid: Vec<Option<usize>>,
    is_inter_grid: Vec<Option<bool>>,
    reference_frame_grid: Vec<Option<u8>>,
    reference_frame_type_grid: Vec<Option<u8>>,
    inter_new_mv_grid: Vec<Option<bool>>,
    motion_vector_grid: Vec<Option<(i32, i32)>>,
    interpolation_filter_grid: Vec<Option<(InterpolationFilter, InterpolationFilter)>>,
    motion_block_size_grid: Vec<Option<BlockSize>>,
    interintra_grid: Vec<Option<bool>>,
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
    compound_group_idx_grid: Vec<Option<u8>>,
    compound_idx_grid: Vec<Option<u8>>,
    segmentation_map: Vec<u8>,
    above_partition_context: Vec<u8>,
    left_partition_context: Vec<u8>,
    above_txfm_context: Vec<usize>,
    left_txfm_context: Vec<usize>,
    reconstructed_mi_grid: [Vec<bool>; 3],
    luma_tx_type_grid: Vec<TxType>,
    plane_entropy_contexts: [PlaneEntropyContexts; 3],
    cdef_units: Vec<post_filter_state::CdefUnit>,
    cdef_blocks: Vec<post_filter_state::CdefBlockIndex>,
    transform_boundaries: Vec<post_filter_state::TransformBoundary>,
    restoration_units: Vec<post_filter_state::RestorationUnit>,
    block_filter_states: Vec<post_filter_state::BlockFilterState>,
}

impl TileDecoderStorage {
    fn fresh_vec<T: Default + Clone>(count: usize, label: &str) -> Result<Vec<T>, DecoderError> {
        let replacement = fresh_replacement::<T>(count, label)?;
        let mut values = replacement.into_vec();
        values.resize(count, T::default());
        Ok(values)
    }

    fn fresh_capacity<T>(count: usize, label: &str) -> Result<Vec<T>, DecoderError> {
        Ok(fresh_replacement::<T>(count, label)?.into_vec())
    }

    fn fresh_tx_types(count: usize, label: &str) -> Result<Vec<TxType>, DecoderError> {
        let replacement = fresh_replacement::<TxType>(count, label)?;
        let mut values = replacement.into_vec();
        values.resize(count, TxType::DctDct);
        Ok(values)
    }

    fn legacy(
        mi_cols: usize,
        mi_rows: usize,
        mi_count: usize,
        cdef_capacity: usize,
        block_filter_capacity: usize,
    ) -> Self {
        Self {
            y_mode_grid: vec![None; mi_count],
            is_inter_grid: vec![None; mi_count],
            reference_frame_grid: vec![None; mi_count],
            reference_frame_type_grid: vec![None; mi_count],
            inter_new_mv_grid: vec![None; mi_count],
            motion_vector_grid: vec![None; mi_count],
            interpolation_filter_grid: vec![None; mi_count],
            motion_block_size_grid: vec![None; mi_count],
            interintra_grid: vec![None; mi_count],
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
            compound_group_idx_grid: vec![None; mi_count],
            compound_idx_grid: vec![None; mi_count],
            segmentation_map: vec![0; mi_count],
            above_partition_context: vec![0; mi_cols],
            left_partition_context: vec![0; mi_rows],
            above_txfm_context: vec![0; mi_cols],
            left_txfm_context: vec![64; mi_rows],
            reconstructed_mi_grid: std::array::from_fn(|_| vec![false; mi_count]),
            luma_tx_type_grid: vec![TxType::DctDct; mi_count],
            plane_entropy_contexts: std::array::from_fn(|_| PlaneEntropyContexts {
                above: vec![0; mi_cols],
                left: vec![0; mi_rows],
            }),
            cdef_units: Vec::with_capacity(cdef_capacity),
            cdef_blocks: Vec::with_capacity(cdef_capacity),
            transform_boundaries: Vec::with_capacity(block_filter_capacity),
            restoration_units: Vec::new(),
            block_filter_states: Vec::with_capacity(block_filter_capacity),
        }
    }

    fn allocate(
        plan: &TileDecoderMemoryPlan,
        budget: &mut DecodeBudget,
    ) -> Result<(Self, AllocationTicket), DecoderError> {
        budget.check_additional_frame(plan.storage_bytes_checked())?;
        let reconstructed_mi_grid = [
            Self::fresh_vec(plan.mi_count, "native AV1 reconstruction grid")?,
            Self::fresh_vec(plan.mi_count, "native AV1 reconstruction grid")?,
            Self::fresh_vec(plan.mi_count, "native AV1 reconstruction grid")?,
        ];
        let plane_entropy_contexts = [
            PlaneEntropyContexts {
                above: Self::fresh_vec(plan.mi_cols, "native AV1 plane above context")?,
                left: Self::fresh_vec(plan.mi_rows, "native AV1 plane left context")?,
            },
            PlaneEntropyContexts {
                above: Self::fresh_vec(plan.mi_cols, "native AV1 plane above context")?,
                left: Self::fresh_vec(plan.mi_rows, "native AV1 plane left context")?,
            },
            PlaneEntropyContexts {
                above: Self::fresh_vec(plan.mi_cols, "native AV1 plane above context")?,
                left: Self::fresh_vec(plan.mi_rows, "native AV1 plane left context")?,
            },
        ];
        let storage = Self {
            y_mode_grid: Self::fresh_vec(plan.mi_count, "native AV1 y mode grid")?,
            is_inter_grid: Self::fresh_vec(plan.mi_count, "native AV1 inter grid")?,
            reference_frame_grid: Self::fresh_vec(plan.mi_count, "native AV1 reference grid")?,
            reference_frame_type_grid: Self::fresh_vec(
                plan.mi_count,
                "native AV1 reference type grid",
            )?,
            inter_new_mv_grid: Self::fresh_vec(plan.mi_count, "native AV1 new MV grid")?,
            motion_vector_grid: Self::fresh_vec(plan.mi_count, "native AV1 motion grid")?,
            interpolation_filter_grid: Self::fresh_vec(
                plan.mi_count,
                "native AV1 interpolation grid",
            )?,
            motion_block_size_grid: Self::fresh_vec(plan.mi_count, "native AV1 block size grid")?,
            interintra_grid: Self::fresh_vec(plan.mi_count, "native AV1 inter-intra grid")?,
            reference_frame_secondary_grid: Self::fresh_vec(
                plan.mi_count,
                "native AV1 secondary reference grid",
            )?,
            reference_frame_secondary_type_grid: Self::fresh_vec(
                plan.mi_count,
                "native AV1 secondary reference type grid",
            )?,
            motion_vector_secondary_grid: Self::fresh_vec(
                plan.mi_count,
                "native AV1 secondary motion grid",
            )?,
            intra_bc_mv_grid: Self::fresh_vec(plan.mi_count, "native AV1 intra BC grid")?,
            y_palette_size_grid: Self::fresh_vec(plan.mi_count, "native AV1 Y palette size grid")?,
            uv_palette_size_grid: Self::fresh_vec(
                plan.mi_count,
                "native AV1 UV palette size grid",
            )?,
            y_palette_colors_grid: Self::fresh_vec(
                plan.mi_count,
                "native AV1 Y palette colors grid",
            )?,
            u_palette_colors_grid: Self::fresh_vec(
                plan.mi_count,
                "native AV1 U palette colors grid",
            )?,
            y_smooth_grid: Self::fresh_vec(plan.mi_count, "native AV1 Y smooth grid")?,
            uv_smooth_grid: Self::fresh_vec(plan.mi_count, "native AV1 UV smooth grid")?,
            skip_grid: Self::fresh_vec(plan.mi_count, "native AV1 skip grid")?,
            skip_mode_grid: Self::fresh_vec(plan.mi_count, "native AV1 skip mode grid")?,
            compound_group_idx_grid: Self::fresh_vec(
                plan.mi_count,
                "native AV1 compound group grid",
            )?,
            compound_idx_grid: Self::fresh_vec(plan.mi_count, "native AV1 compound grid")?,
            segmentation_map: Self::fresh_vec(plan.mi_count, "native AV1 segmentation map")?,
            above_partition_context: Self::fresh_vec(
                plan.mi_cols,
                "native AV1 above partition context",
            )?,
            left_partition_context: Self::fresh_vec(
                plan.mi_rows,
                "native AV1 left partition context",
            )?,
            above_txfm_context: Self::fresh_vec(
                plan.mi_cols,
                "native AV1 above transform context",
            )?,
            left_txfm_context: Self::fresh_vec(plan.mi_rows, "native AV1 left transform context")?,
            reconstructed_mi_grid,
            luma_tx_type_grid: Self::fresh_tx_types(
                plan.mi_count,
                "native AV1 transform type grid",
            )?,
            plane_entropy_contexts,
            cdef_units: Self::fresh_capacity(plan.cdef_capacity, "native AV1 CDEF units")?,
            cdef_blocks: Self::fresh_capacity(plan.cdef_capacity, "native AV1 CDEF blocks")?,
            transform_boundaries: Self::fresh_capacity(
                plan.block_filter_capacity,
                "native AV1 transform boundaries",
            )?,
            restoration_units: Self::fresh_capacity(
                plan.restoration_capacity,
                "native AV1 restoration units",
            )?,
            block_filter_states: Self::fresh_capacity(
                plan.block_filter_capacity,
                "native AV1 block filter states",
            )?,
        };
        let actual_bytes = storage.actual_bytes()?;
        let ticket = budget.reserve_existing_bytes(
            AllocationClass::Frame,
            actual_bytes,
            "native AV1 tile decoder state",
        )?;
        Ok((storage, ticket))
    }

    fn actual_bytes(&self) -> Result<usize, DecoderError> {
        let mut total = 0usize;
        fn vec_bytes<T>(values: &Vec<T>, label: &str) -> Result<usize, DecoderError> {
            capacity_bytes::<T>(values.capacity(), label)
        }
        macro_rules! add {
            ($value:expr, $label:literal) => {
                total = total
                    .checked_add(vec_bytes(&$value, $label)?)
                    .ok_or_else(|| {
                        DecoderError::InvalidParam("AV1 tile state capacity overflows".to_string())
                    })?;
            };
        }
        add!(self.y_mode_grid, "tile y mode grid");
        add!(self.is_inter_grid, "tile inter grid");
        add!(self.reference_frame_grid, "tile reference grid");
        add!(self.reference_frame_type_grid, "tile reference type grid");
        add!(self.inter_new_mv_grid, "tile new MV grid");
        add!(self.motion_vector_grid, "tile motion grid");
        add!(self.interpolation_filter_grid, "tile interpolation grid");
        add!(self.motion_block_size_grid, "tile block grid");
        add!(self.interintra_grid, "tile inter-intra grid");
        add!(
            self.reference_frame_secondary_grid,
            "tile secondary reference grid"
        );
        add!(
            self.reference_frame_secondary_type_grid,
            "tile secondary reference type grid"
        );
        add!(
            self.motion_vector_secondary_grid,
            "tile secondary motion grid"
        );
        add!(self.intra_bc_mv_grid, "tile intra BC grid");
        add!(self.y_palette_size_grid, "tile Y palette size grid");
        add!(self.uv_palette_size_grid, "tile UV palette size grid");
        add!(self.y_palette_colors_grid, "tile Y palette colors grid");
        add!(self.u_palette_colors_grid, "tile U palette colors grid");
        add!(self.y_smooth_grid, "tile Y smooth grid");
        add!(self.uv_smooth_grid, "tile UV smooth grid");
        add!(self.skip_grid, "tile skip grid");
        add!(self.skip_mode_grid, "tile skip mode grid");
        add!(self.compound_group_idx_grid, "tile compound group grid");
        add!(self.compound_idx_grid, "tile compound grid");
        add!(self.segmentation_map, "tile segmentation map");
        add!(self.above_partition_context, "tile above partition context");
        add!(self.left_partition_context, "tile left partition context");
        add!(self.above_txfm_context, "tile above transform context");
        add!(self.left_txfm_context, "tile left transform context");
        for grid in &self.reconstructed_mi_grid {
            add!(grid, "tile reconstruction grid");
        }
        add!(self.luma_tx_type_grid, "tile transform type grid");
        for contexts in &self.plane_entropy_contexts {
            add!(contexts.above, "tile plane above context");
            add!(contexts.left, "tile plane left context");
        }
        add!(self.cdef_units, "tile CDEF units");
        add!(self.cdef_blocks, "tile CDEF blocks");
        add!(self.transform_boundaries, "tile transform boundaries");
        add!(self.restoration_units, "tile restoration units");
        add!(self.block_filter_states, "tile block filter states");
        Ok(total)
    }
}

const TILE_SCRATCH_SAMPLES: usize = 64 * 64;

#[derive(Debug, Default)]
pub(super) struct PaletteScratchLedger {
    bytes: Cell<usize>,
}

impl PaletteScratchLedger {
    pub(super) fn reset(&self) {
        self.bytes.set(0);
    }

    pub(super) fn bytes(&self) -> usize {
        self.bytes.get()
    }

    pub(super) fn add(&self, bytes: usize) -> Result<(), DecoderError> {
        let total = self.bytes.get().checked_add(bytes).ok_or_else(|| {
            DecoderError::InvalidParam("native AV1 palette scratch size overflows".to_string())
        })?;
        self.bytes.set(total);
        Ok(())
    }
}

impl TileScratch {
    fn legacy() -> Self {
        Self {
            dequant: vec![0; TILE_SCRATCH_SAMPLES],
            residual: vec![0; TILE_SCRATCH_SAMPLES],
            prediction: vec![0; TILE_SCRATCH_SAMPLES],
            inter_intra: vec![0; TILE_SCRATCH_SAMPLES],
            reconstruction: vec![0; TILE_SCRATCH_SAMPLES],
        }
    }

    fn strict(
        budget: &mut DecodeBudget,
    ) -> Result<(Self, [AllocationTicket; 5]), TileScratchFailure> {
        let mut allocated: Option<(Self, [Option<AllocationTicket>; 5])> = Some((
            Self {
                dequant: Vec::new(),
                residual: Vec::new(),
                prediction: Vec::new(),
                inter_intra: Vec::new(),
                reconstruction: Vec::new(),
            },
            std::array::from_fn(|_| None),
        ));
        let result = (|| {
            let (scratch, tickets) = allocated.as_mut().expect("scratch transaction");
            macro_rules! admit {
                ($field:ident, $ty:ty, $index:expr, $label:literal) => {{
                    let (replacement, ticket) = admit_fresh_with(
                        budget,
                        TILE_SCRATCH_SAMPLES,
                        AllocationClass::Frame,
                        $label,
                        fresh_replacement::<$ty>,
                    )?;
                    let mut values = replacement.into_vec();
                    values.resize(TILE_SCRATCH_SAMPLES, <$ty>::default());
                    scratch.$field = values;
                    tickets[$index] = Some(ticket);
                }};
            }
            admit!(dequant, i32, 0, "native AV1 tile dequant scratch");
            admit!(residual, i32, 1, "native AV1 tile residual scratch");
            admit!(prediction, u16, 2, "native AV1 tile prediction scratch");
            admit!(inter_intra, u16, 3, "native AV1 tile inter-intra scratch");
            admit!(
                reconstruction,
                u16,
                4,
                "native AV1 tile reconstruction scratch"
            );
            Ok(())
        })();
        if let Err(error) = result {
            let (scratch, tickets) = allocated.take().expect("scratch transaction");
            // Defer cleanup to the caller.  The coefficient scratch is owned
            // by the outer constructor, so all successful tile prefixes must
            // remain alive until it can drop every owner before releasing any
            // of the corresponding accounting tickets.
            return Err(TileScratchFailure {
                error,
                scratch,
                tickets,
            });
        }
        let (scratch, tickets) = allocated.take().expect("scratch transaction committed");
        Ok((
            scratch,
            tickets.map(|ticket| ticket.expect("all tile scratch tickets admitted")),
        ))
    }
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
    pub(crate) order_hint_bits: u8,
    pub(crate) order_hint: u32,
    pub(crate) reference_order_hints: [Option<u32>; 7],
    pub(crate) reference_frame_indices: [u8; 7],
    pub(crate) projected: bool,
    pub(crate) reference_frames: Vec<Option<u8>>,
    pub(crate) motion_vectors: Vec<Option<(i32, i32)>>,
    pub(crate) reference_offsets: Vec<Option<i32>>,
}

impl MotionField {
    pub(crate) fn empty(mi_cols: usize, mi_rows: usize) -> Self {
        let count = mi_cols.saturating_mul(mi_rows);
        Self {
            mi_cols,
            mi_rows,
            order_hint_bits: 0,
            order_hint: 0,
            reference_order_hints: [None; 7],
            reference_frame_indices: [0; 7],
            projected: false,
            reference_frames: vec![None; count],
            motion_vectors: vec![None; count],
            reference_offsets: vec![None; count],
        }
    }

    pub(crate) fn merge(&mut self, tile: Self) {
        if self.mi_cols != tile.mi_cols || self.mi_rows != tile.mi_rows {
            return;
        }
        self.order_hint_bits = tile.order_hint_bits;
        self.order_hint = tile.order_hint;
        self.reference_order_hints = tile.reference_order_hints;
        self.reference_frame_indices = tile.reference_frame_indices;
        self.projected = tile.projected;
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
        for (destination, source) in self
            .reference_offsets
            .iter_mut()
            .zip(tile.reference_offsets)
        {
            if source.is_some() {
                *destination = source;
            }
        }
    }

    pub(crate) fn projected_motion(
        &self,
        mi_col: usize,
        mi_row: usize,
        blk_col: isize,
        blk_row: isize,
        reference_type: usize,
        current_order_hint: u32,
        current_reference_order_hints: &[Option<u32>; 7],
    ) -> Option<(usize, (i32, i32))> {
        let sample_row = if mi_row & 1 == 1 {
            blk_row
        } else {
            blk_row + 1
        };
        let sample_col = if mi_col & 1 == 1 {
            blk_col
        } else {
            blk_col + 1
        };
        let source_row = mi_row.checked_add_signed(sample_row)? >> 1;
        let source_col = mi_col.checked_add_signed(sample_col)? >> 1;
        let source_row = source_row.checked_mul(2)?;
        let source_col = source_col.checked_mul(2)?;
        if source_row >= self.mi_rows || source_col >= self.mi_cols {
            return None;
        }
        let index = source_row * self.mi_cols + source_col;
        let motion_vector = self.motion_vectors.get(index).copied().flatten()?;
        if self.projected {
            let reference_hint = current_reference_order_hints
                .get(reference_type)
                .copied()
                .flatten()?;
            let reference_offset = self.reference_offsets.get(index).copied().flatten()?;
            let current_offset = relative_order_hint_distance(
                self.order_hint_bits,
                current_order_hint,
                reference_hint,
            );
            if reference_offset <= 0 {
                return None;
            }
            return Some((
                index,
                project_temporal_motion_vector(motion_vector, current_offset, reference_offset),
            ));
        }
        let previous_reference_type =
            usize::from(self.reference_frames.get(index).copied().flatten()?);
        let previous_reference_hint = self
            .reference_order_hints
            .get(previous_reference_type)
            .copied()
            .flatten()?;
        let current_reference_hint = current_reference_order_hints
            .get(reference_type)
            .copied()
            .flatten()?;
        let previous_offset = relative_order_hint_distance(
            self.order_hint_bits,
            previous_reference_hint,
            self.order_hint,
        );
        let current_offset = relative_order_hint_distance(
            self.order_hint_bits,
            current_reference_hint,
            current_order_hint,
        );
        if previous_offset == 0 {
            return None;
        }
        Some((
            index,
            project_temporal_motion_vector(motion_vector, current_offset, previous_offset),
        ))
    }
}

fn relative_order_hint_distance(bits: u8, reference: u32, current: u32) -> i32 {
    if bits == 0 {
        return 0;
    }
    let modulo = 1i32 << bits;
    let mask = modulo - 1;
    let mut distance = (reference as i32 - current as i32) & mask;
    if distance & (modulo >> 1) != 0 {
        distance -= modulo;
    }
    distance
}

fn project_temporal_motion_vector(
    motion_vector: (i32, i32),
    numerator: i32,
    denominator: i32,
) -> (i32, i32) {
    const DIV_MULT: [i32; 32] = [
        0, 16384, 8192, 5461, 4096, 3276, 2730, 2340, 2048, 1820, 1638, 1489, 1365, 1260, 1170,
        1092, 1024, 963, 910, 862, 819, 780, 744, 712, 682, 655, 630, 606, 585, 564, 546, 528,
    ];
    fn scale(value: i32, numerator: i32, denominator: i32, div_mult: &[i32; 32]) -> i32 {
        let numerator = numerator.clamp(-31, 31);
        let denominator = denominator.clamp(0, 31);
        let product =
            i64::from(value) * i64::from(numerator) * i64::from(div_mult[denominator as usize]);
        let rounded = if product < 0 {
            -((-product + (1 << 13)) >> 14)
        } else {
            (product + (1 << 13)) >> 14
        };
        rounded.clamp(-32767, 32767) as i32
    }
    (
        scale(motion_vector.0, numerator, denominator, &DIV_MULT),
        scale(motion_vector.1, numerator, denominator, &DIV_MULT),
    )
}

fn lower_motion_vector_precision(
    motion_vector: (i32, i32),
    allow_high_precision_mv: bool,
    force_integer_mv: bool,
) -> (i32, i32) {
    let lower = |value: i32| {
        if force_integer_mv {
            let remainder = value % 8;
            let mut rounded = value - remainder;
            if remainder.abs() > 4 {
                rounded += if remainder > 0 { 8 } else { -8 };
            }
            rounded
        } else if !allow_high_precision_mv && value & 1 != 0 {
            value + if value > 0 { -1 } else { 1 }
        } else {
            value
        }
    };
    (lower(motion_vector.0), lower(motion_vector.1))
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
    interintra_grid: Vec<Option<bool>>,
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
    compound_group_idx_grid: Vec<Option<u8>>,
    compound_idx_grid: Vec<Option<u8>>,
    segmentation_map: Vec<u8>,
    above_partition_context: Vec<u8>,
    left_partition_context: Vec<u8>,
    cdef_transmitted: [bool; 4],
    above_txfm_context: Vec<usize>,
    left_txfm_context: Vec<usize>,
    reconstructed_mi_grid: [Vec<bool>; 3],
    luma_tx_type_grid: Vec<TxType>,
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
    scratch: TileScratch,
    coefficient_scan_cache: CoefficientScanCache,
    palette_scratch: PaletteScratchLedger,
    strict_dynamic_reserved_bytes: usize,
    strict_dynamic_enabled: bool,
    current_qindex: u8,
    current_delta_lf: [i8; 4],
    reference_buffers: [Option<Arc<FrameBuffers>>; 8],
    temporal_motion_field: Option<Arc<MotionField>>,
    reference_frame_indices: [u8; 7],
    reference_order_hints: [Option<u32>; 7],
    order_hint: u32,
    order_hint_bits: u8,
    current_inter_compound: bool,
    allow_high_precision_mv: bool,
    force_integer_mv: bool,
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
        Self::new_with_references_and_cdf_and_scratch(
            payload,
            frame,
            reference_buffers,
            initial_cdf,
            Vec::with_capacity(TxSize::Tx64x64.sample_count()),
            TileScratch::legacy(),
            CoefficientScanCache::new(),
            0,
            false,
        )
    }

    /// Strict-native constructor that admits the reusable coefficient scratch
    /// before any allocation becomes reachable from the decoder.  The legacy
    /// constructors deliberately retain their historical untracked scratch.
    pub(crate) fn new_with_references_and_cdf_with_budget(
        payload: &'a [u8],
        frame: &FrameHeader,
        reference_buffers: [Option<Arc<FrameBuffers>>; 8],
        initial_cdf: Option<CdfContext>,
        budget: &mut DecodeBudget,
    ) -> Result<(Self, [AllocationTicket; 6]), DecoderError> {
        let needed = TxSize::Tx64x64.sample_count();
        let (scratch, mut ticket) = admit_fresh_with(
            budget,
            needed,
            AllocationClass::Frame,
            "native AV1 coefficient scratch",
            fresh_replacement::<i32>,
        )?;
        let (tile_scratch, tile_tickets) = match TileScratch::strict(budget) {
            Ok(value) => value,
            Err(TileScratchFailure {
                error,
                scratch: tile_scratch,
                tickets: tile_tickets,
            }) => {
                // `scratch` owns the coefficient candidate represented by
                // `ticket`. Keep all tile prefix owners alive until every
                // owner can be dropped together before ticket release.
                drop(scratch);
                drop(tile_scratch);
                for mut tile_ticket in tile_tickets.into_iter().flatten() {
                    budget.release_token(&mut tile_ticket)?;
                }
                budget.release_token(&mut ticket)?;
                return Err(error);
            }
        };
        let result = Self::new_with_references_and_cdf_and_scratch(
            payload,
            frame,
            reference_buffers,
            initial_cdf,
            scratch.into_vec(),
            tile_scratch,
            CoefficientScanCache::new(),
            0,
            false,
        );
        match result {
            Ok(decoder) => {
                let [dequant, residual, prediction, inter_intra, reconstruction] = tile_tickets;
                Ok((
                    decoder,
                    [
                        ticket,
                        dequant,
                        residual,
                        prediction,
                        inter_intra,
                        reconstruction,
                    ],
                ))
            }
            Err(error) => {
                budget.release_token(&mut ticket)?;
                for mut tile_ticket in tile_tickets {
                    budget.release_token(&mut tile_ticket)?;
                }
                Err(error)
            }
        }
    }

    /// Strict-native constructor used by the stateful AVIS path.  Unlike the
    /// compatibility strict constructor above, this admits every decoder
    /// context and post-filter vector from one checked memory plan before the
    /// decoder becomes reachable by the caller.
    pub(crate) fn new_with_references_and_cdf_with_budget_for_tile(
        payload: &'a [u8],
        frame: &FrameHeader,
        tile: &crate::av1::TileDecodePlan,
        reference_buffers: [Option<Arc<FrameBuffers>>; 8],
        initial_cdf: Option<CdfContext>,
        budget: &mut DecodeBudget,
    ) -> Result<(Self, [AllocationTicket; 6], AllocationTicket), DecoderError> {
        let plan = TileDecoderMemoryPlan::for_frame(frame, Some(tile))?;
        // Admit the complete decoder-local peak before the first vector is
        // created.  `TileDecoderStorage::allocate` repeats its narrower
        // storage check after this guard so the helper remains independently
        // safe when used by focused tests.
        budget.check_additional_frame(plan.requested_bytes)?;
        let mut dynamic_ticket = budget.reserve_existing_bytes(
            AllocationClass::Frame,
            plan.dynamic_bytes,
            "native AV1 palette and coefficient cache",
        )?;
        let coefficient_scan_cache = match CoefficientScanCache::new_strict(
            coefficient::CoefficientScanCache::strict_lazy_bytes()?,
        ) {
            Ok(cache) => cache,
            Err(error) => {
                budget.release_token(&mut dynamic_ticket)?;
                return Err(error);
            }
        };
        let (storage, mut memory_ticket) = match TileDecoderStorage::allocate(&plan, budget) {
            Ok(value) => value,
            Err(error) => {
                budget.release_token(&mut dynamic_ticket)?;
                return Err(error);
            }
        };
        let dynamic_bytes = dynamic_ticket.charged_capacity_bytes;
        memory_ticket.charged_capacity_bytes = match memory_ticket
            .charged_capacity_bytes
            .checked_add(dynamic_bytes)
        {
            Some(bytes) => bytes,
            None => {
                budget.release_token(&mut memory_ticket)?;
                budget.release_token(&mut dynamic_ticket)?;
                return Err(DecoderError::InvalidParam(
                    "native AV1 tile memory ticket overflows".to_string(),
                ));
            }
        };
        let (scratch, mut coefficient_ticket) = match admit_fresh_with(
            budget,
            plan.coefficient_samples,
            AllocationClass::Frame,
            "native AV1 coefficient scratch",
            fresh_replacement::<i32>,
        ) {
            Ok(value) => value,
            Err(error) => {
                drop(storage);
                let mut ticket = memory_ticket;
                budget.release_token(&mut ticket)?;
                return Err(error);
            }
        };
        let (tile_scratch, tile_tickets) = match TileScratch::strict(budget) {
            Ok(value) => value,
            Err(TileScratchFailure {
                error,
                scratch: tile_scratch,
                tickets: tile_tickets,
            }) => {
                drop(scratch);
                drop(tile_scratch);
                for mut tile_ticket in tile_tickets.into_iter().flatten() {
                    budget.release_token(&mut tile_ticket)?;
                }
                budget.release_token(&mut coefficient_ticket)?;
                drop(storage);
                let mut ticket = memory_ticket;
                budget.release_token(&mut ticket)?;
                return Err(error);
            }
        };
        let result = Self::new_with_references_and_cdf_and_scratch_and_storage(
            payload,
            frame,
            reference_buffers,
            initial_cdf,
            scratch.into_vec(),
            tile_scratch,
            storage,
            coefficient_scan_cache,
            plan.dynamic_bytes,
            false,
        );
        match result {
            Ok(mut decoder) => {
                decoder.strict_dynamic_enabled = true;
                let [dequant, residual, prediction, inter_intra, reconstruction] = tile_tickets;
                Ok((
                    decoder,
                    [
                        coefficient_ticket,
                        dequant,
                        residual,
                        prediction,
                        inter_intra,
                        reconstruction,
                    ],
                    memory_ticket,
                ))
            }
            Err(error) => {
                budget.release_token(&mut memory_ticket)?;
                budget.release_token(&mut coefficient_ticket)?;
                for mut tile_ticket in tile_tickets {
                    budget.release_token(&mut tile_ticket)?;
                }
                Err(error)
            }
        }
    }

    fn new_with_references_and_cdf_and_scratch(
        payload: &'a [u8],
        frame: &FrameHeader,
        reference_buffers: [Option<Arc<FrameBuffers>>; 8],
        initial_cdf: Option<CdfContext>,
        coefficient_scratch: Vec<i32>,
        scratch: TileScratch,
        coefficient_scan_cache: CoefficientScanCache,
        strict_dynamic_reserved_bytes: usize,
        strict_dynamic_enabled: bool,
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
        let storage = TileDecoderStorage::legacy(
            mi_cols,
            mi_rows,
            mi_count,
            cdef_capacity,
            block_filter_capacity,
        );
        Self::new_with_references_and_cdf_and_scratch_and_storage(
            payload,
            frame,
            reference_buffers,
            initial_cdf,
            coefficient_scratch,
            scratch,
            storage,
            coefficient_scan_cache,
            strict_dynamic_reserved_bytes,
            strict_dynamic_enabled,
        )
    }

    fn new_with_references_and_cdf_and_scratch_and_storage(
        payload: &'a [u8],
        frame: &FrameHeader,
        reference_buffers: [Option<Arc<FrameBuffers>>; 8],
        initial_cdf: Option<CdfContext>,
        coefficient_scratch: Vec<i32>,
        scratch: TileScratch,
        storage: TileDecoderStorage,
        coefficient_scan_cache: CoefficientScanCache,
        strict_dynamic_reserved_bytes: usize,
        strict_dynamic_enabled: bool,
    ) -> Result<Self, DecoderError> {
        let mi_cols = storage.above_partition_context.len();
        let mi_rows = storage.left_partition_context.len();
        Ok(Self {
            reader: EntropyDecoder::new(payload, frame.disable_cdf_update)?,
            cdf: initial_cdf.unwrap_or_else(|| CdfContext::new(frame.base_q_idx)),
            mi_cols,
            mi_rows,
            tile_mi_col_start: 0,
            tile_mi_row_start: 0,
            y_mode_grid: storage.y_mode_grid,
            is_inter_grid: storage.is_inter_grid,
            reference_frame_grid: storage.reference_frame_grid,
            reference_frame_type_grid: storage.reference_frame_type_grid,
            inter_new_mv_grid: storage.inter_new_mv_grid,
            motion_vector_grid: storage.motion_vector_grid,
            interpolation_filter_grid: storage.interpolation_filter_grid,
            motion_block_size_grid: storage.motion_block_size_grid,
            interintra_grid: storage.interintra_grid,
            reference_frame_secondary_grid: storage.reference_frame_secondary_grid,
            reference_frame_secondary_type_grid: storage.reference_frame_secondary_type_grid,
            motion_vector_secondary_grid: storage.motion_vector_secondary_grid,
            intra_bc_mv_grid: storage.intra_bc_mv_grid,
            y_palette_size_grid: storage.y_palette_size_grid,
            uv_palette_size_grid: storage.uv_palette_size_grid,
            y_palette_colors_grid: storage.y_palette_colors_grid,
            u_palette_colors_grid: storage.u_palette_colors_grid,
            y_smooth_grid: storage.y_smooth_grid,
            uv_smooth_grid: storage.uv_smooth_grid,
            skip_grid: storage.skip_grid,
            skip_mode_grid: storage.skip_mode_grid,
            compound_group_idx_grid: storage.compound_group_idx_grid,
            compound_idx_grid: storage.compound_idx_grid,
            segmentation_map: storage.segmentation_map,
            above_partition_context: storage.above_partition_context,
            left_partition_context: storage.left_partition_context,
            cdef_transmitted: [false; 4],
            above_txfm_context: storage.above_txfm_context,
            left_txfm_context: storage.left_txfm_context,
            reconstructed_mi_grid: storage.reconstructed_mi_grid,
            luma_tx_type_grid: storage.luma_tx_type_grid,
            current_cfl: None,
            plane_entropy_contexts: storage.plane_entropy_contexts,
            plane_entropy_contexts_configured: false,
            plane_subsampling_x: [0; 3],
            plane_subsampling_y: [0; 3],
            restoration: frame.restoration,
            // Chroma Wiener restoration uses the reduced 5-tap window, so
            // its outer coefficient is implicit zero and is not signaled.
            wiener_refs: [[[3, -7, 15]; 2], [[0, -7, 15]; 2], [[0, -7, 15]; 2]],
            sgrproj_refs: [[-32, 31]; 3],
            cdef_units: storage.cdef_units,
            cdef_blocks: storage.cdef_blocks,
            transform_boundaries: storage.transform_boundaries,
            restoration_units: storage.restoration_units,
            block_filter_states: storage.block_filter_states,
            coefficient_scratch,
            scratch,
            coefficient_scan_cache,
            palette_scratch: PaletteScratchLedger::default(),
            strict_dynamic_reserved_bytes,
            strict_dynamic_enabled,
            current_qindex: frame.segmentation.effective_qindex(frame.base_q_idx),
            current_delta_lf: [0; 4],
            reference_buffers,
            temporal_motion_field: None,
            reference_frame_indices: frame.reference_frame_indices,
            reference_order_hints: frame.reference_order_hints,
            order_hint: frame.order_hint,
            order_hint_bits: 8,
            current_inter_compound: false,
            allow_high_precision_mv: frame.allow_high_precision_mv,
            force_integer_mv: frame.force_integer_mv == 1,
        })
    }

    pub(crate) fn cdf_snapshot(&self) -> CdfContext {
        self.cdf.clone()
    }

    pub(super) fn strict_dynamic_headroom(
        &self,
        additional: usize,
        label: &str,
    ) -> Result<(), DecoderError> {
        if !self.strict_dynamic_enabled {
            return Ok(());
        }
        let current = self.strict_dynamic_bytes()?;
        let requested = current.checked_add(additional).ok_or_else(|| {
            DecoderError::InvalidParam(format!("native AV1 {label} size overflows"))
        })?;
        if requested > self.strict_dynamic_reserved_bytes {
            return Err(DecoderError::InvalidParam(format!(
                "native AV1 {label} exceeds admitted budget"
            )));
        }
        Ok(())
    }

    pub(super) fn strict_dynamic_vec_with_scratch<T: Default + Clone>(
        &self,
        count: usize,
        label: &str,
        scratch: &PaletteScratchLedger,
    ) -> Result<Vec<T>, DecoderError> {
        if !self.strict_dynamic_enabled {
            return Err(DecoderError::InvalidParam(
                "strict scratch allocation requested on legacy tile decoder".to_string(),
            ));
        }
        let requested = capacity_bytes::<T>(count, label)?;
        let current = self
            .strict_dynamic_bytes()?
            .checked_add(scratch.bytes())
            .and_then(|bytes| bytes.checked_add(requested))
            .ok_or_else(|| {
                DecoderError::InvalidParam(format!("native AV1 {label} size overflows"))
            })?;
        if current > self.strict_dynamic_reserved_bytes {
            return Err(DecoderError::InvalidParam(format!(
                "native AV1 {label} exceeds admitted budget"
            )));
        }
        let replacement = fresh_replacement::<T>(count, label)?;
        let actual = replacement.actual_bytes();
        let actual_total = self
            .strict_dynamic_bytes()?
            .checked_add(scratch.bytes())
            .and_then(|bytes| bytes.checked_add(actual))
            .ok_or_else(|| {
                DecoderError::InvalidParam(format!("native AV1 {label} size overflows"))
            })?;
        if actual_total > self.strict_dynamic_reserved_bytes {
            return Err(DecoderError::InvalidParam(format!(
                "native AV1 {label} actual capacity exceeds admitted budget"
            )));
        }
        scratch.add(actual)?;
        let mut values = replacement.into_vec();
        values.resize(count, T::default());
        Ok(values)
    }

    pub(super) fn strict_dynamic_clone_with_scratch<T: Default + Clone>(
        &self,
        source: &[T],
        label: &str,
        scratch: &PaletteScratchLedger,
    ) -> Result<Vec<T>, DecoderError> {
        if !self.strict_dynamic_enabled {
            return Err(DecoderError::InvalidParam(
                "strict scratch clone requested on legacy tile decoder".to_string(),
            ));
        }
        let requested = capacity_bytes::<T>(source.len(), label)?;
        let current = self
            .strict_dynamic_bytes()?
            .checked_add(scratch.bytes())
            .and_then(|bytes| bytes.checked_add(requested))
            .ok_or_else(|| {
                DecoderError::InvalidParam(format!("native AV1 {label} size overflows"))
            })?;
        if current > self.strict_dynamic_reserved_bytes {
            return Err(DecoderError::InvalidParam(format!(
                "native AV1 {label} exceeds admitted budget"
            )));
        }
        let replacement = fresh_replacement::<T>(source.len(), label)?;
        let actual = replacement.actual_bytes();
        let actual_total = self
            .strict_dynamic_bytes()?
            .checked_add(scratch.bytes())
            .and_then(|bytes| bytes.checked_add(actual))
            .ok_or_else(|| {
                DecoderError::InvalidParam(format!("native AV1 {label} size overflows"))
            })?;
        if actual_total > self.strict_dynamic_reserved_bytes {
            return Err(DecoderError::InvalidParam(format!(
                "native AV1 {label} actual capacity exceeds admitted budget"
            )));
        }
        let mut values = replacement.into_vec();
        values.extend_from_slice(source);
        Ok(values)
    }

    pub(super) fn strict_dynamic_bytes(&self) -> Result<usize, DecoderError> {
        let palette = self
            .y_palette_colors_grid
            .iter()
            .chain(self.u_palette_colors_grid.iter())
            .flatten()
            .try_fold(0usize, |total, values| {
                total
                    .checked_add(capacity_bytes::<u16>(
                        values.capacity(),
                        "tile nested palette colors",
                    )?)
                    .ok_or_else(|| {
                        DecoderError::InvalidParam(
                            "native AV1 nested palette size overflows".to_string(),
                        )
                    })
            })?;
        palette
            .checked_add(self.coefficient_scan_cache.actual_bytes()?)
            .ok_or_else(|| {
                DecoderError::InvalidParam("native AV1 dynamic tile size overflows".to_string())
            })
    }

    pub(super) fn strict_dynamic_reserved_bytes(&self) -> usize {
        self.strict_dynamic_reserved_bytes
    }

    pub(super) fn strict_dynamic_enabled(&self) -> bool {
        self.strict_dynamic_enabled
    }

    pub(super) fn disable_strict_dynamic(&mut self) {
        self.strict_dynamic_enabled = false;
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

    pub(super) fn set_order_hint_bits(&mut self, order_hint_bits: u8) {
        self.order_hint_bits = order_hint_bits;
    }

    pub(super) fn set_current_inter_compound(&mut self, compound: bool) {
        self.current_inter_compound = compound;
    }

    pub(super) fn motion_field(&self, frame: &FrameHeader, order_hint_bits: u8) -> MotionField {
        // AOM stores the frame-MV map at 8x8-MI granularity.  Keep the
        // public dimensions unchanged for tile merging, but populate only
        // the corresponding even 4x4-MI coordinates so temporal samples can
        // distinguish an unavailable source block from a propagated MV.
        let mut reference_frames = vec![None; self.mi_cols * self.mi_rows];
        let mut motion_vectors = vec![None; self.mi_cols * self.mi_rows];
        for row in (0..self.mi_rows).step_by(2) {
            for col in (0..self.mi_cols).step_by(2) {
                // `av1_copy_frame_mvs` writes every decoded 4x4 block into
                // the containing 8x8 frame-MV cell.  In raster order the
                // bottom-right 4x4 block is therefore the value retained for
                // a complete 8x8 cell (with edge clamping for odd dimensions).
                let source_row = (row + 1).min(self.mi_rows.saturating_sub(1));
                let source_col = (col + 1).min(self.mi_cols.saturating_sub(1));
                let source_index = source_row * self.mi_cols + source_col;
                let destination_index = row * self.mi_cols + col;
                let mut selected = None;
                for (reference_type, motion_vector) in [
                    (
                        self.reference_frame_type_grid[source_index],
                        self.motion_vector_grid[source_index],
                    ),
                    (
                        self.reference_frame_secondary_type_grid[source_index],
                        self.motion_vector_secondary_grid[source_index],
                    ),
                ] {
                    let (Some(reference_type), Some(motion_vector)) =
                        (reference_type, motion_vector)
                    else {
                        continue;
                    };
                    let is_usable = frame
                        .reference_order_hints
                        .get(usize::from(reference_type))
                        .copied()
                        .flatten()
                        .map(|hint| {
                            // `av1_copy_frame_mvs` excludes both future and
                            // same-order references (`ref_frame_side != 0`).
                            // Retaining a same-order MV creates temporal stack
                            // candidates that are not present in the coded
                            // frame's entropy contexts.
                            relative_order_hint_distance(order_hint_bits, hint, frame.order_hint)
                                < 0
                        })
                        .unwrap_or(true);
                    if is_usable {
                        selected = Some((reference_type, motion_vector));
                    }
                }
                if let Some((reference_type, motion_vector)) = selected {
                    reference_frames[destination_index] = Some(reference_type);
                    motion_vectors[destination_index] = Some(motion_vector);
                }
            }
        }
        MotionField {
            mi_cols: self.mi_cols,
            mi_rows: self.mi_rows,
            order_hint_bits,
            order_hint: frame.order_hint,
            reference_order_hints: frame.reference_order_hints,
            reference_frame_indices: frame.reference_frame_indices,
            projected: false,
            reference_frames,
            motion_vectors,
            reference_offsets: vec![None; self.mi_cols * self.mi_rows],
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
            && !frame.segmentation.segment_skip[usize::from(segment_id)]
            && frame.segmentation.segment_reference_frame[usize::from(segment_id)].is_none()
            && !frame.segmentation.segment_global_mv[usize::from(segment_id)];
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

    pub(super) fn set_luma_tx_type(&mut self, transform: TransformBlock, tx_type: TxType) {
        let start_col = transform.x >> 2;
        let start_row = transform.y >> 2;
        let end_col = (start_col + (transform.tx_size.width() >> 2)).min(self.mi_cols);
        let end_row = (start_row + (transform.tx_size.height() >> 2)).min(self.mi_rows);
        for row in start_row..end_row {
            self.luma_tx_type_grid[row * self.mi_cols + start_col..row * self.mi_cols + end_col]
                .fill(tx_type);
        }
    }

    pub(super) fn luma_tx_type_at(&self, x: usize, y: usize) -> TxType {
        self.luma_tx_type_grid
            .get((y >> 2) * self.mi_cols + (x >> 2))
            .copied()
            .unwrap_or(TxType::DctDct)
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
                predicted + diff.div_ceil(2)
            } else {
                predicted - diff / 2
            }
        } else {
            diff
        }
    } else if diff <= 2 * (max - predicted - 1) {
        if diff & 1 != 0 {
            predicted + diff.div_ceil(2)
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
#[path = "tile_decode/tests/tile_decode_palette_strict.rs"]
mod palette_strict_tests;

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
