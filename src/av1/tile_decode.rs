use super::cdf::CdfContext;
use super::decode::TileDecodePlan;
use super::entropy::EntropyDecoder;
use super::frame::{FrameHeader, RestorationParams};
use super::quant::QuantState;
use super::syntax::{BlockSize, Partition, TxSize, TxType};
use super::transform::{TransformBlock, coefficient_scan};
use crate::DecoderError;

mod block_syntax;
mod coefficient;
mod coefficient_context;
mod context_grid;
mod decode_flow;
mod diagnostic;
mod palette;
mod partition_syntax;
mod public_api;
mod reconstruction;
mod residual_preview;
mod restoration_syntax;

use coefficient::{CoefficientRead, EntropyCoefficientSource, decode_coefficients};
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
use diagnostic::{CoeffBaseProbe, CoeffBaseRead, CoeffBrProbe, CoeffSignRead, TxTypeProbe};
pub use public_api::{
    decode_first_luma_block, decode_first_luma_transform, decode_luma_root_block_prefix,
    decode_luma_root_blocks, prepare_tile_entropy, probe_first_block_residuals,
    probe_tile_block_modes, probe_tile_partitions,
};
use residual_preview::build_residual_preview;

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

    fn txb_context(&self, block_size: BlockSize, transform: TransformBlock) -> TxbContext {
        let contexts = &self.plane_entropy_contexts[transform.plane];
        txb_context(block_size, transform, &contexts.above, &contexts.left)
    }

    fn set_txb_entropy_context(&mut self, transform: TransformBlock, value: u8) {
        let contexts = &mut self.plane_entropy_contexts[transform.plane];
        set_txb_entropy_context(transform, value, &mut contexts.above, &mut contexts.left);
    }

    fn skip_context(&self, x: usize, y: usize) -> usize {
        usize::from(self.above_skip_context(x, y)) + usize::from(self.left_skip_context(x, y))
    }

    fn above_skip_context(&self, x: usize, y: usize) -> bool {
        if y < 4 {
            return false;
        }
        self.skip_at_mi(x >> 2, (y >> 2).saturating_sub(1))
            .unwrap_or(false)
    }

    fn left_skip_context(&self, x: usize, y: usize) -> bool {
        if x < 4 {
            return false;
        }
        self.skip_at_mi((x >> 2).saturating_sub(1), y >> 2)
            .unwrap_or(false)
    }

    fn skip_at_mi(&self, mi_col: usize, mi_row: usize) -> Option<bool> {
        if mi_col >= self.mi_cols || mi_row >= self.mi_rows {
            return None;
        }
        self.skip_grid[mi_row * self.mi_cols + mi_col]
    }

    fn set_skip_context(&mut self, x: usize, y: usize, block_size: BlockSize, skip: bool) {
        let start_col = x >> 2;
        let start_row = y >> 2;
        let end_col = ((x + block_size.width()).min(self.mi_cols << 2) + 3) >> 2;
        let end_row = ((y + block_size.height()).min(self.mi_rows << 2) + 3) >> 2;
        for mi_row in start_row..end_row.min(self.mi_rows) {
            for mi_col in start_col..end_col.min(self.mi_cols) {
                self.skip_grid[mi_row * self.mi_cols + mi_col] = Some(skip);
            }
        }
    }

    fn tx_size_context(&self, x: usize, y: usize, block_size: BlockSize) -> usize {
        let max_tx_size = block_size.largest_supported_tx_size();
        let has_above = y >= 4;
        let has_left = x >= 4;
        let above = has_above
            && self.above_txfm_context.get(x >> 2).copied().unwrap_or(0) >= max_tx_size.width();
        let left = has_left
            && self.left_txfm_context.get(y >> 2).copied().unwrap_or(0) >= max_tx_size.height();
        match (has_above, has_left) {
            (true, true) => usize::from(above) + usize::from(left),
            (true, false) => usize::from(above),
            (false, true) => usize::from(left),
            (false, false) => 0,
        }
    }

    fn set_txfm_context(&mut self, x: usize, y: usize, block_size: BlockSize, tx_size: TxSize) {
        let start_col = x >> 2;
        let start_row = y >> 2;
        let end_col = ((x + block_size.width()).min(self.mi_cols << 2) + 3) >> 2;
        let end_row = ((y + block_size.height()).min(self.mi_rows << 2) + 3) >> 2;
        for mi_col in start_col..end_col.min(self.mi_cols) {
            self.above_txfm_context[mi_col] = tx_size.width();
        }
        for mi_row in start_row..end_row.min(self.mi_rows) {
            self.left_txfm_context[mi_row] = tx_size.height();
        }
    }

    fn partition_context(
        &self,
        tile: &TileDecodePlan,
        x: usize,
        y: usize,
        block_size: BlockSize,
    ) -> usize {
        let mi_col = x >> 2;
        let mi_row = y >> 2;
        let above = if mi_row <= tile.mi_row_start as usize {
            0
        } else {
            self.above_partition_context
                .get(mi_col)
                .copied()
                .unwrap_or(0)
        };
        let left = if mi_col <= tile.mi_col_start as usize {
            0
        } else {
            self.left_partition_context
                .get(mi_row)
                .copied()
                .unwrap_or(0)
        };
        partition_plane_context(above, left, block_size)
    }

    fn update_partition_context(
        &mut self,
        x: usize,
        y: usize,
        subsize: BlockSize,
        context_size: BlockSize,
    ) {
        let mi_col = x >> 2;
        let mi_row = y >> 2;
        let width_mi = context_size.width() >> 2;
        let height_mi = context_size.height() >> 2;
        let (above, left) = subsize.partition_contexts();
        for context in self
            .above_partition_context
            .iter_mut()
            .skip(mi_col)
            .take(width_mi)
        {
            *context = above;
        }
        for context in self
            .left_partition_context
            .iter_mut()
            .skip(mi_row)
            .take(height_mi)
        {
            *context = left;
        }
    }

    fn update_ext_partition_context(
        &mut self,
        x: usize,
        y: usize,
        block_size: BlockSize,
        partition: Partition,
    ) -> Result<(), DecoderError> {
        if block_size.width() < 8 || block_size.height() < 8 {
            return Ok(());
        }
        let split = block_size.split_subsize().ok_or_else(|| {
            DecoderError::Bitstream(format!(
                "AV1 partition context split size is missing for {block_size:?}"
            ))
        })?;
        let half = block_size.width() / 2;
        match partition {
            Partition::Split if block_size != BlockSize::Block8x8 => {}
            Partition::Split | Partition::None => {
                let subsize = partition_subsize(block_size, partition)?;
                self.update_partition_context(x, y, subsize, block_size);
            }
            Partition::Horizontal
            | Partition::Vertical
            | Partition::Horizontal4
            | Partition::Vertical4 => {
                let subsize = partition_subsize(block_size, partition)?;
                self.update_partition_context(x, y, subsize, block_size);
            }
            Partition::HorizontalA => {
                let subsize = partition_subsize(block_size, partition)?;
                self.update_partition_context(x, y, split, subsize);
                self.update_partition_context(x, y + half, subsize, subsize);
            }
            Partition::HorizontalB => {
                let subsize = partition_subsize(block_size, partition)?;
                self.update_partition_context(x, y, subsize, subsize);
                self.update_partition_context(x, y + half, split, subsize);
            }
            Partition::VerticalA => {
                let subsize = partition_subsize(block_size, partition)?;
                self.update_partition_context(x, y, split, subsize);
                self.update_partition_context(x + half, y, subsize, subsize);
            }
            Partition::VerticalB => {
                let subsize = partition_subsize(block_size, partition)?;
                self.update_partition_context(x, y, subsize, subsize);
                self.update_partition_context(x + half, y, split, subsize);
            }
        }
        Ok(())
    }

    pub fn read_first_transform_residual(
        &mut self,
        tile_id: u32,
        frame: &FrameHeader,
        block_mode: &BlockModeProbe,
        transforms: &[TransformBlock],
        quant_state: QuantState,
        bit_depth: u8,
    ) -> Result<ResidualProbe, DecoderError> {
        let transform_count = transforms.len();
        let first_transform = transforms.first().copied();
        if block_mode.skip {
            for transform in transforms.iter().copied() {
                self.set_txb_entropy_context(transform, 0);
            }
            return Ok(ResidualProbe {
                tile_id,
                block_size: block_mode.block_size,
                skipped: true,
                transform_count,
                zero_transform_count: transform_count,
                first_tx_size: first_transform.map(|transform| transform.tx_size),
                first_non_zero_transform_index: None,
                first_non_zero_transform: None,
                first_non_zero_tx_size: None,
                tx_type_read: false,
                tx_type_set: None,
                tx_type_symbol: None,
                tx_type: None,
                txb_skip_context: None,
                all_zero_symbol: None,
                first_transform_all_zero: true,
                eob_multisize: None,
                eob_pt_symbol: None,
                eob_pt: None,
                eob_base: None,
                eob_extra_context: None,
                eob_extra_symbol: None,
                eob_extra_literal_bits: None,
                eob: None,
                coeff_base_eob_context: None,
                coeff_base_eob_symbol: None,
                coeff_base_eob_level: None,
                regular_coeff_base_count: None,
                regular_coeff_base_decoded_count: None,
                coeff_base_non_zero_count: None,
                coeff_base_range_count: None,
                coeff_br_decoded_count: None,
                first_coeff_br_scan_index: None,
                first_coeff_br_position: None,
                first_coeff_br_context: None,
                first_coeff_br_symbol: None,
                first_coeff_br_level: None,
                sign_decoded_count: None,
                dc_sign_context: None,
                dc_sign_symbol: None,
                first_ac_sign_scan_index: None,
                first_ac_sign_bit: None,
                golomb_decoded_count: None,
                first_golomb_scan_index: None,
                first_golomb_value: None,
                signed_coeff_non_zero_count: None,
                first_signed_coeff_scan_index: None,
                first_signed_coeff_position: None,
                first_signed_coeff_value: None,
                dequant_non_zero_count: None,
                first_dequant_coeff_position: None,
                first_dequant_coeff_value: None,
                residual_preview_tx_type: None,
                residual_preview_sample_count: None,
                first_residual_preview_sample: None,
                first_coeff_base_scan_index: None,
                first_coeff_base_position: None,
                first_coeff_base_context: None,
                first_coeff_base_reference_magnitude: None,
                first_coeff_base_symbol: None,
                first_coeff_base_level: None,
                first_quantized_coefficients: None,
                bit_position_after: block_mode.bit_position_after,
            });
        }

        let Some(first_transform) = first_transform else {
            return Ok(ResidualProbe {
                tile_id,
                block_size: block_mode.block_size,
                skipped: false,
                transform_count,
                zero_transform_count: 0,
                first_tx_size: None,
                first_non_zero_transform_index: None,
                first_non_zero_transform: None,
                first_non_zero_tx_size: None,
                tx_type_read: false,
                tx_type_set: None,
                tx_type_symbol: None,
                tx_type: None,
                txb_skip_context: None,
                all_zero_symbol: None,
                first_transform_all_zero: false,
                eob_multisize: None,
                eob_pt_symbol: None,
                eob_pt: None,
                eob_base: None,
                eob_extra_context: None,
                eob_extra_symbol: None,
                eob_extra_literal_bits: None,
                eob: None,
                coeff_base_eob_context: None,
                coeff_base_eob_symbol: None,
                coeff_base_eob_level: None,
                regular_coeff_base_count: None,
                regular_coeff_base_decoded_count: None,
                coeff_base_non_zero_count: None,
                coeff_base_range_count: None,
                coeff_br_decoded_count: None,
                first_coeff_br_scan_index: None,
                first_coeff_br_position: None,
                first_coeff_br_context: None,
                first_coeff_br_symbol: None,
                first_coeff_br_level: None,
                sign_decoded_count: None,
                dc_sign_context: None,
                dc_sign_symbol: None,
                first_ac_sign_scan_index: None,
                first_ac_sign_bit: None,
                golomb_decoded_count: None,
                first_golomb_scan_index: None,
                first_golomb_value: None,
                signed_coeff_non_zero_count: None,
                first_signed_coeff_scan_index: None,
                first_signed_coeff_position: None,
                first_signed_coeff_value: None,
                dequant_non_zero_count: None,
                first_dequant_coeff_position: None,
                first_dequant_coeff_value: None,
                residual_preview_tx_type: None,
                residual_preview_sample_count: None,
                first_residual_preview_sample: None,
                first_coeff_base_scan_index: None,
                first_coeff_base_position: None,
                first_coeff_base_context: None,
                first_coeff_base_reference_magnitude: None,
                first_coeff_base_symbol: None,
                first_coeff_base_level: None,
                first_quantized_coefficients: None,
                bit_position_after: block_mode.bit_position_after,
            });
        };

        let mut first_txb_skip_context_value = None;
        let mut first_all_zero_symbol = None;
        let mut first_transform_all_zero = true;
        let mut zero_transform_count = 0usize;
        let mut first_non_zero_transform = None;
        let mut first_non_zero_transform_index = None;
        let mut first_non_zero_txb_context = None;

        for (index, transform) in transforms.iter().copied().enumerate() {
            let txb_context = self.txb_context(block_mode.block_size, transform);
            let all_zero_symbol = self.reader.read_symbol(
                self.cdf
                    .txb_skip_cdf_mut(transform.tx_size.coeff_cdf_index(), txb_context.skip),
            )?;
            if index == 0 {
                first_txb_skip_context_value = Some(txb_context.skip);
                first_all_zero_symbol = Some(all_zero_symbol);
                first_transform_all_zero = all_zero_symbol != 0;
            }
            if all_zero_symbol != 0 {
                self.set_txb_entropy_context(transform, 0);
                zero_transform_count += 1;
                continue;
            }
            first_non_zero_transform = Some(transform);
            first_non_zero_transform_index = Some(index);
            first_non_zero_txb_context = Some(txb_context);
            break;
        }

        let (
            eob_multisize,
            eob_pt_symbol,
            eob_pt,
            eob_base,
            eob_extra_context,
            eob_extra_symbol,
            eob_extra_literal_bits,
            eob,
            tx_type_read,
            tx_type_set,
            tx_type_symbol,
            tx_type,
            coeff_base_eob_context,
            coeff_base_eob_symbol,
            coeff_base_eob_level,
            regular_coeff_base_count,
            regular_coeff_base_decoded_count,
            coeff_base_non_zero_count,
            coeff_base_range_count,
            coeff_br_decoded_count,
            first_coeff_br_scan_index,
            first_coeff_br_position,
            first_coeff_br_context,
            first_coeff_br_symbol,
            first_coeff_br_level,
            sign_decoded_count,
            dc_sign_context,
            dc_sign_symbol,
            first_ac_sign_scan_index,
            first_ac_sign_bit,
            golomb_decoded_count,
            first_golomb_scan_index,
            first_golomb_value,
            signed_coeff_non_zero_count,
            first_signed_coeff_scan_index,
            first_signed_coeff_position,
            first_signed_coeff_value,
            dequant_non_zero_count,
            first_dequant_coeff_position,
            first_dequant_coeff_value,
            residual_preview_tx_type,
            residual_preview_sample_count,
            first_residual_preview_sample,
            first_coeff_base_scan_index,
            first_coeff_base_position,
            first_coeff_base_context,
            first_coeff_base_reference_magnitude,
            first_coeff_base_symbol,
            first_coeff_base_level,
            first_quantized_coefficients,
        ) = if let Some(non_zero_transform) = first_non_zero_transform {
            let tx_type_probe = self.read_intra_tx_type(frame, block_mode, non_zero_transform)?;
            let plane_type = usize::from(non_zero_transform.plane > 0);
            let coefficient_read = self.read_coefficient_state(
                non_zero_transform.tx_size,
                tx_type_probe.tx_type,
                plane_type,
                first_non_zero_txb_context
                    .expect("non-zero transform should retain its txb context")
                    .dc_sign,
            )?;
            let coeff_base_read = coefficient_read.base;
            self.set_txb_entropy_context(
                non_zero_transform,
                coefficient_entropy_context(&coeff_base_read.base_levels),
            );
            debug_assert_eq!(
                coeff_base_read.base_levels.len(),
                non_zero_transform.tx_size.sample_count()
            );
            let residual_preview = build_residual_preview(
                non_zero_transform,
                &coeff_base_read.base_levels,
                quant_state,
                bit_depth,
                tx_type_probe.tx_type,
            )?;
            (
                Some(coefficient_read.eob_multisize),
                Some(coefficient_read.eob_pt_symbol),
                Some(coefficient_read.eob_pt),
                Some(coefficient_read.eob_base),
                coefficient_read.eob_extra_context,
                coefficient_read.eob_extra_symbol,
                Some(coefficient_read.eob_extra_literal_bits),
                Some(coefficient_read.eob),
                tx_type_probe.read,
                tx_type_probe.set,
                tx_type_probe.symbol,
                Some(tx_type_probe.tx_type),
                Some(coefficient_read.coeff_base_eob_context),
                Some(coefficient_read.coeff_base_eob_symbol),
                Some(coefficient_read.coeff_base_eob_level),
                Some(coeff_base_read.probe.remaining_count),
                Some(coeff_base_read.probe.decoded_count),
                Some(coeff_base_read.non_zero_count),
                Some(coeff_base_read.base_range_count),
                Some(coeff_base_read.coeff_br_symbol_count),
                coeff_base_read.first_coeff_br.map(|first| first.scan_index),
                coeff_base_read.first_coeff_br.map(|first| first.position),
                coeff_base_read.first_coeff_br.map(|first| first.context),
                coeff_base_read.first_coeff_br.map(|first| first.symbol),
                coeff_base_read
                    .first_coeff_br
                    .map(|first| first.level_after_symbol),
                Some(coeff_base_read.signs.sign_count),
                coeff_base_read.signs.dc_sign_context,
                coeff_base_read.signs.dc_sign_symbol,
                coeff_base_read.signs.first_ac_sign_scan_index,
                coeff_base_read.signs.first_ac_sign_bit,
                Some(coeff_base_read.signs.golomb_count),
                coeff_base_read.signs.first_golomb_scan_index,
                coeff_base_read.signs.first_golomb_value,
                Some(coeff_base_read.signed_non_zero_count),
                coeff_base_read
                    .first_signed_coeff
                    .map(|first| first.scan_index),
                coeff_base_read
                    .first_signed_coeff
                    .map(|first| first.position),
                coeff_base_read.first_signed_coeff.map(|first| first.value),
                residual_preview
                    .as_ref()
                    .map(|preview| preview.dequant_non_zero_count),
                residual_preview
                    .as_ref()
                    .and_then(|preview| preview.first_dequant_coeff)
                    .map(|first| first.position),
                residual_preview
                    .as_ref()
                    .and_then(|preview| preview.first_dequant_coeff)
                    .map(|first| first.value),
                residual_preview.as_ref().map(|preview| preview.tx_type),
                residual_preview
                    .as_ref()
                    .map(|preview| preview.residual_sample_count),
                residual_preview
                    .as_ref()
                    .and_then(|preview| preview.first_residual_sample),
                coeff_base_read.probe.scan_index,
                coeff_base_read.probe.position,
                coeff_base_read.probe.context,
                coeff_base_read.probe.reference_magnitude,
                coeff_base_read.probe.symbol,
                coeff_base_read.probe.level,
                Some(coeff_base_read.base_levels),
            )
        } else {
            (
                None, None, None, None, None, None, None, None, false, None, None, None, None,
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None, None,
            )
        };

        Ok(ResidualProbe {
            tile_id,
            block_size: block_mode.block_size,
            skipped: false,
            transform_count,
            zero_transform_count,
            first_tx_size: Some(first_transform.tx_size),
            first_non_zero_transform_index,
            first_non_zero_transform,
            first_non_zero_tx_size: first_non_zero_transform.map(|transform| transform.tx_size),
            tx_type_read,
            tx_type_set,
            tx_type_symbol,
            tx_type,
            txb_skip_context: first_txb_skip_context_value,
            all_zero_symbol: first_all_zero_symbol,
            first_transform_all_zero,
            eob_multisize,
            eob_pt_symbol,
            eob_pt,
            eob_base,
            eob_extra_context,
            eob_extra_symbol,
            eob_extra_literal_bits,
            eob,
            coeff_base_eob_context,
            coeff_base_eob_symbol,
            coeff_base_eob_level,
            regular_coeff_base_count,
            regular_coeff_base_decoded_count,
            coeff_base_non_zero_count,
            coeff_base_range_count,
            coeff_br_decoded_count,
            first_coeff_br_scan_index,
            first_coeff_br_position,
            first_coeff_br_context,
            first_coeff_br_symbol,
            first_coeff_br_level,
            sign_decoded_count,
            dc_sign_context,
            dc_sign_symbol,
            first_ac_sign_scan_index,
            first_ac_sign_bit,
            golomb_decoded_count,
            first_golomb_scan_index,
            first_golomb_value,
            signed_coeff_non_zero_count,
            first_signed_coeff_scan_index,
            first_signed_coeff_position,
            first_signed_coeff_value,
            dequant_non_zero_count,
            first_dequant_coeff_position,
            first_dequant_coeff_value,
            residual_preview_tx_type,
            residual_preview_sample_count,
            first_residual_preview_sample,
            first_coeff_base_scan_index,
            first_coeff_base_position,
            first_coeff_base_context,
            first_coeff_base_reference_magnitude,
            first_coeff_base_symbol,
            first_coeff_base_level,
            first_quantized_coefficients,
            bit_position_after: self.reader.bit_position(),
        })
    }

    fn read_intra_tx_type(
        &mut self,
        frame: &FrameHeader,
        block_mode: &BlockModeProbe,
        transform: TransformBlock,
    ) -> Result<TxTypeProbe, DecoderError> {
        if transform.plane != 0
            || frame.base_q_idx == 0
            || transform.tx_size.width() >= 32
            || transform.tx_size.height() >= 32
        {
            return Ok(TxTypeProbe {
                read: false,
                set: None,
                symbol: None,
                tx_type: TxType::DctDct,
            });
        }
        let intra_mode = block_mode
            .filter_intra_mode
            .map(filter_intra_mode_to_tx_cdf_mode)
            .transpose()?
            .unwrap_or(block_mode.y_mode_symbol);
        let (set, tx_size_context) =
            intra_ext_tx_set_context(frame.reduced_tx_set, transform.tx_size).ok_or_else(|| {
                DecoderError::Bitstream(format!(
                    "AV1 intra tx_type is not signaled for {:?}",
                    transform.tx_size
                ))
            })?;
        if set == 2 {
            let symbol = self.reader.read_symbol(
                self.cdf
                    .intra_ext_tx_set2_cdf_mut(tx_size_context, intra_mode),
            )?;
            let tx_type = TxType::from_intra_ext_tx_set2_symbol(symbol).ok_or_else(|| {
                DecoderError::Bitstream(format!(
                    "AV1 intra tx_type set2 symbol {symbol} is invalid"
                ))
            })?;
            Ok(TxTypeProbe {
                read: true,
                set: Some(2),
                symbol: Some(symbol),
                tx_type,
            })
        } else {
            let symbol = self.reader.read_symbol(
                self.cdf
                    .intra_ext_tx_set1_cdf_mut(tx_size_context, intra_mode),
            )?;
            let tx_type = TxType::from_intra_ext_tx_set1_symbol(symbol).ok_or_else(|| {
                DecoderError::Bitstream(format!(
                    "AV1 intra tx_type set1 symbol {symbol} is invalid"
                ))
            })?;
            Ok(TxTypeProbe {
                read: true,
                set: Some(1),
                symbol: Some(symbol),
                tx_type,
            })
        }
    }

    fn read_decoded_transform(
        &mut self,
        frame: &FrameHeader,
        block_mode: &BlockModeProbe,
        transform: TransformBlock,
        dc_sign_context: usize,
    ) -> Result<DecodedTransform, DecoderError> {
        let tx_type = self
            .read_intra_tx_type(frame, block_mode, transform)?
            .tx_type;
        let plane_type = usize::from(transform.plane > 0);
        let coefficient_read =
            self.read_coefficient_state(transform.tx_size, tx_type, plane_type, dc_sign_context)?;
        Ok(DecodedTransform {
            transform,
            tx_type,
            coefficients: coefficient_read.base.base_levels,
        })
    }

    fn read_coefficient_state(
        &mut self,
        tx_size: TxSize,
        tx_type: TxType,
        plane_type: usize,
        dc_sign_context: usize,
    ) -> Result<CoefficientRead, DecoderError> {
        let mut source = EntropyCoefficientSource::new(&mut self.reader, &mut self.cdf);
        decode_coefficients(&mut source, tx_size, tx_type, plane_type, dc_sign_context)
    }
}

fn intra_ext_tx_set_context(reduced_tx_set: bool, tx_size: TxSize) -> Option<(usize, usize)> {
    match tx_size {
        TxSize::Tx4x4 => Some((if reduced_tx_set { 2 } else { 1 }, 0)),
        TxSize::Tx8x8 => Some((if reduced_tx_set { 2 } else { 1 }, 1)),
        TxSize::Tx16x16 => Some((2, 2)),
        TxSize::Tx32x32 | TxSize::Tx64x64 => None,
    }
}

fn filter_intra_mode_to_tx_cdf_mode(filter_intra_mode: usize) -> Result<usize, DecoderError> {
    match filter_intra_mode {
        0 => Ok(0),
        1 => Ok(1),
        2 => Ok(2),
        3 => Ok(6),
        4 => Ok(0),
        _ => Err(DecoderError::Bitstream(format!(
            "AV1 filter intra mode {filter_intra_mode} is invalid"
        ))),
    }
}

fn partition_subsize(
    block_size: BlockSize,
    partition: Partition,
) -> Result<BlockSize, DecoderError> {
    let subsize = match partition {
        Partition::None => Some(block_size),
        Partition::Horizontal | Partition::HorizontalA | Partition::HorizontalB => {
            block_size.horizontal_subsize()
        }
        Partition::Vertical | Partition::VerticalA | Partition::VerticalB => {
            block_size.vertical_subsize()
        }
        Partition::Split => block_size.split_subsize(),
        Partition::Horizontal4 => block_size.horizontal_4_subsize(),
        Partition::Vertical4 => block_size.vertical_4_subsize(),
    };
    subsize.ok_or_else(|| {
        DecoderError::Bitstream(format!(
            "AV1 partition {partition:?} has no subsize for {block_size:?}"
        ))
    })
}

fn partition_symbol(partition: Partition) -> usize {
    match partition {
        Partition::None => 0,
        Partition::Horizontal => 1,
        Partition::Vertical => 2,
        Partition::Split => 3,
        Partition::HorizontalA => 4,
        Partition::HorizontalB => 5,
        Partition::VerticalA => 6,
        Partition::VerticalB => 7,
        Partition::Horizontal4 => 8,
        Partition::Vertical4 => 9,
    }
}

fn partition_plane_context(above: u8, left: u8, block_size: BlockSize) -> usize {
    let bit = block_size.width_mi_log2().saturating_sub(1);
    usize::from((above >> bit) & 1) + usize::from((left >> bit) & 1) * 2
}

fn ceil_log2(value: usize) -> usize {
    if value <= 1 {
        0
    } else {
        usize::BITS as usize - (value - 1).leading_zeros() as usize
    }
}

#[cfg(test)]
#[path = "tests/tile_decode_coeff.rs"]
mod coeff_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av1::transform::plan_transform_blocks_with_tx_size;
    use crate::av1::{
        PredictionMode, UvPredictionMode, alloc_frame_buffers, build_still_decode_plan,
        parse_frame_header, parse_sequence_header, parse_tile_group,
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
    fn reads_sample_root_partition_symbol() {
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
        let tile_payload = &tile_group.tiles[0];
        let payload = &frame_payload[tile_payload.offset..tile_payload.offset + tile_payload.len];
        let mut decoder = TileDecoder::new(payload, &frame).unwrap();

        let probe = decoder
            .read_root_partition(&plan.tiles[0], &sequence)
            .unwrap();

        assert_eq!(probe.tile_id, 0);
        assert_eq!(probe.block_size, BlockSize::Block128x128);
        assert_eq!(probe.symbol, 3);
        assert_eq!(probe.partition, Partition::Split);
        assert!(probe.bit_position_after >= 15);
    }

    #[test]
    fn reads_sample_first_block_mode_symbols() {
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

        let probes =
            probe_tile_block_modes(frame_payload, &tile_group, &sequence, &frame, &plan).unwrap();

        assert_eq!(probes.len(), 1);
        assert_eq!(probes[0].tile_id, 0);
        assert_eq!(probes[0].block_size, BlockSize::Block64x64);
        assert_eq!(probes[0].skip_symbol, 0);
        assert_eq!(probes[0].cdef_idx, Some(0));
        assert_eq!(probes[0].y_mode_symbol, 0);
        assert_eq!(probes[0].y_mode, PredictionMode::Dc);
        assert_eq!(probes[0].uv_mode_symbol, Some(0));
        assert_eq!(
            probes[0].uv_mode,
            Some(UvPredictionMode::Intra(PredictionMode::Dc))
        );
        assert_eq!(probes[0].tx_size_symbol, Some(0));
        assert_eq!(probes[0].tx_size, TxSize::Tx64x64);
        assert!(probes[0].bit_position_after > 15);
    }

    #[test]
    fn plans_sample_first_block_transforms() {
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
        let probes =
            probe_tile_block_modes(frame_payload, &tile_group, &sequence, &frame, &plan).unwrap();

        let transforms = plan_transform_blocks_with_tx_size(
            0,
            0,
            0,
            probes[0].block_size,
            probes[0].tx_size,
            plan.width,
            plan.height,
        );

        assert_eq!(transforms.len(), 1);
        assert!(transforms.iter().all(|tx| tx.plane == 0));
        assert!(transforms.iter().all(|tx| tx.tx_size == probes[0].tx_size));
    }

    #[test]
    fn probes_sample_first_block_residual_plan() {
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
        let probes =
            probe_first_block_residuals(frame_payload, &tile_group, &sequence, &frame, &plan)
                .unwrap();

        assert_eq!(probes.len(), 1);
        assert_eq!(probes[0].tile_id, 0);
        assert_eq!(probes[0].block_size, BlockSize::Block64x64);
        let first_tx_size = probes[0]
            .first_tx_size
            .expect("sample first transform size should be known");
        let transform_count = (probes[0].block_size.width() / first_tx_size.width())
            * (probes[0].block_size.height() / first_tx_size.height());
        let tx_sample_count = first_tx_size.sample_count();
        assert_eq!(probes[0].transform_count, transform_count);
        if probes[0].skipped {
            assert_eq!(probes[0].zero_transform_count, transform_count);
            assert_eq!(probes[0].txb_skip_context, None);
            assert_eq!(probes[0].all_zero_symbol, None);
            assert_eq!(probes[0].first_non_zero_transform_index, None);
            assert_eq!(probes[0].first_non_zero_transform, None);
            assert_eq!(probes[0].first_non_zero_tx_size, None);
            assert!(!probes[0].tx_type_read);
            assert_eq!(probes[0].tx_type_set, None);
            assert_eq!(probes[0].tx_type_symbol, None);
            assert_eq!(probes[0].tx_type, None);
            assert_eq!(probes[0].coeff_base_eob_context, None);
            assert_eq!(probes[0].coeff_base_eob_symbol, None);
            assert_eq!(probes[0].coeff_base_eob_level, None);
            assert_eq!(probes[0].regular_coeff_base_count, None);
            assert_eq!(probes[0].regular_coeff_base_decoded_count, None);
            assert_eq!(probes[0].coeff_base_non_zero_count, None);
            assert_eq!(probes[0].coeff_base_range_count, None);
            assert_eq!(probes[0].coeff_br_decoded_count, None);
            assert_eq!(probes[0].first_coeff_br_scan_index, None);
            assert_eq!(probes[0].first_coeff_br_context, None);
            assert_eq!(probes[0].first_coeff_br_symbol, None);
            assert_eq!(probes[0].first_coeff_br_level, None);
            assert_eq!(probes[0].sign_decoded_count, None);
            assert_eq!(probes[0].dc_sign_context, None);
            assert_eq!(probes[0].dc_sign_symbol, None);
            assert_eq!(probes[0].first_ac_sign_scan_index, None);
            assert_eq!(probes[0].first_ac_sign_bit, None);
            assert_eq!(probes[0].golomb_decoded_count, None);
            assert_eq!(probes[0].first_golomb_scan_index, None);
            assert_eq!(probes[0].first_golomb_value, None);
            assert_eq!(probes[0].signed_coeff_non_zero_count, None);
            assert_eq!(probes[0].first_signed_coeff_scan_index, None);
            assert_eq!(probes[0].first_signed_coeff_position, None);
            assert_eq!(probes[0].first_signed_coeff_value, None);
            assert_eq!(probes[0].dequant_non_zero_count, None);
            assert_eq!(probes[0].first_dequant_coeff_position, None);
            assert_eq!(probes[0].first_dequant_coeff_value, None);
            assert_eq!(probes[0].residual_preview_tx_type, None);
            assert_eq!(probes[0].residual_preview_sample_count, None);
            assert_eq!(probes[0].first_residual_preview_sample, None);
            assert_eq!(probes[0].first_coeff_base_scan_index, None);
            assert_eq!(probes[0].first_coeff_base_context, None);
            assert_eq!(probes[0].first_coeff_base_symbol, None);
            assert_eq!(probes[0].first_coeff_base_level, None);
            assert_eq!(probes[0].first_quantized_coefficients, None);
        } else {
            assert!(probes[0].txb_skip_context.unwrap() <= 1);
            assert!(probes[0].all_zero_symbol.unwrap() <= 1);
            assert_eq!(
                probes[0].zero_transform_count,
                probes[0]
                    .first_non_zero_transform_index
                    .unwrap_or(transform_count)
            );
            if probes[0].first_non_zero_transform_index.is_none() {
                assert_eq!(probes[0].eob_multisize, None);
                assert_eq!(probes[0].eob_pt_symbol, None);
                assert_eq!(probes[0].eob_base, None);
                assert_eq!(probes[0].eob_extra_symbol, None);
                assert_eq!(probes[0].eob, None);
                assert_eq!(probes[0].first_non_zero_transform, None);
                assert!(!probes[0].tx_type_read);
                assert_eq!(probes[0].tx_type_set, None);
                assert_eq!(probes[0].tx_type_symbol, None);
                assert_eq!(probes[0].tx_type, None);
                assert_eq!(probes[0].coeff_base_eob_context, None);
                assert_eq!(probes[0].coeff_base_eob_symbol, None);
                assert_eq!(probes[0].coeff_base_eob_level, None);
                assert_eq!(probes[0].regular_coeff_base_count, None);
                assert_eq!(probes[0].regular_coeff_base_decoded_count, None);
                assert_eq!(probes[0].coeff_base_non_zero_count, None);
                assert_eq!(probes[0].coeff_base_range_count, None);
                assert_eq!(probes[0].coeff_br_decoded_count, None);
                assert_eq!(probes[0].first_coeff_br_scan_index, None);
                assert_eq!(probes[0].first_coeff_br_context, None);
                assert_eq!(probes[0].first_coeff_br_symbol, None);
                assert_eq!(probes[0].first_coeff_br_level, None);
                assert_eq!(probes[0].sign_decoded_count, None);
                assert_eq!(probes[0].dc_sign_context, None);
                assert_eq!(probes[0].dc_sign_symbol, None);
                assert_eq!(probes[0].first_ac_sign_scan_index, None);
                assert_eq!(probes[0].first_ac_sign_bit, None);
                assert_eq!(probes[0].golomb_decoded_count, None);
                assert_eq!(probes[0].first_golomb_scan_index, None);
                assert_eq!(probes[0].first_golomb_value, None);
                assert_eq!(probes[0].signed_coeff_non_zero_count, None);
                assert_eq!(probes[0].first_signed_coeff_scan_index, None);
                assert_eq!(probes[0].first_signed_coeff_position, None);
                assert_eq!(probes[0].first_signed_coeff_value, None);
                assert_eq!(probes[0].dequant_non_zero_count, None);
                assert_eq!(probes[0].first_dequant_coeff_position, None);
                assert_eq!(probes[0].first_dequant_coeff_value, None);
                assert_eq!(probes[0].residual_preview_tx_type, None);
                assert_eq!(probes[0].residual_preview_sample_count, None);
                assert_eq!(probes[0].first_residual_preview_sample, None);
                assert_eq!(probes[0].first_coeff_base_scan_index, None);
                assert_eq!(probes[0].first_coeff_base_context, None);
                assert_eq!(probes[0].first_coeff_base_symbol, None);
                assert_eq!(probes[0].first_coeff_base_level, None);
                assert_eq!(probes[0].first_quantized_coefficients, None);
            } else {
                assert!(probes[0].first_non_zero_transform_index.unwrap() < transform_count);
                assert_eq!(
                    probes[0].first_non_zero_transform.unwrap().tx_size,
                    first_tx_size
                );
                assert_eq!(probes[0].first_non_zero_tx_size, Some(first_tx_size));
                assert_eq!(
                    probes[0].eob_multisize,
                    Some(eob_multisize(probes[0].first_non_zero_transform.unwrap()))
                );
                assert!(probes[0].eob_pt_symbol.unwrap() < 11);
                assert_eq!(
                    probes[0].eob_pt.unwrap(),
                    probes[0].eob_pt_symbol.unwrap() + 1
                );
                assert!(probes[0].eob_base.unwrap() > 0);
                assert_eq!(
                    probes[0].eob_extra_context,
                    probes[0].eob_pt.filter(|pt| *pt >= 3).map(|pt| pt - 3)
                );
                assert!(probes[0].eob_extra_symbol.unwrap_or(0) <= 1);
                assert_eq!(
                    probes[0].eob_extra_literal_bits,
                    Some(probes[0].eob_pt.unwrap().saturating_sub(3))
                );
                assert!(probes[0].eob.unwrap() >= probes[0].eob_base.unwrap());
                assert!(probes[0].eob.unwrap() <= tx_sample_count);
                assert!(!probes[0].tx_type_read);
                assert_eq!(probes[0].tx_type_set, None);
                assert_eq!(probes[0].tx_type_symbol, None);
                assert_eq!(probes[0].tx_type, Some(TxType::DctDct));
                assert!(probes[0].coeff_base_eob_context.unwrap() < 4);
                assert!(probes[0].coeff_base_eob_symbol.unwrap() < 3);
                assert_eq!(
                    probes[0].coeff_base_eob_level.unwrap(),
                    probes[0].coeff_base_eob_symbol.unwrap() + 1
                );
                assert_eq!(
                    probes[0].regular_coeff_base_count,
                    Some(probes[0].eob.unwrap() - 1)
                );
                assert_eq!(
                    probes[0].regular_coeff_base_decoded_count,
                    probes[0].regular_coeff_base_count
                );
                assert!(probes[0].coeff_base_non_zero_count.unwrap() >= 1);
                assert!(probes[0].coeff_base_non_zero_count.unwrap() <= probes[0].eob.unwrap());
                assert!(
                    probes[0].coeff_base_range_count.unwrap()
                        <= probes[0].coeff_base_non_zero_count.unwrap()
                );
                assert!(
                    probes[0].coeff_br_decoded_count.unwrap()
                        >= probes[0].coeff_base_range_count.unwrap()
                );
                assert_eq!(
                    probes[0].sign_decoded_count,
                    probes[0].coeff_base_non_zero_count
                );
                assert_eq!(
                    probes[0].signed_coeff_non_zero_count,
                    probes[0].coeff_base_non_zero_count
                );
                assert!(probes[0].first_signed_coeff_scan_index.unwrap() < probes[0].eob.unwrap());
                assert!(probes[0].first_signed_coeff_position.unwrap() < tx_sample_count);
                assert_ne!(probes[0].first_signed_coeff_value.unwrap(), 0);
                assert_eq!(
                    probes[0].dequant_non_zero_count,
                    probes[0].signed_coeff_non_zero_count
                );
                assert!(probes[0].first_dequant_coeff_position.unwrap() < tx_sample_count);
                assert_ne!(probes[0].first_dequant_coeff_value.unwrap(), 0);
                if matches!(
                    probes[0].tx_type,
                    Some(
                        TxType::DctDct
                            | TxType::Identity
                            | TxType::VerticalDct
                            | TxType::HorizontalDct
                    )
                ) {
                    assert_eq!(probes[0].residual_preview_tx_type, probes[0].tx_type);
                    assert_eq!(
                        probes[0].residual_preview_sample_count,
                        Some(tx_sample_count)
                    );
                    assert!(probes[0].first_residual_preview_sample.is_some());
                } else {
                    assert_eq!(probes[0].residual_preview_tx_type, None);
                    assert_eq!(probes[0].residual_preview_sample_count, None);
                    assert_eq!(probes[0].first_residual_preview_sample, None);
                }
                if probes[0].dc_sign_symbol.is_some() {
                    assert!(probes[0].dc_sign_context.unwrap() < 3);
                    assert!(probes[0].dc_sign_symbol.unwrap() <= 1);
                }
                assert!(
                    probes[0].golomb_decoded_count.unwrap()
                        <= probes[0].sign_decoded_count.unwrap()
                );
                if probes[0].sign_decoded_count.unwrap()
                    > usize::from(probes[0].dc_sign_symbol.is_some())
                {
                    assert!(probes[0].first_ac_sign_scan_index.unwrap() < probes[0].eob.unwrap());
                    assert!(probes[0].first_ac_sign_bit.unwrap() <= 1);
                } else {
                    assert_eq!(probes[0].first_ac_sign_scan_index, None);
                    assert_eq!(probes[0].first_ac_sign_bit, None);
                }
                if probes[0].golomb_decoded_count.unwrap() > 0 {
                    assert!(probes[0].first_golomb_scan_index.unwrap() < probes[0].eob.unwrap());
                    assert!(probes[0].first_golomb_value.is_some());
                } else {
                    assert_eq!(probes[0].first_golomb_scan_index, None);
                    assert_eq!(probes[0].first_golomb_value, None);
                }
                if probes[0].coeff_base_range_count.unwrap() > 0 {
                    assert!(probes[0].first_coeff_br_scan_index.unwrap() < probes[0].eob.unwrap());
                    assert!(probes[0].first_coeff_br_position.unwrap() < tx_sample_count);
                    assert!(probes[0].first_coeff_br_context.unwrap() < 21);
                    assert!(probes[0].first_coeff_br_symbol.unwrap() < 4);
                    assert!(probes[0].first_coeff_br_level.unwrap() >= 3);
                } else {
                    assert_eq!(probes[0].first_coeff_br_scan_index, None);
                    assert_eq!(probes[0].first_coeff_br_context, None);
                    assert_eq!(probes[0].first_coeff_br_symbol, None);
                    assert_eq!(probes[0].first_coeff_br_level, None);
                }
                if probes[0].regular_coeff_base_count.unwrap() > 0 {
                    assert_eq!(
                        probes[0].first_coeff_base_scan_index,
                        Some(probes[0].eob.unwrap() - 2)
                    );
                    assert!(probes[0].first_coeff_base_position.unwrap() < tx_sample_count);
                    assert!(probes[0].first_coeff_base_context.unwrap() < 42);
                    assert!(probes[0].first_coeff_base_reference_magnitude.unwrap() <= 15);
                    assert!(probes[0].first_coeff_base_symbol.unwrap() < 4);
                    assert_eq!(
                        probes[0].first_coeff_base_level,
                        probes[0].first_coeff_base_symbol
                    );
                }
                assert_eq!(
                    probes[0]
                        .first_quantized_coefficients
                        .as_ref()
                        .unwrap()
                        .len(),
                    tx_sample_count
                );
                let coefficients = probes[0].first_quantized_coefficients.as_ref().unwrap();
                assert_eq!(coefficients[0], -468);
                assert_eq!(coefficients.iter().filter(|value| **value != 0).count(), 1);
            }
        }
        assert_eq!(probes[0].first_tx_size, Some(first_tx_size));
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
    fn intra_ext_tx_set_context_uses_set2_for_tx16() {
        assert_eq!(intra_ext_tx_set_context(false, TxSize::Tx4x4), Some((1, 0)));
        assert_eq!(intra_ext_tx_set_context(false, TxSize::Tx8x8), Some((1, 1)));
        assert_eq!(
            intra_ext_tx_set_context(false, TxSize::Tx16x16),
            Some((2, 2))
        );
        assert_eq!(intra_ext_tx_set_context(true, TxSize::Tx4x4), Some((2, 0)));
        assert_eq!(intra_ext_tx_set_context(true, TxSize::Tx8x8), Some((2, 1)));
        assert_eq!(intra_ext_tx_set_context(false, TxSize::Tx32x32), None);
    }

    #[test]
    fn filter_intra_mode_selects_normative_tx_cdf_mode() {
        assert_eq!(filter_intra_mode_to_tx_cdf_mode(0).unwrap(), 0);
        assert_eq!(filter_intra_mode_to_tx_cdf_mode(1).unwrap(), 1);
        assert_eq!(filter_intra_mode_to_tx_cdf_mode(2).unwrap(), 2);
        assert_eq!(filter_intra_mode_to_tx_cdf_mode(3).unwrap(), 6);
        assert_eq!(filter_intra_mode_to_tx_cdf_mode(4).unwrap(), 0);
        assert!(filter_intra_mode_to_tx_cdf_mode(5).is_err());
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
