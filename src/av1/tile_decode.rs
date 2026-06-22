use super::cdf::CdfContext;
use super::decode::{FrameBuffers, FrameDecodePlan, PlaneBuffer, TileDecodePlan};
use super::entropy::EntropyDecoder;
use super::frame::{FrameHeader, TxMode};
use super::predict::{
    IntraEdges, predict_filter_intra, predict_intra, predict_intra_with_edge_filter,
};
use super::quant::{QuantState, dequantize_coefficients};
use super::reconstruct::{read_intra_edges, write_plane_block};
use super::sequence::SequenceHeader;
use super::syntax::{BlockSize, Partition, PredictionMode, TxSize, TxType, UvPredictionMode};
use super::tile_group::TileGroup;
use super::transform::{
    QuantizedTransform, TransformBlock, coefficient_scan, inverse_transform,
    plan_transform_blocks_with_tx_size, reconstruct_transform_block,
};
use crate::DecoderError;

mod coefficient;

use coefficient::{CoefficientRead, EntropyCoefficientSource, decode_coefficients};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockModeProbe {
    pub tile_id: u32,
    pub block_size: BlockSize,
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
    pub first_tx_size: Option<super::syntax::TxSize>,
    pub first_non_zero_transform_index: Option<usize>,
    pub first_non_zero_transform: Option<TransformBlock>,
    pub first_non_zero_tx_size: Option<super::syntax::TxSize>,
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
    pub transforms: Vec<DecodedTransform>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedBlockPrefix {
    pub blocks: Vec<DecodedLumaBlock>,
    pub next_unsupported: Option<DecoderError>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TxbContext {
    skip: usize,
    dc_sign: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PlaneEntropyContexts {
    above: Vec<u8>,
    left: Vec<u8>,
}

pub struct TileDecoder<'a> {
    reader: EntropyDecoder<'a>,
    cdf: CdfContext,
    mi_cols: usize,
    mi_rows: usize,
    y_mode_grid: Vec<Option<usize>>,
    y_smooth_grid: Vec<Option<bool>>,
    uv_smooth_grid: Vec<Option<bool>>,
    skip_grid: Vec<Option<bool>>,
    above_partition_context: Vec<u8>,
    left_partition_context: Vec<u8>,
    cdef_transmitted: [bool; 4],
    above_txfm_context: Vec<usize>,
    left_txfm_context: Vec<usize>,
    plane_entropy_contexts: [PlaneEntropyContexts; 3],
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

    pub fn read_root_partition(
        &mut self,
        tile: &TileDecodePlan,
    ) -> Result<PartitionProbe, DecoderError> {
        self.read_partition(tile, BlockSize::Block128x128, tile.pixel_x, tile.pixel_y)
    }

    fn read_partition(
        &mut self,
        tile: &TileDecodePlan,
        block_size: BlockSize,
        x: usize,
        y: usize,
    ) -> Result<PartitionProbe, DecoderError> {
        let context = self.partition_context(tile, x, y, block_size);
        let half_mi = block_size.width() / 8;
        let mi_col = x >> 2;
        let mi_row = y >> 2;
        let has_rows = mi_row + half_mi < tile.mi_row_end as usize;
        let has_cols = mi_col + half_mi < tile.mi_col_end as usize;
        let (symbol, partition) = if !has_rows && !has_cols {
            (3, Partition::Split)
        } else if has_rows && has_cols {
            let symbol = self.reader.read_symbol(
                self.cdf
                    .partition_cdf_mut(block_size.width_mi_log2(), context),
            )?;
            let partition = Partition::from_symbol(block_size, symbol).ok_or_else(|| {
                DecoderError::Bitstream(format!(
                    "AV1 partition symbol {symbol} is invalid for {block_size:?}"
                ))
            })?;
            (symbol, partition)
        } else {
            if block_size.width() <= 8 {
                return Err(DecoderError::Bitstream(format!(
                    "AV1 edge partition reached invalid block size {block_size:?}"
                )));
            }
            let source = self
                .cdf
                .partition_cdf_mut(block_size.width_mi_log2(), context)
                .to_vec();
            let mut cdf = restricted_partition_cdf(&source, has_rows);
            let symbol = self.reader.read_symbol(&mut cdf)?;
            let partition = if symbol == 0 {
                if has_rows {
                    Partition::Vertical
                } else {
                    Partition::Horizontal
                }
            } else {
                Partition::Split
            };
            (partition_symbol(partition), partition)
        };
        Ok(PartitionProbe {
            tile_id: tile.tile_id,
            block_size,
            context,
            symbol,
            partition,
            bit_position_after: self.reader.bit_position(),
        })
    }

    pub fn read_intra_frame_block_mode(
        &mut self,
        sequence: &SequenceHeader,
        frame: &FrameHeader,
        tile: &TileDecodePlan,
        block_size: BlockSize,
        x: usize,
        y: usize,
    ) -> Result<BlockModeProbe, DecoderError> {
        if frame.delta_q.present {
            return Err(DecoderError::Unsupported(
                "AV1 delta-q block syntax is not supported yet".to_string(),
            ));
        }
        if frame.delta_lf.present {
            return Err(DecoderError::Unsupported(
                "AV1 delta loop-filter block syntax is not supported yet".to_string(),
            ));
        }
        if frame.allow_intrabc {
            return Err(DecoderError::Unsupported(
                "AV1 intrabc block syntax is not supported yet".to_string(),
            ));
        }

        let skip_context = self.skip_context(x, y);
        let skip_symbol = self
            .reader
            .read_symbol(self.cdf.skip_cdf_mut(skip_context))?;
        let skip = skip_symbol != 0;
        let cdef_idx = self.read_cdef_index(sequence, frame, skip, x, y)?;

        let y_above_context = self.above_y_mode_context(x, y);
        let y_left_context = self.left_y_mode_context(x, y);
        let y_mode_symbol = self.reader.read_symbol(
            self.cdf
                .intra_frame_y_mode_cdf_mut(y_above_context, y_left_context),
        )?;
        let y_mode = PredictionMode::from_intra_symbol(y_mode_symbol).ok_or_else(|| {
            DecoderError::Bitstream(format!("AV1 y_mode symbol {y_mode_symbol} is invalid"))
        })?;
        let use_angle_delta = block_size.width() >= 8 && block_size.height() >= 8;
        let angle_delta_y = if use_angle_delta && y_mode.is_directional() {
            Some(self.read_angle_delta(y_mode.directional_index().unwrap())?)
        } else {
            None
        };
        let mut filter_intra_mode = None;
        if sequence.enable_filter_intra
            && block_size.width() <= 32
            && block_size.height() <= 32
            && y_mode == PredictionMode::Dc
        {
            let use_filter_intra = self.reader.read_symbol(
                self.cdf
                    .use_filter_intra_cdf_mut(block_size.filter_intra_cdf_index()),
            )? != 0;
            if use_filter_intra {
                filter_intra_mode = Some(
                    self.reader
                        .read_symbol(self.cdf.filter_intra_mode_cdf_mut())?,
                );
            }
        }

        let has_chroma = !sequence.color_config.monochrome;
        let (uv_mode_symbol, uv_mode, angle_delta_uv) = if has_chroma {
            let uv_symbol = self
                .reader
                .read_symbol(self.cdf.uv_mode_cfl_not_allowed_cdf_mut(y_mode_symbol))?;
            let uv_mode = UvPredictionMode::from_symbol(uv_symbol).ok_or_else(|| {
                DecoderError::Bitstream(format!("AV1 uv_mode symbol {uv_symbol} is invalid"))
            })?;
            if uv_mode == UvPredictionMode::Cfl {
                return Err(DecoderError::Unsupported(
                    "AV1 CFL chroma prediction is not supported yet".to_string(),
                ));
            }
            let angle_delta = if use_angle_delta && uv_mode.is_directional() {
                Some(self.read_angle_delta(uv_mode.directional_index().unwrap())?)
            } else {
                None
            };
            (Some(uv_symbol), Some(uv_mode), angle_delta)
        } else {
            (None, None, None)
        };
        let y_smooth_neighbour = self.has_smooth_intra_neighbour(0, x, y);
        let uv_smooth_neighbour = self.has_smooth_intra_neighbour(1, x, y);
        self.set_y_mode_context(x, y, block_size, intra_mode_context(y_mode));
        self.set_smooth_context(
            x,
            y,
            block_size,
            y_mode.is_smooth(),
            matches!(uv_mode, Some(UvPredictionMode::Intra(mode)) if mode.is_smooth()),
        );
        self.set_skip_context(x, y, block_size, skip);
        let (tx_size_context, tx_size_symbol, tx_size) =
            self.read_intra_tx_size(frame, block_size, skip, x, y)?;

        Ok(BlockModeProbe {
            tile_id: tile.tile_id,
            block_size,
            skip_context,
            skip_symbol,
            skip,
            cdef_idx,
            y_above_context,
            y_left_context,
            y_mode_symbol,
            y_mode,
            angle_delta_y,
            y_smooth_neighbour,
            filter_intra_mode,
            uv_mode_symbol,
            uv_mode,
            angle_delta_uv,
            uv_smooth_neighbour,
            tx_size_context,
            tx_size_symbol,
            tx_size,
            bit_position_after: self.reader.bit_position(),
        })
    }

    fn read_intra_tx_size(
        &mut self,
        frame: &FrameHeader,
        block_size: BlockSize,
        skip: bool,
        x: usize,
        y: usize,
    ) -> Result<(Option<usize>, Option<usize>, TxSize), DecoderError> {
        let tx_size = match frame.tx_mode {
            TxMode::Only4x4 => TxSize::Tx4x4,
            TxMode::Largest => block_size.largest_supported_tx_size(),
            TxMode::Select if block_size.signals_tx_size() && !skip => {
                let context = self.tx_size_context(x, y, block_size);
                let category = block_size.tx_size_category();
                let symbol = self
                    .reader
                    .read_symbol(self.cdf.tx_size_cdf_mut(category, context))?;
                if symbol > block_size.max_tx_size_depth() {
                    return Err(DecoderError::Bitstream(format!(
                        "AV1 tx_size symbol {symbol} exceeds max depth for {block_size:?}"
                    )));
                }
                let tx_size = block_size.tx_size_from_depth(symbol);
                self.set_txfm_context(x, y, block_size, tx_size);
                return Ok((Some(context), Some(symbol), tx_size));
            }
            TxMode::Select => TxSize::Tx4x4,
        };
        self.set_txfm_context(x, y, block_size, tx_size);
        Ok((None, None, tx_size))
    }

    fn read_cdef_index(
        &mut self,
        sequence: &SequenceHeader,
        frame: &FrameHeader,
        skip: bool,
        x: usize,
        y: usize,
    ) -> Result<Option<u32>, DecoderError> {
        if !frame.cdef.enabled || frame.allow_intrabc {
            return Ok(None);
        }
        let superblock_mi = if sequence.use_128x128_superblock {
            32
        } else {
            16
        };
        let mi_col = x >> 2;
        let mi_row = y >> 2;
        if mi_col % superblock_mi == 0 && mi_row % superblock_mi == 0 {
            self.cdef_transmitted = [false; 4];
        }
        let index = if sequence.use_128x128_superblock {
            usize::from((mi_col & 16) != 0) + usize::from((mi_row & 16) != 0) * 2
        } else {
            0
        };
        if self.cdef_transmitted[index] || skip {
            return Ok(None);
        }
        let value = self
            .reader
            .read_literal(frame.cdef.bits as usize)
            .map_err(|err| DecoderError::Bitstream(format!("AV1 cdef_idx: {err}")))?;
        self.cdef_transmitted[index] = true;
        Ok(Some(value))
    }

    fn read_angle_delta(&mut self, directional_index: usize) -> Result<i8, DecoderError> {
        let symbol = self
            .reader
            .read_symbol(self.cdf.angle_delta_cdf_mut(directional_index))?;
        Ok(symbol as i8 - 3)
    }

    fn above_y_mode_context(&self, x: usize, y: usize) -> usize {
        if y < 4 {
            return 0;
        }
        self.y_mode_at_mi(x >> 2, (y >> 2).saturating_sub(1))
            .unwrap_or(0)
    }

    fn left_y_mode_context(&self, x: usize, y: usize) -> usize {
        if x < 4 {
            return 0;
        }
        self.y_mode_at_mi((x >> 2).saturating_sub(1), y >> 2)
            .unwrap_or(0)
    }

    fn y_mode_at_mi(&self, mi_col: usize, mi_row: usize) -> Option<usize> {
        if mi_col >= self.mi_cols || mi_row >= self.mi_rows {
            return None;
        }
        self.y_mode_grid[mi_row * self.mi_cols + mi_col]
    }

    fn set_y_mode_context(&mut self, x: usize, y: usize, block_size: BlockSize, symbol: usize) {
        fill_mi_grid(
            &mut self.y_mode_grid,
            self.mi_cols,
            self.mi_rows,
            x,
            y,
            block_size,
            symbol,
        );
    }

    fn set_smooth_context(
        &mut self,
        x: usize,
        y: usize,
        block_size: BlockSize,
        y_smooth: bool,
        uv_smooth: bool,
    ) {
        fill_mi_grid(
            &mut self.y_smooth_grid,
            self.mi_cols,
            self.mi_rows,
            x,
            y,
            block_size,
            y_smooth,
        );
        fill_mi_grid(
            &mut self.uv_smooth_grid,
            self.mi_cols,
            self.mi_rows,
            x,
            y,
            block_size,
            uv_smooth,
        );
    }

    fn has_smooth_intra_neighbour(&self, plane: usize, x: usize, y: usize) -> bool {
        let grid = if plane == 0 {
            &self.y_smooth_grid
        } else {
            &self.uv_smooth_grid
        };
        has_smooth_neighbour(grid, self.mi_cols, self.mi_rows, x, y)
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
            let plane_quant = quant_state.plane(non_zero_transform.plane);
            let dequant = dequantize_coefficients(
                &coeff_base_read.base_levels,
                plane_quant,
                bit_depth,
                non_zero_transform.tx_size.dq_denom(),
            );
            let dequant_non_zero_count = dequant.iter().filter(|value| **value != 0).count();
            let first_dequant_coeff =
                dequant
                    .iter()
                    .copied()
                    .enumerate()
                    .find_map(|(position, value)| {
                        (value != 0).then_some(DequantCoeffProbe { position, value })
                    });
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
                Some(dequant_non_zero_count),
                first_dequant_coeff.map(|first| first.position),
                first_dequant_coeff.map(|first| first.value),
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

fn decode_luma_root_block(
    decoder: &mut TileDecoder<'_>,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    tile_plan: &TileDecodePlan,
    plan: &FrameDecodePlan,
    buffers: &mut FrameBuffers,
    x: usize,
    y: usize,
) -> Result<DecodedLumaBlock, DecoderError> {
    let partition = decoder.read_root_partition(tile_plan)?;
    if partition.partition != Partition::None {
        return Err(DecoderError::Unsupported(format!(
            "AV1 luma root block decode for partition {:?} is not supported yet",
            partition.partition
        )));
    }

    decode_luma_leaf_block(
        decoder,
        sequence,
        frame,
        tile_plan,
        plan,
        buffers,
        partition.block_size,
        x,
        y,
    )
}

fn decode_luma_leaf_block(
    decoder: &mut TileDecoder<'_>,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    tile_plan: &TileDecodePlan,
    plan: &FrameDecodePlan,
    buffers: &mut FrameBuffers,
    block_size: BlockSize,
    x: usize,
    y: usize,
) -> Result<DecodedLumaBlock, DecoderError> {
    let block_mode =
        decoder.read_intra_frame_block_mode(sequence, frame, tile_plan, block_size, x, y)?;
    let quant_state =
        QuantState::from_params(&frame.quantization, sequence.color_config.bit_depth)?;
    let decoded = decode_plane_block(
        decoder,
        sequence,
        frame,
        plan,
        buffers,
        &block_mode,
        0,
        block_mode.y_mode,
        block_mode.angle_delta_y,
        block_mode.filter_intra_mode,
        block_mode.y_smooth_neighbour,
        x,
        y,
        quant_state,
    )?;

    if !sequence.color_config.monochrome {
        let uv_mode = block_mode.uv_mode.ok_or_else(|| {
            DecoderError::Bitstream("AV1 chroma block mode is missing".to_string())
        })?;
        let UvPredictionMode::Intra(chroma_mode) = uv_mode else {
            return Err(DecoderError::Unsupported(
                "AV1 CFL chroma prediction is not supported yet".to_string(),
            ));
        };
        decode_plane_block(
            decoder,
            sequence,
            frame,
            plan,
            buffers,
            &block_mode,
            1,
            chroma_mode,
            block_mode.angle_delta_uv,
            None,
            block_mode.uv_smooth_neighbour,
            x,
            y,
            quant_state,
        )?;
        decode_plane_block(
            decoder,
            sequence,
            frame,
            plan,
            buffers,
            &block_mode,
            2,
            chroma_mode,
            block_mode.angle_delta_uv,
            None,
            block_mode.uv_smooth_neighbour,
            x,
            y,
            quant_state,
        )?;
    }

    Ok(DecodedLumaBlock {
        x,
        y,
        block_size: block_mode.block_size,
        transforms: decoded,
    })
}

fn decode_plane_block(
    decoder: &mut TileDecoder<'_>,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    plan: &FrameDecodePlan,
    buffers: &mut FrameBuffers,
    block_mode: &BlockModeProbe,
    plane_index: usize,
    prediction_mode: PredictionMode,
    angle_delta: Option<i8>,
    filter_intra_mode: Option<usize>,
    smooth_neighbour: bool,
    x: usize,
    y: usize,
    quant_state: QuantState,
) -> Result<Vec<DecodedTransform>, DecoderError> {
    let layout = plan.planes.get(plane_index).ok_or_else(|| {
        DecoderError::Bitstream(format!("AV1 plane {plane_index} decode plan is missing"))
    })?;
    if layout.subsampling_x != 0 || layout.subsampling_y != 0 {
        return Err(DecoderError::Unsupported(
            "AV1 subsampled chroma block reconstruction is not supported yet".to_string(),
        ));
    }
    let plane = buffers.planes.get_mut(plane_index).ok_or_else(|| {
        DecoderError::Bitstream(format!("AV1 plane {plane_index} buffer is missing"))
    })?;
    let transforms = plan_transform_blocks_with_tx_size(
        plane_index,
        x,
        y,
        block_mode.block_size,
        block_mode.tx_size,
        layout.width,
        layout.height,
    );
    if block_mode.skip {
        for transform in &transforms {
            decoder.set_txb_entropy_context(*transform, 0);
            let prediction = predict_block(
                plane,
                prediction_mode,
                transform.x,
                transform.y,
                transform.tx_size.width(),
                transform.tx_size.height(),
                angle_delta,
                filter_intra_mode,
                sequence.color_config.bit_depth,
                sequence.enable_intra_edge_filter,
                smooth_neighbour,
            )?;
            write_plane_block(
                plane,
                transform.x,
                transform.y,
                transform.tx_size.width(),
                transform.tx_size.height(),
                &prediction,
            )?;
        }
        return Ok(Vec::new());
    }

    let mut decoded = Vec::new();
    for transform in transforms {
        let txb_context = decoder.txb_context(block_mode.block_size, transform);
        let all_zero_symbol = decoder.reader.read_symbol(
            decoder
                .cdf
                .txb_skip_cdf_mut(transform.tx_size.coeff_cdf_index(), txb_context.skip),
        )?;
        if all_zero_symbol != 0 {
            decoder.set_txb_entropy_context(transform, 0);
            let prediction = predict_block(
                plane,
                prediction_mode,
                transform.x,
                transform.y,
                transform.tx_size.width(),
                transform.tx_size.height(),
                angle_delta,
                filter_intra_mode,
                sequence.color_config.bit_depth,
                sequence.enable_intra_edge_filter,
                smooth_neighbour,
            )?;
            write_plane_block(
                plane,
                transform.x,
                transform.y,
                transform.tx_size.width(),
                transform.tx_size.height(),
                &prediction,
            )?;
            continue;
        }

        let decoded_transform =
            decoder.read_decoded_transform(frame, &block_mode, transform, txb_context.dc_sign)?;
        decoder.set_txb_entropy_context(
            transform,
            coefficient_entropy_context(&decoded_transform.coefficients),
        );
        let prediction = predict_block(
            plane,
            prediction_mode,
            transform.x,
            transform.y,
            transform.tx_size.width(),
            transform.tx_size.height(),
            angle_delta,
            filter_intra_mode,
            sequence.color_config.bit_depth,
            sequence.enable_intra_edge_filter,
            smooth_neighbour,
        )?;
        let quantized = QuantizedTransform {
            block: decoded_transform.transform,
            tx_type: decoded_transform.tx_type,
            coefficients: decoded_transform.coefficients.clone(),
        };
        reconstruct_transform_block(
            plane,
            &quantized,
            quant_state.plane(transform.plane),
            &prediction,
            sequence.color_config.bit_depth,
        )?;
        decoded.push(decoded_transform);
    }

    Ok(decoded)
}

fn predict_block(
    plane: &PlaneBuffer,
    prediction_mode: PredictionMode,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    angle_delta: Option<i8>,
    filter_intra_mode: Option<usize>,
    bit_depth: u8,
    enable_intra_edge_filter: bool,
    smooth_neighbour: bool,
) -> Result<Vec<u16>, DecoderError> {
    let mut edges = read_intra_edges(plane, x, y, width, height, bit_depth);
    let midpoint = 1u16 << (bit_depth - 1);
    let above_left = match (edges.above_available, edges.left_available) {
        (true, true) => edges.above_left,
        (true, false) => edges.above[0],
        (false, true) => edges.left[0],
        (false, false) => midpoint,
    };
    if !edges.above_available && edges.left_available {
        edges.above.fill(edges.left[0]);
    }
    if !edges.left_available && edges.above_available {
        edges.left.fill(edges.above[0]);
    }
    let edges = if prediction_mode == PredictionMode::Dc && filter_intra_mode.is_none() {
        IntraEdges {
            above: edges.above_available.then_some(edges.above.as_slice()),
            left: edges.left_available.then_some(edges.left.as_slice()),
            above_left: Some(above_left),
            bit_depth,
        }
    } else {
        IntraEdges {
            above: Some(&edges.above),
            left: Some(&edges.left),
            above_left: Some(above_left),
            bit_depth,
        }
    };
    if let Some(filter_intra_mode) = filter_intra_mode {
        return predict_filter_intra(filter_intra_mode, width, height, edges);
    }
    predict_intra_with_edge_filter(
        prediction_mode,
        angle_delta,
        width,
        height,
        edges,
        enable_intra_edge_filter,
        smooth_neighbour,
    )
}

fn decode_luma_block_tree(
    decoder: &mut TileDecoder<'_>,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    tile_plan: &TileDecodePlan,
    plan: &FrameDecodePlan,
    buffers: &mut FrameBuffers,
    block_size: BlockSize,
    x: usize,
    y: usize,
    block_budget: &mut usize,
) -> Result<Vec<DecodedLumaBlock>, DecoderError> {
    if *block_budget == 0 || x >= plan.width || y >= plan.height {
        return Ok(Vec::new());
    }
    if block_size == BlockSize::Block4x4 {
        let block = decode_luma_leaf_block(
            decoder, sequence, frame, tile_plan, plan, buffers, block_size, x, y,
        )?;
        *block_budget -= 1;
        return Ok(vec![block]);
    }
    let partition = decoder
        .read_partition(tile_plan, block_size, x, y)?
        .partition;
    let decoded = match partition {
        Partition::None => {
            let block = decode_luma_leaf_block(
                decoder, sequence, frame, tile_plan, plan, buffers, block_size, x, y,
            )?;
            *block_budget -= 1;
            Ok(vec![block])
        }
        Partition::Horizontal => {
            let subsize = block_size.horizontal_subsize().ok_or_else(|| {
                DecoderError::Unsupported(format!(
                    "AV1 horizontal partition for {block_size:?} is not supported yet"
                ))
            })?;
            decode_luma_partition_runs(
                decoder,
                sequence,
                frame,
                tile_plan,
                plan,
                buffers,
                &[(subsize, x, y), (subsize, x, y + subsize.height())],
                block_budget,
            )
        }
        Partition::Vertical => {
            let subsize = block_size.vertical_subsize().ok_or_else(|| {
                DecoderError::Unsupported(format!(
                    "AV1 vertical partition for {block_size:?} is not supported yet"
                ))
            })?;
            decode_luma_partition_runs(
                decoder,
                sequence,
                frame,
                tile_plan,
                plan,
                buffers,
                &[(subsize, x, y), (subsize, x + subsize.width(), y)],
                block_budget,
            )
        }
        Partition::Split => {
            let subsize = block_size.split_subsize().ok_or_else(|| {
                DecoderError::Unsupported(format!(
                    "AV1 split partition for {block_size:?} is not supported yet"
                ))
            })?;
            decode_luma_partition_children(
                decoder,
                sequence,
                frame,
                tile_plan,
                plan,
                buffers,
                subsize,
                &[
                    (x, y),
                    (x + subsize.width(), y),
                    (x, y + subsize.height()),
                    (x + subsize.width(), y + subsize.height()),
                ],
                block_budget,
            )
        }
        Partition::HorizontalA => {
            let split_subsize = block_size.split_subsize().ok_or_else(|| {
                DecoderError::Unsupported(format!(
                    "AV1 horizontal-a partition for {block_size:?} is not supported yet"
                ))
            })?;
            let horizontal_subsize = block_size.horizontal_subsize().ok_or_else(|| {
                DecoderError::Unsupported(format!(
                    "AV1 horizontal-a partition tail for {block_size:?} is not supported yet"
                ))
            })?;
            decode_luma_partition_runs(
                decoder,
                sequence,
                frame,
                tile_plan,
                plan,
                buffers,
                &[
                    (split_subsize, x, y),
                    (split_subsize, x + split_subsize.width(), y),
                    (horizontal_subsize, x, y + split_subsize.height()),
                ],
                block_budget,
            )
        }
        Partition::HorizontalB => {
            let horizontal_subsize = block_size.horizontal_subsize().ok_or_else(|| {
                DecoderError::Unsupported(format!(
                    "AV1 horizontal-b partition head for {block_size:?} is not supported yet"
                ))
            })?;
            let split_subsize = block_size.split_subsize().ok_or_else(|| {
                DecoderError::Unsupported(format!(
                    "AV1 horizontal-b partition for {block_size:?} is not supported yet"
                ))
            })?;
            decode_luma_partition_runs(
                decoder,
                sequence,
                frame,
                tile_plan,
                plan,
                buffers,
                &[
                    (horizontal_subsize, x, y),
                    (split_subsize, x, y + horizontal_subsize.height()),
                    (
                        split_subsize,
                        x + split_subsize.width(),
                        y + horizontal_subsize.height(),
                    ),
                ],
                block_budget,
            )
        }
        Partition::VerticalA => {
            let split_subsize = block_size.split_subsize().ok_or_else(|| {
                DecoderError::Unsupported(format!(
                    "AV1 vertical-a partition head for {block_size:?} is not supported yet"
                ))
            })?;
            let vertical_subsize = block_size.vertical_subsize().ok_or_else(|| {
                DecoderError::Unsupported(format!(
                    "AV1 vertical-a partition tail for {block_size:?} is not supported yet"
                ))
            })?;
            decode_luma_partition_runs(
                decoder,
                sequence,
                frame,
                tile_plan,
                plan,
                buffers,
                &[
                    (split_subsize, x, y),
                    (split_subsize, x, y + split_subsize.height()),
                    (vertical_subsize, x + split_subsize.width(), y),
                ],
                block_budget,
            )
        }
        Partition::VerticalB => {
            let vertical_subsize = block_size.vertical_subsize().ok_or_else(|| {
                DecoderError::Unsupported(format!(
                    "AV1 vertical-b partition head for {block_size:?} is not supported yet"
                ))
            })?;
            let split_subsize = block_size.split_subsize().ok_or_else(|| {
                DecoderError::Unsupported(format!(
                    "AV1 vertical-b partition for {block_size:?} is not supported yet"
                ))
            })?;
            decode_luma_partition_runs(
                decoder,
                sequence,
                frame,
                tile_plan,
                plan,
                buffers,
                &[
                    (vertical_subsize, x, y),
                    (split_subsize, x + vertical_subsize.width(), y),
                    (
                        split_subsize,
                        x + vertical_subsize.width(),
                        y + split_subsize.height(),
                    ),
                ],
                block_budget,
            )
        }
        Partition::Horizontal4 => {
            let subsize = block_size.horizontal_4_subsize().ok_or_else(|| {
                DecoderError::Unsupported(format!(
                    "AV1 horizontal4 partition for {block_size:?} is not supported yet"
                ))
            })?;
            decode_luma_partition_runs(
                decoder,
                sequence,
                frame,
                tile_plan,
                plan,
                buffers,
                &[
                    (subsize, x, y),
                    (subsize, x, y + subsize.height()),
                    (subsize, x, y + subsize.height() * 2),
                    (subsize, x, y + subsize.height() * 3),
                ],
                block_budget,
            )
        }
        Partition::Vertical4 => {
            let subsize = block_size.vertical_4_subsize().ok_or_else(|| {
                DecoderError::Unsupported(format!(
                    "AV1 vertical4 partition for {block_size:?} is not supported yet"
                ))
            })?;
            decode_luma_partition_runs(
                decoder,
                sequence,
                frame,
                tile_plan,
                plan,
                buffers,
                &[
                    (subsize, x, y),
                    (subsize, x + subsize.width(), y),
                    (subsize, x + subsize.width() * 2, y),
                    (subsize, x + subsize.width() * 3, y),
                ],
                block_budget,
            )
        }
    }?;
    decoder.update_ext_partition_context(x, y, block_size, partition)?;
    Ok(decoded)
}

fn decode_luma_partition_children(
    decoder: &mut TileDecoder<'_>,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    tile_plan: &TileDecodePlan,
    plan: &FrameDecodePlan,
    buffers: &mut FrameBuffers,
    subsize: BlockSize,
    children: &[(usize, usize)],
    block_budget: &mut usize,
) -> Result<Vec<DecodedLumaBlock>, DecoderError> {
    let mut blocks = Vec::new();
    for &(sub_x, sub_y) in children {
        if *block_budget == 0 {
            return Ok(blocks);
        }
        match decode_luma_block_tree(
            decoder,
            sequence,
            frame,
            tile_plan,
            plan,
            buffers,
            subsize,
            sub_x,
            sub_y,
            block_budget,
        ) {
            Ok(decoded) => blocks.extend(decoded),
            Err(DecoderError::Unsupported(_)) if !blocks.is_empty() => return Ok(blocks),
            Err(err) => return Err(err),
        }
    }
    Ok(blocks)
}

fn decode_luma_partition_runs(
    decoder: &mut TileDecoder<'_>,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    tile_plan: &TileDecodePlan,
    plan: &FrameDecodePlan,
    buffers: &mut FrameBuffers,
    children: &[(BlockSize, usize, usize)],
    block_budget: &mut usize,
) -> Result<Vec<DecodedLumaBlock>, DecoderError> {
    let mut blocks = Vec::new();
    for &(subsize, sub_x, sub_y) in children {
        if *block_budget == 0 {
            return Ok(blocks);
        }
        if sub_x >= plan.width || sub_y >= plan.height {
            continue;
        }
        match decode_luma_leaf_block(
            decoder, sequence, frame, tile_plan, plan, buffers, subsize, sub_x, sub_y,
        ) {
            Ok(decoded) => {
                *block_budget -= 1;
                blocks.push(decoded);
            }
            Err(DecoderError::Unsupported(_)) if !blocks.is_empty() => return Ok(blocks),
            Err(err) => return Err(err),
        }
    }
    Ok(blocks)
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

fn restricted_partition_cdf(source: &[u16], has_rows: bool) -> [u16; 3] {
    let symbols = source.len().saturating_sub(1);
    let alike = if has_rows {
        [1usize, 3, 4, 5, 6, 8]
    } else {
        [2usize, 3, 4, 6, 7, 9]
    };
    let alike_probability = alike
        .into_iter()
        .filter(|symbol| *symbol < symbols)
        .map(|symbol| cdf_symbol_probability(source, symbol))
        .sum::<u16>();
    [32768u16.saturating_sub(alike_probability), 32768, 0]
}

fn cdf_symbol_probability(cdf: &[u16], symbol: usize) -> u16 {
    let lower = if symbol == 0 { 0 } else { cdf[symbol - 1] };
    cdf[symbol].saturating_sub(lower)
}

const COEFF_CONTEXT_BITS: u8 = 3;
const COEFF_CONTEXT_MASK: u8 = (1 << COEFF_CONTEXT_BITS) - 1;
const TXB_SKIP_CONTEXTS: [[usize; 5]; 5] = [
    [1, 2, 2, 2, 3],
    [2, 4, 4, 4, 5],
    [2, 4, 4, 4, 5],
    [2, 4, 4, 4, 5],
    [3, 5, 5, 5, 6],
];

fn txb_context(
    block_size: BlockSize,
    transform: TransformBlock,
    above: &[u8],
    left: &[u8],
) -> TxbContext {
    let col = transform.x >> 2;
    let row = transform.y >> 2;
    let width_units = transform.tx_size.width() >> 2;
    let height_units = transform.tx_size.height() >> 2;
    let above_contexts = above
        .get(col..col.saturating_add(width_units).min(above.len()))
        .unwrap_or(&[]);
    let left_contexts = left
        .get(row..row.saturating_add(height_units).min(left.len()))
        .unwrap_or(&[]);

    let dc_sign_sum = above_contexts
        .iter()
        .chain(left_contexts)
        .map(|value| match value >> COEFF_CONTEXT_BITS {
            1 => -1,
            2 => 1,
            _ => 0,
        })
        .sum::<i32>();
    let dc_sign = match dc_sign_sum.cmp(&0) {
        std::cmp::Ordering::Less => 1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 2,
    };

    let skip = if transform.plane == 0 {
        if block_size.width() == transform.tx_size.width()
            && block_size.height() == transform.tx_size.height()
        {
            0
        } else {
            let top = above_contexts
                .iter()
                .fold(0, |value, context| value | context)
                & COEFF_CONTEXT_MASK;
            let left = left_contexts
                .iter()
                .fold(0, |value, context| value | context)
                & COEFF_CONTEXT_MASK;
            TXB_SKIP_CONTEXTS[usize::from(top.min(4))][usize::from(left.min(4))]
        }
    } else {
        let base = usize::from(above_contexts.iter().any(|value| *value != 0))
            + usize::from(left_contexts.iter().any(|value| *value != 0));
        let offset = if block_size.width() * block_size.height()
            > transform.tx_size.width() * transform.tx_size.height()
        {
            10
        } else {
            7
        };
        base + offset
    };

    TxbContext { skip, dc_sign }
}

fn coefficient_entropy_context(coefficients: &[i32]) -> u8 {
    let mut context = coefficients
        .iter()
        .map(|coefficient| coefficient.unsigned_abs() as u64)
        .sum::<u64>()
        .min(u64::from(COEFF_CONTEXT_MASK)) as u8;
    if let Some(dc) = coefficients.first() {
        if *dc < 0 {
            context |= 1 << COEFF_CONTEXT_BITS;
        } else if *dc > 0 {
            context += 2 << COEFF_CONTEXT_BITS;
        }
    }
    context
}

fn set_txb_entropy_context(
    transform: TransformBlock,
    value: u8,
    above: &mut [u8],
    left: &mut [u8],
) {
    let col = transform.x >> 2;
    let row = transform.y >> 2;
    let width_units = transform.tx_size.width() >> 2;
    let height_units = transform.tx_size.height() >> 2;
    let above_end = col.saturating_add(width_units).min(above.len());
    if let Some(contexts) = above.get_mut(col..above_end) {
        contexts.fill(value);
    }
    let left_end = row.saturating_add(height_units).min(left.len());
    if let Some(contexts) = left.get_mut(row..left_end) {
        contexts.fill(value);
    }
}

fn intra_mode_context(mode: PredictionMode) -> usize {
    match mode {
        PredictionMode::Dc => 0,
        PredictionMode::Vertical => 1,
        PredictionMode::Horizontal => 2,
        PredictionMode::Smooth
        | PredictionMode::SmoothVertical
        | PredictionMode::SmoothHorizontal => 3,
        _ => 4,
    }
}

fn smooth_mode_at(
    grid: &[Option<bool>],
    mi_cols: usize,
    mi_rows: usize,
    mi_col: usize,
    mi_row: usize,
) -> bool {
    if mi_col >= mi_cols || mi_row >= mi_rows {
        return false;
    }
    grid[mi_row * mi_cols + mi_col].unwrap_or(false)
}

fn has_smooth_neighbour(
    grid: &[Option<bool>],
    mi_cols: usize,
    mi_rows: usize,
    x: usize,
    y: usize,
) -> bool {
    let mi_col = x >> 2;
    let mi_row = y >> 2;
    let above = y >= 4 && smooth_mode_at(grid, mi_cols, mi_rows, mi_col, mi_row - 1);
    let left = x >= 4 && smooth_mode_at(grid, mi_cols, mi_rows, mi_col - 1, mi_row);
    above || left
}

fn fill_mi_grid<T: Copy>(
    grid: &mut [Option<T>],
    mi_cols: usize,
    mi_rows: usize,
    x: usize,
    y: usize,
    block_size: BlockSize,
    value: T,
) {
    let start_col = x >> 2;
    let start_row = y >> 2;
    let end_col = ((x + block_size.width()).min(mi_cols << 2) + 3) >> 2;
    let end_row = ((y + block_size.height()).min(mi_rows << 2) + 3) >> 2;
    for mi_row in start_row..end_row.min(mi_rows) {
        for mi_col in start_col..end_col.min(mi_cols) {
            grid[mi_row * mi_cols + mi_col] = Some(value);
        }
    }
}

fn eob_tx_class_context(tx_type: TxType) -> usize {
    usize::from(matches!(
        tx_type,
        TxType::VerticalDct | TxType::HorizontalDct
    ))
}

#[cfg(test)]
fn eob_multisize(transform: TransformBlock) -> usize {
    usize::from(transform.tx_size.width_log2().min(5) + transform.tx_size.height_log2().min(5) - 4)
}

fn eob_base_from_pt(eob_pt: usize) -> usize {
    if eob_pt < 2 {
        eob_pt
    } else {
        (1 << (eob_pt - 2)) + 1
    }
}

fn coeff_base_eob_context(tx_size: super::syntax::TxSize, scan_index: usize) -> usize {
    let samples = coeff_scan_sample_count(tx_size);
    if scan_index == 0 {
        0
    } else if scan_index <= samples / 8 {
        1
    } else if scan_index <= samples / 4 {
        2
    } else {
        3
    }
}

fn coeff_scan_sample_count(tx_size: super::syntax::TxSize) -> usize {
    tx_size.width().min(32) * tx_size.height().min(32)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CoeffBaseProbe {
    remaining_count: usize,
    decoded_count: usize,
    scan_index: Option<usize>,
    position: Option<usize>,
    context: Option<usize>,
    reference_magnitude: Option<usize>,
    symbol: Option<usize>,
    level: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TxTypeProbe {
    read: bool,
    set: Option<usize>,
    symbol: Option<usize>,
    tx_type: TxType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CoeffBaseRead {
    probe: CoeffBaseProbe,
    base_levels: Vec<i32>,
    non_zero_count: usize,
    base_range_count: usize,
    coeff_br_symbol_count: usize,
    first_coeff_br: Option<CoeffBrProbe>,
    signs: CoeffSignRead,
    signed_non_zero_count: usize,
    first_signed_coeff: Option<SignedCoeffProbe>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CoeffBrProbe {
    scan_index: usize,
    position: usize,
    context: usize,
    symbol: usize,
    level_after_symbol: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CoeffSignRead {
    sign_count: usize,
    dc_sign_context: Option<usize>,
    dc_sign_symbol: Option<usize>,
    first_ac_sign_scan_index: Option<usize>,
    first_ac_sign_bit: Option<usize>,
    golomb_count: usize,
    first_golomb_scan_index: Option<usize>,
    first_golomb_value: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SignedCoeffProbe {
    scan_index: usize,
    position: usize,
    value: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResidualPreview {
    tx_type: TxType,
    dequant_non_zero_count: usize,
    first_dequant_coeff: Option<DequantCoeffProbe>,
    residual_sample_count: usize,
    first_residual_sample: Option<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DequantCoeffProbe {
    position: usize,
    value: i32,
}

fn coeff_base_non_zero_count(base_levels: &[i32]) -> usize {
    let mut non_zero_count = 0usize;
    for level in base_levels.iter().copied() {
        let magnitude = level.unsigned_abs() as usize;
        if magnitude != 0 {
            non_zero_count += 1;
        }
    }
    non_zero_count
}

fn first_signed_coeff(
    eob: usize,
    scan: &[usize],
    coefficients: &[i32],
) -> Result<Option<SignedCoeffProbe>, DecoderError> {
    if eob == 0 || eob > scan.len() {
        return Err(DecoderError::InvalidParam(
            "AV1 signed coefficient eob exceeds scan".to_string(),
        ));
    }
    for scan_index in 0..eob {
        let position = scan[scan_index];
        let value = coefficients[position];
        if value != 0 {
            return Ok(Some(SignedCoeffProbe {
                scan_index,
                position,
                value,
            }));
        }
    }
    Ok(None)
}

fn build_residual_preview(
    transform: TransformBlock,
    coefficients: &[i32],
    quant_state: QuantState,
    bit_depth: u8,
    tx_type: TxType,
) -> Result<Option<ResidualPreview>, DecoderError> {
    if coefficients.len() != transform.tx_size.sample_count() {
        return Err(DecoderError::InvalidParam(
            "AV1 residual preview coefficient count does not match transform size".to_string(),
        ));
    }
    if !matches!(
        tx_type,
        TxType::DctDct
            | TxType::AdstDct
            | TxType::DctAdst
            | TxType::AdstAdst
            | TxType::Identity
            | TxType::VerticalDct
            | TxType::HorizontalDct
    ) {
        return Ok(None);
    }
    let plane_quant = quant_state.plane(transform.plane);
    let dequant = dequantize_coefficients(
        coefficients,
        plane_quant,
        bit_depth,
        transform.tx_size.dq_denom(),
    );
    let first_dequant_coeff = dequant
        .iter()
        .copied()
        .enumerate()
        .find_map(|(position, value)| {
            (value != 0).then_some(DequantCoeffProbe { position, value })
        });
    let dequant_non_zero_count = dequant.iter().filter(|value| **value != 0).count();
    let residual = inverse_transform(tx_type, transform.tx_size, &dequant, bit_depth)?;
    let first_residual_sample = residual.first().copied();
    Ok(Some(ResidualPreview {
        tx_type,
        dequant_non_zero_count,
        first_dequant_coeff,
        residual_sample_count: residual.len(),
        first_residual_sample,
    }))
}

const NUM_BASE_LEVELS: usize = 2;
const COEFF_BASE_RANGE: usize = 12;
const BR_CDF_SIZE: usize = 4;
const COEFF_BR_CDF_ROUNDS: usize = COEFF_BASE_RANGE / (BR_CDF_SIZE - 1);
const MAX_BASE_BR_RANGE: usize = NUM_BASE_LEVELS + COEFF_BASE_RANGE + 1;
const BR_LEVEL_CAP: usize = COEFF_BASE_RANGE + NUM_BASE_LEVELS + 1;
const COEFFICIENT_LEVEL_MASK: usize = (1 << 20) - 1;

fn clamp_coefficient_level(level: usize) -> usize {
    level & COEFFICIENT_LEVEL_MASK
}

#[cfg(test)]
#[path = "tests/tile_decode_coeff.rs"]
mod coeff_tests;

const MAG_REF_OFFSET_WITH_TX_CLASS_2D: [(usize, usize); 3] = [(0, 1), (1, 0), (1, 1)];
const SIG_REF_DIFF_OFFSET_2D: [(usize, usize); 5] = [(0, 1), (1, 0), (1, 1), (0, 2), (2, 0)];
const COEFF_BASE_CTX_OFFSET_SQUARE: [[[usize; 5]; 5]; 5] = [
    [
        [0, 1, 6, 6, 0],
        [1, 6, 6, 21, 0],
        [6, 6, 21, 21, 0],
        [6, 21, 21, 21, 0],
        [0, 0, 0, 0, 0],
    ],
    [
        [0, 1, 6, 6, 21],
        [1, 6, 6, 21, 21],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [
        [0, 1, 6, 6, 21],
        [1, 6, 6, 21, 21],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [
        [0, 1, 6, 6, 21],
        [1, 6, 6, 21, 21],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    [
        [0, 1, 6, 6, 21],
        [1, 6, 6, 21, 21],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
];

fn coeff_base_context_2d(
    tx_size: super::syntax::TxSize,
    position: usize,
    quant: &[i32],
) -> Result<(usize, usize), DecoderError> {
    if quant.len() != tx_size.sample_count() {
        return Err(DecoderError::InvalidParam(
            "AV1 coeff_base context quant buffer size does not match transform size".to_string(),
        ));
    }
    if position >= quant.len() {
        return Err(DecoderError::InvalidParam(
            "AV1 coeff_base context position exceeds transform size".to_string(),
        ));
    }

    let width = tx_size.width();
    let height = tx_size.height();
    let row = position / width;
    let col = position % width;
    let mut magnitude = 0usize;
    for (row_offset, col_offset) in SIG_REF_DIFF_OFFSET_2D {
        let ref_row = row + row_offset;
        let ref_col = col + col_offset;
        if ref_row < height && ref_col < width {
            magnitude += quant[ref_row * width + ref_col].unsigned_abs().min(3) as usize;
        }
    }

    if row == 0 && col == 0 {
        return Ok((0, magnitude));
    }

    let context_delta = ((magnitude + 1) >> 1).min(4);
    let offset = COEFF_BASE_CTX_OFFSET_SQUARE[tx_size.coeff_cdf_index()][row.min(4)][col.min(4)];
    Ok((context_delta + offset, magnitude))
}

fn coeff_base_context_1d(
    tx_size: super::syntax::TxSize,
    tx_type: TxType,
    position: usize,
    quant: &[i32],
) -> Result<(usize, usize), DecoderError> {
    if quant.len() != tx_size.sample_count() || position >= quant.len() {
        return Err(DecoderError::InvalidParam(
            "AV1 1D coeff_base context input is invalid".to_string(),
        ));
    }
    let width = tx_size.width();
    let height = tx_size.height();
    let row = position / width;
    let col = position % width;
    let offsets: [(usize, usize); 5] = match tx_type {
        TxType::VerticalDct => [(0, 1), (1, 0), (0, 2), (0, 3), (0, 4)],
        TxType::HorizontalDct => [(0, 1), (1, 0), (2, 0), (3, 0), (4, 0)],
        _ => {
            return Err(DecoderError::InvalidParam(
                "AV1 1D coeff_base context requires a directional transform".to_string(),
            ));
        }
    };
    let magnitude = offsets
        .into_iter()
        .filter_map(|(dy, dx)| {
            let y = row + dy;
            let x = col + dx;
            (y < height && x < width).then(|| quant[y * width + x].unsigned_abs().min(3) as usize)
        })
        .sum::<usize>();
    if position == 0 {
        return Ok((0, magnitude));
    }
    let delta = ((magnitude + 1) >> 1).min(4);
    let axis = if tx_type == TxType::HorizontalDct {
        row
    } else {
        col
    };
    let offset = if axis == 0 {
        26
    } else if axis == 1 {
        31
    } else {
        36
    };
    Ok((offset + delta, magnitude))
}

fn coeff_br_context_2d(
    tx_size: super::syntax::TxSize,
    position: usize,
    quant: &[i32],
) -> Result<usize, DecoderError> {
    if quant.len() != tx_size.sample_count() {
        return Err(DecoderError::InvalidParam(
            "AV1 coeff_br context quant buffer size does not match transform size".to_string(),
        ));
    }
    if position >= quant.len() {
        return Err(DecoderError::InvalidParam(
            "AV1 coeff_br context position exceeds transform size".to_string(),
        ));
    }

    let width = tx_size.width();
    let height = tx_size.height();
    let row = position / width;
    let col = position % width;
    let mut magnitude = 0usize;
    for (row_offset, col_offset) in MAG_REF_OFFSET_WITH_TX_CLASS_2D {
        let ref_row = row + row_offset;
        let ref_col = col + col_offset;
        if ref_row < height && ref_col < width {
            magnitude +=
                (quant[ref_row * width + ref_col].unsigned_abs() as usize).min(BR_LEVEL_CAP);
        }
    }

    let magnitude_context = ((magnitude + 1) >> 1).min(6);
    if position == 0 {
        Ok(magnitude_context)
    } else if row < 2 && col < 2 {
        Ok(magnitude_context + 7)
    } else {
        Ok(magnitude_context + 14)
    }
}

fn coeff_br_context_1d(
    tx_size: super::syntax::TxSize,
    tx_type: TxType,
    position: usize,
    quant: &[i32],
) -> Result<usize, DecoderError> {
    if quant.len() != tx_size.sample_count() || position >= quant.len() {
        return Err(DecoderError::InvalidParam(
            "AV1 1D coeff_br context input is invalid".to_string(),
        ));
    }
    let width = tx_size.width();
    let height = tx_size.height();
    let row = position / width;
    let col = position % width;
    let offsets = match tx_type {
        TxType::VerticalDct => [(0, 1), (1, 0), (0, 2)],
        TxType::HorizontalDct => [(0, 1), (1, 0), (2, 0)],
        _ => {
            return Err(DecoderError::InvalidParam(
                "AV1 1D coeff_br context requires a directional transform".to_string(),
            ));
        }
    };
    let magnitude = offsets
        .into_iter()
        .filter_map(|(dy, dx)| {
            let y = row + dy;
            let x = col + dx;
            (y < height && x < width)
                .then(|| (quant[y * width + x].unsigned_abs() as usize).min(BR_LEVEL_CAP))
        })
        .sum::<usize>();
    let delta = ((magnitude + 1) >> 1).min(6);
    if position == 0 {
        Ok(delta)
    } else if (tx_type == TxType::HorizontalDct && row == 0)
        || (tx_type == TxType::VerticalDct && col == 0)
    {
        Ok(delta + 7)
    } else {
        Ok(delta + 14)
    }
}

pub fn prepare_tile_entropy(
    data: &[u8],
    tile_group: &TileGroup,
    frame: &FrameHeader,
) -> Result<Vec<TileEntropyState>, DecoderError> {
    if tile_group.tiles.is_empty() {
        return Err(DecoderError::Bitstream(
            "AV1 tile group has no tile payloads".to_string(),
        ));
    }

    let mut states = Vec::with_capacity(tile_group.tiles.len());
    for tile in &tile_group.tiles {
        let end = tile
            .offset
            .checked_add(tile.len)
            .ok_or_else(|| DecoderError::Bitstream("AV1 tile payload end overflow".to_string()))?;
        let payload = data.get(tile.offset..end).ok_or_else(|| {
            DecoderError::NotEnoughData("AV1 tile payload extends beyond tile group".to_string())
        })?;
        let decoder = EntropyDecoder::new(payload, frame.disable_cdf_update)?;
        states.push(TileEntropyState {
            tile_id: tile.tile_id,
            payload_offset: tile.offset,
            payload_len: tile.len,
            entropy_start_bits: decoder.bit_position(),
        });
    }
    Ok(states)
}

pub fn probe_tile_partitions(
    data: &[u8],
    tile_group: &TileGroup,
    frame: &FrameHeader,
    plan: &FrameDecodePlan,
) -> Result<Vec<PartitionProbe>, DecoderError> {
    let mut probes = Vec::with_capacity(tile_group.tiles.len());
    for (index, tile_payload) in tile_group.tiles.iter().enumerate() {
        let end = tile_payload
            .offset
            .checked_add(tile_payload.len)
            .ok_or_else(|| DecoderError::Bitstream("AV1 tile payload end overflow".to_string()))?;
        let payload = data.get(tile_payload.offset..end).ok_or_else(|| {
            DecoderError::NotEnoughData("AV1 tile payload extends beyond tile group".to_string())
        })?;
        let tile_plan = plan.tiles.get(index).ok_or_else(|| {
            DecoderError::Bitstream("AV1 tile decode plan is missing a tile".to_string())
        })?;
        let mut decoder = TileDecoder::new(payload, frame)?;
        probes.push(decoder.read_root_partition(tile_plan)?);
    }
    Ok(probes)
}

pub fn probe_tile_block_modes(
    data: &[u8],
    tile_group: &TileGroup,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    plan: &FrameDecodePlan,
) -> Result<Vec<BlockModeProbe>, DecoderError> {
    let mut probes = Vec::with_capacity(tile_group.tiles.len());
    for (index, tile_payload) in tile_group.tiles.iter().enumerate() {
        let end = tile_payload
            .offset
            .checked_add(tile_payload.len)
            .ok_or_else(|| DecoderError::Bitstream("AV1 tile payload end overflow".to_string()))?;
        let payload = data.get(tile_payload.offset..end).ok_or_else(|| {
            DecoderError::NotEnoughData("AV1 tile payload extends beyond tile group".to_string())
        })?;
        let tile_plan = plan.tiles.get(index).ok_or_else(|| {
            DecoderError::Bitstream("AV1 tile decode plan is missing a tile".to_string())
        })?;
        let mut decoder = TileDecoder::new(payload, frame)?;
        let partition = decoder.read_root_partition(tile_plan)?;
        if partition.partition == Partition::None {
            probes.push(decoder.read_intra_frame_block_mode(
                sequence,
                frame,
                tile_plan,
                partition.block_size,
                tile_plan.pixel_x,
                tile_plan.pixel_y,
            )?);
        }
    }
    Ok(probes)
}

pub fn probe_first_block_residuals(
    data: &[u8],
    tile_group: &TileGroup,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    plan: &FrameDecodePlan,
) -> Result<Vec<ResidualProbe>, DecoderError> {
    let quant_state =
        QuantState::from_params(&frame.quantization, sequence.color_config.bit_depth)?;
    let mut probes = Vec::with_capacity(tile_group.tiles.len());
    for (index, tile_payload) in tile_group.tiles.iter().enumerate() {
        let end = tile_payload
            .offset
            .checked_add(tile_payload.len)
            .ok_or_else(|| DecoderError::Bitstream("AV1 tile payload end overflow".to_string()))?;
        let payload = data.get(tile_payload.offset..end).ok_or_else(|| {
            DecoderError::NotEnoughData("AV1 tile payload extends beyond tile group".to_string())
        })?;
        let tile_plan = plan.tiles.get(index).ok_or_else(|| {
            DecoderError::Bitstream("AV1 tile decode plan is missing a tile".to_string())
        })?;
        let mut decoder = TileDecoder::new(payload, frame)?;
        let partition = decoder.read_root_partition(tile_plan)?;
        if partition.partition == Partition::None {
            let block_mode = decoder.read_intra_frame_block_mode(
                sequence,
                frame,
                tile_plan,
                partition.block_size,
                tile_plan.pixel_x,
                tile_plan.pixel_y,
            )?;
            let transforms = plan_transform_blocks_with_tx_size(
                0,
                0,
                0,
                block_mode.block_size,
                block_mode.tx_size,
                plan.width,
                plan.height,
            );
            probes.push(decoder.read_first_transform_residual(
                tile_plan.tile_id,
                frame,
                &block_mode,
                &transforms,
                quant_state,
                sequence.color_config.bit_depth,
            )?);
        }
    }
    Ok(probes)
}

pub fn decode_first_luma_transform(
    data: &[u8],
    tile_group: &TileGroup,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    plan: &FrameDecodePlan,
    buffers: &mut FrameBuffers,
) -> Result<ResidualProbe, DecoderError> {
    let quant_state =
        QuantState::from_params(&frame.quantization, sequence.color_config.bit_depth)?;
    let tile_payload = tile_group.tiles.first().ok_or_else(|| {
        DecoderError::Bitstream("AV1 tile group has no tile payloads".to_string())
    })?;
    let end = tile_payload
        .offset
        .checked_add(tile_payload.len)
        .ok_or_else(|| DecoderError::Bitstream("AV1 tile payload end overflow".to_string()))?;
    let payload = data.get(tile_payload.offset..end).ok_or_else(|| {
        DecoderError::NotEnoughData("AV1 tile payload extends beyond tile group".to_string())
    })?;
    let tile_plan = plan
        .tiles
        .first()
        .ok_or_else(|| DecoderError::Bitstream("AV1 tile decode plan is missing".to_string()))?;
    let mut decoder = TileDecoder::new(payload, frame)?;
    let partition = decoder.read_root_partition(tile_plan)?;
    if partition.partition != Partition::None {
        return Err(DecoderError::Unsupported(format!(
            "AV1 first luma transform decode for root partition {:?} is not supported yet",
            partition.partition
        )));
    }

    let block_mode = decoder.read_intra_frame_block_mode(
        sequence,
        frame,
        tile_plan,
        partition.block_size,
        tile_plan.pixel_x,
        tile_plan.pixel_y,
    )?;
    let transforms = plan_transform_blocks_with_tx_size(
        0,
        tile_plan.pixel_x,
        tile_plan.pixel_y,
        block_mode.block_size,
        block_mode.tx_size,
        plan.width,
        plan.height,
    );
    let residual = decoder.read_first_transform_residual(
        tile_plan.tile_id,
        frame,
        &block_mode,
        &transforms,
        quant_state,
        sequence.color_config.bit_depth,
    )?;

    if residual.skipped || residual.first_non_zero_transform.is_none() {
        return Ok(residual);
    }
    let transform = residual
        .first_non_zero_transform
        .expect("checked first_non_zero_transform");
    let tx_type = residual
        .tx_type
        .ok_or_else(|| DecoderError::Bitstream("AV1 residual tx_type is missing".to_string()))?;
    let coefficients = residual
        .first_quantized_coefficients
        .as_ref()
        .ok_or_else(|| {
            DecoderError::Bitstream("AV1 residual quantized coefficients are missing".to_string())
        })?;
    let mid = 1u16 << (sequence.color_config.bit_depth - 1);
    let above = vec![mid; transform.tx_size.width()];
    let left = vec![mid; transform.tx_size.height()];
    let prediction = predict_intra(
        block_mode.y_mode,
        block_mode.angle_delta_y,
        transform.tx_size.width(),
        transform.tx_size.height(),
        IntraEdges {
            above: Some(&above),
            left: Some(&left),
            above_left: Some(mid),
            bit_depth: sequence.color_config.bit_depth,
        },
    )?;
    let quantized = QuantizedTransform {
        block: transform,
        tx_type,
        coefficients: coefficients.clone(),
    };
    let luma = buffers
        .planes
        .get_mut(0)
        .ok_or_else(|| DecoderError::Bitstream("AV1 luma plane is missing".to_string()))?;
    reconstruct_transform_block(
        luma,
        &quantized,
        quant_state.plane(transform.plane),
        &prediction,
        sequence.color_config.bit_depth,
    )?;

    Ok(residual)
}

pub fn decode_first_luma_block(
    data: &[u8],
    tile_group: &TileGroup,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    plan: &FrameDecodePlan,
    buffers: &mut FrameBuffers,
) -> Result<Vec<DecodedTransform>, DecoderError> {
    let tile_payload = tile_group.tiles.first().ok_or_else(|| {
        DecoderError::Bitstream("AV1 tile group has no tile payloads".to_string())
    })?;
    let end = tile_payload
        .offset
        .checked_add(tile_payload.len)
        .ok_or_else(|| DecoderError::Bitstream("AV1 tile payload end overflow".to_string()))?;
    let payload = data.get(tile_payload.offset..end).ok_or_else(|| {
        DecoderError::NotEnoughData("AV1 tile payload extends beyond tile group".to_string())
    })?;
    let tile_plan = plan
        .tiles
        .first()
        .ok_or_else(|| DecoderError::Bitstream("AV1 tile decode plan is missing".to_string()))?;
    let mut decoder = TileDecoder::new(payload, frame)?;
    let block = decode_luma_root_block(
        &mut decoder,
        sequence,
        frame,
        tile_plan,
        plan,
        buffers,
        tile_plan.pixel_x,
        tile_plan.pixel_y,
    )?;
    Ok(block.transforms)
}

pub fn decode_luma_root_blocks(
    data: &[u8],
    tile_group: &TileGroup,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    plan: &FrameDecodePlan,
    buffers: &mut FrameBuffers,
    max_blocks: usize,
) -> Result<Vec<DecodedLumaBlock>, DecoderError> {
    Ok(
        decode_luma_root_block_prefix(
            data, tile_group, sequence, frame, plan, buffers, max_blocks,
        )?
        .blocks,
    )
}

pub fn decode_luma_root_block_prefix(
    data: &[u8],
    tile_group: &TileGroup,
    sequence: &SequenceHeader,
    frame: &FrameHeader,
    plan: &FrameDecodePlan,
    buffers: &mut FrameBuffers,
    max_blocks: usize,
) -> Result<DecodedBlockPrefix, DecoderError> {
    let tile_payload = tile_group.tiles.first().ok_or_else(|| {
        DecoderError::Bitstream("AV1 tile group has no tile payloads".to_string())
    })?;
    let end = tile_payload
        .offset
        .checked_add(tile_payload.len)
        .ok_or_else(|| DecoderError::Bitstream("AV1 tile payload end overflow".to_string()))?;
    let payload = data.get(tile_payload.offset..end).ok_or_else(|| {
        DecoderError::NotEnoughData("AV1 tile payload extends beyond tile group".to_string())
    })?;
    let tile_plan = plan
        .tiles
        .first()
        .ok_or_else(|| DecoderError::Bitstream("AV1 tile decode plan is missing".to_string()))?;
    let mut decoder = TileDecoder::new(payload, frame)?;
    let mut blocks = Vec::new();
    let mut block_budget = max_blocks;
    for sb_row in tile_plan.sb_row_start..tile_plan.sb_row_end {
        for sb_col in tile_plan.sb_col_start..tile_plan.sb_col_end {
            if block_budget == 0 {
                return Ok(DecodedBlockPrefix {
                    blocks,
                    next_unsupported: None,
                });
            }
            let x = (sb_col as usize * plan.superblock_size).min(plan.width);
            let y = (sb_row as usize * plan.superblock_size).min(plan.height);
            let decoded = match decode_luma_block_tree(
                &mut decoder,
                sequence,
                frame,
                tile_plan,
                plan,
                buffers,
                BlockSize::Block128x128,
                x,
                y,
                &mut block_budget,
            ) {
                Ok(blocks) => blocks,
                Err(err @ DecoderError::Unsupported(_)) if !blocks.is_empty() => {
                    return Ok(DecodedBlockPrefix {
                        blocks,
                        next_unsupported: Some(err),
                    });
                }
                Err(err) => return Err(err),
            };
            blocks.extend(decoded);
        }
    }

    Ok(DecodedBlockPrefix {
        blocks,
        next_unsupported: None,
    })
}

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
    fn smooth_mode_grid_tracks_above_and_left_neighbours() {
        let mut grid = vec![None; 16];
        fill_mi_grid(&mut grid, 4, 4, 4, 4, BlockSize::Block8x8, true);

        assert!(has_smooth_neighbour(&grid, 4, 4, 4, 12));
        assert!(has_smooth_neighbour(&grid, 4, 4, 12, 4));
        assert!(!has_smooth_neighbour(&grid, 4, 4, 0, 0));
        assert!(PredictionMode::Smooth.is_smooth());
        assert!(!PredictionMode::Vertical.is_smooth());
    }

    #[test]
    fn transform_prediction_reads_reconstructed_neighbor() {
        let layout = super::super::decode::PlaneLayout {
            plane: 0,
            width: 8,
            height: 4,
            subsampling_x: 0,
            subsampling_y: 0,
            sample_count: 32,
        };
        let mut samples = vec![0; 32];
        for (row, value) in [10, 20, 30, 40].into_iter().enumerate() {
            samples[row * 8 + 3] = value;
        }
        let plane = PlaneBuffer { layout, samples };

        let prediction = predict_block(
            &plane,
            PredictionMode::Horizontal,
            4,
            0,
            4,
            4,
            None,
            None,
            8,
            false,
            false,
        )
        .unwrap();

        assert_eq!(
            prediction,
            vec![
                10, 10, 10, 10, //
                20, 20, 20, 20, //
                30, 30, 30, 30, //
                40, 40, 40, 40,
            ]
        );
    }

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

        let probe = decoder.read_root_partition(&plan.tiles[0]).unwrap();

        assert_eq!(probe.tile_id, 0);
        assert_eq!(probe.block_size, BlockSize::Block128x128);
        assert_eq!(probe.symbol, 0);
        assert_eq!(probe.partition, Partition::None);
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
        assert_eq!(probes[0].block_size, BlockSize::Block128x128);
        assert!(probes[0].y_mode_symbol < 13);
        assert!(probes[0].uv_mode_symbol.is_some());
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

        assert_eq!(transforms.len(), 16);
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
        assert_eq!(probes[0].block_size, BlockSize::Block128x128);
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
                assert!(probes[0].tx_type_read);
                assert_eq!(probes[0].tx_type_set, Some(1));
                assert!(probes[0].tx_type_symbol.unwrap() < 7);
                assert_eq!(probes[0].tx_type, Some(TxType::VerticalDct));
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
            assert!(buffers.planes[0].samples.iter().any(|sample| *sample != 0));
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
        assert!(buffers.planes[0].samples.iter().any(|sample| *sample != 0));
        assert!(buffers.planes[1].samples.iter().any(|sample| *sample != 0));
        assert!(buffers.planes[2].samples.iter().any(|sample| *sample != 0));
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
        assert_eq!((blocks[1].x, blocks[1].y), (128, 0));
        assert!(blocks.iter().any(|block| !block.transforms.is_empty()));
        assert!(buffers.planes[1].samples.iter().any(|sample| *sample != 0));
        assert!(buffers.planes[2].samples.iter().any(|sample| *sample != 0));
        assert_eq!(prefix.next_unsupported, None);
    }

    #[test]
    fn decodes_all_sample_blocks_without_unsupported_syntax() {
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

        assert_eq!(prefix.blocks.len(), 915);
        assert!(
            buffers
                .planes
                .iter()
                .all(|plane| { plane.samples.iter().any(|sample| *sample != 0) })
        );
        assert_eq!(prefix.next_unsupported, None);
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
