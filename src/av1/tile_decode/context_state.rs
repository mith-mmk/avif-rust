use super::TileDecoder;
use super::partition_syntax::{partition_plane_context, partition_subsize};
use crate::DecoderError;
use crate::av1::decode::TileDecodePlan;
use crate::av1::syntax::{BlockSize, Partition, TxSize};

impl<'a> TileDecoder<'a> {
    pub(super) fn intra_bc_mv_predictor(&self, x: usize, y: usize) -> Option<(i32, i32)> {
        let mi_col = x >> 2;
        let mi_row = y >> 2;
        let left = (mi_col > self.tile_mi_col_start)
            .then(|| self.intra_bc_mv_at(mi_col - 1, mi_row))
            .flatten();
        let above = (mi_row > self.tile_mi_row_start)
            .then(|| self.intra_bc_mv_at(mi_col, mi_row - 1))
            .flatten();
        left.or(above)
    }

    pub(super) fn set_intra_bc_mv(
        &mut self,
        x: usize,
        y: usize,
        block_size: BlockSize,
        mv: (i32, i32),
    ) {
        super::context_grid::fill_mi_grid(
            &mut self.intra_bc_mv_grid,
            self.mi_cols,
            self.mi_rows,
            x,
            y,
            block_size,
            mv,
        );
    }

    fn intra_bc_mv_at(&self, mi_col: usize, mi_row: usize) -> Option<(i32, i32)> {
        if mi_col >= self.mi_cols || mi_row >= self.mi_rows {
            return None;
        }
        self.intra_bc_mv_grid[mi_row * self.mi_cols + mi_col]
    }

    pub(super) fn reset_left_superblock_contexts(&mut self) {
        self.left_partition_context.fill(0);
        self.left_txfm_context.fill(64);
        for contexts in &mut self.plane_entropy_contexts {
            contexts.left.fill(0);
        }
    }

    pub(super) fn skip_context(&self, x: usize, y: usize) -> usize {
        usize::from(self.above_skip_context(x, y)) + usize::from(self.left_skip_context(x, y))
    }

    fn above_skip_context(&self, x: usize, y: usize) -> bool {
        if (y >> 2) <= self.tile_mi_row_start {
            return false;
        }
        self.skip_at_mi(x >> 2, (y >> 2).saturating_sub(1))
            .unwrap_or(false)
    }

    fn left_skip_context(&self, x: usize, y: usize) -> bool {
        if (x >> 2) <= self.tile_mi_col_start {
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

    pub(super) fn set_skip_context(
        &mut self,
        x: usize,
        y: usize,
        block_size: BlockSize,
        skip: bool,
    ) {
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

    pub(super) fn tx_size_context(&self, x: usize, y: usize, block_size: BlockSize) -> usize {
        let (max_tx_width, max_tx_height) = block_size.largest_supported_tx_dimensions();
        let has_above = (y >> 2) > self.tile_mi_row_start;
        let has_left = (x >> 2) > self.tile_mi_col_start;
        let above =
            has_above && self.above_txfm_context.get(x >> 2).copied().unwrap_or(0) >= max_tx_width;
        let left =
            has_left && self.left_txfm_context.get(y >> 2).copied().unwrap_or(0) >= max_tx_height;
        match (has_above, has_left) {
            (true, true) => usize::from(above) + usize::from(left),
            (true, false) => usize::from(above),
            (false, true) => usize::from(left),
            (false, false) => 0,
        }
    }

    pub(super) fn set_txfm_context(
        &mut self,
        x: usize,
        y: usize,
        block_size: BlockSize,
        tx_size: TxSize,
    ) {
        self.set_txfm_context_dimensions(x, y, block_size, tx_size.width(), tx_size.height());
    }

    pub(super) fn set_txfm_context_dimensions(
        &mut self,
        x: usize,
        y: usize,
        block_size: BlockSize,
        width: usize,
        height: usize,
    ) {
        let start_col = x >> 2;
        let start_row = y >> 2;
        let end_col = ((x + block_size.width()).min(self.mi_cols << 2) + 3) >> 2;
        let end_row = ((y + block_size.height()).min(self.mi_rows << 2) + 3) >> 2;
        for mi_col in start_col..end_col.min(self.mi_cols) {
            self.above_txfm_context[mi_col] = width;
        }
        for mi_row in start_row..end_row.min(self.mi_rows) {
            self.left_txfm_context[mi_row] = height;
        }
    }

    pub(super) fn partition_context(
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

    pub(super) fn update_ext_partition_context(
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
}
