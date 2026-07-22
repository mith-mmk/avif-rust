use super::TileDecoder;
use super::partition_syntax::{partition_plane_context, partition_subsize};
use crate::DecoderError;
use crate::av1::decode::TileDecodePlan;
use crate::av1::syntax::{BlockSize, Partition, TxSize};

fn intra_bc_candidate_offsets(block_size: BlockSize) -> [(isize, isize); 9] {
    let n8_w = block_size.width().max(8) / 8;
    let n8_h = block_size.height().max(8) / 8;
    [
        ((n8_h.saturating_sub(1) * 2) as isize, -2),
        (-2isize, n8_w.saturating_sub(1) as isize * 2),
        (-2, ((n8_w.saturating_sub(1) / 2) * 2) as isize),
        (((n8_h.saturating_sub(1) / 2) * 2) as isize, -2),
        (-2, -2),
        (-2, (n8_w * 2) as isize),
        ((n8_h * 2) as isize, -2),
        (-2, -6),
        ((n8_h.saturating_sub(1) * 2) as isize, -6),
    ]
}

impl<'a> TileDecoder<'a> {
    fn inter_mv_neighbor_candidates(
        &self,
        x: usize,
        y: usize,
        block_size: BlockSize,
        reference_frame: u8,
    ) -> [Option<(i32, i32)>; 4] {
        let mi_col = x >> 2;
        let mi_row = y >> 2;
        let block_mi_width = (block_size.width() / 4).max(1);
        let candidates = [
            (mi_col, mi_row.saturating_sub(1)),
            (mi_col.saturating_sub(1), mi_row),
            (mi_col.saturating_sub(1), mi_row.saturating_sub(1)),
            (
                mi_col.saturating_add(block_mi_width),
                mi_row.saturating_sub(1),
            ),
        ];
        let mut values = [None; 4];
        for (candidate_index, (candidate_col, candidate_row)) in candidates.into_iter().enumerate()
        {
            if candidate_col < self.tile_mi_col_start
                || candidate_row < self.tile_mi_row_start
                || candidate_col >= self.mi_cols
                || candidate_row >= self.mi_rows
            {
                continue;
            }
            let index = candidate_row * self.mi_cols + candidate_col;
            if self.reference_frame_grid[index] == Some(reference_frame) {
                values[candidate_index] = self.motion_vector_grid[index];
            }
        }
        values
    }

    pub(super) fn inter_mv_predictor(
        &self,
        x: usize,
        y: usize,
        block_size: BlockSize,
        reference_frame: u8,
    ) -> (i32, i32) {
        self.inter_mv_neighbor_candidates(x, y, block_size, reference_frame)
            .into_iter()
            .flatten()
            .next()
            .unwrap_or((0, 0))
    }

    pub(super) fn inter_mv_candidate(
        &self,
        x: usize,
        y: usize,
        block_size: BlockSize,
        reference_frame: u8,
        candidate_index: usize,
    ) -> (i32, i32) {
        if candidate_index == 0 {
            return self.inter_mv_predictor(x, y, block_size, reference_frame);
        }
        let mut seen = [(0, 0); 4];
        let mut seen_len = 0;
        for mv in self
            .inter_mv_neighbor_candidates(x, y, block_size, reference_frame)
            .into_iter()
            .flatten()
        {
            if !seen[..seen_len].contains(&mv) {
                if seen_len == candidate_index {
                    return mv;
                }
                seen[seen_len] = mv;
                seen_len += 1;
            }
        }
        (0, 0)
    }

    pub(super) fn inter_mv_candidate_count(
        &self,
        x: usize,
        y: usize,
        block_size: BlockSize,
        reference_frame: u8,
    ) -> usize {
        let mut seen = [(0, 0); 4];
        let mut seen_len = 0;
        for mv in self
            .inter_mv_neighbor_candidates(x, y, block_size, reference_frame)
            .into_iter()
            .flatten()
        {
            if !seen[..seen_len].contains(&mv) {
                seen[seen_len] = mv;
                seen_len += 1;
            }
        }
        seen_len.max(1)
    }

    pub(super) fn set_inter_mv(
        &mut self,
        x: usize,
        y: usize,
        block_size: BlockSize,
        reference_frame: u8,
        motion_vector: (i32, i32),
    ) {
        super::context_grid::fill_mi_grid(
            &mut self.reference_frame_grid,
            self.mi_cols,
            self.mi_rows,
            x,
            y,
            block_size,
            reference_frame,
        );
        super::context_grid::fill_mi_grid(
            &mut self.motion_vector_grid,
            self.mi_cols,
            self.mi_rows,
            x,
            y,
            block_size,
            motion_vector,
        );
    }

    pub(super) fn intra_bc_mv_predictor(
        &self,
        x: usize,
        y: usize,
        block_size: BlockSize,
    ) -> Option<(i32, i32)> {
        let mi_col = x >> 2;
        let mi_row = y >> 2;
        for (row_offset, col_offset) in intra_bc_candidate_offsets(block_size) {
            let candidate_row = mi_row.checked_add_signed(row_offset);
            let candidate_col = mi_col.checked_add_signed(col_offset);
            let (Some(candidate_row), Some(candidate_col)) = (candidate_row, candidate_col) else {
                continue;
            };
            if candidate_row < self.tile_mi_row_start
                || candidate_col < self.tile_mi_col_start
                || candidate_row >= self.mi_rows
                || candidate_col >= self.mi_cols
            {
                continue;
            }
            let Some(mv) = self.intra_bc_mv_at(candidate_col, candidate_row) else {
                continue;
            };
            return Some(mv);
        }
        None
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

#[cfg(test)]
mod tests {
    use super::intra_bc_candidate_offsets;
    use crate::av1::syntax::BlockSize;

    #[test]
    fn intrabc_candidate_offsets_follow_block_geometry() {
        assert_eq!(
            intra_bc_candidate_offsets(BlockSize::Block16x16),
            [
                (2, -2),
                (-2, 2),
                (-2, 0),
                (0, -2),
                (-2, -2),
                (-2, 4),
                (4, -2),
                (-2, -6),
                (2, -6),
            ]
        );
        assert_eq!(
            intra_bc_candidate_offsets(BlockSize::Block4x16)[..4],
            [(2, -2), (-2, 0), (-2, 0), (0, -2)]
        );
    }
}
