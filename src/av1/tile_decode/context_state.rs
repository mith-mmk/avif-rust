use super::TileDecoder;
use super::diagnostic::{LocalWarpSample, ObmcNeighbor, ObmcNeighbors};
use super::partition_syntax::{partition_plane_context, partition_subsize};
use crate::DecoderError;
use crate::av1::decode::TileDecodePlan;
use crate::av1::frame::InterpolationFilter;
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

fn push_local_warp_offset(
    offsets: &mut [(isize, isize); 8],
    offset_len: &mut usize,
    delta_row: isize,
    delta_col: isize,
) {
    if *offset_len < offsets.len() {
        offsets[*offset_len] = (delta_row, delta_col);
        *offset_len += 1;
    }
}

fn push_mv_candidate(
    values: &mut [Option<(i32, i32)>; 4],
    value_len: &mut usize,
    candidate: Option<(i32, i32)>,
) {
    let Some(candidate) = candidate else {
        return;
    };
    if *value_len == 4 || values[..*value_len].contains(&Some(candidate)) {
        return;
    }
    values[*value_len] = Some(candidate);
    *value_len += 1;
}

impl<'a> TileDecoder<'a> {
    pub(super) fn obmc_neighbors(
        &self,
        x: usize,
        y: usize,
        block_size: BlockSize,
        reference_frame: u8,
    ) -> ObmcNeighbors {
        ObmcNeighbors {
            above: self.collect_obmc_above(x, y, block_size, reference_frame),
            left: self.collect_obmc_left(x, y, block_size, reference_frame),
        }
    }

    fn collect_obmc_above(
        &self,
        x: usize,
        y: usize,
        block_size: BlockSize,
        reference_frame: u8,
    ) -> [Option<ObmcNeighbor>; 4] {
        let mi_col = x / 4;
        let mi_row = y / 4;
        if mi_row <= self.tile_mi_row_start {
            return [None; 4];
        }
        let end_col = mi_col
            .saturating_add((block_size.width() / 4).max(1))
            .min(self.mi_cols);
        let sample_row = mi_row - 1;
        let mut result = [None; 4];
        let mut result_len = 0usize;
        let mut candidate_col = mi_col.max(self.tile_mi_col_start);
        while candidate_col < end_col && result_len < result.len() {
            let index = sample_row * self.mi_cols + candidate_col;
            let Some(source_size) = self.motion_block_size_grid[index] else {
                candidate_col += 1;
                continue;
            };
            let source_w4 = (source_size.width() / 4).max(1);
            let source_h4 = (source_size.height() / 4).max(1);
            let origin_col = candidate_col & !(source_w4 - 1);
            let origin_row = sample_row & !(source_h4 - 1);
            if origin_row + source_h4 != mi_row {
                candidate_col += 1;
                continue;
            }
            let origin_index = origin_row * self.mi_cols + origin_col;
            let same_reference = self.reference_frame_grid[origin_index] == Some(reference_frame)
                && self.reference_frame_secondary_grid[origin_index].is_none();
            if same_reference {
                if let Some(motion_vector) = self.motion_vector_grid[origin_index] {
                    let neighbor = ObmcNeighbor {
                        origin_x: origin_col * 4,
                        origin_y: origin_row * 4,
                        width: source_w4 * 4,
                        height: source_h4 * 4,
                        motion_vector,
                        interpolation_filters: self.interpolation_filter_grid[origin_index]
                            .unwrap_or((
                                InterpolationFilter::Regular,
                                InterpolationFilter::Regular,
                            )),
                    };
                    if !result[..result_len].contains(&Some(neighbor)) {
                        result[result_len] = Some(neighbor);
                        result_len += 1;
                    }
                }
            }
            candidate_col = candidate_col
                .saturating_add(1)
                .max(origin_col.saturating_add(source_w4));
        }
        result
    }

    fn collect_obmc_left(
        &self,
        x: usize,
        y: usize,
        block_size: BlockSize,
        reference_frame: u8,
    ) -> [Option<ObmcNeighbor>; 4] {
        let mi_col = x / 4;
        let mi_row = y / 4;
        if mi_col <= self.tile_mi_col_start {
            return [None; 4];
        }
        let end_row = mi_row
            .saturating_add((block_size.height() / 4).max(1))
            .min(self.mi_rows);
        let sample_col = mi_col - 1;
        let mut result = [None; 4];
        let mut result_len = 0usize;
        let mut candidate_row = mi_row.max(self.tile_mi_row_start);
        while candidate_row < end_row && result_len < result.len() {
            let index = candidate_row * self.mi_cols + sample_col;
            let Some(source_size) = self.motion_block_size_grid[index] else {
                candidate_row += 1;
                continue;
            };
            let source_w4 = (source_size.width() / 4).max(1);
            let source_h4 = (source_size.height() / 4).max(1);
            let origin_col = sample_col & !(source_w4 - 1);
            let origin_row = candidate_row & !(source_h4 - 1);
            if origin_col + source_w4 != mi_col {
                candidate_row += 1;
                continue;
            }
            let origin_index = origin_row * self.mi_cols + origin_col;
            let same_reference = self.reference_frame_grid[origin_index] == Some(reference_frame)
                && self.reference_frame_secondary_grid[origin_index].is_none();
            if same_reference {
                if let Some(motion_vector) = self.motion_vector_grid[origin_index] {
                    let neighbor = ObmcNeighbor {
                        origin_x: origin_col * 4,
                        origin_y: origin_row * 4,
                        width: source_w4 * 4,
                        height: source_h4 * 4,
                        motion_vector,
                        interpolation_filters: self.interpolation_filter_grid[origin_index]
                            .unwrap_or((
                                InterpolationFilter::Regular,
                                InterpolationFilter::Regular,
                            )),
                    };
                    if !result[..result_len].contains(&Some(neighbor)) {
                        result[result_len] = Some(neighbor);
                        result_len += 1;
                    }
                }
            }
            candidate_row = candidate_row
                .saturating_add(1)
                .max(origin_row.saturating_add(source_h4));
        }
        result
    }

    pub(super) fn inter_mv_neighbor_candidates(
        &self,
        x: usize,
        y: usize,
        block_size: BlockSize,
        reference_frame: u8,
        secondary: bool,
    ) -> [Option<(i32, i32)>; 4] {
        let mi_col = x >> 2;
        let mi_row = y >> 2;
        let block_mi_width = (block_size.width() / 4).max(1);
        let mut values = [None; 4];
        let mut value_len = 0usize;
        let reference_grid = if secondary {
            &self.reference_frame_secondary_type_grid
        } else {
            &self.reference_frame_type_grid
        };
        let motion_grid = if secondary {
            &self.motion_vector_secondary_grid
        } else {
            &self.motion_vector_grid
        };
        if mi_row > self.tile_mi_row_start {
            let candidate_row = mi_row - 1;
            let mut candidate_col = mi_col;
            let end_col = mi_col.saturating_add(block_mi_width).min(self.mi_cols);
            while candidate_col < end_col && value_len < 4 {
                let index = candidate_row * self.mi_cols + candidate_col;
                let Some(source_size) = self.motion_block_size_grid[index] else {
                    candidate_col += 1;
                    continue;
                };
                let source_w4 = (source_size.width() / 4).max(1);
                let source_h4 = (source_size.height() / 4).max(1);
                let origin_col = candidate_col & !(source_w4 - 1);
                let origin_row = candidate_row & !(source_h4 - 1);
                if origin_row + source_h4 == mi_row {
                    let origin_index = origin_row * self.mi_cols + origin_col;
                    if reference_grid[origin_index] == Some(reference_frame) {
                        push_mv_candidate(
                            &mut values,
                            &mut value_len,
                            motion_grid[origin_index],
                        );
                    }
                }
                candidate_col = candidate_col
                    .saturating_add(1)
                    .max(origin_col.saturating_add(source_w4));
            }
        }
        if mi_col > self.tile_mi_col_start {
            let candidate_col = mi_col - 1;
            let mut candidate_row = mi_row;
            let end_row = mi_row
                .saturating_add((block_size.height() / 4).max(1))
                .min(self.mi_rows);
            while candidate_row < end_row && value_len < 4 {
                let index = candidate_row * self.mi_cols + candidate_col;
                let Some(source_size) = self.motion_block_size_grid[index] else {
                    candidate_row += 1;
                    continue;
                };
                let source_w4 = (source_size.width() / 4).max(1);
                let source_h4 = (source_size.height() / 4).max(1);
                let origin_col = candidate_col & !(source_w4 - 1);
                let origin_row = candidate_row & !(source_h4 - 1);
                if origin_col + source_w4 == mi_col {
                    let origin_index = origin_row * self.mi_cols + origin_col;
                    if reference_grid[origin_index] == Some(reference_frame) {
                        push_mv_candidate(
                            &mut values,
                            &mut value_len,
                            motion_grid[origin_index],
                        );
                    }
                }
                candidate_row = candidate_row
                    .saturating_add(1)
                    .max(origin_row.saturating_add(source_h4));
            }
        }
        if mi_row > self.tile_mi_row_start {
            let candidate_row = mi_row - 1;
            let candidate_col = mi_col.saturating_add(block_mi_width);
            if candidate_col < self.mi_cols {
                let index = candidate_row * self.mi_cols + candidate_col;
                if let Some(source_size) = self.motion_block_size_grid[index] {
                    let source_w4 = (source_size.width() / 4).max(1);
                    let source_h4 = (source_size.height() / 4).max(1);
                    let origin_col = candidate_col & !(source_w4 - 1);
                    let origin_row = candidate_row & !(source_h4 - 1);
                    let origin_index = origin_row * self.mi_cols + origin_col;
                    if origin_row + source_h4 == mi_row
                        && reference_grid[origin_index] == Some(reference_frame)
                    {
                        push_mv_candidate(
                            &mut values,
                            &mut value_len,
                            motion_grid[origin_index],
                        );
                    }
                }
            }
        }
        values
    }


    pub(super) fn local_warp_sample_candidates(
        &self,
        x: usize,
        y: usize,
        block_size: BlockSize,
        reference_frame: u8,
        motion_vector: (i32, i32),
    ) -> [Option<LocalWarpSample>; 8] {
        let mi_col = x >> 2;
        let mi_row = y >> 2;
        let w4 = (block_size.width() / 4).max(1);
        let h4 = (block_size.height() / 4).max(1);
        let threshold = block_size.width().max(block_size.height()).clamp(16, 112);
        let mut offsets = [(0isize, 0isize); 8];
        let mut offset_len = 0usize;
        let mut do_top_left = true;
        let mut do_top_right = true;

        if mi_row > self.tile_mi_row_start {
            if let Some(source_size) = self.motion_block_size_at(mi_col, mi_row - 1) {
                let source_w4 = (source_size.width() / 4).max(1);
                if w4 <= source_w4 {
                    let col_offset = -((mi_col & (source_w4 - 1)) as isize);
                    if col_offset < 0 {
                        do_top_left = false;
                    }
                    if col_offset + source_w4 as isize > w4 as isize {
                        do_top_right = false;
                    }
                    push_local_warp_offset(&mut offsets, &mut offset_len, -1, 0);
                } else {
                    let mut i = 0usize;
                    while i < w4.min(self.mi_cols.saturating_sub(mi_col))
                        && offset_len < offsets.len()
                    {
                        let source_w4 = self
                            .motion_block_size_at(mi_col + i, mi_row - 1)
                            .map(|size| (size.width() / 4).max(1))
                            .unwrap_or(1);
                        let mi_step = w4.min(source_w4).max(1);
                        push_local_warp_offset(&mut offsets, &mut offset_len, -1, i as isize);
                        i = i.saturating_add(mi_step);
                    }
                }
            }
        }
        if mi_col > self.tile_mi_col_start {
            if let Some(source_size) = self.motion_block_size_at(mi_col - 1, mi_row) {
                let source_h4 = (source_size.height() / 4).max(1);
                if h4 <= source_h4 {
                    let row_offset = -((mi_row & (source_h4 - 1)) as isize);
                    if row_offset < 0 {
                        do_top_left = false;
                    }
                    push_local_warp_offset(&mut offsets, &mut offset_len, 0, -1);
                } else {
                    let mut i = 0usize;
                    while i < h4.min(self.mi_rows.saturating_sub(mi_row))
                        && offset_len < offsets.len()
                    {
                        let source_h4 = self
                            .motion_block_size_at(mi_col - 1, mi_row + i)
                            .map(|size| (size.height() / 4).max(1))
                            .unwrap_or(1);
                        let mi_step = h4.min(source_h4).max(1);
                        push_local_warp_offset(&mut offsets, &mut offset_len, i as isize, -1);
                        i = i.saturating_add(mi_step);
                    }
                }
            }
        }
        if do_top_left {
            push_local_warp_offset(&mut offsets, &mut offset_len, -1, -1);
        }
        if do_top_right && w4.max(h4) <= 16 {
            push_local_warp_offset(&mut offsets, &mut offset_len, -1, w4 as isize);
        }

        let mut samples = [None; 8];
        let mut sample_len = 0usize;
        let mut scanned = 0usize;
        for &(delta_row, delta_col) in &offsets[..offset_len] {
            if scanned >= 8 {
                break;
            }
            let Some(mv_row) = mi_row.checked_add_signed(delta_row) else {
                continue;
            };
            let Some(mv_col) = mi_col.checked_add_signed(delta_col) else {
                continue;
            };
            let Some((sample, valid)) = self.local_warp_candidate_at(
                mv_row,
                mv_col,
                reference_frame,
                motion_vector,
                threshold,
            ) else {
                continue;
            };
            if samples[..sample_len].contains(&Some(sample)) {
                continue;
            }
            scanned += 1;
            if !valid && scanned > 1 {
                break;
            }
            samples[sample_len] = Some(sample);
            sample_len += 1;
            if !valid {
                break;
            }
        }
        samples
    }

    fn motion_block_size_at(&self, mi_col: usize, mi_row: usize) -> Option<BlockSize> {
        if mi_col >= self.mi_cols || mi_row >= self.mi_rows {
            return None;
        }
        self.motion_block_size_grid[mi_row * self.mi_cols + mi_col]
    }

    fn local_warp_candidate_at(
        &self,
        mv_row: usize,
        mv_col: usize,
        reference_frame: u8,
        motion_vector: (i32, i32),
        threshold: usize,
    ) -> Option<(LocalWarpSample, bool)> {
        if mv_row < self.tile_mi_row_start
            || mv_col < self.tile_mi_col_start
            || mv_row >= self.mi_rows
            || mv_col >= self.mi_cols
        {
            return None;
        }
        let index = mv_row * self.mi_cols + mv_col;
        if self.reference_frame_type_grid[index] != Some(reference_frame)
            || self.reference_frame_secondary_type_grid[index].is_some()
        {
            return None;
        }
        let candidate_size = self.motion_block_size_grid[index]?;
        let candidate_w4 = (candidate_size.width() / 4).max(1);
        let candidate_h4 = (candidate_size.height() / 4).max(1);
        let candidate_row = mv_row & !(candidate_h4 - 1);
        let candidate_col = mv_col & !(candidate_w4 - 1);
        let candidate_index = candidate_row
            .checked_mul(self.mi_cols)?
            .checked_add(candidate_col)?;
        let candidate_mv = self.motion_vector_grid[candidate_index]?;
        let mid_y = candidate_row
            .checked_mul(4)?
            .checked_add(candidate_h4 * 2)?
            - 1;
        let mid_x = candidate_col
            .checked_mul(4)?
            .checked_add(candidate_w4 * 2)?
            - 1;
        let source = (
            i32::try_from(mid_y.checked_mul(8)?).ok()?,
            i32::try_from(mid_x.checked_mul(8)?).ok()?,
        );
        let destination = (
            source.0.checked_add(candidate_mv.0)?,
            source.1.checked_add(candidate_mv.1)?,
        );
        let valid = (candidate_mv.0.abs_diff(motion_vector.0) as usize)
            .saturating_add(candidate_mv.1.abs_diff(motion_vector.1) as usize)
            <= threshold;
        Some((
            LocalWarpSample {
                source,
                destination,
            },
            valid,
        ))
    }

    pub(super) fn inter_mv_predictor(
        &self,
        x: usize,
        y: usize,
        block_size: BlockSize,
        reference_frame: u8,
    ) -> (i32, i32) {
        self.inter_mv_neighbor_candidates(x, y, block_size, reference_frame, false)
            .into_iter()
            .flatten()
            .next()
            .unwrap_or((0, 0))
    }

    pub(super) fn inter_mv_predictor_secondary(
        &self,
        x: usize,
        y: usize,
        block_size: BlockSize,
        reference_frame: u8,
    ) -> (i32, i32) {
        self.inter_mv_neighbor_candidates(x, y, block_size, reference_frame, true)
            .into_iter()
            .flatten()
            .next()
            .unwrap_or((0, 0))
    }

    pub(super) fn inter_mode_context(
        &self,
        x: usize,
        y: usize,
        block_size: BlockSize,
        reference_frame: u8,
    ) -> usize {
        let mi_col = x / 4;
        let mi_row = y / 4;
        let block_w = (block_size.width() / 4).max(1);
        let block_h = (block_size.height() / 4).max(1);
        let mut row_matches = 0usize;
        let mut col_matches = 0usize;
        let mut new_mv_count = 0usize;
        let mut seen = [(0usize, 0usize, false); 8];
        let mut seen_len = 0usize;

        let visit = |candidate_col: usize,
                         candidate_row: usize,
                         row: bool,
                         row_matches: &mut usize,
                         col_matches: &mut usize,
                         new_mv_count: &mut usize,
                         seen: &mut [(usize, usize, bool); 8],
                         seen_len: &mut usize| {
            if candidate_col < self.tile_mi_col_start
                || candidate_row < self.tile_mi_row_start
                || candidate_col >= self.mi_cols
                || candidate_row >= self.mi_rows
            {
                return;
            }
            let index = candidate_row * self.mi_cols + candidate_col;
            let Some(source_size) = self.motion_block_size_grid[index] else {
                return;
            };
            let source_w = (source_size.width() / 4).max(1);
            let source_h = (source_size.height() / 4).max(1);
            let origin_col = candidate_col & !(source_w - 1);
            let origin_row = candidate_row & !(source_h - 1);
            let origin_index = origin_row * self.mi_cols + origin_col;
            if origin_col + source_w > self.mi_cols
                || origin_row + source_h > self.mi_rows
                || self.reference_frame_type_grid[origin_index] != Some(reference_frame)
                || seen[..*seen_len].contains(&(origin_col, origin_row, row))
            {
                return;
            }
            if *seen_len < seen.len() {
                seen[*seen_len] = (origin_col, origin_row, row);
                *seen_len += 1;
            }
            if row {
                *row_matches += 1;
            } else {
                *col_matches += 1;
            }
            if self.inter_new_mv_grid[origin_index].unwrap_or(false) {
                *new_mv_count += 1;
            }
        };

        if mi_row > self.tile_mi_row_start {
            let above_row = mi_row - 1;
            for offset in 0..block_w {
                visit(
                    mi_col + offset,
                    above_row,
                    true,
                    &mut row_matches,
                    &mut col_matches,
                    &mut new_mv_count,
                    &mut seen,
                    &mut seen_len,
                );
            }
        }
        if mi_col > self.tile_mi_col_start {
            let left_col = mi_col - 1;
            for offset in 0..block_h {
                visit(
                    left_col,
                    mi_row + offset,
                    false,
                    &mut row_matches,
                    &mut col_matches,
                    &mut new_mv_count,
                    &mut seen,
                    &mut seen_len,
                );
            }
        }
        if mi_row > self.tile_mi_row_start {
            visit(
                mi_col + block_w,
                mi_row - 1,
                true,
                &mut row_matches,
                &mut col_matches,
                &mut new_mv_count,
                &mut seen,
                &mut seen_len,
            );
        }
        let nearest_match = usize::from(row_matches > 0) + usize::from(col_matches > 0);
        let ref_match_count = nearest_match;
        let new_context = match nearest_match {
            0 => usize::from(ref_match_count >= 1),
            1 => if new_mv_count > 0 { 2 } else { 3 },
            _ => if new_mv_count > 0 { 4 } else { 5 },
        };
        let ref_context = match nearest_match {
            0 => match ref_match_count {
                0 => 0,
                1 => 1,
                _ => 2,
            },
            1 => if ref_match_count >= 2 { 4 } else { 3 },
            _ => 5,
        };
        let global_context = self
            .temporal_motion_field
            .as_deref()
            .and_then(|field| {
                let sample_row = (mi_row) | 1;
                let sample_col = (mi_col) | 1;
                if sample_row >= field.mi_rows || sample_col >= field.mi_cols {
                    return None;
                }
                let index = sample_row / 2 * field.mi_cols + sample_col / 2;
                (field.reference_frames[index] == Some(reference_frame))
                    .then(|| field.motion_vectors[index])
                    .flatten()
            })
            .map(|motion_vector| {
                usize::from(
                    motion_vector.0.abs() >= 16
                        || motion_vector.1.abs() >= 16,
                )
            })
            .unwrap_or(0);
        new_context | (global_context << 3) | (ref_context << 4)
    }

    pub(super) fn inter_mv_candidate(
        &self,
        x: usize,
        y: usize,
        block_size: BlockSize,
        reference_frame: u8,
        candidate_index: usize,
        secondary: bool,
    ) -> (i32, i32) {
        if candidate_index == 0 {
            return if secondary {
                self.inter_mv_predictor_secondary(x, y, block_size, reference_frame)
            } else {
                self.inter_mv_predictor(x, y, block_size, reference_frame)
            };
        }
        let mut seen = [(0, 0); 4];
        let mut seen_len = 0;
        for mv in self
            .inter_mv_neighbor_candidates(x, y, block_size, reference_frame, secondary)
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
        secondary: bool,
    ) -> usize {
        let mut seen = [(0, 0); 4];
        let mut seen_len = 0;
        for mv in self
            .inter_mv_neighbor_candidates(x, y, block_size, reference_frame, secondary)
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
        reference_frame_type: u8,
        motion_vector: (i32, i32),
        has_new_mv: bool,
        reference_frame_secondary: Option<u8>,
        reference_frame_secondary_type: Option<u8>,
        motion_vector_secondary: Option<(i32, i32)>,
        interpolation_filters: (InterpolationFilter, InterpolationFilter),
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
            &mut self.inter_new_mv_grid,
            self.mi_cols,
            self.mi_rows,
            x,
            y,
            block_size,
            has_new_mv,
        );
        super::context_grid::fill_mi_grid(
            &mut self.reference_frame_type_grid,
            self.mi_cols,
            self.mi_rows,
            x,
            y,
            block_size,
            reference_frame_type,
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
        super::context_grid::fill_mi_grid(
            &mut self.interpolation_filter_grid,
            self.mi_cols,
            self.mi_rows,
            x,
            y,
            block_size,
            interpolation_filters,
        );
        super::context_grid::fill_mi_grid(
            &mut self.motion_block_size_grid,
            self.mi_cols,
            self.mi_rows,
            x,
            y,
            block_size,
            block_size,
        );
        super::context_grid::fill_mi_grid_clone(
            &mut self.reference_frame_secondary_grid,
            self.mi_cols,
            self.mi_rows,
            x,
            y,
            block_size,
            reference_frame_secondary,
        );
        super::context_grid::fill_mi_grid_clone(
            &mut self.reference_frame_secondary_type_grid,
            self.mi_cols,
            self.mi_rows,
            x,
            y,
            block_size,
            reference_frame_secondary_type,
        );
        super::context_grid::fill_mi_grid_clone(
            &mut self.motion_vector_secondary_grid,
            self.mi_cols,
            self.mi_rows,
            x,
            y,
            block_size,
            motion_vector_secondary,
        );
    }

    pub(super) fn clear_inter_mv(&mut self, x: usize, y: usize, block_size: BlockSize) {
        super::context_grid::fill_mi_grid_clone(
            &mut self.reference_frame_grid,
            self.mi_cols,
            self.mi_rows,
            x,
            y,
            block_size,
            None,
        );
        super::context_grid::fill_mi_grid_clone(
            &mut self.reference_frame_type_grid,
            self.mi_cols,
            self.mi_rows,
            x,
            y,
            block_size,
            None,
        );
        super::context_grid::fill_mi_grid_clone(
            &mut self.inter_new_mv_grid,
            self.mi_cols,
            self.mi_rows,
            x,
            y,
            block_size,
            None,
        );
        super::context_grid::fill_mi_grid_clone(
            &mut self.motion_vector_grid,
            self.mi_cols,
            self.mi_rows,
            x,
            y,
            block_size,
            None,
        );
        super::context_grid::fill_mi_grid_clone(
            &mut self.interpolation_filter_grid,
            self.mi_cols,
            self.mi_rows,
            x,
            y,
            block_size,
            None,
        );
        super::context_grid::fill_mi_grid_clone(
            &mut self.motion_block_size_grid,
            self.mi_cols,
            self.mi_rows,
            x,
            y,
            block_size,
            None,
        );
        super::context_grid::fill_mi_grid_clone(
            &mut self.reference_frame_secondary_grid,
            self.mi_cols,
            self.mi_rows,
            x,
            y,
            block_size,
            None,
        );
        super::context_grid::fill_mi_grid_clone(
            &mut self.reference_frame_secondary_type_grid,
            self.mi_cols,
            self.mi_rows,
            x,
            y,
            block_size,
            None,
        );
        super::context_grid::fill_mi_grid_clone(
            &mut self.motion_vector_secondary_grid,
            self.mi_cols,
            self.mi_rows,
            x,
            y,
            block_size,
            None,
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
