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

const MAX_INTER_MV_CANDIDATES: usize = 8;

fn push_compound_mv_candidate(
    primary: &mut [Option<(i32, i32)>; MAX_INTER_MV_CANDIDATES],
    secondary: &mut [Option<(i32, i32)>; MAX_INTER_MV_CANDIDATES],
    weights: &mut [u16; MAX_INTER_MV_CANDIDATES],
    len: &mut usize,
    first: (i32, i32),
    second: (i32, i32),
    weight: u16,
) {
    if let Some(index) =
        (0..*len).find(|index| primary[*index] == Some(first) && secondary[*index] == Some(second))
    {
        weights[index] = weights[index].saturating_add(weight);
    } else if *len < MAX_INTER_MV_CANDIDATES {
        primary[*len] = Some(first);
        secondary[*len] = Some(second);
        weights[*len] = weight;
        *len += 1;
    }
}

#[allow(clippy::too_many_arguments)]
fn add_compound_temporal_candidate(
    field: &super::MotionField,
    mi_col: usize,
    mi_row: usize,
    blk_col: isize,
    blk_row: isize,
    primary_reference: usize,
    secondary_reference: usize,
    order_hint: u32,
    reference_order_hints: &[Option<u32>; 7],
    primary: &mut [Option<(i32, i32)>; MAX_INTER_MV_CANDIDATES],
    secondary: &mut [Option<(i32, i32)>; MAX_INTER_MV_CANDIDATES],
    weights: &mut [u16; MAX_INTER_MV_CANDIDATES],
    len: &mut usize,
) {
    let Some((primary_index, primary_motion)) = field.projected_motion(
        mi_col,
        mi_row,
        blk_col,
        blk_row,
        primary_reference,
        order_hint,
        reference_order_hints,
    ) else {
        return;
    };
    let Some((secondary_index, secondary_motion)) = field.projected_motion(
        mi_col,
        mi_row,
        blk_col,
        blk_row,
        secondary_reference,
        order_hint,
        reference_order_hints,
    ) else {
        return;
    };
    if primary_index == secondary_index {
        push_compound_mv_candidate(
            primary,
            secondary,
            weights,
            len,
            primary_motion,
            secondary_motion,
            2,
        );
    }
}

#[derive(Clone, Copy)]
struct InterMvStack {
    values: [Option<(i32, i32)>; MAX_INTER_MV_CANDIDATES],
    weights: [u16; MAX_INTER_MV_CANDIDATES],
    len: usize,
    nearest_len: usize,
    row_matches: usize,
    col_matches: usize,
    nearest_row_matches: usize,
    nearest_col_matches: usize,
    new_mv_count: usize,
}

impl Default for InterMvStack {
    fn default() -> Self {
        Self {
            values: [None; MAX_INTER_MV_CANDIDATES],
            weights: [0; MAX_INTER_MV_CANDIDATES],
            len: 0,
            nearest_len: 0,
            row_matches: 0,
            col_matches: 0,
            nearest_row_matches: 0,
            nearest_col_matches: 0,
            new_mv_count: 0,
        }
    }
}

impl InterMvStack {
    fn add(
        &mut self,
        candidate: Option<(i32, i32)>,
        weight: u16,
        new_mv: bool,
        row: bool,
        nearest: bool,
    ) {
        let Some(candidate) = candidate else {
            return;
        };
        if row {
            self.row_matches += 1;
            if nearest {
                self.nearest_row_matches += 1;
            }
        } else {
            self.col_matches += 1;
            if nearest {
                self.nearest_col_matches += 1;
            }
        }
        if new_mv {
            self.new_mv_count += 1;
        }
        if let Some(index) = self.values[..self.len]
            .iter()
            .position(|value| *value == Some(candidate))
        {
            self.weights[index] = self.weights[index].saturating_add(weight);
            return;
        }
        if self.len < MAX_INTER_MV_CANDIDATES {
            self.values[self.len] = Some(candidate);
            self.weights[self.len] = weight;
            self.len += 1;
        }
    }

    fn add_temporal(&mut self, candidate: (i32, i32), weight: u16) {
        if let Some(index) = self.values[..self.len]
            .iter()
            .position(|value| *value == Some(candidate))
        {
            self.weights[index] = self.weights[index].saturating_add(weight);
            return;
        }
        if self.len < MAX_INTER_MV_CANDIDATES {
            self.values[self.len] = Some(candidate);
            self.weights[self.len] = weight;
            self.len += 1;
        }
    }

    fn rank(&mut self) {
        let nearest_len = self.nearest_len.min(self.len);
        for end in (1..nearest_len).rev() {
            for index in 0..end {
                if self.weights[index] < self.weights[index + 1] {
                    self.values.swap(index, index + 1);
                    self.weights.swap(index, index + 1);
                }
            }
        }
        for end in (nearest_len + 1..self.len).rev() {
            for index in nearest_len..end {
                if self.weights[index] < self.weights[index + 1] {
                    self.values.swap(index, index + 1);
                    self.weights.swap(index, index + 1);
                }
            }
        }
    }
}

impl<'a> TileDecoder<'a> {
    fn neighbor_reference_block(&self, mi_col: usize, mi_row: usize) -> Option<(u8, Option<u8>)> {
        if mi_col < self.tile_mi_col_start
            || mi_row < self.tile_mi_row_start
            || mi_col >= self.mi_cols
            || mi_row >= self.mi_rows
        {
            return None;
        }
        let index = mi_row * self.mi_cols + mi_col;
        self.motion_block_size_grid[index]?;
        self.reference_frame_type_grid[index]
            .map(|primary| (primary, self.reference_frame_secondary_type_grid[index]))
    }

    fn neighbor_block(&self, mi_col: usize, mi_row: usize) -> Option<(Option<u8>, Option<u8>)> {
        if mi_col < self.tile_mi_col_start
            || mi_row < self.tile_mi_row_start
            || mi_col >= self.mi_cols
            || mi_row >= self.mi_rows
        {
            return None;
        }
        let index = mi_row * self.mi_cols + mi_col;
        self.motion_block_size_grid[index]?;
        Some((
            self.reference_frame_type_grid[index],
            self.reference_frame_secondary_type_grid[index],
        ))
    }

    fn neighbor_reference_counts(&self, x: usize, y: usize) -> [usize; 7] {
        let mi_col = x / 4;
        let mi_row = y / 4;
        let mut counts = [0usize; 7];
        let neighbors = [
            (Some(mi_col), mi_row.checked_sub(1)),
            (mi_col.checked_sub(1), Some(mi_row)),
        ];
        for (column, row) in neighbors {
            let Some((column, row)) = column.zip(row) else {
                continue;
            };
            let Some((primary, secondary)) = self.neighbor_reference_block(column, row) else {
                continue;
            };
            if let Some(count) = counts.get_mut(usize::from(primary)) {
                *count += 1;
            }
            if let Some(secondary) = secondary {
                if let Some(count) = counts.get_mut(usize::from(secondary)) {
                    *count += 1;
                }
            }
        }
        counts
    }

    pub(super) fn reference_mode_context(&self, x: usize, y: usize) -> usize {
        let mi_col = x / 4;
        let mi_row = y / 4;
        let above = mi_row
            .checked_sub(1)
            .and_then(|row| self.neighbor_block(mi_col, row));
        let left = mi_col
            .checked_sub(1)
            .and_then(|column| self.neighbor_block(column, mi_row));
        // Neighbour grids store AV1 reference types, not remapped reference
        // slots. Forward/backward classification therefore follows the type
        // (BWDREF and later), even when ref_frame_idx[] shuffles the slots.
        let backward =
            |reference: Option<u8>| reference.is_some_and(|reference| reference >= 4) as usize;
        match (above, left) {
            (Some((above, above_secondary)), Some((left, left_secondary))) => {
                match (above_secondary.is_some(), left_secondary.is_some()) {
                    (false, false) => backward(above) ^ backward(left),
                    (false, true) => 2 + usize::from(above.is_none() || backward(above) != 0),
                    (true, false) => 2 + usize::from(left.is_none() || backward(left) != 0),
                    (true, true) => 4,
                }
            }
            (Some((reference, secondary)), None) | (None, Some((reference, secondary))) => {
                if secondary.is_some() {
                    3
                } else {
                    backward(reference)
                }
            }
            (None, None) => 1,
        }
    }

    fn is_backward_reference(reference: u8) -> bool {
        reference >= 4
    }

    fn is_unidirectional_compound(primary: u8, secondary: u8) -> bool {
        Self::is_backward_reference(primary) == Self::is_backward_reference(secondary)
    }

    pub(super) fn compound_reference_type_context(&self, x: usize, y: usize) -> usize {
        let mi_col = x / 4;
        let mi_row = y / 4;
        let above = mi_row
            .checked_sub(1)
            .and_then(|row| self.neighbor_block(mi_col, row));
        let left = mi_col
            .checked_sub(1)
            .and_then(|column| self.neighbor_block(column, mi_row));
        let backward = Self::is_backward_reference;
        let compound_is_uni = |reference: (Option<u8>, Option<u8>)| {
            reference
                .0
                .zip(reference.1)
                .is_some_and(|(primary, secondary)| {
                    Self::is_unidirectional_compound(primary, secondary)
                })
        };

        match (above, left) {
            (Some(above), Some(left)) => {
                let above_intra = above.0.is_none();
                let left_intra = left.0.is_none();
                if above_intra && left_intra {
                    2
                } else if above_intra || left_intra {
                    let inter = if above_intra { left } else { above };
                    if inter.0.is_some() && inter.1.is_none() {
                        2
                    } else {
                        1 + 2 * usize::from(compound_is_uni(inter))
                    }
                } else {
                    let above_single = above.1.is_none();
                    let left_single = left.1.is_none();
                    if above_single && left_single {
                        1 + 2 * usize::from(backward(above.0.unwrap()) == backward(left.0.unwrap()))
                    } else if above_single || left_single {
                        let compound = if above_single { left } else { above };
                        if !compound_is_uni(compound) {
                            1
                        } else {
                            3 + usize::from(backward(above.0.unwrap()) == backward(left.0.unwrap()))
                        }
                    } else {
                        let above_uni = compound_is_uni(above);
                        let left_uni = compound_is_uni(left);
                        if !above_uni && !left_uni {
                            0
                        } else if above_uni != left_uni {
                            2
                        } else {
                            3 + usize::from(backward(above.0.unwrap()) == backward(left.0.unwrap()))
                        }
                    }
                }
            }
            (Some(reference), None) | (None, Some(reference)) => {
                if reference.0.is_none() || reference.1.is_none() {
                    2
                } else {
                    4 * usize::from(compound_is_uni(reference))
                }
            }
            (None, None) => 2,
        }
    }

    pub(super) fn compound_reference_contexts(&self, x: usize, y: usize) -> [usize; 8] {
        let counts = self.neighbor_reference_counts(x, y);
        let compare = |left: usize, right: usize| {
            if left == right {
                1
            } else if left < right {
                0
            } else {
                2
            }
        };
        [
            compare(counts[0] + counts[1], counts[2] + counts[3]),
            compare(counts[0], counts[1]),
            compare(counts[2], counts[3]),
            compare(counts[4] + counts[5], counts[6]),
            compare(counts[4], counts[5]),
            compare(
                counts[0] + counts[1] + counts[2] + counts[3],
                counts[4] + counts[5] + counts[6],
            ),
            compare(counts[1], counts[2] + counts[3]),
            compare(counts[2], counts[3]),
        ]
    }

    pub(super) fn switchable_interpolation_context(
        &self,
        x: usize,
        y: usize,
        reference_frame: u8,
        compound: bool,
        direction: usize,
    ) -> usize {
        let neighbor_filter = |mi_col: usize, mi_row: usize| {
            if mi_col >= self.mi_cols || mi_row >= self.mi_rows {
                return 3usize;
            }
            let index = mi_row * self.mi_cols + mi_col;
            let Some(source_size) = self.motion_block_size_grid[index] else {
                return 3;
            };
            let source_w4 = (source_size.width() / 4).max(1);
            let source_h4 = (source_size.height() / 4).max(1);
            let origin_col = mi_col & !(source_w4 - 1);
            let origin_row = mi_row & !(source_h4 - 1);
            let origin_index = origin_row * self.mi_cols + origin_col;
            let primary_matches =
                self.reference_frame_type_grid[origin_index] == Some(reference_frame);
            let secondary_matches =
                self.reference_frame_secondary_type_grid[origin_index] == Some(reference_frame);
            if !(primary_matches || secondary_matches) {
                return 3;
            }
            let filters = self.interpolation_filter_grid[origin_index].unwrap_or((
                InterpolationFilter::Switchable,
                InterpolationFilter::Switchable,
            ));
            let filter = if direction == 0 { filters.1 } else { filters.0 };
            match filter {
                InterpolationFilter::Regular => 0,
                InterpolationFilter::Smooth => 1,
                InterpolationFilter::Sharp => 2,
                InterpolationFilter::Bilinear | InterpolationFilter::Switchable => 3,
            }
        };

        let mi_col = x / 4;
        let mi_row = y / 4;
        let left = (mi_col > self.tile_mi_col_start)
            .then(|| neighbor_filter(mi_col - 1, mi_row))
            .unwrap_or(3);
        let above = (mi_row > self.tile_mi_row_start)
            .then(|| neighbor_filter(mi_col, mi_row - 1))
            .unwrap_or(3);
        let offset = usize::from(compound) * 4 + direction * 8;
        offset
            + if left == above {
                left
            } else if left == 3 {
                above
            } else if above == 3 {
                left
            } else {
                3
            }
    }

    pub(super) fn single_reference_contexts(&self, x: usize, y: usize) -> [usize; 6] {
        let counts = self.neighbor_reference_counts(x, y);
        let compare = |left: usize, right: usize| {
            if left == right {
                1
            } else if left < right {
                0
            } else {
                2
            }
        };
        [
            compare(
                counts[0] + counts[1] + counts[2] + counts[3],
                counts[4] + counts[5] + counts[6],
            ),
            compare(counts[4] + counts[5], counts[6]),
            compare(counts[0] + counts[1], counts[2] + counts[3]),
            compare(counts[0], counts[1]),
            compare(counts[2], counts[3]),
            compare(counts[4], counts[5]),
        ]
    }

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
        let stack = self.inter_mv_stack(x, y, block_size, reference_frame, secondary);
        let mut values = [None; 4];
        values.copy_from_slice(&stack.values[..4]);
        values
    }

    fn inter_mv_stack(
        &self,
        x: usize,
        y: usize,
        block_size: BlockSize,
        reference_frame: u8,
        secondary: bool,
    ) -> InterMvStack {
        let mi_col = x / 4;
        let mi_row = y / 4;
        let block_w4 = (block_size.width() / 4).max(1);
        let block_h4 = (block_size.height() / 4).max(1);
        let mut stack = InterMvStack::default();
        let target_reference = self
            .reference_frame_indices
            .get(usize::from(reference_frame))
            .copied()
            .unwrap_or(reference_frame);
        let sign_bias = |reference: u8| {
            let hint = self
                .reference_frame_indices
                .iter()
                .position(|candidate| *candidate == reference)
                .and_then(|index| self.reference_order_hints[index]);
            let Some(hint) = hint else {
                return false;
            };
            let bits = self.order_hint_bits.max(1).min(31);
            let half = 1i32 << (bits - 1);
            let mask = half - 1;
            let mut distance = hint as i32 - self.order_hint as i32;
            distance = (distance & mask) - (distance & half);
            distance > 0
        };
        let normalize_motion = |motion: (i32, i32), candidate_reference: u8| {
            if sign_bias(candidate_reference) == sign_bias(target_reference) {
                motion
            } else {
                (-motion.0, -motion.1)
            }
        };

        let candidate = |candidate_col: usize,
                         candidate_row: usize,
                         reference_frame: u8,
                         secondary: bool|
         -> Option<(Option<(i32, i32)>, bool)> {
            if candidate_col < self.tile_mi_col_start
                || candidate_row < self.tile_mi_row_start
                || candidate_col >= self.mi_cols
                || candidate_row >= self.mi_rows
            {
                return None;
            }
            let index = candidate_row * self.mi_cols + candidate_col;
            let _source_size = self.motion_block_size_grid[index]?;
            let origin_index = index;
            let candidate_reference = self
                .reference_frame_indices
                .get(usize::from(reference_frame))
                .copied()
                .unwrap_or(reference_frame);
            let primary =
                self.reference_frame_grid[origin_index].zip(self.motion_vector_grid[origin_index]);
            let secondary_motion = self.reference_frame_secondary_grid[origin_index]
                .zip(self.motion_vector_secondary_grid[origin_index]);
            let motion = if secondary {
                secondary_motion.filter(|(reference, _)| *reference == candidate_reference)
            } else {
                primary
                    .filter(|(reference, _)| *reference == candidate_reference)
                    .or_else(|| {
                        secondary_motion.filter(|(reference, _)| *reference == candidate_reference)
                    })
            };
            Some((
                motion.map(|(reference, motion)| normalize_motion(motion, reference)),
                self.inter_new_mv_grid[origin_index].unwrap_or(false),
            ))
        };

        let row_adj = usize::from(block_h4 < 2 && (mi_row & 1) == 1) as isize;
        let col_adj = usize::from(block_w4 < 2 && (mi_col & 1) == 1) as isize;
        let max_row_offset = if mi_row > self.tile_mi_row_start {
            let limit = if block_h4 < 2 { -4 } else { -6 } + row_adj;
            limit
                .max(self.tile_mi_row_start as isize - mi_row as isize)
                .min(self.mi_rows as isize - mi_row as isize - 1)
        } else {
            0
        };
        let max_col_offset = if mi_col > self.tile_mi_col_start {
            let limit = if block_w4 < 2 { -4 } else { -6 } + col_adj;
            limit
                .max(self.tile_mi_col_start as isize - mi_col as isize)
                .min(self.mi_cols as isize - mi_col as isize - 1)
        } else {
            0
        };
        let mut processed_rows = 0isize;
        let mut processed_cols = 0isize;
        let scan_row = |row_offset: isize,
                        nearest: bool,
                        stack: &mut InterMvStack,
                        processed_rows: &mut isize| {
            let Some(row) = mi_row.checked_add_signed(row_offset) else {
                return;
            };
            if row < self.tile_mi_row_start || row >= self.mi_rows {
                return;
            }
            let end_mi = block_w4.min(self.mi_cols.saturating_sub(mi_col)).min(16);
            let mut col_offset = 0usize;
            if row_offset.abs() > 1 {
                col_offset = 1;
                if block_w4 < 2 && (mi_col & 1) == 1 {
                    col_offset = 0;
                }
            }
            let mut i = 0usize;
            while i < end_mi {
                let Some(col) = mi_col
                    .checked_add(col_offset)
                    .and_then(|value| value.checked_add(i))
                else {
                    break;
                };
                let Some(source_size) = self.motion_block_size_at(col, row) else {
                    i += 1;
                    continue;
                };
                let source_w4 = (source_size.width() / 4).max(1);
                let source_h4 = (source_size.height() / 4).max(1);
                let mut len = block_w4.min(source_w4);
                if block_w4 >= 16 {
                    len = len.max(4);
                } else if row_offset.abs() > 1 {
                    len = len.max(2);
                }
                let mut weight = 2isize;
                if block_w4 >= 2 && block_w4 <= source_w4 {
                    let inc = (-max_row_offset + row_offset + 1).min(source_h4 as isize);
                    weight = weight.max(inc);
                    *processed_rows = inc - row_offset - 1;
                }
                if let Some((mv, new_mv)) = candidate(col, row, reference_frame, secondary) {
                    stack.add(
                        mv,
                        (len as isize * weight).max(0).min(u16::MAX as isize) as u16,
                        new_mv && nearest,
                        true,
                        nearest,
                    );
                }
                i = i.saturating_add(len);
            }
            if nearest {
                stack.nearest_len = stack.len;
            }
        };
        let scan_col = |col_offset: isize,
                        nearest: bool,
                        stack: &mut InterMvStack,
                        processed_cols: &mut isize| {
            let Some(col) = mi_col.checked_add_signed(col_offset) else {
                return;
            };
            if col < self.tile_mi_col_start || col >= self.mi_cols {
                return;
            }
            let end_mi = block_h4.min(self.mi_rows.saturating_sub(mi_row)).min(16);
            let mut row_offset = 0usize;
            if col_offset.abs() > 1 {
                row_offset = 1;
                if block_h4 < 2 && (mi_row & 1) == 1 {
                    row_offset = 0;
                }
            }
            let mut i = 0usize;
            while i < end_mi {
                let Some(row) = mi_row
                    .checked_add(row_offset)
                    .and_then(|value| value.checked_add(i))
                else {
                    break;
                };
                let Some(source_size) = self.motion_block_size_at(col, row) else {
                    i += 1;
                    continue;
                };
                let source_w4 = (source_size.width() / 4).max(1);
                let source_h4 = (source_size.height() / 4).max(1);
                let mut len = block_h4.min(source_h4);
                if block_h4 >= 16 {
                    len = len.max(4);
                } else if col_offset.abs() > 1 {
                    len = len.max(2);
                }
                let mut weight = 2isize;
                if block_h4 >= 2 && block_h4 <= source_h4 {
                    let inc = (-max_col_offset + col_offset + 1).min(source_w4 as isize);
                    weight = weight.max(inc);
                    *processed_cols = inc - col_offset - 1;
                }
                if let Some((mv, new_mv)) = candidate(col, row, reference_frame, secondary) {
                    stack.add(
                        mv,
                        (len as isize * weight).max(0).min(u16::MAX as isize) as u16,
                        new_mv && nearest,
                        false,
                        nearest,
                    );
                }
                i = i.saturating_add(len);
            }
            if nearest {
                stack.nearest_len = stack.len;
            }
        };

        if mi_row > self.tile_mi_row_start {
            scan_row(-1, true, &mut stack, &mut processed_rows);
        }
        if mi_col > self.tile_mi_col_start {
            scan_col(-1, true, &mut stack, &mut processed_cols);
        }
        if mi_row > self.tile_mi_row_start {
            let top_right = mi_col.saturating_add(block_w4);
            if top_right < self.mi_cols {
                if let Some((mv, new_mv)) =
                    candidate(top_right, mi_row - 1, reference_frame, secondary)
                {
                    stack.add(mv, 4, new_mv, true, true);
                }
            }
        }
        stack.nearest_len = stack.len;
        for weight in &mut stack.weights[..stack.nearest_len] {
            *weight = weight.saturating_add(640);
        }

        if let Some(field) = self.temporal_motion_field.as_deref() {
            let blk_row_end = block_h4.min(16);
            let blk_col_end = block_w4.min(16);
            let step_row = if block_h4 >= 16 { 4 } else { 2 };
            let step_col = if block_w4 >= 16 { 4 } else { 2 };
            let reference_type = usize::from(reference_frame);
            let mut add_temporal = |blk_row: isize, blk_col: isize| {
                if let Some((_, motion_vector)) = field.projected_motion(
                    mi_col,
                    mi_row,
                    blk_col,
                    blk_row,
                    reference_type,
                    self.order_hint,
                    &self.reference_order_hints,
                ) {
                    stack.add_temporal(motion_vector, 2);
                }
            };
            for blk_row in (0..blk_row_end).step_by(step_row) {
                for blk_col in (0..blk_col_end).step_by(step_col) {
                    add_temporal(blk_row as isize, blk_col as isize);
                }
            }
            if block_h4 >= 2 && block_h4 < 16 && block_w4 >= 2 && block_w4 < 16 {
                let vertical_offset = block_h4.max(2) as isize;
                let horizontal_offset = block_w4.max(2) as isize;
                for (blk_row, blk_col) in [
                    (vertical_offset, -2),
                    (vertical_offset, horizontal_offset),
                    (vertical_offset - 2, horizontal_offset),
                ] {
                    let sb_row = (mi_row & 15) as isize + blk_row;
                    let sb_col = (mi_col & 15) as isize + blk_col;
                    if (0..16).contains(&sb_row) && (0..16).contains(&sb_col) {
                        add_temporal(blk_row, blk_col);
                    }
                }
            }
        }

        if mi_row > self.tile_mi_row_start && mi_col > self.tile_mi_col_start {
            if let Some((mv, _new_mv)) =
                candidate(mi_col - 1, mi_row - 1, reference_frame, secondary)
            {
                stack.add(mv, 4, false, true, false);
            }
        }
        for idx in 2..=3 {
            let row_offset = -(idx as isize * 2) + 1 + row_adj;
            let col_offset = -(idx as isize * 2) + 1 + col_adj;
            if row_offset.abs() <= max_row_offset.abs() && row_offset.abs() > processed_rows {
                scan_row(row_offset, false, &mut stack, &mut processed_rows);
            }
            if col_offset.abs() <= max_col_offset.abs() && col_offset.abs() > processed_cols {
                scan_col(col_offset, false, &mut stack, &mut processed_cols);
            }
        }
        stack.rank();
        if !self.current_inter_compound && stack.len < 2 {
            let mi_width = block_w4.min(16).min(self.mi_cols.saturating_sub(mi_col));
            let mi_height = block_h4.min(16).min(self.mi_rows.saturating_sub(mi_row));
            let mi_size = mi_width.min(mi_height);
            let add_neighbor =
                |candidate_col: usize, candidate_row: usize, stack: &mut InterMvStack| {
                    let index = candidate_row * self.mi_cols + candidate_col;
                    for candidate in [
                        self.reference_frame_grid[index].zip(self.motion_vector_grid[index]),
                        self.reference_frame_secondary_grid[index]
                            .zip(self.motion_vector_secondary_grid[index]),
                    ]
                    .into_iter()
                    .flatten()
                    {
                        stack.add_temporal(normalize_motion(candidate.1, candidate.0), 2);
                        if stack.len >= 2 {
                            break;
                        }
                    }
                };

            if max_row_offset != 0 {
                let mut offset = 0;
                while offset < mi_size && stack.len < 2 {
                    let candidate_col = mi_col + offset;
                    let candidate_row = mi_row - 1;
                    add_neighbor(candidate_col, candidate_row, &mut stack);
                    let step = self.motion_block_size_grid
                        [candidate_row * self.mi_cols + candidate_col]
                        .map(|size| (size.width() / 4).max(1))
                        .unwrap_or(1);
                    offset += step;
                }
            }
            if max_col_offset != 0 {
                let mut offset = 0;
                while offset < mi_size && stack.len < 2 {
                    let candidate_col = mi_col - 1;
                    let candidate_row = mi_row + offset;
                    add_neighbor(candidate_col, candidate_row, &mut stack);
                    let step = self.motion_block_size_grid
                        [candidate_row * self.mi_cols + candidate_col]
                        .map(|size| (size.height() / 4).max(1))
                        .unwrap_or(1);
                    offset += step;
                }
            }
        }
        stack
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
        if self.interintra_grid[index] == Some(true)
            || self.reference_frame_type_grid[index] != Some(reference_frame)
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
        let stack = self.inter_mv_stack(x, y, block_size, reference_frame, false);
        let nearest_match =
            usize::from(stack.nearest_row_matches > 0) + usize::from(stack.nearest_col_matches > 0);
        let ref_match_count =
            usize::from(stack.row_matches > 0) + usize::from(stack.col_matches > 0);
        let (new_context, ref_context) = match nearest_match {
            0 => (
                usize::from(ref_match_count >= 1),
                match ref_match_count {
                    0 => 0,
                    1 => 1,
                    _ => 2,
                },
            ),
            1 => (
                2 + usize::from(stack.new_mv_count == 0),
                match ref_match_count {
                    1 => 3,
                    count if count >= 2 => 4,
                    _ => 0,
                },
            ),
            _ => (if stack.new_mv_count >= 1 { 4 } else { 5 }, 5),
        };
        new_context | (ref_context << 4)
    }

    pub(super) fn compound_inter_mode_context(
        &self,
        x: usize,
        y: usize,
        block_size: BlockSize,
        primary_reference: u8,
        secondary_reference: u8,
    ) -> usize {
        let mi_col = x / 4;
        let mi_row = y / 4;
        let block_w4 = (block_size.width() / 4).max(1);
        let block_h4 = (block_size.height() / 4).max(1);
        let target_primary = self
            .reference_frame_indices
            .get(usize::from(primary_reference))
            .copied()
            .unwrap_or(primary_reference);
        let target_secondary = self
            .reference_frame_indices
            .get(usize::from(secondary_reference))
            .copied()
            .unwrap_or(secondary_reference);
        let matches = |column: usize, row: usize| {
            if column < self.tile_mi_col_start
                || row < self.tile_mi_row_start
                || column >= self.mi_cols
                || row >= self.mi_rows
            {
                return None;
            }
            let index = row * self.mi_cols + column;
            let size = self.motion_block_size_grid[index]?;
            let width = (size.width() / 4).max(1);
            let height = (size.height() / 4).max(1);
            if self.reference_frame_grid[index] != Some(target_primary)
                || self.reference_frame_secondary_grid[index] != Some(target_secondary)
            {
                return None;
            }
            Some((
                width,
                height,
                self.inter_new_mv_grid[index].unwrap_or(false),
            ))
        };
        let row_adj = usize::from(block_h4 < 2 && (mi_row & 1) == 1) as isize;
        let col_adj = usize::from(block_w4 < 2 && (mi_col & 1) == 1) as isize;
        let max_row_offset = if mi_row > self.tile_mi_row_start {
            (if block_h4 < 2 { -4 } else { -6 } + row_adj)
                .max(self.tile_mi_row_start as isize - mi_row as isize)
                .min(self.mi_rows as isize - mi_row as isize - 1)
        } else {
            0
        };
        let max_col_offset = if mi_col > self.tile_mi_col_start {
            (if block_w4 < 2 { -4 } else { -6 } + col_adj)
                .max(self.tile_mi_col_start as isize - mi_col as isize)
                .min(self.mi_cols as isize - mi_col as isize - 1)
        } else {
            0
        };
        let row_matches = std::cell::Cell::new(0usize);
        let col_matches = std::cell::Cell::new(0usize);
        let nearest_row_matches = std::cell::Cell::new(0usize);
        let nearest_col_matches = std::cell::Cell::new(0usize);
        let new_mv_count = std::cell::Cell::new(0usize);
        let processed_rows = std::cell::Cell::new(0isize);
        let processed_cols = std::cell::Cell::new(0isize);

        let scan_row = |row_offset: isize, nearest: bool| {
            let Some(row) = mi_row.checked_add_signed(row_offset) else {
                return;
            };
            if row < self.tile_mi_row_start || row >= self.mi_rows {
                return;
            }
            let end_mi = block_w4.min(self.mi_cols.saturating_sub(mi_col)).min(16);
            let mut col_offset = if row_offset.abs() > 1 { 1 } else { 0 };
            if block_w4 < 2 && (mi_col & 1) == 1 && row_offset.abs() > 1 {
                col_offset = 0;
            }
            let use_step_16 = block_w4 >= 16;
            let mut i = 0usize;
            while i < end_mi {
                let Some(column) = mi_col
                    .checked_add(col_offset)
                    .and_then(|value| value.checked_add(i))
                else {
                    break;
                };
                let Some((source_w4, source_h4, has_new_mv)) = matches(column, row) else {
                    i += 1;
                    continue;
                };
                let mut len = block_w4.min(source_w4);
                if use_step_16 {
                    len = len.max(4);
                } else if row_offset.abs() > 1 {
                    len = len.max(2);
                }
                let mut weight = 2isize;
                if block_w4 >= 2 && block_w4 <= source_w4 {
                    let inc = (-max_row_offset + row_offset + 1).min(source_h4 as isize);
                    weight = weight.max(inc);
                    processed_rows.set(inc - row_offset - 1);
                }
                let _ = weight;
                row_matches.set(row_matches.get() + 1);
                if nearest {
                    nearest_row_matches.set(nearest_row_matches.get() + 1);
                    if has_new_mv {
                        new_mv_count.set(new_mv_count.get() + 1);
                    }
                }
                i = i.saturating_add(len.max(1));
            }
        };
        let scan_col = |col_offset: isize, nearest: bool| {
            let Some(column) = mi_col.checked_add_signed(col_offset) else {
                return;
            };
            if column < self.tile_mi_col_start || column >= self.mi_cols {
                return;
            }
            let end_mi = block_h4.min(self.mi_rows.saturating_sub(mi_row)).min(16);
            let mut row_offset = if col_offset.abs() > 1 { 1 } else { 0 };
            if block_h4 < 2 && (mi_row & 1) == 1 && col_offset.abs() > 1 {
                row_offset = 0;
            }
            let use_step_16 = block_h4 >= 16;
            let mut i = 0usize;
            while i < end_mi {
                let Some(row) = mi_row
                    .checked_add(row_offset)
                    .and_then(|value| value.checked_add(i))
                else {
                    break;
                };
                let Some((source_w4, source_h4, has_new_mv)) = matches(column, row) else {
                    i += 1;
                    continue;
                };
                let mut len = block_h4.min(source_h4);
                if use_step_16 {
                    len = len.max(4);
                } else if col_offset.abs() > 1 {
                    len = len.max(2);
                }
                let mut weight = 2isize;
                if block_h4 >= 2 && block_h4 <= source_h4 {
                    let inc = (-max_col_offset + col_offset + 1).min(source_w4 as isize);
                    weight = weight.max(inc);
                    processed_cols.set(inc - col_offset - 1);
                }
                let _ = weight;
                col_matches.set(col_matches.get() + 1);
                if nearest {
                    nearest_col_matches.set(nearest_col_matches.get() + 1);
                    if has_new_mv {
                        new_mv_count.set(new_mv_count.get() + 1);
                    }
                }
                i = i.saturating_add(len.max(1));
            }
        };

        if mi_row > self.tile_mi_row_start {
            scan_row(-1, true);
        }
        if mi_col > self.tile_mi_col_start {
            scan_col(-1, true);
        }
        if mi_row > self.tile_mi_row_start {
            let top_right = mi_col.saturating_add(block_w4);
            if matches(top_right, mi_row - 1).is_some() {
                row_matches.set(row_matches.get() + 1);
                nearest_row_matches.set(nearest_row_matches.get() + 1);
            }
        }
        if mi_row > self.tile_mi_row_start && mi_col > self.tile_mi_col_start {
            let _ = matches(mi_col - 1, mi_row - 1).is_some().then(|| {
                row_matches.set(row_matches.get() + 1);
                col_matches.set(col_matches.get() + 1);
            });
        }
        for idx in 2..=3 {
            let row_offset = -(idx as isize * 2) + 1 + row_adj;
            let col_offset = -(idx as isize * 2) + 1 + col_adj;
            if row_offset.abs() <= max_row_offset.abs() && row_offset.abs() > processed_rows.get() {
                scan_row(row_offset, false);
            }
            if col_offset.abs() <= max_col_offset.abs() && col_offset.abs() > processed_cols.get() {
                scan_col(col_offset, false);
            }
        }

        let nearest_match =
            usize::from(nearest_row_matches.get() > 0) + usize::from(nearest_col_matches.get() > 0);
        let ref_match_count =
            usize::from(row_matches.get() > 0) + usize::from(col_matches.get() > 0);
        let (new_context, ref_context) = match nearest_match {
            0 => (
                usize::from(ref_match_count >= 1),
                match ref_match_count {
                    0 => 0,
                    1 => 1,
                    _ => 2,
                },
            ),
            1 => (
                2 + usize::from(new_mv_count.get() == 0),
                match ref_match_count {
                    1 => 3,
                    count if count >= 2 => 4,
                    _ => 0,
                },
            ),
            _ => (if new_mv_count.get() >= 1 { 4 } else { 5 }, 5),
        };
        new_context | (ref_context << 4)
    }

    pub(super) fn compound_mv_candidates(
        &self,
        x: usize,
        y: usize,
        block_size: BlockSize,
        primary_reference: u8,
        secondary_reference: u8,
        global_motion: [(i32, i32); 2],
    ) -> (
        [Option<(i32, i32)>; MAX_INTER_MV_CANDIDATES],
        [Option<(i32, i32)>; MAX_INTER_MV_CANDIDATES],
        [u16; MAX_INTER_MV_CANDIDATES],
        usize,
    ) {
        let mi_col = x / 4;
        let mi_row = y / 4;
        let block_w4 = (block_size.width() / 4).max(1);
        let block_h4 = (block_size.height() / 4).max(1);
        let target = [
            self.reference_frame_indices
                .get(usize::from(primary_reference))
                .copied()
                .unwrap_or(primary_reference),
            self.reference_frame_indices
                .get(usize::from(secondary_reference))
                .copied()
                .unwrap_or(secondary_reference),
        ];
        let mut primary = [None; MAX_INTER_MV_CANDIDATES];
        let mut secondary = [None; MAX_INTER_MV_CANDIDATES];
        let mut weights = [0u16; MAX_INTER_MV_CANDIDATES];
        let mut len = 0usize;
        let row_adj = usize::from(block_h4 < 2 && (mi_row & 1) == 1) as isize;
        let col_adj = usize::from(block_w4 < 2 && (mi_col & 1) == 1) as isize;
        let max_row_offset = if mi_row > self.tile_mi_row_start {
            (if block_h4 < 2 { -4 } else { -6 } + row_adj)
                .max(self.tile_mi_row_start as isize - mi_row as isize)
        } else {
            0
        };
        let max_col_offset = if mi_col > self.tile_mi_col_start {
            (if block_w4 < 2 { -4 } else { -6 } + col_adj)
                .max(self.tile_mi_col_start as isize - mi_col as isize)
        } else {
            0
        };

        let mut visit_pair = |column: usize, row: usize, weight: u16| {
            if column < self.tile_mi_col_start
                || row < self.tile_mi_row_start
                || column >= self.mi_cols
                || row >= self.mi_rows
            {
                return;
            }
            let index = row * self.mi_cols + column;
            if self.reference_frame_type_grid[index] != Some(primary_reference)
                || self.reference_frame_secondary_type_grid[index] != Some(secondary_reference)
            {
                return;
            }
            let Some(first) = self.motion_vector_grid[index] else {
                return;
            };
            let Some(second) = self.motion_vector_secondary_grid[index] else {
                return;
            };
            push_compound_mv_candidate(
                &mut primary,
                &mut secondary,
                &mut weights,
                &mut len,
                first,
                second,
                weight,
            );
        };
        if mi_row > self.tile_mi_row_start {
            let mut offset = 0usize;
            while offset < block_w4.min(self.mi_cols.saturating_sub(mi_col)) {
                let column = mi_col + offset;
                let source_size = self.motion_block_size_at(column, mi_row - 1);
                let source_w4 = source_size
                    .map(|size| (size.width() / 4).max(1))
                    .unwrap_or(1);
                let source_h4 = source_size
                    .map(|size| (size.height() / 4).max(1))
                    .unwrap_or(1);
                let weight = if block_w4 >= 2 && block_w4 <= source_w4 {
                    2isize.max((-max_row_offset).min(source_h4 as isize)) as usize
                } else {
                    2
                };
                visit_pair(
                    column,
                    mi_row - 1,
                    block_w4.min(source_w4).saturating_mul(weight) as u16,
                );
                offset = offset.saturating_add(block_w4.min(source_w4).max(1));
            }
        }
        if mi_col > self.tile_mi_col_start {
            let mut offset = 0usize;
            while offset < block_h4.min(self.mi_rows.saturating_sub(mi_row)) {
                let row = mi_row + offset;
                let source_size = self.motion_block_size_at(mi_col - 1, row);
                let source_w4 = source_size
                    .map(|size| (size.width() / 4).max(1))
                    .unwrap_or(1);
                let source_h4 = source_size
                    .map(|size| (size.height() / 4).max(1))
                    .unwrap_or(1);
                let weight = if block_h4 >= 2 && block_h4 <= source_h4 {
                    2isize.max((-max_col_offset).min(source_w4 as isize)) as usize
                } else {
                    2
                };
                visit_pair(
                    mi_col - 1,
                    row,
                    block_h4.min(source_h4).saturating_mul(weight) as u16,
                );
                offset = offset.saturating_add(block_h4.min(source_h4).max(1));
            }
        }
        if mi_row > self.tile_mi_row_start {
            visit_pair(mi_col.saturating_add(block_w4), mi_row - 1, 4);
        }
        drop(visit_pair);
        for index in 0..len {
            weights[index] = weights[index].saturating_add(640);
        }
        for end in (1..len).rev() {
            for index in 0..end {
                if weights[index] < weights[index + 1] {
                    primary.swap(index, index + 1);
                    secondary.swap(index, index + 1);
                    weights.swap(index, index + 1);
                }
            }
        }
        let nearest_len = len;

        // A compound temporal candidate must keep both projections from the
        // same collocated motion-field sample. Building the two sides from
        // independent spatial lists changes NEW_NEWMV predictors.
        if let Some(field) = self.temporal_motion_field.as_deref() {
            let blk_row_end = block_h4.min(16);
            let blk_col_end = block_w4.min(16);
            let step_row = if block_h4 >= 16 { 4 } else { 2 };
            let step_col = if block_w4 >= 16 { 4 } else { 2 };
            for blk_row in (0..blk_row_end).step_by(step_row) {
                for blk_col in (0..blk_col_end).step_by(step_col) {
                    add_compound_temporal_candidate(
                        field,
                        mi_col,
                        mi_row,
                        blk_col as isize,
                        blk_row as isize,
                        usize::from(primary_reference),
                        usize::from(secondary_reference),
                        self.order_hint,
                        &self.reference_order_hints,
                        &mut primary,
                        &mut secondary,
                        &mut weights,
                        &mut len,
                    );
                }
            }
            if block_h4 >= 2 && block_h4 < 16 && block_w4 >= 2 && block_w4 < 16 {
                let vertical_offset = block_h4.max(2) as isize;
                let horizontal_offset = block_w4.max(2) as isize;
                for (blk_row, blk_col) in [
                    (vertical_offset, -2),
                    (vertical_offset, horizontal_offset),
                    (vertical_offset - 2, horizontal_offset),
                ] {
                    let sb_row = (mi_row & 15) as isize + blk_row;
                    let sb_col = (mi_col & 15) as isize + blk_col;
                    if (0..16).contains(&sb_row) && (0..16).contains(&sb_col) {
                        add_compound_temporal_candidate(
                            field,
                            mi_col,
                            mi_row,
                            blk_col,
                            blk_row,
                            usize::from(primary_reference),
                            usize::from(secondary_reference),
                            self.order_hint,
                            &self.reference_order_hints,
                            &mut primary,
                            &mut secondary,
                            &mut weights,
                            &mut len,
                        );
                    }
                }
            }
        }

        // Continue with the non-nearest spatial positions. These candidates
        // remain below the REF_CAT_LEVEL nearest group but precede the
        // different-reference fallback.
        {
            let mut visit_pair = |column: usize, row: usize, weight: u16| {
                if column < self.tile_mi_col_start
                    || row < self.tile_mi_row_start
                    || column >= self.mi_cols
                    || row >= self.mi_rows
                {
                    return;
                }
                let index = row * self.mi_cols + column;
                if self.reference_frame_type_grid[index] != Some(primary_reference)
                    || self.reference_frame_secondary_type_grid[index] != Some(secondary_reference)
                {
                    return;
                }
                let Some(first) = self.motion_vector_grid[index] else {
                    return;
                };
                let Some(second) = self.motion_vector_secondary_grid[index] else {
                    return;
                };
                push_compound_mv_candidate(
                    &mut primary,
                    &mut secondary,
                    &mut weights,
                    &mut len,
                    first,
                    second,
                    weight,
                );
            };
            if mi_row > self.tile_mi_row_start && mi_col > self.tile_mi_col_start {
                visit_pair(mi_col - 1, mi_row - 1, 4);
            }
            for index in 2..=3 {
                let row_offset = -(index as isize * 2) + 1 + row_adj;
                if let Some(row) = mi_row.checked_add_signed(row_offset) {
                    let mut offset = usize::from(row_offset.abs() > 1);
                    if block_w4 < 2 && (mi_col & 1) == 1 {
                        offset = 0;
                    }
                    while offset < block_w4.min(self.mi_cols.saturating_sub(mi_col)) {
                        let column = mi_col + offset;
                        let source_w4 = self
                            .motion_block_size_at(column, row)
                            .map(|size| (size.width() / 4).max(1))
                            .unwrap_or(1);
                        let step = block_w4.min(source_w4).max(2);
                        visit_pair(column, row, step.saturating_mul(2) as u16);
                        offset = offset.saturating_add(step);
                    }
                }
                let col_offset = -(index as isize * 2) + 1 + col_adj;
                if let Some(column) = mi_col.checked_add_signed(col_offset) {
                    let mut offset = usize::from(col_offset.abs() > 1);
                    if block_h4 < 2 && (mi_row & 1) == 1 {
                        offset = 0;
                    }
                    while offset < block_h4.min(self.mi_rows.saturating_sub(mi_row)) {
                        let row = mi_row + offset;
                        let source_h4 = self
                            .motion_block_size_at(column, row)
                            .map(|size| (size.height() / 4).max(1))
                            .unwrap_or(1);
                        let step = block_h4.min(source_h4).max(2);
                        visit_pair(column, row, step.saturating_mul(2) as u16);
                        offset = offset.saturating_add(step);
                    }
                }
            }
        }
        for end in (nearest_len + 1..len).rev() {
            for index in nearest_len..end {
                if weights[index] < weights[index + 1] {
                    primary.swap(index, index + 1);
                    secondary.swap(index, index + 1);
                    weights.swap(index, index + 1);
                }
            }
        }

        let sign_bias = |reference: u8| {
            let hint = self
                .reference_frame_indices
                .iter()
                .position(|candidate| *candidate == reference)
                .and_then(|index| self.reference_order_hints[index]);
            let Some(hint) = hint else {
                return false;
            };
            let bits = self.order_hint_bits.max(1).min(31);
            let half = 1i32 << (bits - 1);
            let mask = half - 1;
            let mut distance = hint as i32 - self.order_hint as i32;
            distance = (distance & mask) - (distance & half);
            distance > 0
        };
        let normalize = |motion: (i32, i32), reference: u8, wanted: u8| {
            if sign_bias(reference) == sign_bias(wanted) {
                motion
            } else {
                (-motion.0, -motion.1)
            }
        };
        let mut exact = [[None; 2]; 2];
        let mut different = [[None; 2]; 2];
        let mut exact_len = [0usize; 2];
        let mut different_len = [0usize; 2];
        let mut add = |reference: u8, motion: (i32, i32)| {
            for target_index in 0..2 {
                let value = normalize(motion, reference, target[target_index]);
                if reference == target[target_index] {
                    if exact_len[target_index] < 2 {
                        exact[target_index][exact_len[target_index]] = Some(value);
                        exact_len[target_index] += 1;
                    }
                } else if different_len[target_index] < 2 {
                    different[target_index][different_len[target_index]] = Some(value);
                    different_len[target_index] += 1;
                }
            }
        };
        let mut visit = |column: usize, row: usize| {
            if column < self.tile_mi_col_start
                || row < self.tile_mi_row_start
                || column >= self.mi_cols
                || row >= self.mi_rows
            {
                return;
            }
            let index = row * self.mi_cols + column;
            let Some(size) = self.motion_block_size_grid[index] else {
                return;
            };
            let origin_col = column & !((size.width() / 4).max(1) - 1);
            let origin_row = row & !((size.height() / 4).max(1) - 1);
            let origin = origin_row * self.mi_cols + origin_col;
            if let Some(motion) = self.motion_vector_grid[origin] {
                if let Some(reference) = self.reference_frame_grid[origin] {
                    add(reference, motion);
                }
            }
            if let Some(motion) = self.motion_vector_secondary_grid[origin] {
                if let Some(reference) = self.reference_frame_secondary_grid[origin] {
                    add(reference, motion);
                }
            }
        };
        let mi_size = block_w4
            .min(block_h4)
            .min(self.mi_cols.saturating_sub(mi_col))
            .min(self.mi_rows.saturating_sub(mi_row))
            .min(16);
        if mi_row > self.tile_mi_row_start {
            let mut offset = 0usize;
            while offset < mi_size {
                visit(mi_col + offset, mi_row - 1);
                let Some(size) = self.motion_block_size_at(mi_col + offset, mi_row - 1) else {
                    offset += 1;
                    continue;
                };
                offset += (size.width() / 4).max(1);
            }
        }
        if mi_col > self.tile_mi_col_start {
            let mut offset = 0usize;
            while offset < mi_size {
                visit(mi_col - 1, mi_row + offset);
                let Some(size) = self.motion_block_size_at(mi_col - 1, mi_row + offset) else {
                    offset += 1;
                    continue;
                };
                offset += (size.height() / 4).max(1);
            }
        }
        let compound_list = std::array::from_fn::<_, 2, _>(|index| {
            std::array::from_fn::<_, 2, _>(|candidate| {
                exact[index][candidate]
                    .or_else(|| {
                        candidate
                            .checked_sub(exact_len[index])
                            .and_then(|different_index| different[index][different_index])
                    })
                    .unwrap_or(global_motion[index])
            })
        });
        if len == 1 {
            let first = (compound_list[0][0], compound_list[1][0]);
            let candidate = if primary[0] == Some(first.0) && secondary[0] == Some(first.1) {
                (compound_list[0][1], compound_list[1][1])
            } else {
                first
            };
            primary[len] = Some(candidate.0);
            secondary[len] = Some(candidate.1);
            weights[len] = 2;
            len += 1;
        } else if len == 0 {
            for candidate in 0..2 {
                primary[len] = Some(compound_list[0][candidate]);
                secondary[len] = Some(compound_list[1][candidate]);
                weights[len] = 2;
                len += 1;
            }
        }
        (primary, secondary, weights, len)
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
        let mut seen = [(0, 0); MAX_INTER_MV_CANDIDATES];
        let mut seen_len = 0;
        let stack = self.inter_mv_stack(x, y, block_size, reference_frame, secondary);
        for mv in stack.values.into_iter().take(stack.len).flatten() {
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
        let mut seen = [(0, 0); MAX_INTER_MV_CANDIDATES];
        let mut seen_len = 0;
        let stack = self.inter_mv_stack(x, y, block_size, reference_frame, secondary);
        for mv in stack.values.into_iter().take(stack.len).flatten() {
            if !seen[..seen_len].contains(&mv) {
                seen[seen_len] = mv;
                seen_len += 1;
            }
        }
        seen_len.max(1)
    }

    pub(super) fn inter_mv_candidate_weights(
        &self,
        x: usize,
        y: usize,
        block_size: BlockSize,
        reference_frame: u8,
        secondary: bool,
    ) -> [u16; MAX_INTER_MV_CANDIDATES] {
        self.inter_mv_stack(x, y, block_size, reference_frame, secondary)
            .weights
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
        interintra: bool,
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
        super::context_grid::fill_mi_grid(
            &mut self.interintra_grid,
            self.mi_cols,
            self.mi_rows,
            x,
            y,
            block_size,
            interintra,
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
        super::context_grid::fill_mi_grid(
            &mut self.interintra_grid,
            self.mi_cols,
            self.mi_rows,
            x,
            y,
            block_size,
            false,
        );
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

    pub(super) fn set_motion_block_size(&mut self, x: usize, y: usize, block_size: BlockSize) {
        super::context_grid::fill_mi_grid(
            &mut self.motion_block_size_grid,
            self.mi_cols,
            self.mi_rows,
            x,
            y,
            block_size,
            block_size,
        );
    }

    pub(super) fn set_compound_context(
        &mut self,
        x: usize,
        y: usize,
        block_size: BlockSize,
        compound_group_idx: Option<u8>,
        compound_idx: Option<u8>,
    ) {
        super::context_grid::fill_mi_grid_clone(
            &mut self.compound_group_idx_grid,
            self.mi_cols,
            self.mi_rows,
            x,
            y,
            block_size,
            compound_group_idx,
        );
        super::context_grid::fill_mi_grid_clone(
            &mut self.compound_idx_grid,
            self.mi_cols,
            self.mi_rows,
            x,
            y,
            block_size,
            compound_idx,
        );
    }

    pub(super) fn compound_group_idx_context(&self, x: usize, y: usize) -> usize {
        let mi_col = x / 4;
        let mi_row = y / 4;
        let neighbor_context = |column: usize, row: usize| {
            if column < self.tile_mi_col_start
                || row < self.tile_mi_row_start
                || column >= self.mi_cols
                || row >= self.mi_rows
            {
                return 0usize;
            }
            let index = row * self.mi_cols + column;
            if self.reference_frame_secondary_type_grid[index].is_some() {
                usize::from(self.compound_group_idx_grid[index].unwrap_or(0))
            } else if self.reference_frame_type_grid[index] == Some(6) {
                3
            } else {
                0
            }
        };
        let above = mi_row
            .checked_sub(1)
            .map(|row| neighbor_context(mi_col, row))
            .unwrap_or(0);
        let left = mi_col
            .checked_sub(1)
            .map(|column| neighbor_context(column, mi_row))
            .unwrap_or(0);
        (above + left).min(5)
    }

    pub(super) fn compound_idx_context(
        &self,
        x: usize,
        y: usize,
        primary_reference: usize,
        secondary_reference: usize,
    ) -> usize {
        let mi_col = x / 4;
        let mi_row = y / 4;
        let neighbor_context = |column: usize, row: usize| {
            if column < self.tile_mi_col_start
                || row < self.tile_mi_row_start
                || column >= self.mi_cols
                || row >= self.mi_rows
            {
                return 0usize;
            }
            let index = row * self.mi_cols + column;
            if self.reference_frame_secondary_type_grid[index].is_some() {
                usize::from(self.compound_idx_grid[index].unwrap_or(0))
            } else if self.reference_frame_type_grid[index] == Some(6) {
                1
            } else {
                0
            }
        };
        let above = mi_row
            .checked_sub(1)
            .map(|row| neighbor_context(mi_col, row))
            .unwrap_or(0);
        let left = mi_col
            .checked_sub(1)
            .map(|column| neighbor_context(column, mi_row))
            .unwrap_or(0);
        let relative_distance = |reference: u32| {
            let bits = self.order_hint_bits.max(1).min(31);
            let half = 1i32 << (bits - 1);
            let mask = half - 1;
            let distance = reference as i32 - self.order_hint as i32;
            (distance & mask) - (distance & half)
        };
        let equally_distant = self
            .reference_order_hints
            .get(primary_reference)
            .copied()
            .flatten()
            .zip(
                self.reference_order_hints
                    .get(secondary_reference)
                    .copied()
                    .flatten(),
            )
            .is_some_and(|(primary, secondary)| {
                relative_distance(primary).unsigned_abs()
                    == relative_distance(secondary).unsigned_abs()
            });
        above + left + 3 * usize::from(equally_distant)
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

    pub(super) fn set_txfm_context_leaf(
        &mut self,
        x: usize,
        y: usize,
        tx_size: TxSize,
        txb_size: TxSize,
    ) {
        let start_col = x >> 2;
        let start_row = y >> 2;
        let end_col = ((x + txb_size.width()).min(self.mi_cols << 2) + 3) >> 2;
        let end_row = ((y + txb_size.height()).min(self.mi_rows << 2) + 3) >> 2;
        for mi_col in start_col..end_col.min(self.mi_cols) {
            self.above_txfm_context[mi_col] = tx_size.width();
        }
        for mi_row in start_row..end_row.min(self.mi_rows) {
            self.left_txfm_context[mi_row] = tx_size.height();
        }
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
