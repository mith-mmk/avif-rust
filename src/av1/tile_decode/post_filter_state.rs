use super::TileDecoder;
use crate::av1::frame::FrameHeader;
use crate::av1::syntax::TxType;
use crate::av1::tile_decode::DecodedLumaBlock;
use crate::av1::transform::TransformBlock;

/// Apply the AV1 CDEF constrained-difference function to one neighbour delta.
///
/// The sign is preserved while the magnitude is limited by the selected
/// strength and damping. Keeping this primitive independent from plane
/// traversal makes the later 8x8 edge application easy to test against the
/// normative scalar vectors.
#[allow(dead_code)]
pub(crate) fn cdef_constrain(diff: i32, threshold: u8, damping: u8) -> i32 {
    if diff == 0 || threshold == 0 {
        return 0;
    }
    let magnitude = diff.unsigned_abs() as i32;
    let shift = damping.saturating_sub(threshold.ilog2() as u8);
    let limit = (threshold as i32 - (magnitude >> shift)).max(0);
    let constrained = magnitude.min(limit);
    if diff < 0 { -constrained } else { constrained }
}

#[allow(dead_code)]
const CDEF_DIRECTIONS: [[(isize, isize); 2]; 8] = [
    [(1, -1), (2, -2)],
    [(1, 0), (2, -1)],
    [(1, 0), (2, 0)],
    [(1, 0), (2, 1)],
    [(1, 1), (2, 2)],
    [(0, 1), (1, 2)],
    [(0, 1), (0, 2)],
    [(0, 1), (-1, 2)],
];

/// Filter one CDEF block using a caller-selected direction and strengths.
/// Edge samples are replicated at the supplied plane bounds; frame-level
/// orchestration is responsible for selecting the direction and CDEF index.
#[allow(dead_code)]
pub(crate) fn cdef_filter_block(
    source: &[u16],
    width: usize,
    height: usize,
    origin_x: usize,
    origin_y: usize,
    block_width: usize,
    block_height: usize,
    direction: usize,
    primary_strength: u8,
    secondary_strength: u8,
    damping: u8,
) -> Vec<u16> {
    let mut output = source.to_vec();
    let direction = direction & 7;
    let primary_taps = if primary_strength & 1 == 0 {
        [4, 2]
    } else {
        [3, 3]
    };
    let secondary_taps = [2, 1];
    let sample = |x: isize, y: isize| -> i32 {
        let x = x.clamp(0, width.saturating_sub(1) as isize) as usize;
        let y = y.clamp(0, height.saturating_sub(1) as isize) as usize;
        source[y * width + x] as i32
    };
    for row in 0..block_height {
        for col in 0..block_width {
            let x = origin_x + col;
            let y = origin_y + row;
            if x >= width || y >= height {
                continue;
            }
            let center = sample(x as isize, y as isize);
            let mut sum = 0i32;
            let mut min_value = center;
            let mut max_value = center;
            for (tap_index, &(dy, dx)) in CDEF_DIRECTIONS[direction].iter().enumerate() {
                for sign in [-1isize, 1] {
                    let value = sample(x as isize + sign * dx, y as isize + sign * dy);
                    sum += primary_taps[tap_index]
                        * cdef_constrain(value - center, primary_strength, damping);
                    min_value = min_value.min(value);
                    max_value = max_value.max(value);
                }
            }
            for secondary_direction in [(direction + 2) & 7, (direction + 6) & 7] {
                for (tap_index, &(dy, dx)) in
                    CDEF_DIRECTIONS[secondary_direction].iter().enumerate()
                {
                    for sign in [-1isize, 1] {
                        let value = sample(x as isize + sign * dx, y as isize + sign * dy);
                        sum += secondary_taps[tap_index]
                            * cdef_constrain(value - center, secondary_strength, damping);
                        min_value = min_value.min(value);
                        max_value = max_value.max(value);
                    }
                }
            }
            let filtered = (center + ((8 + sum - i32::from(sum < 0)) >> 4))
                .clamp(min_value, max_value)
                .clamp(0, u16::MAX as i32) as u16;
            output[y * width + x] = filtered;
        }
    }
    output
}

#[allow(dead_code)]
pub(crate) fn cdef_find_direction(
    source: &[u16],
    width: usize,
    height: usize,
    origin_x: usize,
    origin_y: usize,
    coeff_shift: u8,
) -> usize {
    const DIV: [i32; 9] = [0, 840, 420, 280, 210, 168, 140, 120, 105];
    let sample = |x: usize, y: usize| -> i32 {
        let x = (origin_x + x).min(width.saturating_sub(1));
        let y = (origin_y + y).min(height.saturating_sub(1));
        (source[y * width + x] >> coeff_shift) as i32 - 128
    };
    let mut partial = [[0i32; 15]; 8];
    for y in 0..8 {
        for x in 0..8 {
            let value = sample(x, y);
            partial[0][y + x] += value;
            partial[1][y + x / 2] += value;
            partial[2][y] += value;
            partial[3][3 + y - x / 2] += value;
            partial[4][7 + y - x] += value;
            partial[5][3 - y / 2 + x] += value;
            partial[6][x] += value;
            partial[7][y / 2 + x] += value;
        }
    }
    let mut cost = [0i32; 8];
    for i in 0..8 {
        cost[2] += partial[2][i] * partial[2][i];
        cost[6] += partial[6][i] * partial[6][i];
    }
    cost[2] *= DIV[8];
    cost[6] *= DIV[8];
    for i in 0..7 {
        cost[0] +=
            (partial[0][i] * partial[0][i] + partial[0][14 - i] * partial[0][14 - i]) * DIV[i + 1];
        cost[4] +=
            (partial[4][i] * partial[4][i] + partial[4][14 - i] * partial[4][14 - i]) * DIV[i + 1];
    }
    cost[0] += partial[0][7] * partial[0][7] * DIV[8];
    cost[4] += partial[4][7] * partial[4][7] * DIV[8];
    for direction in (1..8).step_by(2) {
        for i in 0..5 {
            cost[direction] += partial[direction][3 + i] * partial[direction][3 + i];
        }
        cost[direction] *= DIV[8];
        for i in 0..3 {
            cost[direction] += (partial[direction][i] * partial[direction][i]
                + partial[direction][10 - i] * partial[direction][10 - i])
                * DIV[2 * i + 2];
        }
    }
    let mut best = 0;
    for direction in 1..8 {
        if cost[direction] > cost[best] {
            best = direction;
        }
    }
    best
}

#[allow(dead_code)]
pub(crate) fn wiener_filter_unit(
    source: &[u16],
    width: usize,
    height: usize,
    origin_x: usize,
    origin_y: usize,
    unit_width: usize,
    unit_height: usize,
    filters: [[i16; 3]; 2],
) -> Vec<u16> {
    let sample = |x: isize, y: isize| -> i32 {
        let x = x.clamp(0, width.saturating_sub(1) as isize) as usize;
        let y = y.clamp(0, height.saturating_sub(1) as isize) as usize;
        source[y * width + x] as i32
    };
    let mut horizontal = source.to_vec();
    let taps = |axis: usize| {
        [
            filters[axis][0] as i32,
            filters[axis][1] as i32,
            filters[axis][2] as i32,
        ]
    };
    for y in origin_y..(origin_y + unit_height).min(height) {
        for x in origin_x..(origin_x + unit_width).min(width) {
            let [a, b, c] = taps(0);
            let center = 128 - 2 * (a + b + c);
            let value = a * sample(x as isize - 3, y as isize)
                + b * sample(x as isize - 2, y as isize)
                + c * sample(x as isize - 1, y as isize)
                + center * sample(x as isize, y as isize)
                + c * sample(x as isize + 1, y as isize)
                + b * sample(x as isize + 2, y as isize)
                + a * sample(x as isize + 3, y as isize);
            horizontal[y * width + x] = ((value + 64) >> 7).clamp(0, u16::MAX as i32) as u16;
        }
    }
    let sample_horizontal = |x: isize, y: isize| -> i32 {
        let x = x.clamp(0, width.saturating_sub(1) as isize) as usize;
        let y = y.clamp(0, height.saturating_sub(1) as isize) as usize;
        horizontal[y * width + x] as i32
    };
    let mut output = source.to_vec();
    for y in origin_y..(origin_y + unit_height).min(height) {
        for x in origin_x..(origin_x + unit_width).min(width) {
            let [a, b, c] = taps(1);
            let center = 128 - 2 * (a + b + c);
            let value = a * sample_horizontal(x as isize, y as isize - 3)
                + b * sample_horizontal(x as isize, y as isize - 2)
                + c * sample_horizontal(x as isize, y as isize - 1)
                + center * sample_horizontal(x as isize, y as isize)
                + c * sample_horizontal(x as isize, y as isize + 1)
                + b * sample_horizontal(x as isize, y as isize + 2)
                + a * sample_horizontal(x as isize, y as isize + 3);
            output[y * width + x] = ((value + 64) >> 7).clamp(0, u16::MAX as i32) as u16;
        }
    }
    output
}

#[allow(dead_code)]
pub(crate) fn sgrproj_filter_unit(
    source: &[u16],
    width: usize,
    height: usize,
    origin_x: usize,
    origin_y: usize,
    unit_width: usize,
    unit_height: usize,
    sgr_index: u8,
    xqd: [i16; 2],
) -> Vec<u16> {
    const RADII: [[usize; 2]; 16] = [
        [2, 1],
        [2, 1],
        [2, 1],
        [2, 1],
        [2, 1],
        [2, 1],
        [2, 1],
        [2, 1],
        [2, 1],
        [2, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [2, 0],
        [2, 0],
    ];
    const S: [[i32; 2]; 16] = [
        [140, 3236],
        [112, 2158],
        [93, 1618],
        [80, 1438],
        [70, 1295],
        [58, 1177],
        [47, 1079],
        [37, 996],
        [30, 925],
        [25, 863],
        [-1, 2589],
        [-1, 1618],
        [-1, 1177],
        [-1, 925],
        [56, -1],
        [22, -1],
    ];
    let index = usize::from(sgr_index.min(15));
    let sample = |x: isize, y: isize| -> i32 {
        let x = x.clamp(0, width.saturating_sub(1) as isize) as usize;
        let y = y.clamp(0, height.saturating_sub(1) as isize) as usize;
        source[y * width + x] as i32
    };
    let mut output = source.to_vec();
    let xq0 = if RADII[index][0] == 0 {
        0
    } else {
        i32::from(xqd[0])
    };
    let xq1 = if RADII[index][1] == 0 {
        0
    } else if RADII[index][0] == 0 {
        128 - i32::from(xqd[1])
    } else {
        128 - xq0 - i32::from(xqd[1])
    };
    let filter_at = |x: usize, y: usize, radius_index: usize| -> i32 {
        let radius = RADII[index][radius_index];
        if radius == 0 {
            return sample(x as isize, y as isize) << 4;
        }
        let coeff = |px: isize, py: isize| -> (i32, i32) {
            let side = radius * 2 + 1;
            let n = (side * side) as i32;
            let mut sum = 0i32;
            let mut sum_sq = 0i32;
            for dy in -(radius as isize)..=(radius as isize) {
                for dx in -(radius as isize)..=(radius as isize) {
                    let value = sample(px + dx, py + dy);
                    sum += value;
                    sum_sq += value * value;
                }
            }
            let p = (sum_sq * n - sum * sum).max(0);
            let z = ((p * S[index][radius_index] + (1 << 19)) >> 20).clamp(0, 255);
            let a = ((256 * z + z + 1) / (z + 1)).clamp(1, 256);
            let recip = ((4096 + n / 2) / n).max(1);
            let b = (((256 - a) * sum * recip) + 2048) >> 12;
            (a, b)
        };
        let mut a = 0i32;
        let mut b = 0i32;
        let taps = if radius == 2 {
            &[
                (0, 0, 4),
                (-1, 0, 4),
                (1, 0, 4),
                (0, -1, 4),
                (0, 1, 4),
                (-1, -1, 3),
                (-1, 1, 3),
                (1, -1, 3),
                (1, 1, 3),
            ][..]
        } else {
            &[
                (0, 0, 4),
                (-1, 0, 4),
                (1, 0, 4),
                (0, -1, 4),
                (0, 1, 4),
                (-1, -1, 3),
                (-1, 1, 3),
                (1, -1, 3),
                (1, 1, 3),
            ][..]
        };
        for &(dy, dx, weight) in taps {
            let (local_a, local_b) = coeff(x as isize + dx, y as isize + dy);
            a += local_a * weight;
            b += local_b * weight;
        }
        ((a * (sample(x as isize, y as isize) << 4) + b) + (1 << 7)) >> 8
    };
    for y in origin_y..(origin_y + unit_height).min(height) {
        for x in origin_x..(origin_x + unit_width).min(width) {
            let u = sample(x as isize, y as isize) << 4;
            let f0 = filter_at(x, y, 0);
            let f1 = filter_at(x, y, 1);
            let value = (u << 7) + xq0 * (f0 - u) + xq1 * (f1 - u);
            output[y * width + x] = ((value + (1 << 10)) >> 11).clamp(0, u16::MAX as i32) as u16;
        }
    }
    output
}

#[allow(dead_code)]
pub(crate) fn deblock_filter_edge(
    samples: &mut [u16],
    width: usize,
    height: usize,
    edge_x: usize,
    edge_y: usize,
    vertical: bool,
    level: u8,
    sharpness: u8,
    bit_depth: u8,
) {
    if level == 0
        || (vertical && (edge_x < 2 || edge_x + 1 >= width))
        || (!vertical && (edge_y < 2 || edge_y + 1 >= height))
    {
        return;
    }
    let shift = u32::from(bit_depth.saturating_sub(8));
    let mut inside = i32::from(level) >> u32::from((sharpness > 0) as u8 + (sharpness > 4) as u8);
    inside = inside.min(9 - i32::from(sharpness)).max(1) << shift;
    let blimit = (2 * (i32::from(level) + 2) + (inside >> shift)) << shift;
    let limit = inside;
    let hev = (i32::from(level) >> 4) << shift;
    let max_sample = ((1u32 << bit_depth.min(16)) - 1) as i32;
    for lane in 0..4 {
        let (p1, p0, q0, q1) = if vertical {
            let x = edge_x;
            let y = edge_y + lane;
            if y >= height {
                continue;
            }
            (
                samples[y * width + x - 2] as i32,
                samples[y * width + x - 1] as i32,
                samples[y * width + x] as i32,
                samples[y * width + x + 1] as i32,
            )
        } else {
            let x = edge_x + lane;
            let y = edge_y;
            if x >= width {
                continue;
            }
            (
                samples[(y - 2) * width + x] as i32,
                samples[(y - 1) * width + x] as i32,
                samples[y * width + x] as i32,
                samples[(y + 1) * width + x] as i32,
            )
        };
        let mask = (i32::abs(p1 - p0) <= limit
            && i32::abs(q1 - q0) <= limit
            && 2 * i32::abs(p0 - q0) + i32::abs(p1 - q1) / 2 <= blimit) as i32;
        if mask == 0 {
            continue;
        }
        let hev_mask = (i32::abs(p1 - p0) > hev || i32::abs(q1 - q0) > hev) as i32;
        let mut filter = (p1 - q1) * hev_mask + 3 * (q0 - p0);
        filter = filter.clamp(-128 << shift, 127 << shift);
        let f1 = (filter + 4).div_euclid(8);
        let f2 = (filter + 3).div_euclid(8);
        let np0 = (p0 + f2).clamp(0, max_sample) as u16;
        let nq0 = (q0 - f1).clamp(0, max_sample) as u16;
        if vertical {
            samples[(edge_y + lane) * width + edge_x - 1] = np0;
            samples[(edge_y + lane) * width + edge_x] = nq0;
        } else {
            samples[(edge_y - 1) * width + edge_x + lane] = np0;
            samples[edge_y * width + edge_x + lane] = nq0;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CdefUnit {
    pub(crate) x: usize,
    pub(crate) y: usize,
    pub(crate) index: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TransformBoundary {
    pub(crate) block: TransformBlock,
    pub(crate) tx_type: TxType,
    pub(crate) non_zero_coefficients: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RestorationUnit {
    pub(crate) x: usize,
    pub(crate) y: usize,
    pub(crate) plane: usize,
    pub(crate) restoration_type: u8,
    pub(crate) wiener: Option<[[i16; 3]; 2]>,
    pub(crate) sgrproj: Option<[i16; 2]>,
    pub(crate) sgrproj_index: Option<u8>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct PostFilterState {
    pub(crate) cdef_units: Vec<CdefUnit>,
    pub(crate) transform_boundaries: Vec<TransformBoundary>,
    pub(crate) restoration_units: Vec<RestorationUnit>,
}

impl PostFilterState {
    pub(crate) fn merge(&mut self, other: Self) {
        for unit in other.cdef_units {
            if let Some(existing) = self
                .cdef_units
                .iter_mut()
                .find(|existing| existing.x == unit.x && existing.y == unit.y)
            {
                existing.index = unit.index;
            } else {
                self.cdef_units.push(unit);
            }
        }
        for boundary in other.transform_boundaries {
            if let Some(existing) = self
                .transform_boundaries
                .iter_mut()
                .find(|existing| existing.block == boundary.block)
            {
                *existing = boundary;
            } else {
                self.transform_boundaries.push(boundary);
            }
        }
        for unit in other.restoration_units {
            if let Some(existing) = self.restoration_units.iter_mut().find(|existing| {
                existing.x == unit.x && existing.y == unit.y && existing.plane == unit.plane
            }) {
                *existing = unit;
            } else {
                self.restoration_units.push(unit);
            }
        }
    }

    pub(crate) fn record_luma_blocks(&mut self, blocks: &[DecodedLumaBlock]) {
        for block in blocks {
            for transform in &block.transforms {
                let boundary = TransformBoundary {
                    block: transform.transform,
                    tx_type: transform.tx_type,
                    non_zero_coefficients: transform
                        .coefficients
                        .iter()
                        .filter(|coefficient| **coefficient != 0)
                        .count(),
                };
                if !self
                    .transform_boundaries
                    .iter()
                    .any(|existing| existing.block == boundary.block)
                {
                    self.transform_boundaries.push(boundary);
                }
            }
        }
    }
}

impl<'a> TileDecoder<'a> {
    pub(super) fn take_post_filter_state(self) -> PostFilterState {
        PostFilterState {
            cdef_units: self.cdef_units,
            transform_boundaries: Vec::new(),
            restoration_units: self.restoration_units,
        }
    }

    pub(super) fn record_cdef_index(
        &mut self,
        frame: &FrameHeader,
        x: usize,
        y: usize,
        index: Option<u32>,
    ) {
        if !frame.cdef.enabled || frame.allow_intrabc {
            return;
        }

        let (unit_x, unit_y) = cdef_unit_origin(x, y);
        store_cdef_unit(&mut self.cdef_units, unit_x, unit_y, index);
    }
}

fn cdef_unit_origin(x: usize, y: usize) -> (usize, usize) {
    (x & !63, y & !63)
}

fn store_cdef_unit(units: &mut Vec<CdefUnit>, x: usize, y: usize, index: Option<u32>) {
    if let Some(unit) = units.iter_mut().find(|unit| unit.x == x && unit.y == y) {
        if let Some(index) = index {
            unit.index = index;
        }
        return;
    }
    units.push(CdefUnit {
        x,
        y,
        index: index.unwrap_or(0),
    });
}

#[cfg(test)]
mod tests {
    use super::{
        CdefUnit, PostFilterState, cdef_constrain, cdef_filter_block, cdef_find_direction,
        cdef_unit_origin, deblock_filter_edge, store_cdef_unit,
    };

    #[test]
    fn cdef_filter_keeps_constant_block_unchanged() {
        let source = vec![128u16; 16 * 16];
        let filtered = cdef_filter_block(&source, 16, 16, 4, 4, 8, 8, 0, 4, 2, 3);
        assert_eq!(filtered, source);
    }

    #[test]
    fn cdef_direction_is_stable_for_a_constant_block() {
        let source = vec![128u16; 8 * 8];
        assert_eq!(cdef_find_direction(&source, 8, 8, 0, 0, 0), 0);
    }

    #[test]
    fn wiener_filter_preserves_a_constant_unit_for_normalized_taps() {
        let source = vec![200u16; 16 * 16];
        let filtered =
            super::wiener_filter_unit(&source, 16, 16, 0, 0, 16, 16, [[1, 2, 3], [3, -2, 1]]);
        assert_eq!(filtered, source);
    }

    #[test]
    fn sgrproj_filter_preserves_a_constant_unit_with_zero_projection() {
        let source = vec![200u16; 16 * 16];
        let filtered = super::sgrproj_filter_unit(&source, 16, 16, 0, 0, 16, 16, 0, [0, 128]);
        assert_eq!(filtered, source);
    }

    #[test]
    fn deblock_filter_keeps_constant_edges_unchanged() {
        let mut samples = vec![128u16; 16 * 16];
        deblock_filter_edge(&mut samples, 16, 16, 8, 0, true, 20, 0, 8);
        deblock_filter_edge(&mut samples, 16, 16, 0, 8, false, 20, 0, 8);
        assert!(samples.iter().all(|sample| *sample == 128));
    }

    #[test]
    fn cdef_filter_clips_an_impulse_to_neighbour_range() {
        let mut source = vec![100u16; 8 * 8];
        source[4 * 8 + 4] = 500;
        let filtered = cdef_filter_block(&source, 8, 8, 0, 0, 8, 8, 2, 8, 4, 3);
        assert!(filtered[4 * 8 + 4] >= 100);
        assert!(filtered[4 * 8 + 4] <= 500);
    }

    #[test]
    fn cdef_constrain_preserves_small_deltas_and_sign() {
        assert_eq!(cdef_constrain(2, 4, 3), 2);
        assert_eq!(cdef_constrain(-2, 4, 3), -2);
    }

    #[test]
    fn cdef_constrain_clips_large_deltas_with_damping() {
        assert_eq!(cdef_constrain(10, 4, 3), 0);
        assert_eq!(cdef_constrain(-10, 4, 3), 0);
        assert_eq!(cdef_constrain(5, 8, 3), 3);
    }

    #[test]
    fn cdef_unit_is_addressed_on_a_64_pixel_grid() {
        let (x, y) = cdef_unit_origin(126, 191);
        let unit = CdefUnit { x, y, index: 3 };
        assert_eq!((unit.x, unit.y, unit.index), (64, 128, 3));
    }

    #[test]
    fn later_transmitted_index_replaces_skipped_unit_default() {
        let mut units = Vec::new();
        store_cdef_unit(&mut units, 0, 0, None);
        store_cdef_unit(&mut units, 0, 0, Some(2));
        assert_eq!(
            units,
            vec![CdefUnit {
                x: 0,
                y: 0,
                index: 2
            }]
        );
    }

    #[test]
    fn post_filter_state_merges_tile_units_by_origin() {
        let mut state = PostFilterState {
            cdef_units: vec![CdefUnit {
                x: 0,
                y: 0,
                index: 1,
            }],
            transform_boundaries: Vec::new(),
            restoration_units: Vec::new(),
        };
        state.merge(PostFilterState {
            cdef_units: vec![
                CdefUnit {
                    x: 0,
                    y: 0,
                    index: 3,
                },
                CdefUnit {
                    x: 64,
                    y: 0,
                    index: 2,
                },
            ],
            transform_boundaries: Vec::new(),
            restoration_units: Vec::new(),
        });

        assert_eq!(
            state.cdef_units,
            vec![
                CdefUnit {
                    x: 0,
                    y: 0,
                    index: 3,
                },
                CdefUnit {
                    x: 64,
                    y: 0,
                    index: 2,
                },
            ]
        );
    }

    #[test]
    fn post_filter_state_merges_restoration_units_by_plane_and_origin() {
        let mut state = PostFilterState::default();
        state.merge(PostFilterState {
            cdef_units: Vec::new(),
            transform_boundaries: Vec::new(),
            restoration_units: vec![super::RestorationUnit {
                x: 0,
                y: 0,
                plane: 1,
                restoration_type: 1,
                wiener: Some([[1, 2, 3], [4, 5, 6]]),
                sgrproj: None,
                sgrproj_index: None,
            }],
        });
        state.merge(PostFilterState {
            cdef_units: Vec::new(),
            transform_boundaries: Vec::new(),
            restoration_units: vec![super::RestorationUnit {
                x: 0,
                y: 0,
                plane: 1,
                restoration_type: 2,
                wiener: None,
                sgrproj: Some([7, 8]),
                sgrproj_index: Some(3),
            }],
        });
        assert_eq!(state.restoration_units.len(), 1);
        assert_eq!(state.restoration_units[0].restoration_type, 2);
        assert_eq!(state.restoration_units[0].sgrproj, Some([7, 8]));
        assert_eq!(state.restoration_units[0].sgrproj_index, Some(3));
    }
}
