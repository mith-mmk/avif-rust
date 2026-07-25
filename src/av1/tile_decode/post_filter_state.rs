use super::TileDecoder;
use crate::av1::frame::FrameHeader;
use crate::av1::syntax::{BlockSize, PredictionMode, TxType, UvPredictionMode};
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
    [(0, 1), (-1, 2)],
    [(0, 1), (0, 2)],
    [(0, 1), (1, 2)],
    [(1, 1), (2, 2)],
    [(1, 0), (2, 1)],
    [(1, 0), (2, 0)],
    [(1, 0), (2, -1)],
];

/// Filter one CDEF block using a caller-selected direction and strengths.
/// Edge samples are replicated at the supplied plane bounds; frame-level
/// orchestration is responsible for selecting the direction and CDEF index.
#[allow(dead_code)]
#[expect(
    clippy::too_many_arguments,
    reason = "scalar CDEF kernel parameters mirror the normative filter inputs"
)]
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
    cdef_filter_block_with_edge_mode(
        source,
        width,
        height,
        origin_x,
        origin_y,
        block_width,
        block_height,
        direction,
        primary_strength,
        secondary_strength,
        damping,
        false,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "scalar CDEF kernel parameters mirror the normative filter inputs"
)]
pub(crate) fn cdef_filter_block_with_edge_mode(
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
    use_edge_sentinel: bool,
) -> Vec<u16> {
    let mut output = source.to_vec();
    let filtered = cdef_filter_block_region_with_edge_mode(
        source,
        width,
        height,
        origin_x,
        origin_y,
        block_width,
        block_height,
        direction,
        primary_strength,
        secondary_strength,
        damping,
        use_edge_sentinel,
    );
    for row in 0..block_height {
        for col in 0..block_width {
            let x = origin_x + col;
            let y = origin_y + row;
            if x < width && y < height {
                output[y * width + x] = filtered[row * block_width + col];
            }
        }
    }
    output
}

pub(crate) fn cdef_filter_block_region_with_edge_mode(
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
    use_edge_sentinel: bool,
) -> Vec<u16> {
    let mut output = vec![0u16; block_width * block_height];
    cdef_filter_block_region_with_edge_mode_into(
        source,
        width,
        height,
        origin_x,
        origin_y,
        block_width,
        block_height,
        direction,
        primary_strength,
        secondary_strength,
        damping,
        use_edge_sentinel,
        &mut output,
    );
    output
}

#[expect(
    clippy::too_many_arguments,
    reason = "scalar CDEF kernel parameters mirror the normative filter inputs"
)]
pub(crate) fn cdef_filter_block_region_with_edge_mode_into(
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
    use_edge_sentinel: bool,
    output: &mut [u16],
) {
    cdef_filter_block_region_with_edge_mode_into_bit_depth(
        source,
        width,
        height,
        origin_x,
        origin_y,
        block_width,
        block_height,
        direction,
        primary_strength,
        secondary_strength,
        damping,
        0,
        use_edge_sentinel,
        output,
    );
}

#[expect(
    clippy::too_many_arguments,
    reason = "scalar CDEF kernel parameters mirror the normative filter inputs"
)]
pub(crate) fn cdef_filter_block_region_with_edge_mode_into_bit_depth(
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
    coeff_shift: u8,
    use_edge_sentinel: bool,
    output: &mut [u16],
) {
    cdef_filter_block_region_with_edge_mode_into_bit_depth_visible(
        source,
        width,
        height,
        width,
        height,
        origin_x,
        origin_y,
        block_width,
        block_height,
        direction,
        primary_strength,
        secondary_strength,
        damping,
        coeff_shift,
        use_edge_sentinel,
        output,
    );
}

#[expect(
    clippy::too_many_arguments,
    reason = "scalar CDEF kernel parameters mirror the normative filter inputs"
)]
pub(crate) fn cdef_filter_block_region_with_edge_mode_into_bit_depth_visible(
    source: &[u16],
    width: usize,
    height: usize,
    visible_width: usize,
    visible_height: usize,
    origin_x: usize,
    origin_y: usize,
    block_width: usize,
    block_height: usize,
    direction: usize,
    primary_strength: u8,
    secondary_strength: u8,
    damping: u8,
    coeff_shift: u8,
    use_edge_sentinel: bool,
    output: &mut [u16],
) {
    let strength_scale = 1u8.checked_shl(u32::from(coeff_shift)).unwrap_or(u8::MAX);
    let scaled_primary_strength = primary_strength.saturating_mul(strength_scale);
    let scaled_secondary_strength = secondary_strength.saturating_mul(strength_scale);
    let scaled_damping = damping.saturating_add(coeff_shift);
    cdef_filter_block_region_with_edge_mode_into_bit_depth_visible_scaled(
        source,
        width,
        height,
        visible_width,
        visible_height,
        origin_x,
        origin_y,
        block_width,
        block_height,
        direction,
        scaled_primary_strength,
        scaled_secondary_strength,
        scaled_damping,
        coeff_shift,
        use_edge_sentinel,
        output,
    );
}

#[expect(
    clippy::too_many_arguments,
    reason = "scalar CDEF kernel parameters mirror the normative filter inputs"
)]
pub(crate) fn cdef_filter_block_region_with_edge_mode_into_bit_depth_visible_scaled(
    source: &[u16],
    width: usize,
    height: usize,
    visible_width: usize,
    visible_height: usize,
    origin_x: usize,
    origin_y: usize,
    block_width: usize,
    block_height: usize,
    direction: usize,
    scaled_primary_strength: u8,
    scaled_secondary_strength: u8,
    scaled_damping: u8,
    coeff_shift: u8,
    use_edge_sentinel: bool,
    output: &mut [u16],
) {
    if output.len() < block_width.saturating_mul(block_height) {
        return;
    }
    const CDEF_VERY_LARGE: i32 = 0x4000;
    let direction = direction & 7;
    let enable_primary = scaled_primary_strength != 0;
    let enable_secondary = scaled_secondary_strength != 0;
    let clipping_required = enable_primary && enable_secondary;
    let primary_taps = if (scaled_primary_strength >> coeff_shift) & 1 == 0 {
        [4, 2]
    } else {
        [3, 3]
    };
    let secondary_taps = [2, 1];
    let sample = |x: isize, y: isize| -> i32 {
        if use_edge_sentinel
            && (x < 0 || y < 0 || x >= visible_width as isize || y >= visible_height as isize)
        {
            return CDEF_VERY_LARGE;
        }
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
                if enable_primary {
                    for sign in [-1isize, 1] {
                        let value = sample(x as isize + sign * dx, y as isize + sign * dy);
                        sum += primary_taps[tap_index]
                            * cdef_constrain(
                                value - center,
                                scaled_primary_strength,
                                scaled_damping,
                            );
                        if clipping_required {
                            min_value = min_value.min(value);
                            if value != CDEF_VERY_LARGE {
                                max_value = max_value.max(value);
                            }
                        }
                    }
                }
            }
            for secondary_direction in [(direction + 2) & 7, (direction + 6) & 7] {
                for (tap_index, &(dy, dx)) in
                    CDEF_DIRECTIONS[secondary_direction].iter().enumerate()
                {
                    if enable_secondary {
                        for sign in [-1isize, 1] {
                            let value = sample(x as isize + sign * dx, y as isize + sign * dy);
                            sum += secondary_taps[tap_index]
                                * cdef_constrain(
                                    value - center,
                                    scaled_secondary_strength,
                                    scaled_damping,
                                );
                            if clipping_required {
                                min_value = min_value.min(value);
                                if value != CDEF_VERY_LARGE {
                                    max_value = max_value.max(value);
                                }
                            }
                        }
                    }
                }
            }
            let filtered = (center + ((8 + sum - i32::from(sum < 0)) >> 4))
                .clamp(
                    if clipping_required {
                        min_value
                    } else {
                        i32::MIN
                    },
                    if clipping_required {
                        max_value
                    } else {
                        i32::MAX
                    },
                )
                .clamp(0, u16::MAX as i32) as u16;
            output[row * block_width + col] = filtered;
        }
    }
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
    cdef_find_direction_with_variance(
        source,
        width,
        height,
        origin_x,
        origin_y,
        coeff_shift,
        false,
    )
    .0
}

pub(crate) fn cdef_find_direction_with_variance(
    source: &[u16],
    width: usize,
    height: usize,
    origin_x: usize,
    origin_y: usize,
    coeff_shift: u8,
    use_edge_sentinel: bool,
) -> (usize, i32) {
    cdef_find_direction_with_variance_visible(
        source,
        width,
        height,
        width,
        height,
        origin_x,
        origin_y,
        coeff_shift,
        use_edge_sentinel,
    )
}

pub(crate) fn cdef_find_direction_with_variance_visible(
    source: &[u16],
    width: usize,
    height: usize,
    visible_width: usize,
    visible_height: usize,
    origin_x: usize,
    origin_y: usize,
    coeff_shift: u8,
    use_edge_sentinel: bool,
) -> (usize, i32) {
    const CDEF_VERY_LARGE: i64 = 0x4000;
    const DIV: [i64; 9] = [0, 840, 420, 280, 210, 168, 140, 120, 105];
    let sample = |x: usize, y: usize| -> i64 {
        if use_edge_sentinel && (origin_x + x >= visible_width || origin_y + y >= visible_height) {
            return (CDEF_VERY_LARGE >> coeff_shift) - 128;
        }
        let x = (origin_x + x).min(width.saturating_sub(1));
        let y = (origin_y + y).min(height.saturating_sub(1));
        i64::from(source[y * width + x] >> coeff_shift) - 128
    };
    let mut partial = [[0i64; 15]; 8];
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
    let mut cost = [0i64; 8];
    for (&partial_2, &partial_6) in partial[2].iter().zip(&partial[6]) {
        cost[2] += partial_2 * partial_2;
        cost[6] += partial_6 * partial_6;
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
    let variance = ((cost[best] - cost[(best + 4) & 7]) >> 10).clamp(0, i64::from(i32::MAX)) as i32;
    (best, variance)
}

pub(crate) fn cdef_adjust_primary_strength(strength: u8, variance: i32) -> u8 {
    if variance == 0 {
        return 0;
    }
    let scaled = variance >> 6;
    let i = if scaled == 0 {
        0
    } else {
        (i32::BITS - 1 - scaled.leading_zeros()).min(12) as i32
    };
    ((i32::from(strength) * (4 + i) + 8) >> 4) as u8
}

/// Map the luma CDEF direction to a subsampled chroma plane.
///
/// AV1 derives direction and variance from the luma 8x8 block.  4:2:0 keeps
/// that direction, while the asymmetric 4:2:2 and 4:4:0 layouts use the
/// normative plane-specific lookup tables.
pub(crate) fn cdef_chroma_direction(
    direction: usize,
    subsampling_x: bool,
    subsampling_y: bool,
) -> usize {
    const CONV_422: [usize; 8] = [7, 0, 2, 4, 5, 6, 6, 6];
    const CONV_440: [usize; 8] = [1, 2, 2, 2, 3, 4, 6, 0];
    let direction = direction & 7;
    match (subsampling_x, subsampling_y) {
        (true, false) => CONV_422[direction],
        (false, true) => CONV_440[direction],
        _ => direction,
    }
}

/// Read a restoration stripe with AOM's three-row stripe halo. Loop
/// restoration stores only two rows at an internal stripe boundary and
/// duplicates the outer row to provide the three convolution rows required by
/// the 7-tap kernel. Frame edges continue to use the extended outermost row.
#[cfg(test)]
fn restoration_sample(
    source: &[u16],
    width: usize,
    height: usize,
    x: isize,
    y: isize,
    origin_y: usize,
    stripe_height: usize,
) -> i32 {
    restoration_sample_with_visible_bounds(
        source,
        width,
        width,
        height,
        x,
        y,
        origin_y,
        stripe_height,
    )
}

fn restoration_sample_with_visible_bounds(
    source: &[u16],
    stride: usize,
    visible_width: usize,
    visible_height: usize,
    x: isize,
    y: isize,
    origin_y: usize,
    stripe_height: usize,
) -> i32 {
    let mut sample_y = y;
    let stripe_start = origin_y as isize;
    if origin_y > 0 && (sample_y == stripe_start - 3 || sample_y == stripe_start - 2) {
        sample_y = stripe_start - 2;
    }
    let stripe_end = (origin_y + stripe_height).min(visible_height) as isize;
    if stripe_end < visible_height as isize && sample_y == stripe_end + 2 {
        sample_y = stripe_end + 1;
    }
    let sample_x = x.clamp(0, visible_width.saturating_sub(1) as isize) as usize;
    let sample_y = sample_y.clamp(0, visible_height.saturating_sub(1) as isize) as usize;
    source[sample_y * stride + sample_x] as i32
}

#[allow(dead_code)]
#[expect(
    clippy::too_many_arguments,
    reason = "scalar Wiener kernel parameters mirror the normative restoration inputs"
)]
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
    let mut output = source.to_vec();
    wiener_filter_unit_into(
        source,
        &mut output,
        width,
        height,
        origin_x,
        origin_y,
        unit_width,
        unit_height,
        filters,
    );
    output
}

#[expect(
    clippy::too_many_arguments,
    reason = "scalar Wiener kernel parameters mirror the normative restoration inputs"
)]
pub(crate) fn wiener_filter_unit_into(
    source: &[u16],
    output: &mut [u16],
    width: usize,
    height: usize,
    origin_x: usize,
    origin_y: usize,
    unit_width: usize,
    unit_height: usize,
    filters: [[i16; 3]; 2],
) {
    let mut horizontal_scratch = Vec::new();
    wiener_filter_unit_into_with_scratch(
        source,
        output,
        width,
        height,
        origin_x,
        origin_y,
        unit_width,
        unit_height,
        filters,
        &mut horizontal_scratch,
    );
}

#[expect(
    clippy::too_many_arguments,
    reason = "scalar Wiener kernel parameters and reusable scratch stay explicit"
)]
pub(crate) fn wiener_filter_unit_into_with_scratch(
    source: &[u16],
    output: &mut [u16],
    width: usize,
    height: usize,
    origin_x: usize,
    origin_y: usize,
    unit_width: usize,
    unit_height: usize,
    filters: [[i16; 3]; 2],
    horizontal_scratch: &mut Vec<i32>,
) {
    wiener_filter_unit_into_with_scratch_bit_depth(
        source,
        output,
        width,
        height,
        origin_x,
        origin_y,
        unit_width,
        unit_height,
        filters,
        8,
        horizontal_scratch,
    );
}

#[expect(
    clippy::too_many_arguments,
    reason = "scalar Wiener kernel parameters and reusable scratch stay explicit"
)]
pub(crate) fn wiener_filter_unit_into_with_scratch_bit_depth(
    source: &[u16],
    output: &mut [u16],
    width: usize,
    height: usize,
    origin_x: usize,
    origin_y: usize,
    unit_width: usize,
    unit_height: usize,
    filters: [[i16; 3]; 2],
    bit_depth: u8,
    horizontal_scratch: &mut Vec<i32>,
) {
    wiener_filter_unit_into_with_scratch_bit_depth_visible(
        source,
        output,
        width,
        height,
        width,
        height,
        origin_x,
        origin_y,
        unit_width,
        unit_height,
        filters,
        bit_depth,
        horizontal_scratch,
    );
}

#[expect(
    clippy::too_many_arguments,
    reason = "scalar Wiener kernel keeps coded stride and visible bounds explicit"
)]
pub(crate) fn wiener_filter_unit_into_with_scratch_bit_depth_visible(
    source: &[u16],
    output: &mut [u16],
    width: usize,
    _height: usize,
    visible_width: usize,
    visible_height: usize,
    origin_x: usize,
    origin_y: usize,
    unit_width: usize,
    unit_height: usize,
    filters: [[i16; 3]; 2],
    bit_depth: u8,
    horizontal_scratch: &mut Vec<i32>,
) {
    const FILTER_BITS: u32 = 7;
    const ROUND_0_BITS: u32 = 3;
    const ROUND_1_BITS: u32 = 2 * FILTER_BITS - ROUND_0_BITS;
    let output_width = unit_width.min(visible_width.saturating_sub(origin_x));
    let output_height = unit_height.min(visible_height.saturating_sub(origin_y));
    if output_width == 0 || output_height == 0 {
        return;
    }
    let sample = |x: isize, y: isize| {
        restoration_sample_with_visible_bounds(
            source,
            width,
            visible_width,
            visible_height,
            x,
            y,
            origin_y,
            output_height,
        )
    };
    let residual_kernel = |axis: usize| {
        let [outer, middle, inner] = filters[axis].map(i32::from);
        [
            outer,
            middle,
            inner,
            -2 * (outer + middle + inner),
            inner,
            middle,
            outer,
        ]
    };
    // AV1 transmits the vertical filter first and the horizontal filter
    // second. The separable convolution keeps extra precision between passes
    // and includes the implicit +128 center tap through add-src offsets.
    let horizontal_kernel = residual_kernel(1);
    let vertical_kernel = residual_kernel(0);
    let intermediate_height = output_height + 6;
    horizontal_scratch.resize(output_width * intermediate_height, 0);
    let horizontal = &mut horizontal_scratch[..output_width * intermediate_height];
    let horizontal_offset = 1_i32 << (u32::from(bit_depth) + FILTER_BITS - 1);
    let horizontal_limit = (1_i32 << (u32::from(bit_depth) + 1 + FILTER_BITS - ROUND_0_BITS)) - 1;
    for intermediate_y in 0..intermediate_height {
        let source_y = origin_y as isize + intermediate_y as isize - 3;
        for local_x in 0..output_width {
            let source_x = origin_x + local_x;
            let center = sample(source_x as isize, source_y);
            let residual = horizontal_kernel
                .iter()
                .enumerate()
                .map(|(tap, coefficient)| {
                    coefficient * sample(source_x as isize + tap as isize - 3, source_y)
                })
                .sum::<i32>();
            horizontal[intermediate_y * output_width + local_x] = ((residual
                + (center << FILTER_BITS)
                + horizontal_offset
                + (1 << (ROUND_0_BITS - 1)))
                >> ROUND_0_BITS)
                .clamp(0, horizontal_limit);
        }
    }

    let vertical_offset = 1_i32 << (u32::from(bit_depth) + ROUND_1_BITS - 1);
    let max_sample = ((1_u32 << u32::from(bit_depth.min(16))) - 1) as i32;
    for local_y in 0..output_height {
        for local_x in 0..output_width {
            let center = horizontal[(local_y + 3) * output_width + local_x];
            let residual = vertical_kernel
                .iter()
                .enumerate()
                .map(|(tap, coefficient)| {
                    coefficient * horizontal[(local_y + tap) * output_width + local_x]
                })
                .sum::<i32>();
            let value = residual + (center << FILTER_BITS) - vertical_offset;
            output[(origin_y + local_y) * width + origin_x + local_x] =
                ((value + (1 << (ROUND_1_BITS - 1))) >> ROUND_1_BITS).clamp(0, max_sample) as u16;
        }
    }
}

#[allow(dead_code)]
#[expect(
    clippy::too_many_arguments,
    reason = "scalar SGRPROJ kernel parameters mirror the normative restoration inputs"
)]
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
    let mut output = source.to_vec();
    sgrproj_filter_unit_into(
        source,
        &mut output,
        width,
        height,
        origin_x,
        origin_y,
        unit_width,
        unit_height,
        sgr_index,
        xqd,
    );
    output
}

#[expect(
    clippy::too_many_arguments,
    reason = "scalar SGRPROJ kernel parameters mirror the normative restoration inputs"
)]
pub(crate) fn sgrproj_filter_unit_into(
    source: &[u16],
    output: &mut [u16],
    width: usize,
    height: usize,
    origin_x: usize,
    origin_y: usize,
    unit_width: usize,
    unit_height: usize,
    sgr_index: u8,
    xqd: [i16; 2],
) {
    let mut scratch = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    sgrproj_filter_unit_into_with_scratch(
        source,
        output,
        width,
        height,
        origin_x,
        origin_y,
        unit_width,
        unit_height,
        sgr_index,
        xqd,
        &mut scratch,
    );
}

#[expect(
    clippy::too_many_arguments,
    reason = "scalar SGRPROJ kernel parameters and reusable scratch stay explicit"
)]
pub(crate) fn sgrproj_filter_unit_into_with_scratch(
    source: &[u16],
    output: &mut [u16],
    width: usize,
    height: usize,
    origin_x: usize,
    origin_y: usize,
    unit_width: usize,
    unit_height: usize,
    sgr_index: u8,
    xqd: [i16; 2],
    scratch: &mut [Vec<i32>; 4],
) {
    sgrproj_filter_unit_into_with_scratch_bit_depth(
        source,
        output,
        width,
        height,
        origin_x,
        origin_y,
        unit_width,
        unit_height,
        sgr_index,
        xqd,
        8,
        scratch,
    );
}

#[expect(
    clippy::too_many_arguments,
    reason = "scalar SGRPROJ kernel parameters and reusable scratch stay explicit"
)]
pub(crate) fn sgrproj_filter_unit_into_with_scratch_bit_depth(
    source: &[u16],
    output: &mut [u16],
    width: usize,
    height: usize,
    origin_x: usize,
    origin_y: usize,
    unit_width: usize,
    unit_height: usize,
    sgr_index: u8,
    xqd: [i16; 2],
    bit_depth: u8,
    scratch: &mut [Vec<i32>; 4],
) {
    sgrproj_filter_unit_into_with_scratch_bit_depth_visible(
        source,
        output,
        width,
        height,
        width,
        height,
        origin_x,
        origin_y,
        unit_width,
        unit_height,
        sgr_index,
        xqd,
        bit_depth,
        scratch,
    );
}

#[expect(
    clippy::too_many_arguments,
    reason = "scalar SGRPROJ kernel keeps coded stride and visible bounds explicit"
)]
pub(crate) fn sgrproj_filter_unit_into_with_scratch_bit_depth_visible(
    source: &[u16],
    output: &mut [u16],
    width: usize,
    _height: usize,
    visible_width: usize,
    visible_height: usize,
    origin_x: usize,
    origin_y: usize,
    unit_width: usize,
    unit_height: usize,
    sgr_index: u8,
    xqd: [i16; 2],
    bit_depth: u8,
    scratch: &mut [Vec<i32>; 4],
) {
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
    let output_width = unit_width.min(visible_width.saturating_sub(origin_x));
    let output_height = unit_height.min(visible_height.saturating_sub(origin_y));
    if output_width == 0 || output_height == 0 {
        return;
    }
    let sample = |x: isize, y: isize| {
        restoration_sample_with_visible_bounds(
            source,
            width,
            visible_width,
            visible_height,
            x,
            y,
            origin_y,
            output_height,
        )
    };
    let round_shift = |value: i64, shift: u32| -> i32 {
        ((value + (1_i64 << shift.saturating_sub(1))) >> shift) as i32
    };
    let round_shift_i64 =
        |value: i64, shift: u32| -> i64 { (value + (1_i64 << shift.saturating_sub(1))) >> shift };
    let bd_shift = u32::from(bit_depth.saturating_sub(8));
    let max_sample = ((1_u32 << u32::from(bit_depth.min(16))) - 1) as i32;
    let intermediate = |radius: usize, scale: i32, a: &mut Vec<i32>, b: &mut Vec<i32>| -> usize {
        let stride = output_width + 4;
        let scratch_len = (output_height + 4) * stride;
        a.resize(scratch_len, 0);
        b.resize(scratch_len, 0);
        let side = radius * 2 + 1;
        let n = (side * side) as i64;
        if bit_depth == 8 {
            let n = n as i32;
            for yy in 0..output_height + 4 {
                for xx in 0..output_width + 4 {
                    let cx = xx as isize + origin_x as isize - 2;
                    let cy = yy as isize + origin_y as isize - 2;
                    let mut sum = 0i32;
                    let mut sum_sq = 0i32;
                    for dy in -(radius as isize)..=(radius as isize) {
                        for dx in -(radius as isize)..=(radius as isize) {
                            let value = sample(cx + dx, cy + dy);
                            sum += value;
                            sum_sq += value * value;
                        }
                    }
                    let p = (sum_sq * n - sum * sum).max(0) as i64;
                    let z = round_shift(p * i64::from(scale), 20).clamp(0, 255);
                    let af = sgr_x_by_xplus1(z);
                    let bf = round_shift(
                        i64::from(256 - af) * i64::from(sum) * i64::from((4096 + n / 2) / n),
                        12,
                    );
                    let index = yy * stride + xx;
                    a[index] = af;
                    b[index] = bf;
                }
            }
            return stride;
        }
        for yy in 0..output_height + 4 {
            for xx in 0..output_width + 4 {
                let cx = xx as isize + origin_x as isize - 2;
                let cy = yy as isize + origin_y as isize - 2;
                let mut sum = 0i64;
                let mut sum_sq = 0i64;
                for dy in -(radius as isize)..=(radius as isize) {
                    for dx in -(radius as isize)..=(radius as isize) {
                        let value = sample(cx + dx, cy + dy);
                        let value = i64::from(value);
                        sum += value;
                        sum_sq += value * value;
                    }
                }
                // AOM normalizes the highbd box sums before calculating the
                // variance. This keeps the fixed-point range identical to
                // the 8-bit kernel while avoiding overflow in sum_sq * n.
                let normalized_sum = round_shift_i64(sum, bd_shift);
                let normalized_sum_sq = round_shift_i64(sum_sq, bd_shift * 2);
                let p = (normalized_sum_sq * n - normalized_sum * normalized_sum).max(0);
                let z = round_shift_i64(p * i64::from(scale), 20).clamp(0, 255) as i32;
                // AOM's av1_x_by_xplus1 table maps zero to one and rounds
                // 256*z/(z+1) to nearest for every other variance value.
                let af = sgr_x_by_xplus1(z);
                let bf = round_shift(
                    i64::from(256 - af) * normalized_sum * ((4096 + n / 2) / n),
                    12,
                );
                let index = yy * stride + xx;
                a[index] = af;
                b[index] = bf;
            }
        }
        stride
    };
    let stride0 = if RADII[index][0] == 0 {
        0
    } else {
        let (a, b) = scratch.split_at_mut(1);
        intermediate(RADII[index][0], S[index][0], &mut a[0], &mut b[0])
    };
    let stride1 = if RADII[index][1] == 0 {
        0
    } else {
        let (a, b) = scratch.split_at_mut(3);
        intermediate(RADII[index][1], S[index][1], &mut a[2], &mut b[0])
    };
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
            return sample((origin_x + x) as isize, (origin_y + y) as isize) << 4;
        }
        let (a, b, stride) = if radius_index == 0 {
            (&scratch[0], &scratch[1], stride0)
        } else {
            (&scratch[2], &scratch[3], stride1)
        };
        let k = (y + 2) * stride + x + 2;
        let pixel = sample((origin_x + x) as isize, (origin_y + y) as isize);
        if radius == 2 {
            let aa = if y & 1 == 0 {
                (a[k - stride] + a[k + stride]) * 6
                    + (a[k - stride - 1]
                        + a[k - stride + 1]
                        + a[k + stride - 1]
                        + a[k + stride + 1])
                        * 5
            } else {
                a[k] * 6 + (a[k - 1] + a[k + 1]) * 5
            };
            let bb = if y & 1 == 0 {
                (b[k - stride] + b[k + stride]) * 6
                    + (b[k - stride - 1]
                        + b[k - stride + 1]
                        + b[k + stride - 1]
                        + b[k + stride + 1])
                        * 5
            } else {
                b[k] * 6 + (b[k - 1] + b[k + 1]) * 5
            };
            let shift = if y & 1 == 0 { 9 } else { 8 };
            round_shift(i64::from(aa) * i64::from(pixel) + i64::from(bb), shift)
        } else {
            let aa = (a[k] + a[k - 1] + a[k + 1] + a[k - stride] + a[k + stride]) * 4
                + (a[k - stride - 1] + a[k - stride + 1] + a[k + stride - 1] + a[k + stride + 1])
                    * 3;
            let bb = (b[k] + b[k - 1] + b[k + 1] + b[k - stride] + b[k + stride]) * 4
                + (b[k - stride - 1] + b[k - stride + 1] + b[k + stride - 1] + b[k + stride + 1])
                    * 3;
            round_shift(i64::from(aa) * i64::from(pixel) + i64::from(bb), 9)
        }
    };
    for local_y in 0..output_height {
        for local_x in 0..output_width {
            let x = origin_x + local_x;
            let y = origin_y + local_y;
            let u = sample(x as isize, y as isize) << 4;
            let f0 = filter_at(local_x, local_y, 0);
            let f1 = filter_at(local_x, local_y, 1);
            let value = (u << 7) + xq0 * (f0 - u) + xq1 * (f1 - u);
            output[y * width + x] = ((value + (1 << 10)) >> 11).clamp(0, max_sample) as u16;
        }
    }
}

fn sgr_x_by_xplus1(z: i32) -> i32 {
    if z == 0 {
        1
    } else if z >= 255 {
        256
    } else {
        ((256 * z + (z + 1) / 2) / (z + 1)).clamp(1, 256)
    }
}

#[allow(dead_code)]
#[expect(
    clippy::too_many_arguments,
    reason = "scalar deblock kernel parameters mirror the normative edge inputs"
)]
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
    deblock_filter_edge_with_length(
        samples, width, height, edge_x, edge_y, vertical, level, sharpness, bit_depth, 4,
    );
}

#[expect(
    clippy::too_many_arguments,
    reason = "scalar deblock kernel parameters mirror the normative edge inputs"
)]
pub(crate) fn deblock_filter_edge_with_length(
    samples: &mut [u16],
    width: usize,
    height: usize,
    edge_x: usize,
    edge_y: usize,
    vertical: bool,
    level: u8,
    sharpness: u8,
    bit_depth: u8,
    filter_length: u8,
) {
    deblock_filter_edge_with_visible_bounds(
        samples,
        width,
        height,
        width,
        height,
        edge_x,
        edge_y,
        vertical,
        level,
        sharpness,
        bit_depth,
        filter_length,
    );
}

#[expect(
    clippy::too_many_arguments,
    reason = "scalar deblock kernel parameters mirror the normative edge inputs"
)]
pub(crate) fn deblock_filter_edge_with_visible_bounds(
    samples: &mut [u16],
    width: usize,
    _height: usize,
    visible_width: usize,
    visible_height: usize,
    edge_x: usize,
    edge_y: usize,
    vertical: bool,
    level: u8,
    sharpness: u8,
    bit_depth: u8,
    filter_length: u8,
) {
    let radius = match filter_length {
        14 => 6,
        8 => 3,
        6 => 2,
        _ => 1,
    };
    if level == 0
        || (vertical && (edge_x < radius || edge_x + radius >= visible_width))
        || (!vertical && (edge_y < radius || edge_y + radius >= visible_height))
    {
        return;
    }
    let shift = u32::from(bit_depth.saturating_sub(8));
    let mut inside = i32::from(level) >> u32::from((sharpness > 0) as u8 + (sharpness > 4) as u8);
    if sharpness > 0 {
        inside = inside.min(9 - i32::from(sharpness));
    }
    inside = inside.max(1) << shift;
    let blimit = (2 * (i32::from(level) + 2) + (inside >> shift)) << shift;
    let limit = inside;
    let hev = (i32::from(level) >> 4) << shift;
    let max_sample = ((1u32 << bit_depth.min(16)) - 1) as i32;
    for lane in 0..4 {
        let (lane_x, lane_y, p1, p0, q0, q1) = if vertical {
            let x = edge_x;
            let y = edge_y + lane;
            if y >= visible_height {
                continue;
            }
            (
                x,
                y,
                samples[y * width + x - 2] as i32,
                samples[y * width + x - 1] as i32,
                samples[y * width + x] as i32,
                samples[y * width + x + 1] as i32,
            )
        } else {
            let x = edge_x + lane;
            let y = edge_y;
            if x >= visible_width {
                continue;
            }
            (
                x,
                y,
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
        let filter_length = filter_length.min(14);
        let (p2, q2) = if filter_length >= 6 {
            if vertical {
                (
                    samples[lane_y * width + lane_x - 3] as i32,
                    samples[lane_y * width + lane_x + 2] as i32,
                )
            } else {
                (
                    samples[(lane_y - 3) * width + lane_x] as i32,
                    samples[(lane_y + 2) * width + lane_x] as i32,
                )
            }
        } else {
            (0, 0)
        };
        if filter_length >= 8 {
            let (p3, q3) = if vertical {
                (
                    samples[lane_y * width + lane_x - 4] as i32,
                    samples[lane_y * width + lane_x + 3] as i32,
                )
            } else {
                (
                    samples[(lane_y - 4) * width + lane_x] as i32,
                    samples[(lane_y + 3) * width + lane_x] as i32,
                )
            };
            let mask8 = i32::abs(p3 - p2) <= limit
                && i32::abs(p2 - p1) <= limit
                && i32::abs(p1 - p0) <= limit
                && i32::abs(q1 - q0) <= limit
                && i32::abs(q2 - q1) <= limit
                && i32::abs(q3 - q2) <= limit
                && 2 * i32::abs(p0 - q0) + i32::abs(p1 - q1) / 2 <= blimit;
            if !mask8 {
                continue;
            }
            let flat8 = i32::abs(p1 - p0) <= 1
                && i32::abs(q1 - q0) <= 1
                && i32::abs(p2 - p0) <= 1
                && i32::abs(q2 - q0) <= 1
                && i32::abs(p3 - p0) <= 1
                && i32::abs(q3 - q0) <= 1;
            if filter_length == 14 {
                let (p4, p5, p6, q4, q5, q6) = if vertical {
                    (
                        samples[lane_y * width + lane_x - 5] as i32,
                        samples[lane_y * width + lane_x - 6] as i32,
                        samples[lane_y * width + lane_x - 7] as i32,
                        samples[lane_y * width + lane_x + 4] as i32,
                        samples[lane_y * width + lane_x + 5] as i32,
                        samples[lane_y * width + lane_x + 6] as i32,
                    )
                } else {
                    (
                        samples[(lane_y - 5) * width + lane_x] as i32,
                        samples[(lane_y - 6) * width + lane_x] as i32,
                        samples[(lane_y - 7) * width + lane_x] as i32,
                        samples[(lane_y + 4) * width + lane_x] as i32,
                        samples[(lane_y + 5) * width + lane_x] as i32,
                        samples[(lane_y + 6) * width + lane_x] as i32,
                    )
                };
                let flat2 = i32::abs(p4 - p0) <= 1
                    && i32::abs(q4 - q0) <= 1
                    && i32::abs(p5 - p0) <= 1
                    && i32::abs(q5 - q0) <= 1
                    && i32::abs(p6 - p0) <= 1
                    && i32::abs(q6 - q0) <= 1;
                if mask8 && flat8 && flat2 {
                    let filtered = [
                        (p6 * 7 + p5 * 2 + p4 * 2 + p3 + p2 + p1 + p0 + q0 + 8) >> 4,
                        (p6 * 5 + p5 * 2 + p4 * 2 + p3 * 2 + p2 + p1 + p0 + q0 + q1 + 8) >> 4,
                        (p6 * 4 + p5 + p4 * 2 + p3 * 2 + p2 * 2 + p1 + p0 + q0 + q1 + q2 + 8) >> 4,
                        (p6 * 3 + p5 + p4 + p3 * 2 + p2 * 2 + p1 * 2 + p0 + q0 + q1 + q2 + q3 + 8)
                            >> 4,
                        (p6 * 2
                            + p5
                            + p4
                            + p3
                            + p2 * 2
                            + p1 * 2
                            + p0 * 2
                            + q0
                            + q1
                            + q2
                            + q3
                            + q4
                            + 8)
                            >> 4,
                        (p6 + p5
                            + p4
                            + p3
                            + p2
                            + p1 * 2
                            + p0 * 2
                            + q0 * 2
                            + q1
                            + q2
                            + q3
                            + q4
                            + q5
                            + 8)
                            >> 4,
                        (p5 + p4
                            + p3
                            + p2
                            + p1
                            + p0 * 2
                            + q0 * 2
                            + q1 * 2
                            + q2
                            + q3
                            + q4
                            + q5
                            + q6
                            + 8)
                            >> 4,
                        (p4 + p3
                            + p2
                            + p1
                            + p0
                            + q0 * 2
                            + q1 * 2
                            + q2 * 2
                            + q3
                            + q4
                            + q5
                            + q6 * 2
                            + 8)
                            >> 4,
                        (p3 + p2 + p1 + p0 + q0 + q1 * 2 + q2 * 2 + q3 * 2 + q4 + q5 + q6 * 3 + 8)
                            >> 4,
                        (p2 + p1 + p0 + q0 + q1 + q2 * 2 + q3 * 2 + q4 * 2 + q5 + q6 * 4 + 8) >> 4,
                        (p1 + p0 + q0 + q1 + q2 + q3 * 2 + q4 * 2 + q5 * 2 + q6 * 5 + 8) >> 4,
                        (p0 + q0 + q1 + q2 + q3 + q4 * 2 + q5 * 2 + q6 * 7 + 8) >> 4,
                    ];
                    if vertical {
                        for (offset, value) in filtered.into_iter().enumerate() {
                            samples[lane_y * width + lane_x - 6 + offset] =
                                value.clamp(0, max_sample) as u16;
                        }
                    } else {
                        for (offset, value) in filtered.into_iter().enumerate() {
                            samples[(lane_y - 6 + offset) * width + lane_x] =
                                value.clamp(0, max_sample) as u16;
                        }
                    }
                    continue;
                }
            }
            if mask8 && flat8 {
                let filtered = [
                    (p3 + p3 + p3 + 2 * p2 + p1 + p0 + q0 + 4) >> 3,
                    (p3 + p3 + p2 + 2 * p1 + p0 + q0 + q1 + 4) >> 3,
                    (p3 + p2 + p1 + 2 * p0 + q0 + q1 + q2 + 4) >> 3,
                    (p2 + p1 + p0 + 2 * q0 + q1 + q2 + q3 + 4) >> 3,
                    (p1 + p0 + q0 + 2 * q1 + q2 + q3 + q3 + 4) >> 3,
                    (p0 + q0 + q1 + 2 * q2 + q3 + q3 + q3 + 4) >> 3,
                ];
                if vertical {
                    for (offset, value) in filtered.into_iter().enumerate() {
                        samples[lane_y * width + lane_x - 3 + offset] =
                            value.clamp(0, max_sample) as u16;
                    }
                } else {
                    for (offset, value) in filtered.into_iter().enumerate() {
                        samples[(lane_y - 3 + offset) * width + lane_x] =
                            value.clamp(0, max_sample) as u16;
                    }
                }
                continue;
            }
        }
        if filter_length >= 6 {
            let mask6 = i32::abs(p2 - p1) <= limit
                && i32::abs(p1 - p0) <= limit
                && i32::abs(q1 - q0) <= limit
                && i32::abs(q2 - q1) <= limit
                && 2 * i32::abs(p0 - q0) + i32::abs(p1 - q1) / 2 <= blimit;
            let flat6 = i32::abs(p2 - p0) <= 1
                && i32::abs(p1 - p0) <= 1
                && i32::abs(q1 - q0) <= 1
                && i32::abs(q2 - q0) <= 1;
            if filter_length == 6 && !mask6 {
                continue;
            }
            if filter_length == 6 && mask6 && flat6 {
                let filtered = [
                    (p2 * 3 + p1 * 2 + p0 * 2 + q0 + 4) >> 3,
                    (p2 + p1 * 2 + p0 * 2 + q0 * 2 + q1 + 4) >> 3,
                    (p1 + p0 * 2 + q0 * 2 + q1 * 2 + q2 + 4) >> 3,
                    (p0 + q0 * 2 + q1 * 2 + q2 * 3 + 4) >> 3,
                ];
                if vertical {
                    samples[lane_y * width + lane_x - 2] = filtered[0].clamp(0, max_sample) as u16;
                    samples[lane_y * width + lane_x - 1] = filtered[1].clamp(0, max_sample) as u16;
                    samples[lane_y * width + lane_x] = filtered[2].clamp(0, max_sample) as u16;
                    samples[lane_y * width + lane_x + 1] = filtered[3].clamp(0, max_sample) as u16;
                } else {
                    samples[(lane_y - 2) * width + lane_x] =
                        filtered[0].clamp(0, max_sample) as u16;
                    samples[(lane_y - 1) * width + lane_x] =
                        filtered[1].clamp(0, max_sample) as u16;
                    samples[lane_y * width + lane_x] = filtered[2].clamp(0, max_sample) as u16;
                    samples[(lane_y + 1) * width + lane_x] =
                        filtered[3].clamp(0, max_sample) as u16;
                }
                continue;
            }
        }
        // AOM clamps the outer-tap contribution to signed-byte range before
        // adding the inner-tap term; clamping only the final sum changes the
        // rounding for high-contrast edges.
        let outer_filter = (p1 - q1).clamp(-128, 127) * hev_mask;
        let mut filter = outer_filter + 3 * (q0 - p0);
        filter = filter.clamp(-128 << shift, 127 << shift);
        let f1 = (filter + 4).clamp(-128, 127) >> 3;
        let f2 = (filter + 3).clamp(-128, 127) >> 3;
        let np0 = (p0 + f2).clamp(0, max_sample) as u16;
        let nq0 = (q0 - f1).clamp(0, max_sample) as u16;
        let outer = if hev_mask == 0 { (f1 + 1) >> 1 } else { 0 };
        if vertical {
            samples[lane_y * width + lane_x - 2] = (p1 + outer).clamp(0, max_sample) as u16;
            samples[lane_y * width + lane_x - 1] = np0;
            samples[lane_y * width + lane_x] = nq0;
            samples[lane_y * width + lane_x + 1] = (q1 - outer).clamp(0, max_sample) as u16;
        } else {
            samples[(lane_y - 2) * width + lane_x] = (p1 + outer).clamp(0, max_sample) as u16;
            samples[(lane_y - 1) * width + lane_x] = np0;
            samples[lane_y * width + lane_x] = nq0;
            samples[(lane_y + 1) * width + lane_x] = (q1 - outer).clamp(0, max_sample) as u16;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CdefUnit {
    pub(crate) x: usize,
    pub(crate) y: usize,
    pub(crate) index: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CdefBlockIndex {
    pub(crate) x: usize,
    pub(crate) y: usize,
    pub(crate) index: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TransformBoundary {
    pub(crate) block: TransformBlock,
    pub(crate) tx_type: TxType,
    pub(crate) non_zero_coefficients: usize,
    pub(crate) skip: bool,
    pub(crate) is_inter: bool,
    pub(crate) reference_frame: Option<u8>,
    pub(crate) has_nonzero_mv: bool,
    pub(crate) y_mode: PredictionMode,
    pub(crate) uv_mode: Option<UvPredictionMode>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BlockFilterState {
    pub(crate) x: usize,
    pub(crate) y: usize,
    pub(crate) block_size: BlockSize,
    pub(crate) segment_id: u8,
    pub(crate) skip: bool,
    pub(crate) is_inter: bool,
    pub(crate) reference_frame: Option<u8>,
    pub(crate) has_nonzero_mv: bool,
    pub(crate) y_mode: PredictionMode,
    pub(crate) uv_mode: Option<UvPredictionMode>,
    pub(crate) delta_lf: [i8; 4],
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct PostFilterState {
    pub(crate) cdef_units: Vec<CdefUnit>,
    pub(crate) cdef_blocks: Vec<CdefBlockIndex>,
    pub(crate) transform_boundaries: Vec<TransformBoundary>,
    pub(crate) restoration_units: Vec<RestorationUnit>,
    pub(crate) block_filter_states: Vec<BlockFilterState>,
}

impl PostFilterState {
    pub(crate) fn merge(&mut self, other: Self) {
        if self.is_empty() {
            *self = other;
            return;
        }
        if other.is_empty() {
            return;
        }
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
        for block in other.cdef_blocks {
            if let Some(existing) = self
                .cdef_blocks
                .iter_mut()
                .find(|existing| existing.x == block.x && existing.y == block.y)
            {
                *existing = block;
            } else {
                self.cdef_blocks.push(block);
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
        for state in other.block_filter_states {
            if let Some(existing) = self
                .block_filter_states
                .iter_mut()
                .find(|existing| existing.x == state.x && existing.y == state.y)
            {
                *existing = state;
            } else {
                self.block_filter_states.push(state);
            }
        }
    }

    fn is_empty(&self) -> bool {
        self.cdef_units.is_empty()
            && self.cdef_blocks.is_empty()
            && self.transform_boundaries.is_empty()
            && self.restoration_units.is_empty()
            && self.block_filter_states.is_empty()
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
                    skip: false,
                    is_inter: false,
                    reference_frame: None,
                    has_nonzero_mv: false,
                    y_mode: PredictionMode::Dc,
                    uv_mode: Some(UvPredictionMode::Intra(PredictionMode::Dc)),
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
            cdef_blocks: self.cdef_blocks,
            transform_boundaries: self.transform_boundaries,
            restoration_units: self.restoration_units,
            block_filter_states: self.block_filter_states,
        }
    }

    pub(super) fn record_block_filter_state(
        &mut self,
        x: usize,
        y: usize,
        block_size: BlockSize,
        segment_id: u8,
        skip: bool,
        is_inter: bool,
        reference_frame: Option<u8>,
        has_nonzero_mv: bool,
        y_mode: PredictionMode,
        uv_mode: Option<UvPredictionMode>,
        delta_lf: [i8; 4],
    ) {
        self.block_filter_states.push(BlockFilterState {
            x,
            y,
            block_size,
            segment_id,
            skip,
            is_inter,
            reference_frame,
            has_nonzero_mv,
            y_mode,
            uv_mode,
            delta_lf,
        });
    }

    pub(super) fn record_transform_boundary(
        &mut self,
        block: TransformBlock,
        tx_type: TxType,
        non_zero_coefficients: usize,
        skip: bool,
        is_inter: bool,
        reference_frame: Option<u8>,
        has_nonzero_mv: bool,
        y_mode: PredictionMode,
        uv_mode: Option<UvPredictionMode>,
    ) {
        self.transform_boundaries.push(TransformBoundary {
            block,
            tx_type,
            non_zero_coefficients,
            skip,
            is_inter,
            reference_frame,
            has_nonzero_mv,
            y_mode,
            uv_mode,
        });
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
        if let Some(index) = index {
            store_cdef_block_index(&mut self.cdef_blocks, unit_x, unit_y, index);
        }
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

fn store_cdef_block_index(blocks: &mut Vec<CdefBlockIndex>, x: usize, y: usize, index: u32) {
    if let Some(existing) = blocks
        .iter_mut()
        .find(|existing| existing.x == x && existing.y == y)
    {
        existing.index = index;
    } else {
        blocks.push(CdefBlockIndex { x, y, index });
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CDEF_DIRECTIONS, CdefUnit, PostFilterState, TransformBoundary,
        cdef_adjust_primary_strength, cdef_chroma_direction, cdef_constrain, cdef_filter_block,
        cdef_filter_block_region_with_edge_mode, cdef_filter_block_region_with_edge_mode_into,
        cdef_filter_block_region_with_edge_mode_into_bit_depth, cdef_find_direction,
        cdef_unit_origin, deblock_filter_edge, deblock_filter_edge_with_length,
        deblock_filter_edge_with_visible_bounds, restoration_sample, sgr_x_by_xplus1,
        store_cdef_unit,
    };
    use crate::av1::syntax::{BlockSize, PredictionMode, TxSize, TxType, UvPredictionMode};
    use crate::av1::transform::TransformBlock;

    #[test]
    fn cdef_filter_keeps_constant_block_unchanged() {
        let source = vec![128u16; 16 * 16];
        let filtered = cdef_filter_block(&source, 16, 16, 4, 4, 8, 8, 0, 4, 2, 3);
        assert_eq!(filtered, source);
    }

    #[test]
    fn cdef_into_kernel_matches_allocating_wrapper() {
        let source = (0..32 * 32)
            .map(|index| ((index * 17 + 9) & 255) as u16)
            .collect::<Vec<_>>();
        let expected =
            cdef_filter_block_region_with_edge_mode(&source, 32, 32, 4, 8, 8, 8, 3, 5, 2, 3, true);
        let mut actual = vec![0u16; 64];
        cdef_filter_block_region_with_edge_mode_into(
            &source,
            32,
            32,
            4,
            8,
            8,
            8,
            3,
            5,
            2,
            3,
            true,
            &mut actual,
        );
        assert_eq!(actual, expected);
    }

    #[test]
    fn cdef_high_bit_depth_scales_strengths_and_damping() {
        let source = (0..32 * 32)
            .map(|index| 2048 + ((index * 3) % 128) as u16)
            .collect::<Vec<_>>();
        let unscaled =
            cdef_filter_block_region_with_edge_mode(&source, 32, 32, 8, 8, 8, 8, 3, 3, 2, 3, true);
        let mut scaled = vec![0u16; 64];
        cdef_filter_block_region_with_edge_mode_into_bit_depth(
            &source,
            32,
            32,
            8,
            8,
            8,
            8,
            3,
            3,
            2,
            3,
            2,
            true,
            &mut scaled,
        );

        assert_ne!(scaled, unscaled);
        assert!(scaled.iter().all(|&sample| sample <= 4095));
    }

    #[test]
    fn cdef_direction_is_stable_for_a_constant_block() {
        let source = vec![128u16; 8 * 8];
        assert_eq!(cdef_find_direction(&source, 8, 8, 0, 0, 0), 0);
    }

    #[test]
    fn cdef_direction_handles_high_bit_depth_without_overflow() {
        let source = (0..64)
            .map(|index| 2048u16 + ((index * 37) % 1024) as u16)
            .collect::<Vec<_>>();
        let (direction, variance) =
            super::cdef_find_direction_with_variance(&source, 8, 8, 0, 0, 0, false);
        assert!(direction < 8);
        assert!(variance >= 0);
    }

    #[test]
    fn cdef_direction_uses_sentinel_for_partial_frame_edges() {
        let source = (0..5 * 5)
            .map(|index| ((index * 17 + index / 5 * 9) & 255) as u16)
            .collect::<Vec<_>>();
        let clamped = super::cdef_find_direction_with_variance(&source, 5, 5, 0, 0, 0, false);
        let sentinel = super::cdef_find_direction_with_variance(&source, 5, 5, 0, 0, 0, true);
        assert_ne!(clamped, sentinel);
    }

    #[test]
    fn cdef_directions_match_aom_axis_order() {
        assert_eq!(CDEF_DIRECTIONS[2], [(0, 1), (0, 2)]);
        assert_eq!(CDEF_DIRECTIONS[6], [(1, 0), (2, 0)]);
    }

    #[test]
    fn cdef_primary_strength_uses_directional_variance() {
        assert_eq!(cdef_adjust_primary_strength(8, 0), 0);
        assert_eq!(cdef_adjust_primary_strength(8, 64), 2);
        assert_eq!(cdef_adjust_primary_strength(8, 4096), 5);
        assert_eq!(cdef_adjust_primary_strength(16, 64), 4);
        assert_eq!(cdef_adjust_primary_strength(16, 4096), 10);
    }

    #[test]
    fn cdef_scaled_strength_wrapper_matches_bit_depth_wrapper() {
        let source = (0..16 * 16)
            .map(|index| ((index * 29 + index / 16 * 7) & 1023) as u16)
            .collect::<Vec<_>>();
        let mut wrapper = vec![0u16; 64];
        let mut scaled = vec![0u16; 64];
        super::cdef_filter_block_region_with_edge_mode_into_bit_depth_visible(
            &source,
            16,
            16,
            13,
            12,
            5,
            4,
            8,
            8,
            3,
            3,
            2,
            3,
            2,
            true,
            &mut wrapper,
        );
        super::cdef_filter_block_region_with_edge_mode_into_bit_depth_visible_scaled(
            &source,
            16,
            16,
            13,
            12,
            5,
            4,
            8,
            8,
            3,
            12,
            8,
            5,
            2,
            true,
            &mut scaled,
        );
        assert_eq!(wrapper, scaled);
    }

    #[test]
    fn cdef_chroma_direction_maps_asymmetric_subsampling() {
        let directions = 0..8;
        assert_eq!(
            directions
                .clone()
                .map(|direction| cdef_chroma_direction(direction, true, false))
                .collect::<Vec<_>>(),
            vec![7, 0, 2, 4, 5, 6, 6, 6]
        );
        assert_eq!(
            directions
                .clone()
                .map(|direction| cdef_chroma_direction(direction, false, true))
                .collect::<Vec<_>>(),
            vec![1, 2, 2, 2, 3, 4, 6, 0]
        );
        assert_eq!(
            directions
                .map(|direction| cdef_chroma_direction(direction, true, true))
                .collect::<Vec<_>>(),
            (0..8).collect::<Vec<_>>()
        );
    }

    #[test]
    fn wiener_filter_preserves_a_constant_unit_for_normalized_taps() {
        let source = vec![200u16; 16 * 16];
        let filtered =
            super::wiener_filter_unit(&source, 16, 16, 0, 0, 16, 16, [[1, 2, 3], [3, -2, 1]]);
        assert_eq!(filtered, source);
    }

    #[test]
    fn wiener_zero_residual_filter_is_identity() {
        let source = (0..16 * 16)
            .map(|index| ((index * 37 + 11) & 255) as u16)
            .collect::<Vec<_>>();
        let filtered = super::wiener_filter_unit(&source, 16, 16, 0, 0, 16, 16, [[0; 3]; 2]);
        assert_eq!(filtered, source);
    }

    #[test]
    fn wiener_filter_treats_first_transmitted_axis_as_vertical() {
        let source = (0..16)
            .flat_map(|_| (0..16).map(|x| (x * 13) as u16))
            .collect::<Vec<_>>();
        let filtered =
            super::wiener_filter_unit(&source, 16, 16, 0, 0, 16, 16, [[0, 0, 8], [0; 3]]);
        assert_eq!(filtered, source);
    }

    #[test]
    fn wiener_high_bit_depth_clamps_to_the_declared_sample_range() {
        let source = (0..32 * 32)
            .map(|index| 1024u16 + ((index * 197) % 3072) as u16)
            .collect::<Vec<_>>();
        let mut output = source.clone();
        let mut scratch = Vec::new();
        super::wiener_filter_unit_into_with_scratch_bit_depth(
            &source,
            &mut output,
            32,
            32,
            0,
            0,
            32,
            32,
            [[1, 2, 3], [3, -2, 1]],
            12,
            &mut scratch,
        );
        assert!(output.iter().all(|&sample| sample <= 4095));
        assert!(
            output
                .iter()
                .zip(&source)
                .any(|(&filtered, &original)| filtered != original)
        );
    }

    #[test]
    fn sgrproj_filter_preserves_a_constant_unit_with_zero_projection() {
        let source = vec![200u16; 16 * 16];
        let filtered = super::sgrproj_filter_unit(&source, 16, 16, 0, 0, 16, 16, 0, [0, 128]);
        assert_eq!(filtered, source);
    }

    #[test]
    fn sgrproj_high_bit_depth_arithmetic_stays_in_sample_range() {
        let source = (0..32 * 32)
            .map(|index| 1024u16 + ((index * 197) % 3072) as u16)
            .collect::<Vec<_>>();
        let mut output = source.clone();
        let mut scratch = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        super::sgrproj_filter_unit_into_with_scratch_bit_depth(
            &source,
            &mut output,
            32,
            32,
            0,
            0,
            32,
            32,
            0,
            [12, 64],
            12,
            &mut scratch,
        );
        assert!(output.iter().all(|&sample| sample <= 4095));
        assert!(
            output
                .iter()
                .zip(&source)
                .any(|(&filtered, &original)| filtered != original)
        );
    }

    #[test]
    fn restoration_into_kernels_match_allocating_wrappers() {
        let source = (0..32 * 32)
            .map(|index| ((index * 29 + 17) & 255) as u16)
            .collect::<Vec<_>>();
        let expected_wiener =
            super::wiener_filter_unit(&source, 32, 32, 4, 8, 16, 16, [[1, 2, 3], [3, -2, 1]]);
        let mut actual_wiener = source.clone();
        super::wiener_filter_unit_into(
            &source,
            &mut actual_wiener,
            32,
            32,
            4,
            8,
            16,
            16,
            [[1, 2, 3], [3, -2, 1]],
        );
        assert_eq!(actual_wiener, expected_wiener);

        let mut reused_wiener = source.clone();
        let mut wiener_scratch = Vec::new();
        super::wiener_filter_unit_into_with_scratch(
            &source,
            &mut reused_wiener,
            32,
            32,
            4,
            8,
            16,
            16,
            [[1, 2, 3], [3, -2, 1]],
            &mut wiener_scratch,
        );
        assert_eq!(reused_wiener, expected_wiener);

        let expected_sgr = super::sgrproj_filter_unit(&source, 32, 32, 4, 8, 16, 16, 0, [12, 64]);
        let mut actual_sgr = source.clone();
        super::sgrproj_filter_unit_into(
            &source,
            &mut actual_sgr,
            32,
            32,
            4,
            8,
            16,
            16,
            0,
            [12, 64],
        );
        assert_eq!(actual_sgr, expected_sgr);

        let mut reused_sgr = source.clone();
        let mut sgr_scratch = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        super::sgrproj_filter_unit_into_with_scratch(
            &source,
            &mut reused_sgr,
            32,
            32,
            4,
            8,
            16,
            16,
            0,
            [12, 64],
            &mut sgr_scratch,
        );
        assert_eq!(reused_sgr, expected_sgr);
    }

    #[test]
    fn restoration_scratch_covers_maximum_64x64_unit_with_halo() {
        let source = (0..64 * 64)
            .map(|index| ((index * 29 + 17) & 4095) as u16)
            .collect::<Vec<_>>();

        let mut wiener_output = source.clone();
        let mut wiener_scratch = Vec::new();
        super::wiener_filter_unit_into_with_scratch_bit_depth(
            &source,
            &mut wiener_output,
            64,
            64,
            0,
            0,
            64,
            64,
            [[1, 2, 3], [3, -2, 1]],
            12,
            &mut wiener_scratch,
        );
        assert!(wiener_scratch.capacity() >= 64 * (64 + 6));

        let mut sgr_output = source.clone();
        let mut sgr_scratch = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        super::sgrproj_filter_unit_into_with_scratch_bit_depth(
            &source,
            &mut sgr_output,
            64,
            64,
            0,
            0,
            64,
            64,
            0,
            [12, 64],
            12,
            &mut sgr_scratch,
        );
        let expected = (64 + 4) * (64 + 4);
        assert!(
            sgr_scratch
                .iter()
                .all(|scratch| scratch.capacity() >= expected)
        );
    }

    #[test]
    fn sgrproj_x_by_xplus1_matches_aom_rounding_table() {
        assert_eq!(sgr_x_by_xplus1(0), 1);
        assert_eq!(sgr_x_by_xplus1(1), 128);
        assert_eq!(sgr_x_by_xplus1(2), 171);
        assert_eq!(sgr_x_by_xplus1(3), 192);
        assert_eq!(sgr_x_by_xplus1(255), 256);
    }

    #[test]
    fn restoration_sample_matches_aom_stripe_halo_rules() {
        let width = 8;
        let height = 128;
        let source: Vec<u16> = (0..height)
            .flat_map(|row| std::iter::repeat_n(row as u16, width))
            .collect();
        assert_eq!(
            restoration_sample(&source, width, height, 2, 53, 56, 64),
            54
        );
        assert_eq!(
            restoration_sample(&source, width, height, 2, 54, 56, 64),
            54
        );
        assert_eq!(
            restoration_sample(&source, width, height, 2, 55, 56, 64),
            55
        );
        assert_eq!(
            restoration_sample(&source, width, height, 2, 56, 56, 64),
            56
        );
        assert_eq!(
            restoration_sample(&source, width, height, 2, 120, 56, 64),
            120
        );
        assert_eq!(
            restoration_sample(&source, width, height, 2, 121, 56, 64),
            121
        );
        assert_eq!(
            restoration_sample(&source, width, height, 2, 122, 56, 64),
            121
        );
    }

    #[test]
    fn deblock_filter_keeps_constant_edges_unchanged() {
        let mut samples = vec![128u16; 16 * 16];
        deblock_filter_edge(&mut samples, 16, 16, 8, 0, true, 20, 0, 8);
        deblock_filter_edge(&mut samples, 16, 16, 0, 8, false, 20, 0, 8);
        assert!(samples.iter().all(|sample| *sample == 128));
    }

    #[test]
    fn deblock_sharpness_zero_uses_the_full_level_limit() {
        let mut samples = vec![0u16; 16 * 16];
        for y in 0..4 {
            let row = y * 16;
            samples[row + 6] = 210;
            samples[row + 7] = 198;
            samples[row + 8] = 187;
            samples[row + 9] = 187;
        }
        deblock_filter_edge(&mut samples, 16, 16, 8, 0, true, 23, 0, 8);
        assert_eq!(samples[7], 197);
        assert_eq!(samples[8], 188);
    }

    #[test]
    fn deblock_eight_tap_output_starts_at_p3() {
        let mut samples = vec![0u16; 16 * 16];
        for y in 0..4 {
            let row = y * 16;
            for x in 4..8 {
                samples[row + x] = 40;
            }
            for x in 8..12 {
                samples[row + x] = 38;
            }
        }
        deblock_filter_edge_with_length(&mut samples, 16, 16, 8, 0, true, 23, 0, 8, 8);
        assert_eq!(samples[7], 39);
        assert_eq!(samples[8], 39);
    }

    #[test]
    fn deblock_does_not_filter_coded_padding_outside_visible_width() {
        let mut samples = vec![0u16; 16 * 4];
        for y in 0..4 {
            let row = y * 16;
            samples[row + 6] = 210;
            samples[row + 7] = 198;
            samples[row + 8] = 187;
            samples[row + 9] = 187;
        }
        let original = samples.clone();
        deblock_filter_edge_with_visible_bounds(&mut samples, 16, 4, 8, 4, 8, 0, true, 23, 0, 8, 8);
        assert_eq!(samples, original);
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
            cdef_blocks: Vec::new(),
            transform_boundaries: Vec::new(),
            restoration_units: Vec::new(),
            block_filter_states: Vec::new(),
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
            cdef_blocks: Vec::new(),
            transform_boundaries: Vec::new(),
            restoration_units: Vec::new(),
            block_filter_states: Vec::new(),
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
            cdef_blocks: Vec::new(),
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
            block_filter_states: Vec::new(),
        });
        state.merge(PostFilterState {
            cdef_units: Vec::new(),
            cdef_blocks: Vec::new(),
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
            block_filter_states: Vec::new(),
        });
        assert_eq!(state.restoration_units.len(), 1);
        assert_eq!(state.restoration_units[0].restoration_type, 2);
        assert_eq!(state.restoration_units[0].sgrproj, Some([7, 8]));
        assert_eq!(state.restoration_units[0].sgrproj_index, Some(3));
    }

    #[test]
    fn post_filter_state_retains_block_mode_metadata() {
        let mut state = PostFilterState::default();
        state.merge(PostFilterState {
            cdef_units: Vec::new(),
            cdef_blocks: Vec::new(),
            transform_boundaries: Vec::new(),
            restoration_units: Vec::new(),
            block_filter_states: vec![super::BlockFilterState {
                x: 8,
                y: 4,
                block_size: BlockSize::Block8x8,
                segment_id: 0,
                skip: true,
                is_inter: false,
                reference_frame: None,
                has_nonzero_mv: false,
                y_mode: PredictionMode::Dc,
                uv_mode: Some(UvPredictionMode::Intra(PredictionMode::Dc)),
                delta_lf: [0; 4],
            }],
        });
        assert_eq!(state.block_filter_states.len(), 1);
        assert!(state.block_filter_states[0].skip);
        assert_eq!(state.block_filter_states[0].block_size, BlockSize::Block8x8);
    }

    #[test]
    fn post_filter_state_retains_transform_boundary_mode_metadata() {
        let block = TransformBlock {
            plane: 0,
            x: 16,
            y: 8,
            tx_size: TxSize::Tx8x16,
        };
        let mut state = PostFilterState::default();
        state.merge(PostFilterState {
            cdef_units: Vec::new(),
            cdef_blocks: Vec::new(),
            transform_boundaries: vec![TransformBoundary {
                block,
                tx_type: TxType::DctDct,
                non_zero_coefficients: 3,
                skip: true,
                is_inter: true,
                reference_frame: Some(2),
                has_nonzero_mv: true,
                y_mode: PredictionMode::Dc,
                uv_mode: Some(UvPredictionMode::Intra(PredictionMode::Dc)),
            }],
            restoration_units: Vec::new(),
            block_filter_states: Vec::new(),
        });
        assert_eq!(state.transform_boundaries.len(), 1);
        let boundary = &state.transform_boundaries[0];
        assert!(boundary.skip);
        assert!(boundary.is_inter);
        assert_eq!(boundary.reference_frame, Some(2));
        assert!(boundary.has_nonzero_mv);
    }
}
