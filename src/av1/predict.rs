use super::syntax::PredictionMode;
use crate::DecoderError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntraEdges<'a> {
    pub above: Option<&'a [u16]>,
    pub left: Option<&'a [u16]>,
    pub above_left: Option<u16>,
    pub bit_depth: u8,
}

pub fn predict_intra(
    mode: PredictionMode,
    angle_delta: Option<i8>,
    width: usize,
    height: usize,
    edges: IntraEdges<'_>,
) -> Result<Vec<u16>, DecoderError> {
    predict_intra_with_edge_filter(mode, angle_delta, width, height, edges, false)
}

pub(crate) fn predict_intra_with_edge_filter(
    mode: PredictionMode,
    angle_delta: Option<i8>,
    width: usize,
    height: usize,
    edges: IntraEdges<'_>,
    enable_intra_edge_filter: bool,
) -> Result<Vec<u16>, DecoderError> {
    match mode {
        PredictionMode::Dc => Ok(predict_dc(width, height, edges)),
        PredictionMode::Vertical => copy_above(width, height, edges),
        PredictionMode::Horizontal => copy_left(width, height, edges),
        PredictionMode::D45
        | PredictionMode::D67
        | PredictionMode::D113
        | PredictionMode::D135
        | PredictionMode::D157
        | PredictionMode::D203 => predict_directional(
            mode,
            angle_delta.unwrap_or(0),
            width,
            height,
            edges,
            enable_intra_edge_filter,
        ),
        PredictionMode::Smooth => predict_smooth(width, height, edges),
        PredictionMode::SmoothVertical => predict_smooth_vertical(width, height, edges),
        PredictionMode::SmoothHorizontal => predict_smooth_horizontal(width, height, edges),
        PredictionMode::Paeth => predict_paeth(width, height, edges),
    }
}

pub fn predict_filter_intra(
    mode: usize,
    width: usize,
    height: usize,
    edges: IntraEdges<'_>,
) -> Result<Vec<u16>, DecoderError> {
    if mode >= FILTER_INTRA_TAPS.len() {
        return Err(DecoderError::Bitstream(format!(
            "AV1 filter-intra mode {mode} is invalid"
        )));
    }
    if width > 32 || height > 32 {
        return Err(DecoderError::Unsupported(format!(
            "AV1 filter-intra prediction larger than 32x32 is not supported yet: {width}x{height}"
        )));
    }
    let above = edges.above.ok_or_else(|| {
        DecoderError::Bitstream("AV1 filter-intra prediction requires above edge".to_string())
    })?;
    let left = edges.left.ok_or_else(|| {
        DecoderError::Bitstream("AV1 filter-intra prediction requires left edge".to_string())
    })?;
    if above.is_empty() || left.is_empty() {
        return Err(DecoderError::NotEnoughData(
            "AV1 filter-intra edges are empty".to_string(),
        ));
    }

    let mut buffer = [[0u16; 33]; 33];
    let above_left = edges.above_left.unwrap_or(1u16 << (edges.bit_depth - 1));
    buffer[0][0] = above_left;
    for column in 0..width {
        buffer[0][column + 1] = edge_sample(above, column);
    }
    for row in 0..height {
        buffer[row + 1][0] = edge_sample(left, row);
    }

    for row in (1..=height).step_by(2) {
        for column in (1..=width).step_by(4) {
            let p0 = buffer[row - 1][column - 1];
            let p1 = buffer[row - 1][column];
            let p2 = buffer[row - 1][(column + 1).min(width)];
            let p3 = buffer[row - 1][(column + 2).min(width)];
            let p4 = buffer[row - 1][(column + 3).min(width)];
            let p5 = buffer[row][column - 1];
            let p6 = buffer[(row + 1).min(height)][column - 1];
            for k in 0..8 {
                let out_row = row + (k >> 2);
                let out_column = column + (k & 0x03);
                if out_row > height || out_column > width {
                    continue;
                }
                let taps = FILTER_INTRA_TAPS[mode][k];
                let prediction = i32::from(taps[0]) * i32::from(p0)
                    + i32::from(taps[1]) * i32::from(p1)
                    + i32::from(taps[2]) * i32::from(p2)
                    + i32::from(taps[3]) * i32::from(p3)
                    + i32::from(taps[4]) * i32::from(p4)
                    + i32::from(taps[5]) * i32::from(p5)
                    + i32::from(taps[6]) * i32::from(p6);
                buffer[out_row][out_column] = clip1_signed((prediction + 8) >> 4, edges.bit_depth);
            }
        }
    }

    let mut out = Vec::with_capacity(width * height);
    for row in 0..height {
        out.extend_from_slice(&buffer[row + 1][1..=width]);
    }
    Ok(out)
}

const FILTER_INTRA_TAPS: [[[i8; 8]; 8]; 5] = [
    [
        [-6, 10, 0, 0, 0, 12, 0, 0],
        [-5, 2, 10, 0, 0, 9, 0, 0],
        [-3, 1, 1, 10, 0, 7, 0, 0],
        [-3, 1, 1, 2, 10, 5, 0, 0],
        [-4, 6, 0, 0, 0, 2, 12, 0],
        [-3, 2, 6, 0, 0, 2, 9, 0],
        [-3, 2, 2, 6, 0, 2, 7, 0],
        [-3, 1, 2, 2, 6, 3, 5, 0],
    ],
    [
        [-10, 16, 0, 0, 0, 10, 0, 0],
        [-6, 0, 16, 0, 0, 6, 0, 0],
        [-4, 0, 0, 16, 0, 4, 0, 0],
        [-2, 0, 0, 0, 16, 2, 0, 0],
        [-10, 16, 0, 0, 0, 0, 10, 0],
        [-6, 0, 16, 0, 0, 0, 6, 0],
        [-4, 0, 0, 16, 0, 0, 4, 0],
        [-2, 0, 0, 0, 16, 0, 2, 0],
    ],
    [
        [-8, 8, 0, 0, 0, 16, 0, 0],
        [-8, 0, 8, 0, 0, 16, 0, 0],
        [-8, 0, 0, 8, 0, 16, 0, 0],
        [-8, 0, 0, 0, 8, 16, 0, 0],
        [-4, 4, 0, 0, 0, 0, 16, 0],
        [-4, 0, 4, 0, 0, 0, 16, 0],
        [-4, 0, 0, 4, 0, 0, 16, 0],
        [-4, 0, 0, 0, 4, 0, 16, 0],
    ],
    [
        [-2, 8, 0, 0, 0, 10, 0, 0],
        [-1, 3, 8, 0, 0, 6, 0, 0],
        [-1, 2, 3, 8, 0, 4, 0, 0],
        [0, 1, 2, 3, 8, 2, 0, 0],
        [-1, 4, 0, 0, 0, 3, 10, 0],
        [-1, 3, 4, 0, 0, 4, 6, 0],
        [-1, 2, 3, 4, 0, 4, 4, 0],
        [-1, 2, 2, 3, 4, 3, 3, 0],
    ],
    [
        [-12, 14, 0, 0, 0, 14, 0, 0],
        [-10, 0, 14, 0, 0, 12, 0, 0],
        [-9, 0, 0, 14, 0, 11, 0, 0],
        [-8, 0, 0, 0, 14, 10, 0, 0],
        [-10, 12, 0, 0, 0, 0, 14, 0],
        [-9, 1, 12, 0, 0, 0, 12, 0],
        [-8, 0, 0, 12, 0, 1, 11, 0],
        [-7, 0, 0, 1, 12, 1, 9, 0],
    ],
];

fn predict_dc(width: usize, height: usize, edges: IntraEdges<'_>) -> Vec<u16> {
    let value = match (edges.left, edges.above) {
        (Some(left), Some(above)) => {
            let sum: u32 = left
                .iter()
                .take(height)
                .map(|value| u32::from(*value))
                .sum::<u32>()
                + above
                    .iter()
                    .take(width)
                    .map(|value| u32::from(*value))
                    .sum::<u32>()
                + ((width + height) as u32 >> 1);
            (sum / (width + height) as u32) as u16
        }
        (Some(left), None) => {
            let sum: u32 = left
                .iter()
                .take(height)
                .map(|value| u32::from(*value))
                .sum();
            clip1(
                (sum + (height as u32 >> 1)) >> height.trailing_zeros(),
                edges.bit_depth,
            )
        }
        (None, Some(above)) => {
            let sum: u32 = above
                .iter()
                .take(width)
                .map(|value| u32::from(*value))
                .sum();
            clip1(
                (sum + (width as u32 >> 1)) >> width.trailing_zeros(),
                edges.bit_depth,
            )
        }
        (None, None) => 1u16 << (edges.bit_depth - 1),
    };
    vec![value; width * height]
}

fn copy_above(
    width: usize,
    height: usize,
    edges: IntraEdges<'_>,
) -> Result<Vec<u16>, DecoderError> {
    let above = edges.above.ok_or_else(|| {
        DecoderError::Bitstream("AV1 vertical intra prediction requires above edge".to_string())
    })?;
    if above.len() < width {
        return Err(DecoderError::NotEnoughData(
            "AV1 above edge is shorter than prediction width".to_string(),
        ));
    }
    let mut out = Vec::with_capacity(width * height);
    for _ in 0..height {
        out.extend_from_slice(&above[..width]);
    }
    Ok(out)
}

fn copy_left(width: usize, height: usize, edges: IntraEdges<'_>) -> Result<Vec<u16>, DecoderError> {
    let left = edges.left.ok_or_else(|| {
        DecoderError::Bitstream("AV1 horizontal intra prediction requires left edge".to_string())
    })?;
    if left.len() < height {
        return Err(DecoderError::NotEnoughData(
            "AV1 left edge is shorter than prediction height".to_string(),
        ));
    }
    let mut out = Vec::with_capacity(width * height);
    for value in left.iter().take(height) {
        out.extend(std::iter::repeat(*value).take(width));
    }
    Ok(out)
}

fn predict_directional(
    mode: PredictionMode,
    angle_delta: i8,
    width: usize,
    height: usize,
    edges: IntraEdges<'_>,
    enable_intra_edge_filter: bool,
) -> Result<Vec<u16>, DecoderError> {
    let above = edges.above.ok_or_else(|| {
        DecoderError::Bitstream("AV1 directional intra prediction requires above edge".to_string())
    })?;
    let left = edges.left.ok_or_else(|| {
        DecoderError::Bitstream("AV1 directional intra prediction requires left edge".to_string())
    })?;
    if above.is_empty() || left.is_empty() {
        return Err(DecoderError::NotEnoughData(
            "AV1 directional edges are empty".to_string(),
        ));
    }
    let above_left = edges.above_left.unwrap_or(above[0]);
    let base_angle = directional_base_angle(mode).ok_or_else(|| {
        DecoderError::InvalidParam(format!("AV1 mode {mode:?} is not directional"))
    })?;
    let angle = base_angle + i32::from(angle_delta.clamp(-3, 3)) * 3;
    let dx = directional_dx(angle);
    let dy = directional_dy(angle);

    if angle == 90 {
        return copy_above(width, height, edges);
    }
    if angle == 180 {
        return copy_left(width, height, edges);
    }

    let edge_len = width + height;
    let above = DirectionalEdge::new(
        above,
        above_left,
        edge_len,
        use_directional_edge_upsample(width, height, angle - 90, enable_intra_edge_filter),
        edges.bit_depth,
    );
    let left = DirectionalEdge::new(
        left,
        above_left,
        edge_len,
        use_directional_edge_upsample(height, width, angle - 180, enable_intra_edge_filter),
        edges.bit_depth,
    );

    if angle < 90 {
        Ok(predict_directional_zone1(width, height, &above, dx))
    } else if angle < 180 {
        Ok(predict_directional_zone2(
            width, height, &above, &left, dx, dy,
        ))
    } else {
        Ok(predict_directional_zone3(width, height, &left, dy))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DirectionalEdge {
    samples: Vec<u16>,
    offset: i32,
    upsampled: bool,
}

impl DirectionalEdge {
    fn new(source: &[u16], corner: u16, len: usize, upsampled: bool, bit_depth: u8) -> Self {
        let mut edge = source.to_vec();
        edge.resize(len, *source.last().unwrap_or(&corner));
        if !upsampled {
            let mut samples = Vec::with_capacity(edge.len() + 1);
            samples.push(corner);
            samples.extend_from_slice(&edge);
            return Self {
                samples,
                offset: 1,
                upsampled: false,
            };
        }

        let mut input = Vec::with_capacity(edge.len() + 3);
        input.extend_from_slice(&[corner, corner]);
        input.extend_from_slice(&edge);
        input.push(*edge.last().unwrap_or(&corner));
        let mut samples = Vec::with_capacity(edge.len() * 2 + 1);
        samples.push(corner);
        for index in 0..edge.len() {
            let interpolated = -i32::from(input[index])
                + 9 * i32::from(input[index + 1])
                + 9 * i32::from(input[index + 2])
                - i32::from(input[index + 3]);
            samples.push(clip1_signed((interpolated + 8) >> 4, bit_depth));
            samples.push(edge[index]);
        }
        Self {
            samples,
            offset: 2,
            upsampled: true,
        }
    }

    fn sample(&self, index: i32) -> u16 {
        let position =
            (index + self.offset).clamp(0, self.samples.len().saturating_sub(1) as i32) as usize;
        self.samples[position]
    }
}

fn use_directional_edge_upsample(
    first_size: usize,
    second_size: usize,
    delta: i32,
    enabled: bool,
) -> bool {
    let distance = delta.abs();
    enabled && distance != 0 && distance < 40 && first_size + second_size <= 16
}

fn predict_directional_zone1(
    width: usize,
    height: usize,
    above: &DirectionalEdge,
    dx: i32,
) -> Vec<u16> {
    let upsample = i32::from(above.upsampled);
    let max_base = ((width + height - 1) as i32) << upsample;
    let frac_bits = 6 - upsample;
    let base_increment = 1 << upsample;
    let mut out = vec![0; width * height];
    for row in 0..height {
        let x = (row as i32 + 1) * dx;
        let mut base = x >> frac_bits;
        let shift = (((x << upsample) & 0x3f) >> 1) as u32;
        for column in 0..width {
            out[row * width + column] = if base < max_base {
                directional_interpolate(above.sample(base), above.sample(base + 1), shift)
            } else {
                above.sample(max_base)
            };
            base += base_increment;
        }
    }
    out
}

fn predict_directional_zone2(
    width: usize,
    height: usize,
    above: &DirectionalEdge,
    left: &DirectionalEdge,
    dx: i32,
    dy: i32,
) -> Vec<u16> {
    let mut out = vec![0; width * height];
    for row in 0..height {
        for column in 0..width {
            let x = ((column as i32) << 6) - (row as i32 + 1) * dx;
            let above_upsample = i32::from(above.upsampled);
            let base_x = x >> (6 - above_upsample);
            out[row * width + column] = if base_x >= -(1 << above_upsample) {
                directional_interpolate(
                    above.sample(base_x),
                    above.sample(base_x + 1),
                    (((x << above_upsample) & 0x3f) >> 1) as u32,
                )
            } else {
                let y = ((row as i32) << 6) - (column as i32 + 1) * dy;
                let left_upsample = i32::from(left.upsampled);
                let base_y = y >> (6 - left_upsample);
                directional_interpolate(
                    left.sample(base_y),
                    left.sample(base_y + 1),
                    (((y << left_upsample) & 0x3f) >> 1) as u32,
                )
            };
        }
    }
    out
}

fn predict_directional_zone3(
    width: usize,
    height: usize,
    left: &DirectionalEdge,
    dy: i32,
) -> Vec<u16> {
    let upsample = i32::from(left.upsampled);
    let max_base = ((width + height - 1) as i32) << upsample;
    let frac_bits = 6 - upsample;
    let base_increment = 1 << upsample;
    let mut out = vec![0; width * height];
    for column in 0..width {
        let y = (column as i32 + 1) * dy;
        let mut base = y >> frac_bits;
        let shift = (((y << upsample) & 0x3f) >> 1) as u32;
        for row in 0..height {
            out[row * width + column] = if base < max_base {
                directional_interpolate(left.sample(base), left.sample(base + 1), shift)
            } else {
                left.sample(max_base)
            };
            base += base_increment;
        }
    }
    out
}

fn directional_interpolate(first: u16, second: u16, shift: u32) -> u16 {
    ((u32::from(first) * (32 - shift) + u32::from(second) * shift + 16) >> 5) as u16
}

fn directional_base_angle(mode: PredictionMode) -> Option<i32> {
    match mode {
        PredictionMode::D45 => Some(45),
        PredictionMode::D67 => Some(67),
        PredictionMode::D113 => Some(113),
        PredictionMode::D135 => Some(135),
        PredictionMode::D157 => Some(157),
        PredictionMode::D203 => Some(203),
        _ => None,
    }
}

fn directional_dx(angle: i32) -> i32 {
    if angle < 90 {
        DIRECTIONAL_DERIVATIVE[angle as usize]
    } else if angle < 180 {
        DIRECTIONAL_DERIVATIVE[(180 - angle) as usize]
    } else {
        1
    }
}

fn directional_dy(angle: i32) -> i32 {
    if angle > 90 && angle < 180 {
        DIRECTIONAL_DERIVATIVE[(angle - 90) as usize]
    } else if angle > 180 {
        DIRECTIONAL_DERIVATIVE[(270 - angle) as usize]
    } else {
        1
    }
}

const DIRECTIONAL_DERIVATIVE: [i32; 90] = [
    0, 0, 0, 1023, 0, 0, 547, 0, 0, 372, 0, 0, 0, 0, 273, 0, 0, 215, 0, 0, 178, 0, 0, 151, 0, 0,
    132, 0, 0, 116, 0, 0, 102, 0, 0, 0, 90, 0, 0, 80, 0, 0, 71, 0, 0, 64, 0, 0, 57, 0, 0, 51, 0, 0,
    45, 0, 0, 0, 40, 0, 0, 35, 0, 0, 31, 0, 0, 27, 0, 0, 23, 0, 0, 19, 0, 0, 15, 0, 0, 0, 0, 11, 0,
    0, 7, 0, 0, 3, 0, 0,
];

fn predict_smooth(
    width: usize,
    height: usize,
    edges: IntraEdges<'_>,
) -> Result<Vec<u16>, DecoderError> {
    let above = edges.above.ok_or_else(|| {
        DecoderError::Bitstream("AV1 smooth prediction requires above edge".to_string())
    })?;
    let left = edges.left.ok_or_else(|| {
        DecoderError::Bitstream("AV1 smooth prediction requires left edge".to_string())
    })?;
    if above.len() < width || left.len() < height {
        return Err(DecoderError::NotEnoughData(
            "AV1 smooth edges are shorter than prediction block".to_string(),
        ));
    }
    let weights_x = smooth_weights(width)?;
    let weights_y = smooth_weights(height)?;
    let bottom = left[height - 1];
    let right = above[width - 1];
    let mut out = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let sum = u32::from(weights_y[y]) * u32::from(above[x])
                + (256 - u32::from(weights_y[y])) * u32::from(bottom)
                + u32::from(weights_x[x]) * u32::from(left[y])
                + (256 - u32::from(weights_x[x])) * u32::from(right);
            out.push(((sum + 256) >> 9) as u16);
        }
    }
    Ok(out)
}

fn predict_smooth_vertical(
    width: usize,
    height: usize,
    edges: IntraEdges<'_>,
) -> Result<Vec<u16>, DecoderError> {
    let above = edges.above.ok_or_else(|| {
        DecoderError::Bitstream(
            "AV1 smooth vertical intra prediction requires above edge".to_string(),
        )
    })?;
    let left = edges.left.ok_or_else(|| {
        DecoderError::Bitstream(
            "AV1 smooth vertical intra prediction requires left edge".to_string(),
        )
    })?;
    if above.len() < width || left.is_empty() {
        return Err(DecoderError::NotEnoughData(
            "AV1 smooth vertical edges are shorter than prediction block".to_string(),
        ));
    }
    let weights = smooth_weights(height)?;
    let bottom = left[height - 1];
    let mut out = Vec::with_capacity(width * height);
    for y in 0..height {
        let weight = weights[y];
        for top in above.iter().take(width) {
            out.push(weighted_avg(*top, bottom, weight));
        }
    }
    Ok(out)
}

fn predict_smooth_horizontal(
    width: usize,
    height: usize,
    edges: IntraEdges<'_>,
) -> Result<Vec<u16>, DecoderError> {
    let above = edges.above.ok_or_else(|| {
        DecoderError::Bitstream(
            "AV1 smooth horizontal intra prediction requires above edge".to_string(),
        )
    })?;
    let left = edges.left.ok_or_else(|| {
        DecoderError::Bitstream(
            "AV1 smooth horizontal intra prediction requires left edge".to_string(),
        )
    })?;
    if above.is_empty() || left.len() < height {
        return Err(DecoderError::NotEnoughData(
            "AV1 smooth horizontal edges are shorter than prediction block".to_string(),
        ));
    }
    let weights = smooth_weights(width)?;
    let right = above[width - 1];
    let mut out = Vec::with_capacity(width * height);
    for left_value in left.iter().take(height) {
        for x in 0..width {
            let weight = weights[x];
            out.push(weighted_avg(*left_value, right, weight));
        }
    }
    Ok(out)
}

fn predict_paeth(
    width: usize,
    height: usize,
    edges: IntraEdges<'_>,
) -> Result<Vec<u16>, DecoderError> {
    let above = edges.above.ok_or_else(|| {
        DecoderError::Bitstream("AV1 Paeth intra prediction requires above edge".to_string())
    })?;
    let left = edges.left.ok_or_else(|| {
        DecoderError::Bitstream("AV1 Paeth intra prediction requires left edge".to_string())
    })?;
    let top_left = edges.above_left.ok_or_else(|| {
        DecoderError::Bitstream("AV1 Paeth intra prediction requires top-left edge".to_string())
    })?;
    if above.len() < width || left.len() < height {
        return Err(DecoderError::NotEnoughData(
            "AV1 Paeth edges are shorter than prediction block".to_string(),
        ));
    }

    let mut out = Vec::with_capacity(width * height);
    for left_value in left.iter().take(height) {
        for above_value in above.iter().take(width) {
            let base = i32::from(*above_value) + i32::from(*left_value) - i32::from(top_left);
            let p_left = (base - i32::from(*left_value)).abs();
            let p_top = (base - i32::from(*above_value)).abs();
            let p_top_left = (base - i32::from(top_left)).abs();
            let value = if p_left <= p_top && p_left <= p_top_left {
                *left_value
            } else if p_top <= p_top_left {
                *above_value
            } else {
                top_left
            };
            out.push(value);
        }
    }
    Ok(out)
}

fn edge_sample(edge: &[u16], index: usize) -> u16 {
    edge[index.min(edge.len() - 1)]
}

const SMOOTH_WEIGHTS: [u16; 124] = [
    255, 149, 85, 64, 255, 197, 146, 105, 73, 50, 37, 32, 255, 225, 196, 170, 145, 123, 102, 84,
    68, 54, 43, 33, 26, 20, 17, 16, 255, 240, 225, 210, 196, 182, 169, 157, 145, 133, 122, 111,
    101, 92, 83, 74, 66, 59, 52, 45, 39, 34, 29, 25, 21, 17, 14, 12, 10, 9, 8, 8, 255, 248, 240,
    233, 225, 218, 210, 203, 196, 189, 182, 176, 169, 163, 156, 150, 144, 138, 133, 127, 121, 116,
    111, 106, 101, 96, 91, 86, 82, 77, 73, 69, 65, 61, 57, 54, 50, 47, 44, 41, 38, 35, 32, 29, 27,
    25, 22, 20, 18, 16, 15, 13, 12, 10, 9, 8, 7, 6, 6, 5, 5, 4, 4, 4,
];

fn smooth_weights(len: usize) -> Result<&'static [u16], DecoderError> {
    let offset = match len {
        4 => 0,
        8 => 4,
        16 => 12,
        32 => 28,
        64 => 60,
        _ => {
            return Err(DecoderError::Unsupported(format!(
                "AV1 smooth prediction block dimension {len} is not supported"
            )));
        }
    };
    Ok(&SMOOTH_WEIGHTS[offset..offset + len])
}

fn weighted_avg(primary: u16, secondary: u16, primary_weight: u16) -> u16 {
    let secondary_weight = 256u32 - u32::from(primary_weight);
    ((u32::from(primary) * u32::from(primary_weight)
        + u32::from(secondary) * secondary_weight
        + 128)
        >> 8) as u16
}

fn clip1(value: u32, bit_depth: u8) -> u16 {
    value.min((1u32 << bit_depth) - 1) as u16
}

fn clip1_signed(value: i32, bit_depth: u8) -> u16 {
    value.clamp(0, (1i32 << bit_depth) - 1) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_predicts_midpoint_without_edges() {
        let pred = predict_intra(
            PredictionMode::Dc,
            None,
            4,
            4,
            IntraEdges {
                above: None,
                left: None,
                above_left: None,
                bit_depth: 8,
            },
        )
        .unwrap();

        assert_eq!(pred, vec![128; 16]);
    }

    #[test]
    fn dc_averages_available_edges() {
        let pred = predict_intra(
            PredictionMode::Dc,
            None,
            4,
            2,
            IntraEdges {
                above: Some(&[10, 20, 30, 40]),
                left: Some(&[50, 60]),
                above_left: Some(0),
                bit_depth: 8,
            },
        )
        .unwrap();

        assert_eq!(pred, vec![35; 8]);
    }

    #[test]
    fn filter_intra_predicts_from_above_and_left_edges() {
        let pred = predict_filter_intra(
            0,
            4,
            2,
            IntraEdges {
                above: Some(&[10, 20, 30, 40]),
                left: Some(&[50, 60]),
                above_left: Some(5),
                bit_depth: 8,
            },
        )
        .unwrap();

        assert_eq!(pred.len(), 8);
        assert!(pred.iter().any(|value| *value != pred[0]));
    }

    #[test]
    fn vertical_and_horizontal_copy_edges() {
        let vertical = predict_intra(
            PredictionMode::Vertical,
            None,
            3,
            2,
            IntraEdges {
                above: Some(&[1, 2, 3]),
                left: None,
                above_left: None,
                bit_depth: 8,
            },
        )
        .unwrap();
        let horizontal = predict_intra(
            PredictionMode::Horizontal,
            None,
            3,
            2,
            IntraEdges {
                above: None,
                left: Some(&[4, 5]),
                above_left: None,
                bit_depth: 8,
            },
        )
        .unwrap();

        assert_eq!(vertical, vec![1, 2, 3, 1, 2, 3]);
        assert_eq!(horizontal, vec![4, 4, 4, 5, 5, 5]);
    }

    #[test]
    fn directional_predictions_follow_diagonal_edges() {
        let d45 = predict_intra(
            PredictionMode::D45,
            None,
            3,
            2,
            IntraEdges {
                above: Some(&[1, 2, 3]),
                left: Some(&[4, 5]),
                above_left: Some(0),
                bit_depth: 8,
            },
        )
        .unwrap();
        let d113 = predict_intra(
            PredictionMode::D113,
            None,
            3,
            2,
            IntraEdges {
                above: Some(&[1, 2, 3]),
                left: Some(&[4, 5]),
                above_left: Some(0),
                bit_depth: 8,
            },
        )
        .unwrap();
        let d203 = predict_intra(
            PredictionMode::D203,
            None,
            3,
            2,
            IntraEdges {
                above: Some(&[1, 2, 3]),
                left: Some(&[4, 5]),
                above_left: Some(0),
                bit_depth: 8,
            },
        )
        .unwrap();

        assert_eq!(d45, vec![2, 3, 3, 3, 3, 3]);
        assert_eq!(d113, vec![1, 2, 3, 0, 1, 2]);
        assert_eq!(d203, vec![4, 5, 5, 5, 5, 5]);
    }

    #[test]
    fn directional_angle_delta_uses_three_degree_steps() {
        let edges = IntraEdges {
            above: Some(&[0, 100, 0, 200]),
            left: Some(&[0, 0, 0, 0]),
            above_left: Some(0),
            bit_depth: 8,
        };

        let base = predict_intra(PredictionMode::D45, Some(0), 4, 4, edges).unwrap();
        let plus_three = predict_intra(PredictionMode::D45, Some(1), 4, 4, edges).unwrap();

        assert_eq!(base[0], 100);
        assert_eq!(plus_three[0], 88);
    }

    #[test]
    fn directional_edge_upsample_matches_aom_four_tap_filter() {
        let edge = DirectionalEdge::new(&[10, 20, 30, 40], 0, 4, true, 8);

        assert!(edge.upsampled);
        assert_eq!(edge.sample(-2), 0);
        assert_eq!(edge.sample(-1), 4);
        assert_eq!(edge.sample(0), 10);
        assert_eq!(edge.sample(1), 15);
        assert_eq!(edge.sample(2), 20);
        assert_eq!(edge.sample(3), 25);
    }

    #[test]
    fn directional_edge_upsample_follows_aom_size_and_angle_limits() {
        assert!(use_directional_edge_upsample(4, 4, -23, true));
        assert!(!use_directional_edge_upsample(8, 16, -23, true));
        assert!(!use_directional_edge_upsample(4, 4, -40, true));
        assert!(!use_directional_edge_upsample(4, 4, -23, false));
    }

    #[test]
    fn smooth_predictions_blend_toward_far_edges() {
        let edges = IntraEdges {
            above: Some(&[10, 20, 30, 40]),
            left: Some(&[50, 60, 70, 80]),
            above_left: Some(0),
            bit_depth: 8,
        };

        let smooth = predict_intra(PredictionMode::Smooth, None, 4, 4, edges).unwrap();
        let smooth_horizontal =
            predict_intra(PredictionMode::SmoothHorizontal, None, 4, 4, edges).unwrap();

        assert_eq!(
            smooth,
            vec![
                30, 33, 37, 41, 50, 48, 49, 51, 63, 59, 57, 57, 71, 64, 60, 60
            ]
        );
        assert_eq!(
            smooth_horizontal,
            vec![
                50, 46, 43, 43, 60, 52, 47, 45, 70, 57, 50, 48, 80, 63, 53, 50
            ]
        );
        assert_eq!(smooth_weights(64).unwrap()[63], 4);
    }

    #[test]
    fn paeth_selects_nearest_candidate() {
        let pred = predict_intra(
            PredictionMode::Paeth,
            None,
            2,
            2,
            IntraEdges {
                above: Some(&[10, 100]),
                left: Some(&[20, 90]),
                above_left: Some(15),
                bit_depth: 8,
            },
        )
        .unwrap();

        assert_eq!(pred, vec![15, 100, 90, 100]);
    }
}
