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
    match mode {
        PredictionMode::Dc => Ok(predict_dc(width, height, edges)),
        PredictionMode::Vertical => copy_above(width, height, edges),
        PredictionMode::Horizontal => copy_left(width, height, edges),
        PredictionMode::D45
        | PredictionMode::D67
        | PredictionMode::D113
        | PredictionMode::D135
        | PredictionMode::D157
        | PredictionMode::D203 => {
            predict_directional(mode, angle_delta.unwrap_or(0), width, height, edges)
        }
        PredictionMode::Smooth => predict_smooth(width, height, edges),
        PredictionMode::SmoothVertical => predict_smooth_vertical(width, height, edges),
        PredictionMode::SmoothHorizontal => predict_smooth_horizontal(width, height, edges),
        PredictionMode::Paeth => predict_paeth(width, height, edges),
    }
}

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
    let delta = angle_delta.clamp(-3, 3);
    let mut out = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let adjusted_x = shifted_index(x, delta);
            let adjusted_y = shifted_index(y, delta);
            let value = match mode {
                PredictionMode::D45 => edge_sample(above, adjusted_x + y + 1),
                PredictionMode::D67 => {
                    let primary = edge_sample(above, adjusted_x + ((y + 1) >> 1));
                    let secondary = edge_sample(above, adjusted_x + y + 1);
                    avg2(primary, secondary)
                }
                PredictionMode::D113 => {
                    if adjusted_x + 1 >= y {
                        edge_sample(above, adjusted_x + 1 - y)
                    } else {
                        edge_sample(left, y - adjusted_x - 2)
                    }
                }
                PredictionMode::D135 => {
                    if adjusted_x == adjusted_y {
                        above_left
                    } else if adjusted_x > adjusted_y {
                        edge_sample(above, adjusted_x - adjusted_y - 1)
                    } else {
                        edge_sample(left, adjusted_y - adjusted_x - 1)
                    }
                }
                PredictionMode::D157 => {
                    if adjusted_y + 1 >= x {
                        edge_sample(left, adjusted_y + 1 - x)
                    } else {
                        edge_sample(above, x - adjusted_y - 2)
                    }
                }
                PredictionMode::D203 => edge_sample(left, x + adjusted_y + 1),
                _ => unreachable!(),
            };
            out.push(value);
        }
    }
    Ok(out)
}

fn predict_smooth(
    width: usize,
    height: usize,
    edges: IntraEdges<'_>,
) -> Result<Vec<u16>, DecoderError> {
    let vertical = predict_smooth_vertical(width, height, edges)?;
    let horizontal = predict_smooth_horizontal(width, height, edges)?;
    Ok(vertical
        .iter()
        .zip(horizontal.iter())
        .map(|(a, b)| avg2(*a, *b))
        .collect())
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
    let bottom = left[height.saturating_sub(1).min(left.len() - 1)];
    let mut out = Vec::with_capacity(width * height);
    for y in 0..height {
        let weight = smooth_weight(y, height);
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
    let right = above[width.saturating_sub(1).min(above.len() - 1)];
    let mut out = Vec::with_capacity(width * height);
    for left_value in left.iter().take(height) {
        for x in 0..width {
            let weight = smooth_weight(x, width);
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

fn shifted_index(index: usize, delta: i8) -> usize {
    if delta >= 0 {
        index.saturating_add(delta as usize)
    } else {
        index.saturating_sub((-delta) as usize)
    }
}

fn avg2(a: u16, b: u16) -> u16 {
    ((u32::from(a) + u32::from(b) + 1) >> 1) as u16
}

fn smooth_weight(index: usize, len: usize) -> u16 {
    if len <= 1 {
        return 255;
    }
    let remaining = len - 1 - index.min(len - 1);
    ((remaining * 255 + ((len - 1) >> 1)) / (len - 1)) as u16
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

        assert_eq!(d113, vec![2, 3, 3, 1, 2, 3]);
        assert_eq!(d203, vec![5, 5, 5, 5, 5, 5]);
    }

    #[test]
    fn smooth_predictions_blend_toward_far_edges() {
        let edges = IntraEdges {
            above: Some(&[1, 2, 3]),
            left: Some(&[4, 5]),
            above_left: Some(0),
            bit_depth: 8,
        };

        let smooth = predict_intra(PredictionMode::Smooth, None, 3, 2, edges).unwrap();
        let smooth_horizontal =
            predict_intra(PredictionMode::SmoothHorizontal, None, 3, 2, edges).unwrap();

        assert_eq!(smooth, vec![3, 3, 3, 5, 5, 4]);
        assert_eq!(smooth_horizontal, vec![4, 4, 3, 5, 4, 3]);
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
