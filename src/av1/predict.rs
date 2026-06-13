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
    width: usize,
    height: usize,
    edges: IntraEdges<'_>,
) -> Result<Vec<u16>, DecoderError> {
    match mode {
        PredictionMode::Dc => Ok(predict_dc(width, height, edges)),
        PredictionMode::Vertical => copy_above(width, height, edges),
        PredictionMode::Horizontal => copy_left(width, height, edges),
        PredictionMode::Paeth => predict_paeth(width, height, edges),
        _ => Err(DecoderError::Unsupported(format!(
            "AV1 intra prediction mode {mode:?} is not supported yet"
        ))),
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
    fn paeth_selects_nearest_candidate() {
        let pred = predict_intra(
            PredictionMode::Paeth,
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
