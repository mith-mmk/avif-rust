use super::decode::PlaneBuffer;
use super::quant::{PlaneQuant, dequantize_coefficients};
use super::reconstruct::{add_residual_to_prediction, write_plane_block};
use super::syntax::{BlockSize, TxSize, TxType};
use crate::DecoderError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransformBlock {
    pub plane: usize,
    pub x: usize,
    pub y: usize,
    pub tx_size: TxSize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizedTransform {
    pub block: TransformBlock,
    pub tx_type: TxType,
    pub coefficients: Vec<i32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReconstructedTransform {
    pub block: TransformBlock,
    pub tx_type: TxType,
    pub non_zero_coefficients: usize,
}

pub fn plan_transform_blocks(
    plane: usize,
    x: usize,
    y: usize,
    block_size: BlockSize,
    frame_width: usize,
    frame_height: usize,
) -> Vec<TransformBlock> {
    let tx_size = block_size.largest_supported_tx_size();
    plan_transform_blocks_with_tx_size(plane, x, y, block_size, tx_size, frame_width, frame_height)
}

pub fn plan_transform_blocks_with_tx_size(
    plane: usize,
    x: usize,
    y: usize,
    block_size: BlockSize,
    tx_size: TxSize,
    frame_width: usize,
    frame_height: usize,
) -> Vec<TransformBlock> {
    let tx_width = tx_size.width();
    let tx_height = tx_size.height();
    let block_width = block_size.width().min(frame_width.saturating_sub(x));
    let block_height = block_size.height().min(frame_height.saturating_sub(y));
    let mut blocks = Vec::new();

    let mut offset_y = 0;
    while offset_y < block_height {
        let mut offset_x = 0;
        while offset_x < block_width {
            blocks.push(TransformBlock {
                plane,
                x: x + offset_x,
                y: y + offset_y,
                tx_size,
            });
            offset_x += tx_width;
        }
        offset_y += tx_height;
    }
    blocks
}

pub fn zig_zag_scan(tx_size: TxSize) -> Vec<usize> {
    let full_width = tx_size.width();
    let scan_width = full_width.min(32);
    let scan_height = tx_size.height().min(32);
    let mut scan = Vec::with_capacity(scan_width * scan_height);
    for diagonal in 0..=(scan_width + scan_height - 2) {
        if diagonal % 2 == 1 {
            let mut y = diagonal.min(scan_height - 1);
            let mut x = diagonal - y;
            loop {
                if x < scan_width && y < scan_height {
                    scan.push(y * full_width + x);
                }
                if y == 0 {
                    break;
                }
                y -= 1;
                x += 1;
            }
        } else {
            let mut x = diagonal.min(scan_width - 1);
            let mut y = diagonal - x;
            loop {
                if x < scan_width && y < scan_height {
                    scan.push(y * full_width + x);
                }
                if x == 0 {
                    break;
                }
                x -= 1;
                y += 1;
            }
        }
    }
    scan
}

pub fn coefficients_from_scan(
    tx_size: TxSize,
    scanned_coefficients: &[i32],
) -> Result<Vec<i32>, DecoderError> {
    let scan = zig_zag_scan(tx_size);
    if scanned_coefficients.len() > scan.len() {
        return Err(DecoderError::InvalidParam(
            "AV1 scanned coefficient count exceeds transform size".to_string(),
        ));
    }
    let mut coefficients = vec![0i32; tx_size.sample_count()];
    for (index, coefficient) in scanned_coefficients.iter().enumerate() {
        coefficients[scan[index]] = *coefficient;
    }
    Ok(coefficients)
}

pub fn zero_quantized_transform(block: TransformBlock, tx_type: TxType) -> QuantizedTransform {
    QuantizedTransform {
        block,
        tx_type,
        coefficients: vec![0; block.tx_size.sample_count()],
    }
}

pub fn reconstruct_transform_block(
    plane: &mut PlaneBuffer,
    quantized: &QuantizedTransform,
    plane_quant: PlaneQuant,
    prediction: &[u16],
    bit_depth: u8,
) -> Result<ReconstructedTransform, DecoderError> {
    let tx_size = quantized.block.tx_size;
    if quantized.coefficients.len() != tx_size.sample_count() {
        return Err(DecoderError::InvalidParam(
            "AV1 quantized coefficient count does not match transform size".to_string(),
        ));
    }
    if prediction.len() != tx_size.sample_count() {
        return Err(DecoderError::InvalidParam(
            "AV1 prediction sample count does not match transform size".to_string(),
        ));
    }

    let non_zero_coefficients = quantized
        .coefficients
        .iter()
        .filter(|coefficient| **coefficient != 0)
        .count();
    let dequant = dequantize_coefficients(
        &quantized.coefficients,
        plane_quant,
        bit_depth,
        tx_size.dq_denom(),
    );
    let residual = inverse_transform(quantized.tx_type, tx_size, &dequant, bit_depth)?;
    let reconstructed = add_residual_to_prediction(prediction, &residual, bit_depth)?;
    write_plane_block(
        plane,
        quantized.block.x,
        quantized.block.y,
        tx_size.width(),
        tx_size.height(),
        &reconstructed,
    )?;

    Ok(ReconstructedTransform {
        block: quantized.block,
        tx_type: quantized.tx_type,
        non_zero_coefficients,
    })
}

pub fn inverse_transform(
    tx_type: TxType,
    tx_size: TxSize,
    dequant: &[i32],
    bit_depth: u8,
) -> Result<Vec<i32>, DecoderError> {
    if dequant.len() != tx_size.sample_count() {
        return Err(DecoderError::InvalidParam(
            "AV1 dequant coefficient count does not match transform size".to_string(),
        ));
    }
    match tx_type {
        TxType::DctDct => inverse_separable_transform(
            tx_size,
            dequant,
            bit_depth,
            Transform1d::Dct,
            Transform1d::Dct,
        ),
        TxType::AdstDct => inverse_separable_transform(
            tx_size,
            dequant,
            bit_depth,
            Transform1d::Adst,
            Transform1d::Dct,
        ),
        TxType::DctAdst => inverse_separable_transform(
            tx_size,
            dequant,
            bit_depth,
            Transform1d::Dct,
            Transform1d::Adst,
        ),
        TxType::AdstAdst => inverse_separable_transform(
            tx_size,
            dequant,
            bit_depth,
            Transform1d::Adst,
            Transform1d::Adst,
        ),
        TxType::Identity => Ok(inverse_identity(tx_size, dequant, bit_depth)),
        TxType::VerticalDct => inverse_vertical_dct(tx_size, dequant, bit_depth),
        TxType::HorizontalDct => inverse_horizontal_dct(tx_size, dequant, bit_depth),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Transform1d {
    Dct,
    Adst,
}

fn inverse_separable_transform(
    tx_size: TxSize,
    dequant: &[i32],
    bit_depth: u8,
    vertical: Transform1d,
    horizontal: Transform1d,
) -> Result<Vec<i32>, DecoderError> {
    if (tx_size.width() > 32 || tx_size.height() > 32)
        && (vertical != Transform1d::Dct || horizontal != Transform1d::Dct)
    {
        return Err(DecoderError::Unsupported(
            "AV1 staged separable transforms larger than 32x32 are not supported yet".to_string(),
        ));
    }

    let width = tx_size.width();
    let height = tx_size.height();
    let horizontal_basis = inverse_1d_basis_table(horizontal, width);
    let vertical_basis = inverse_1d_basis_table(vertical, height);
    let mut temp = vec![0.0; width * height];
    let mut out = vec![0i32; width * height];
    let residual_limit = 1i32 << (bit_depth + 7);

    for row in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            for u in 0..width {
                sum += horizontal_basis[x * width + u] * f64::from(dequant[row * width + u]);
            }
            temp[row * width + x] = sum;
        }
    }

    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            for v in 0..height {
                sum += vertical_basis[y * height + v] * temp[v * width + x];
            }
            out[y * width + x] = (sum.round() as i32).clamp(-residual_limit, residual_limit - 1);
        }
    }

    Ok(out)
}

fn inverse_1d_basis_table(transform: Transform1d, len: usize) -> Vec<f64> {
    let mut table = vec![0.0; len * len];
    for sample in 0..len {
        for coeff in 0..len {
            table[sample * len + coeff] = inverse_1d_basis(transform, len, sample, coeff);
        }
    }
    table
}

fn inverse_1d_basis(transform: Transform1d, len: usize, sample: usize, coeff: usize) -> f64 {
    match transform {
        Transform1d::Dct => {
            let scale = (2.0 / len as f64).sqrt();
            let alpha = if coeff == 0 {
                1.0 / 2.0_f64.sqrt()
            } else {
                1.0
            };
            let angle =
                ((2 * sample + 1) * coeff) as f64 * std::f64::consts::PI / (2.0 * len as f64);
            scale * alpha * angle.cos()
        }
        Transform1d::Adst => {
            let scale = (2.0 / (len + 1) as f64).sqrt();
            let angle =
                ((sample + 1) * (coeff + 1)) as f64 * std::f64::consts::PI / ((len + 1) as f64);
            scale * angle.sin()
        }
    }
}

fn inverse_vertical_dct(
    tx_size: TxSize,
    dequant: &[i32],
    bit_depth: u8,
) -> Result<Vec<i32>, DecoderError> {
    if tx_size.height() > 32 {
        return Err(DecoderError::Unsupported(
            "AV1 vertical DCT transforms larger than 32 are not supported yet".to_string(),
        ));
    }

    let width = tx_size.width();
    let height = tx_size.height();
    let mut out = vec![0i32; width * height];
    let height_scale = (2.0 / height as f64).sqrt();
    let residual_limit = 1i32 << (bit_depth + 7);

    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            for v in 0..height {
                let alpha_v = if v == 0 { 1.0 / 2.0_f64.sqrt() } else { 1.0 };
                let cos_v =
                    (((2 * y + 1) * v) as f64 * std::f64::consts::PI / (2.0 * height as f64)).cos();
                sum += alpha_v * f64::from(dequant[v * width + x]) * cos_v;
            }
            let value = (sum * height_scale).round() as i32;
            out[y * width + x] = value.clamp(-residual_limit, residual_limit - 1);
        }
    }
    Ok(out)
}

fn inverse_horizontal_dct(
    tx_size: TxSize,
    dequant: &[i32],
    bit_depth: u8,
) -> Result<Vec<i32>, DecoderError> {
    if tx_size.width() > 32 {
        return Err(DecoderError::Unsupported(
            "AV1 horizontal DCT transforms larger than 32 are not supported yet".to_string(),
        ));
    }

    let width = tx_size.width();
    let height = tx_size.height();
    let mut out = vec![0i32; width * height];
    let width_scale = (2.0 / width as f64).sqrt();
    let residual_limit = 1i32 << (bit_depth + 7);

    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            for u in 0..width {
                let alpha_u = if u == 0 { 1.0 / 2.0_f64.sqrt() } else { 1.0 };
                let cos_u =
                    (((2 * x + 1) * u) as f64 * std::f64::consts::PI / (2.0 * width as f64)).cos();
                sum += alpha_u * f64::from(dequant[y * width + u]) * cos_u;
            }
            let value = (sum * width_scale).round() as i32;
            out[y * width + x] = value.clamp(-residual_limit, residual_limit - 1);
        }
    }
    Ok(out)
}

fn inverse_identity(tx_size: TxSize, dequant: &[i32], bit_depth: u8) -> Vec<i32> {
    let row_shift = tx_size.row_shift();
    let residual_limit = 1i32 << (bit_depth + 7);
    dequant
        .iter()
        .map(|value| {
            round2_signed(*value, row_shift + 4).clamp(-residual_limit, residual_limit - 1)
        })
        .collect()
}

fn round2_signed(value: i32, bits: u8) -> i32 {
    if bits == 0 {
        value
    } else if value >= 0 {
        (value + (1 << (bits - 1))) >> bits
    } else {
        -((-value + (1 << (bits - 1))) >> bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plans_128_block_as_four_64_transforms() {
        let blocks = plan_transform_blocks(0, 0, 0, BlockSize::Block128x128, 900, 900);

        assert_eq!(blocks.len(), 4);
        assert!(blocks.iter().all(|block| block.tx_size == TxSize::Tx64x64));
        assert_eq!(blocks[0].x, 0);
        assert_eq!(blocks[3].x, 64);
        assert_eq!(blocks[3].y, 64);
    }

    #[test]
    fn transform_plan_clips_at_frame_edge() {
        let blocks = plan_transform_blocks(0, 896, 896, BlockSize::Block128x128, 900, 900);

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].x, 896);
        assert_eq!(blocks[0].y, 896);
    }

    #[test]
    fn zig_zag_scan_orders_square_transform() {
        let scan = zig_zag_scan(TxSize::Tx4x4);

        assert_eq!(
            scan,
            vec![0, 4, 1, 2, 5, 8, 12, 9, 6, 3, 7, 10, 13, 14, 11, 15]
        );
    }

    #[test]
    fn tx64_scan_codes_top_left_32_square_only() {
        let scan = zig_zag_scan(TxSize::Tx64x64);

        assert_eq!(scan.len(), TxSize::Tx32x32.sample_count());
        assert_eq!(&scan[..6], &[0, 64, 1, 2, 65, 128]);
        assert!(
            scan.iter()
                .all(|position| position / TxSize::Tx64x64.width() < 32
                    && position % TxSize::Tx64x64.width() < 32)
        );
    }

    #[test]
    fn coefficients_from_scan_places_values_in_raster_slots() {
        let coefficients = coefficients_from_scan(TxSize::Tx4x4, &[1, 2, 3]).unwrap();

        assert_eq!(coefficients[0], 1);
        assert_eq!(coefficients[4], 2);
        assert_eq!(coefficients[1], 3);
        assert_eq!(coefficients.iter().filter(|value| **value != 0).count(), 3);
    }

    #[test]
    fn inverse_dct_dc_only_outputs_constant_residual() {
        let mut coeffs = vec![0; TxSize::Tx4x4.sample_count()];
        coeffs[0] = 16;

        let residual = inverse_transform(TxType::DctDct, TxSize::Tx4x4, &coeffs, 8).unwrap();

        assert_eq!(residual, vec![4; 16]);
    }

    #[test]
    fn identity_transform_rounds_and_clips() {
        let residual = inverse_transform(TxType::Identity, TxSize::Tx4x4, &[64; 16], 8).unwrap();

        assert_eq!(residual, vec![4; 16]);
    }

    #[test]
    fn vertical_dct_only_mixes_columns() {
        let mut coeffs = vec![0; TxSize::Tx4x4.sample_count()];
        coeffs[0] = 16;
        coeffs[1] = 32;

        let residual = inverse_transform(TxType::VerticalDct, TxSize::Tx4x4, &coeffs, 8).unwrap();

        assert_eq!(
            residual,
            vec![8, 16, 0, 0, 8, 16, 0, 0, 8, 16, 0, 0, 8, 16, 0, 0]
        );
    }

    #[test]
    fn horizontal_dct_only_mixes_rows() {
        let mut coeffs = vec![0; TxSize::Tx4x4.sample_count()];
        coeffs[0] = 16;
        coeffs[4] = 32;

        let residual = inverse_transform(TxType::HorizontalDct, TxSize::Tx4x4, &coeffs, 8).unwrap();

        assert_eq!(
            residual,
            vec![8, 8, 8, 8, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0]
        );
    }

    #[test]
    fn adst_dct_staging_transform_outputs_finite_residuals() {
        let mut coeffs = vec![0; TxSize::Tx4x4.sample_count()];
        coeffs[0] = 16;
        coeffs[1] = 8;
        coeffs[4] = -12;

        let residual = inverse_transform(TxType::AdstDct, TxSize::Tx4x4, &coeffs, 8).unwrap();

        assert_eq!(residual.len(), 16);
        assert!(residual.iter().any(|value| *value != 0));
        assert!(residual.iter().all(|value| value.abs() < (1 << 15)));
    }

    #[test]
    fn reconstruct_transform_block_writes_prediction_plus_residual() {
        let layout = super::super::decode::PlaneLayout {
            plane: 0,
            width: 4,
            height: 4,
            subsampling_x: 0,
            subsampling_y: 0,
            sample_count: 16,
        };
        let mut plane = PlaneBuffer {
            layout,
            samples: vec![0; 16],
        };
        let block = TransformBlock {
            plane: 0,
            x: 0,
            y: 0,
            tx_size: TxSize::Tx4x4,
        };
        let quantized = QuantizedTransform {
            block,
            tx_type: TxType::Identity,
            coefficients: vec![64; 16],
        };

        let result = reconstruct_transform_block(
            &mut plane,
            &quantized,
            PlaneQuant { dc: 4, ac: 4 },
            &[100; 16],
            8,
        )
        .unwrap();

        assert_eq!(result.non_zero_coefficients, 16);
        assert_eq!(plane.samples, vec![116; 16]);
    }

    #[test]
    fn zero_quantized_transform_preserves_prediction() {
        let layout = super::super::decode::PlaneLayout {
            plane: 0,
            width: 4,
            height: 4,
            subsampling_x: 0,
            subsampling_y: 0,
            sample_count: 16,
        };
        let mut plane = PlaneBuffer {
            layout,
            samples: vec![0; 16],
        };
        let block = TransformBlock {
            plane: 0,
            x: 0,
            y: 0,
            tx_size: TxSize::Tx4x4,
        };
        let quantized = zero_quantized_transform(block, TxType::DctDct);

        let result = reconstruct_transform_block(
            &mut plane,
            &quantized,
            PlaneQuant { dc: 4, ac: 4 },
            &[77; 16],
            8,
        )
        .unwrap();

        assert_eq!(quantized.coefficients, vec![0; 16]);
        assert_eq!(result.non_zero_coefficients, 0);
        assert_eq!(plane.samples, vec![77; 16]);
    }
}
