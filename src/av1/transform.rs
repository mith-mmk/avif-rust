use super::decode::PlaneBuffer;
use super::quant::{PlaneQuant, dequantize_coefficients};
use super::reconstruct::{add_residual_to_prediction, write_plane_block};
use super::syntax::{BlockSize, TxSize, TxType};
use crate::DecoderError;

mod dct64;

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
    if tx_size.is_rectangular() {
        // AOM stores rectangular transform coefficients column-major.  Its
        // default scan follows diagonals, reversing each diagonal when the
        // transform is wider than tall.
        let mut scan = Vec::with_capacity(scan_width * scan_height);
        for diagonal in 0..=(scan_width + scan_height - 2) {
            let first = (diagonal + 1).saturating_sub(scan_width);
            let last = diagonal.min(scan_height - 1);
            if scan_width >= scan_height {
                for row in (first..=last).rev() {
                    let column = diagonal - row;
                    scan.push(column * tx_size.height() + row);
                }
            } else {
                for row in first..=last {
                    let column = diagonal - row;
                    scan.push(column * tx_size.height() + row);
                }
            }
        }
        return scan;
    }
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

pub fn coefficient_scan(tx_size: TxSize, tx_type: TxType) -> Vec<usize> {
    let width = tx_size.width();
    let scan_width = width.min(32);
    let scan_height = tx_size.height().min(32);
    match tx_type {
        TxType::VerticalDct if tx_size.is_rectangular() => (0..scan_height)
            .flat_map(|row| (0..scan_width).map(move |column| column * tx_size.height() + row))
            .collect(),
        TxType::HorizontalDct if tx_size.is_rectangular() => {
            (0..scan_width * scan_height).collect()
        }
        TxType::VerticalDct => (0..scan_width)
            .flat_map(|column| (0..scan_height).map(move |row| row * width + column))
            .collect(),
        TxType::HorizontalDct => (0..scan_height)
            .flat_map(|row| (0..scan_width).map(move |column| row * width + column))
            .collect(),
        _ => zig_zag_scan(tx_size),
    }
}

#[cfg(test)]
fn normative_square_row_scan(tx_size: TxSize) -> Vec<usize> {
    match tx_size {
        TxSize::Tx4x4 => vec![0, 1, 4, 2, 5, 3, 6, 8, 9, 7, 12, 10, 13, 11, 14, 15],
        TxSize::Tx8x8 => vec![
            0, 1, 2, 8, 9, 3, 16, 10, 4, 17, 11, 24, 5, 18, 25, 12, 19, 26, 32, 6, 13, 20, 33, 27,
            7, 34, 40, 21, 28, 41, 14, 35, 48, 42, 29, 36, 49, 22, 43, 15, 56, 37, 50, 44, 30, 57,
            23, 51, 58, 45, 38, 52, 31, 59, 53, 46, 60, 39, 61, 47, 54, 55, 62, 63,
        ],
        TxSize::Tx16x16 => vec![
            0, 1, 2, 16, 3, 17, 4, 18, 32, 5, 33, 19, 6, 34, 48, 20, 49, 7, 35, 21, 50, 64, 8, 36,
            65, 22, 51, 37, 80, 9, 66, 52, 23, 38, 81, 67, 10, 53, 24, 82, 68, 96, 39, 11, 54, 83,
            97, 69, 25, 98, 84, 40, 112, 55, 12, 70, 99, 113, 85, 26, 41, 56, 114, 100, 13, 71,
            128, 86, 27, 115, 101, 129, 42, 57, 72, 116, 14, 87, 130, 102, 144, 73, 131, 117, 28,
            58, 15, 88, 43, 145, 103, 132, 146, 118, 74, 160, 89, 133, 104, 29, 59, 147, 119, 44,
            161, 148, 90, 105, 134, 162, 120, 176, 75, 135, 149, 30, 60, 163, 177, 45, 121, 91,
            106, 164, 178, 150, 192, 136, 165, 179, 31, 151, 193, 76, 122, 61, 137, 194, 107, 152,
            180, 208, 46, 166, 167, 195, 92, 181, 138, 209, 123, 153, 224, 196, 77, 168, 210, 182,
            240, 108, 197, 62, 154, 225, 183, 169, 211, 47, 139, 93, 184, 226, 212, 241, 198, 170,
            124, 155, 199, 78, 213, 185, 109, 227, 200, 63, 228, 242, 140, 214, 171, 186, 156, 229,
            243, 125, 94, 201, 244, 215, 216, 230, 141, 187, 202, 79, 172, 110, 157, 245, 217, 231,
            95, 246, 232, 126, 203, 247, 233, 173, 218, 142, 111, 158, 188, 248, 127, 234, 219,
            249, 189, 204, 143, 174, 159, 250, 235, 205, 220, 175, 190, 251, 221, 191, 206, 236,
            207, 237, 252, 222, 253, 223, 238, 239, 254, 255,
        ],
        _ => (0..tx_size.sample_count()).collect(),
    }
}

#[cfg(test)]
fn normative_square_col_scan(tx_size: TxSize) -> Vec<usize> {
    match tx_size {
        TxSize::Tx4x4 => vec![0, 4, 8, 1, 12, 5, 2, 9, 6, 3, 13, 10, 7, 14, 11, 15],
        TxSize::Tx8x8 => vec![
            0, 8, 16, 1, 24, 9, 32, 17, 2, 40, 25, 10, 33, 18, 48, 3, 26, 41, 11, 56, 19, 34, 4,
            49, 27, 42, 12, 35, 20, 57, 50, 28, 5, 43, 13, 36, 58, 51, 21, 44, 6, 29, 59, 37, 14,
            52, 22, 7, 45, 60, 30, 15, 38, 53, 23, 46, 31, 61, 39, 54, 47, 62, 55, 63,
        ],
        TxSize::Tx16x16 => vec![
            0, 16, 32, 48, 1, 64, 17, 80, 33, 96, 49, 2, 65, 112, 18, 81, 34, 128, 50, 97, 3, 66,
            144, 19, 113, 35, 82, 160, 98, 51, 129, 4, 67, 176, 20, 114, 145, 83, 36, 99, 130, 52,
            192, 5, 161, 68, 115, 21, 146, 84, 208, 177, 37, 131, 100, 53, 162, 224, 69, 6, 116,
            193, 147, 85, 22, 240, 132, 38, 178, 101, 163, 54, 209, 117, 70, 7, 148, 194, 86, 179,
            225, 23, 133, 39, 164, 8, 102, 210, 241, 55, 195, 118, 149, 71, 180, 24, 87, 226, 134,
            165, 211, 40, 103, 56, 72, 150, 196, 242, 119, 9, 181, 227, 88, 166, 25, 135, 41, 104,
            212, 57, 151, 197, 120, 73, 243, 182, 136, 167, 213, 89, 10, 228, 105, 152, 198, 26,
            42, 121, 183, 244, 168, 58, 137, 229, 74, 214, 90, 153, 199, 184, 11, 106, 245, 27,
            122, 230, 169, 43, 215, 59, 200, 138, 185, 246, 75, 12, 91, 154, 216, 231, 107, 28, 44,
            201, 123, 170, 60, 247, 232, 76, 139, 13, 92, 217, 186, 248, 155, 108, 29, 124, 45,
            202, 233, 171, 61, 14, 77, 140, 15, 249, 93, 30, 187, 156, 218, 46, 109, 125, 62, 172,
            78, 203, 31, 141, 234, 94, 47, 188, 63, 157, 110, 250, 219, 79, 126, 204, 173, 142, 95,
            189, 111, 235, 158, 220, 251, 127, 174, 143, 205, 236, 159, 190, 221, 252, 175, 206,
            237, 191, 253, 222, 238, 207, 254, 223, 239, 255,
        ],
        _ => (0..tx_size.sample_count()).collect(),
    }
}

pub(crate) fn remap_coefficients_for_inverse_storage(
    tx_size: TxSize,
    tx_type: TxType,
    coded_lossless: bool,
    coefficients: &mut [i32],
) {
    let needs_remap = (tx_type == TxType::DctDct
        && !coded_lossless
        && matches!(tx_size, TxSize::Tx4x4 | TxSize::Tx8x8))
        || matches!(
            tx_type,
            TxType::AdstDct
                | TxType::DctAdst
                | TxType::AdstAdst
                | TxType::Identity
                | TxType::VerticalDct
                | TxType::HorizontalDct
        );
    if coefficients.len() != tx_size.sample_count() || tx_size.is_rectangular() || !needs_remap {
        return;
    }
    let side = tx_size.width();
    let column_major = coefficients.to_vec();
    for row in 0..side {
        for column in 0..side {
            coefficients[row * side + column] = column_major[column * side + row];
        }
    }
}

pub fn coefficients_from_scan(
    tx_size: TxSize,
    tx_type: TxType,
    scanned_coefficients: &[i32],
) -> Result<Vec<i32>, DecoderError> {
    let scan = coefficient_scan(tx_size, tx_type);
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

pub fn reconstruct_lossless_transform_block(
    plane: &mut PlaneBuffer,
    quantized: &QuantizedTransform,
    plane_quant: PlaneQuant,
    prediction: &[u16],
    bit_depth: u8,
) -> Result<ReconstructedTransform, DecoderError> {
    if quantized.block.tx_size != TxSize::Tx4x4 {
        return Err(DecoderError::Unsupported(
            "AV1 lossless transform must be 4x4".to_string(),
        ));
    }
    if quantized.coefficients.len() != TxSize::Tx4x4.sample_count()
        || prediction.len() != TxSize::Tx4x4.sample_count()
    {
        return Err(DecoderError::InvalidParam(
            "AV1 lossless transform input size does not match 4x4".to_string(),
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
        TxSize::Tx4x4.dq_denom(),
    );
    let residual = inverse_lossless_transform_4x4(&dequant);
    let reconstructed = add_residual_to_prediction(prediction, &residual, bit_depth)?;
    write_plane_block(
        plane,
        quantized.block.x,
        quantized.block.y,
        4,
        4,
        &reconstructed,
    )?;

    Ok(ReconstructedTransform {
        block: quantized.block,
        tx_type: quantized.tx_type,
        non_zero_coefficients,
    })
}

pub fn inverse_lossless_transform_4x4(dequant: &[i32]) -> Vec<i32> {
    let mut intermediate = [0i32; 16];
    for column in 0..4 {
        let transformed = inverse_wht4(
            dequant[column] >> 2,
            dequant[4 + column] >> 2,
            dequant[8 + column] >> 2,
            dequant[12 + column] >> 2,
        );
        for row in 0..4 {
            intermediate[row * 4 + column] = transformed[row];
        }
    }

    let mut block = [0i32; 16];
    for row in 0..4 {
        let offset = row * 4;
        let transformed = inverse_wht4(
            intermediate[offset],
            intermediate[offset + 1],
            intermediate[offset + 2],
            intermediate[offset + 3],
        );
        for column in 0..4 {
            block[column * 4 + row] = transformed[column];
        }
    }

    block.to_vec()
}

fn inverse_wht4(mut a: i32, mut c: i32, mut d: i32, mut b: i32) -> [i32; 4] {
    a += c;
    d -= b;
    let e = (a - d) >> 1;
    b = e - b;
    c = e - c;
    a -= b;
    d += c;
    [a, b, c, d]
}

pub fn inverse_transform(
    tx_type: TxType,
    tx_size: TxSize,
    dequant: &[i32],
    bit_depth: u8,
) -> Result<Vec<i32>, DecoderError> {
    validate_transform_bit_depth(bit_depth)?;
    if dequant.len() != tx_size.sample_count() {
        return Err(DecoderError::InvalidParam(
            "AV1 dequant coefficient count does not match transform size".to_string(),
        ));
    }
    if dequant.iter().all(|value| *value == 0) {
        return Ok(vec![0; tx_size.sample_count()]);
    }
    if tx_size == TxSize::Tx4x4 {
        return Ok(inverse_transform_4x4(tx_type, dequant, bit_depth));
    }
    if tx_size == TxSize::Tx8x8 {
        return Ok(inverse_transform_8x8(tx_type, dequant, bit_depth));
    }
    if tx_size.is_rectangular() {
        if tx_size.width().max(tx_size.height()) > 16 && tx_type != TxType::DctDct {
            return Err(DecoderError::Unsupported(format!(
                "AV1 {tx_size:?} non-DCT transform is not signaled for intra blocks"
            )));
        }
        return Ok(inverse_transform_rect(tx_type, tx_size, dequant, bit_depth));
    }
    if tx_size == TxSize::Tx16x16 {
        return Ok(inverse_transform_16x16(tx_type, dequant, bit_depth));
    }
    if tx_type != TxType::DctDct && (tx_size.width() >= 32 || tx_size.height() >= 32) {
        return Err(DecoderError::Unsupported(format!(
            "AV1 {tx_size:?} non-DCT transform is not signaled for intra blocks"
        )));
    }
    if tx_type != TxType::DctDct {
        return Err(DecoderError::Unsupported(format!(
            "AV1 {tx_size:?} {tx_type:?} transform is not supported for this size"
        )));
    }
    match tx_size {
        TxSize::Tx32x32 => inverse_transform_32x32_dct(dequant, bit_depth),
        TxSize::Tx64x64 => inverse_transform_64x64_dct(dequant, bit_depth),
        TxSize::Tx4x4 | TxSize::Tx8x8 | TxSize::Tx16x16 => {
            unreachable!("small transforms are dispatched before large DCT fallback handling")
        }
        _ => Err(DecoderError::Unsupported(format!(
            "AV1 {tx_size:?} rectangular transform is not supported yet"
        ))),
    }
}

fn inverse_transform_rect(
    tx_type: TxType,
    tx_size: TxSize,
    dequant: &[i32],
    bit_depth: u8,
) -> Vec<i32> {
    let width = tx_size.width();
    let height = tx_size.height();
    let (vertical, horizontal) = staged_transform_pair(tx_type);
    let row_shift = match tx_size {
        TxSize::Tx4x8 | TxSize::Tx8x4 => 0,
        TxSize::Tx8x16 | TxSize::Tx16x8 | TxSize::Tx4x16 | TxSize::Tx16x4 => 1,
        TxSize::Tx16x32 | TxSize::Tx32x16 | TxSize::Tx32x64 | TxSize::Tx64x32 => 1,
        TxSize::Tx8x32 | TxSize::Tx32x8 | TxSize::Tx16x64 | TxSize::Tx64x16 => 2,
        _ => unreachable!("rectangular transform dimensions are unsupported"),
    };
    let row_range = bit_depth + 8;
    // AOM normalizes rectangular transforms whose log2 aspect ratio is odd
    // with 1/sqrt(2) before the row transform.
    let rect_log_ratio = width.ilog2().abs_diff(height.ilog2());
    let scale_rectangular_input = rect_log_ratio % 2 == 1;
    let mut intermediate = vec![0i32; width * height];
    for row in 0..height {
        let input = (0..width)
            .map(|column| {
                let value = dequant[column * height + row];
                let value = if scale_rectangular_input {
                    round_shift_i64(i64::from(value) * NEW_INV_SQRT2, NEW_SQRT2_BITS) as i32
                } else {
                    value
                };
                clamp_signed(value, bit_depth + 8)
            })
            .collect::<Vec<_>>();
        let values = inverse_staged_dynamic(horizontal, &input, row_range);
        for column in 0..width {
            intermediate[row * width + column] = round2_signed(values[column], row_shift);
        }
    }

    let residual_limit = 1i32 << (bit_depth + 7);
    let mut output = vec![0i32; width * height];
    for column in 0..width {
        let input = (0..height)
            .map(|row| clamp_signed(intermediate[row * width + column], 16))
            .collect::<Vec<_>>();
        let values = inverse_staged_dynamic(vertical, &input, 16);
        for row in 0..height {
            output[row * width + column] =
                round2_signed(values[row], 4).clamp(-residual_limit, residual_limit - 1);
        }
    }
    output
}

fn inverse_staged_dynamic(transform: StagedTransform, input: &[i32], range: u8) -> Vec<i32> {
    match input.len() {
        4 => inverse_staged_4(transform, input.try_into().expect("length checked"), range).to_vec(),
        8 => inverse_staged_8(transform, input.try_into().expect("length checked"), range).to_vec(),
        16 => {
            inverse_staged_16(transform, input.try_into().expect("length checked"), range).to_vec()
        }
        32 if transform == StagedTransform::Dct => {
            inverse_dct32(input.try_into().expect("length checked"), range).to_vec()
        }
        64 if transform == StagedTransform::Dct => inverse_dct64_1d(input, range),
        _ => unreachable!("rectangular transform stage length is unsupported"),
    }
}

fn inverse_dct32(i: [i32; 32], r: u8) -> [i32; 32] {
    const B: u8 = 12;
    let p = [
        i[0], i[16], i[8], i[24], i[4], i[20], i[12], i[28], i[2], i[18], i[10], i[26], i[6],
        i[22], i[14], i[30], i[1], i[17], i[9], i[25], i[5], i[21], i[13], i[29], i[3], i[19],
        i[11], i[27], i[7], i[23], i[15], i[31],
    ];
    let a = [
        p[0],
        p[1],
        p[2],
        p[3],
        p[4],
        p[5],
        p[6],
        p[7],
        p[8],
        p[9],
        p[10],
        p[11],
        p[12],
        p[13],
        p[14],
        p[15],
        half_btf(cospi(62), p[16], -cospi(2), p[31], B),
        half_btf(cospi(30), p[17], -cospi(34), p[30], B),
        half_btf(cospi(46), p[18], -cospi(18), p[29], B),
        half_btf(cospi(14), p[19], -cospi(50), p[28], B),
        half_btf(cospi(54), p[20], -cospi(10), p[27], B),
        half_btf(cospi(22), p[21], -cospi(42), p[26], B),
        half_btf(cospi(38), p[22], -cospi(26), p[25], B),
        half_btf(cospi(6), p[23], -cospi(58), p[24], B),
        half_btf(cospi(58), p[23], cospi(6), p[24], B),
        half_btf(cospi(26), p[22], cospi(38), p[25], B),
        half_btf(cospi(42), p[21], cospi(22), p[26], B),
        half_btf(cospi(10), p[20], cospi(54), p[27], B),
        half_btf(cospi(50), p[19], cospi(14), p[28], B),
        half_btf(cospi(18), p[18], cospi(46), p[29], B),
        half_btf(cospi(34), p[17], cospi(30), p[30], B),
        half_btf(cospi(2), p[16], cospi(62), p[31], B),
    ];
    let b = [
        a[0],
        a[1],
        a[2],
        a[3],
        a[4],
        a[5],
        a[6],
        a[7],
        half_btf(cospi(60), a[8], -cospi(4), a[15], B),
        half_btf(cospi(28), a[9], -cospi(36), a[14], B),
        half_btf(cospi(44), a[10], -cospi(20), a[13], B),
        half_btf(cospi(12), a[11], -cospi(52), a[12], B),
        half_btf(cospi(52), a[11], cospi(12), a[12], B),
        half_btf(cospi(20), a[10], cospi(44), a[13], B),
        half_btf(cospi(36), a[9], cospi(28), a[14], B),
        half_btf(cospi(4), a[8], cospi(60), a[15], B),
        clamp_signed(a[16] + a[17], r),
        clamp_signed(a[16] - a[17], r),
        clamp_signed(-a[18] + a[19], r),
        clamp_signed(a[18] + a[19], r),
        clamp_signed(a[20] + a[21], r),
        clamp_signed(a[20] - a[21], r),
        clamp_signed(-a[22] + a[23], r),
        clamp_signed(a[22] + a[23], r),
        clamp_signed(a[24] + a[25], r),
        clamp_signed(a[24] - a[25], r),
        clamp_signed(-a[26] + a[27], r),
        clamp_signed(a[26] + a[27], r),
        clamp_signed(a[28] + a[29], r),
        clamp_signed(a[28] - a[29], r),
        clamp_signed(-a[30] + a[31], r),
        clamp_signed(a[30] + a[31], r),
    ];
    let c = [
        b[0],
        b[1],
        b[2],
        b[3],
        half_btf(cospi(56), b[4], -cospi(8), b[7], B),
        half_btf(cospi(24), b[5], -cospi(40), b[6], B),
        half_btf(cospi(40), b[5], cospi(24), b[6], B),
        half_btf(cospi(8), b[4], cospi(56), b[7], B),
        clamp_signed(b[8] + b[9], r),
        clamp_signed(b[8] - b[9], r),
        clamp_signed(-b[10] + b[11], r),
        clamp_signed(b[10] + b[11], r),
        clamp_signed(b[12] + b[13], r),
        clamp_signed(b[12] - b[13], r),
        clamp_signed(-b[14] + b[15], r),
        clamp_signed(b[14] + b[15], r),
        b[16],
        half_btf(-cospi(8), b[17], cospi(56), b[30], B),
        half_btf(-cospi(56), b[18], -cospi(8), b[29], B),
        b[19],
        b[20],
        half_btf(-cospi(40), b[21], cospi(24), b[26], B),
        half_btf(-cospi(24), b[22], -cospi(40), b[25], B),
        b[23],
        b[24],
        half_btf(-cospi(40), b[22], cospi(24), b[25], B),
        half_btf(cospi(24), b[21], cospi(40), b[26], B),
        b[27],
        b[28],
        half_btf(-cospi(8), b[18], cospi(56), b[29], B),
        half_btf(cospi(56), b[17], cospi(8), b[30], B),
        b[31],
    ];
    let d = [
        half_btf(cospi(32), c[0], cospi(32), c[1], B),
        half_btf(cospi(32), c[0], -cospi(32), c[1], B),
        half_btf(cospi(48), c[2], -cospi(16), c[3], B),
        half_btf(cospi(16), c[2], cospi(48), c[3], B),
        clamp_signed(c[4] + c[5], r),
        clamp_signed(c[4] - c[5], r),
        clamp_signed(-c[6] + c[7], r),
        clamp_signed(c[6] + c[7], r),
        c[8],
        half_btf(-cospi(16), c[9], cospi(48), c[14], B),
        half_btf(-cospi(48), c[10], -cospi(16), c[13], B),
        c[11],
        c[12],
        half_btf(-cospi(16), c[10], cospi(48), c[13], B),
        half_btf(cospi(48), c[9], cospi(16), c[14], B),
        c[15],
        clamp_signed(c[16] + c[19], r),
        clamp_signed(c[17] + c[18], r),
        clamp_signed(c[17] - c[18], r),
        clamp_signed(c[16] - c[19], r),
        clamp_signed(-c[20] + c[23], r),
        clamp_signed(-c[21] + c[22], r),
        clamp_signed(c[21] + c[22], r),
        clamp_signed(c[20] + c[23], r),
        clamp_signed(c[24] + c[27], r),
        clamp_signed(c[25] + c[26], r),
        clamp_signed(c[25] - c[26], r),
        clamp_signed(c[24] - c[27], r),
        clamp_signed(-c[28] + c[31], r),
        clamp_signed(-c[29] + c[30], r),
        clamp_signed(c[29] + c[30], r),
        clamp_signed(c[28] + c[31], r),
    ];
    let e = [
        clamp_signed(d[0] + d[3], r),
        clamp_signed(d[1] + d[2], r),
        clamp_signed(d[1] - d[2], r),
        clamp_signed(d[0] - d[3], r),
        d[4],
        half_btf(-cospi(32), d[5], cospi(32), d[6], B),
        half_btf(cospi(32), d[5], cospi(32), d[6], B),
        d[7],
        clamp_signed(d[8] + d[11], r),
        clamp_signed(d[9] + d[10], r),
        clamp_signed(d[9] - d[10], r),
        clamp_signed(d[8] - d[11], r),
        clamp_signed(-d[12] + d[15], r),
        clamp_signed(-d[13] + d[14], r),
        clamp_signed(d[13] + d[14], r),
        clamp_signed(d[12] + d[15], r),
        d[16],
        d[17],
        half_btf(-cospi(16), d[18], cospi(48), d[29], B),
        half_btf(-cospi(16), d[19], cospi(48), d[28], B),
        half_btf(-cospi(48), d[20], -cospi(16), d[27], B),
        half_btf(-cospi(48), d[21], -cospi(16), d[26], B),
        d[22],
        d[23],
        d[24],
        d[25],
        half_btf(-cospi(16), d[21], cospi(48), d[26], B),
        half_btf(-cospi(16), d[20], cospi(48), d[27], B),
        half_btf(cospi(48), d[19], cospi(16), d[28], B),
        half_btf(cospi(48), d[18], cospi(16), d[29], B),
        d[30],
        d[31],
    ];
    let f = [
        clamp_signed(e[0] + e[7], r),
        clamp_signed(e[1] + e[6], r),
        clamp_signed(e[2] + e[5], r),
        clamp_signed(e[3] + e[4], r),
        clamp_signed(e[3] - e[4], r),
        clamp_signed(e[2] - e[5], r),
        clamp_signed(e[1] - e[6], r),
        clamp_signed(e[0] - e[7], r),
        e[8],
        e[9],
        half_btf(-cospi(32), e[10], cospi(32), e[13], B),
        half_btf(-cospi(32), e[11], cospi(32), e[12], B),
        half_btf(cospi(32), e[11], cospi(32), e[12], B),
        half_btf(cospi(32), e[10], cospi(32), e[13], B),
        e[14],
        e[15],
        clamp_signed(e[16] + e[23], r),
        clamp_signed(e[17] + e[22], r),
        clamp_signed(e[18] + e[21], r),
        clamp_signed(e[19] + e[20], r),
        clamp_signed(e[19] - e[20], r),
        clamp_signed(e[18] - e[21], r),
        clamp_signed(e[17] - e[22], r),
        clamp_signed(e[16] - e[23], r),
        clamp_signed(-e[24] + e[31], r),
        clamp_signed(-e[25] + e[30], r),
        clamp_signed(-e[26] + e[29], r),
        clamp_signed(-e[27] + e[28], r),
        clamp_signed(e[27] + e[28], r),
        clamp_signed(e[26] + e[29], r),
        clamp_signed(e[25] + e[30], r),
        clamp_signed(e[24] + e[31], r),
    ];
    let g = [
        clamp_signed(f[0] + f[15], r),
        clamp_signed(f[1] + f[14], r),
        clamp_signed(f[2] + f[13], r),
        clamp_signed(f[3] + f[12], r),
        clamp_signed(f[4] + f[11], r),
        clamp_signed(f[5] + f[10], r),
        clamp_signed(f[6] + f[9], r),
        clamp_signed(f[7] + f[8], r),
        clamp_signed(f[7] - f[8], r),
        clamp_signed(f[6] - f[9], r),
        clamp_signed(f[5] - f[10], r),
        clamp_signed(f[4] - f[11], r),
        clamp_signed(f[3] - f[12], r),
        clamp_signed(f[2] - f[13], r),
        clamp_signed(f[1] - f[14], r),
        clamp_signed(f[0] - f[15], r),
        f[16],
        f[17],
        f[18],
        f[19],
        half_btf(-cospi(32), f[20], cospi(32), f[27], B),
        half_btf(-cospi(32), f[21], cospi(32), f[26], B),
        half_btf(-cospi(32), f[22], cospi(32), f[25], B),
        half_btf(-cospi(32), f[23], cospi(32), f[24], B),
        half_btf(cospi(32), f[23], cospi(32), f[24], B),
        half_btf(cospi(32), f[22], cospi(32), f[25], B),
        half_btf(cospi(32), f[21], cospi(32), f[26], B),
        half_btf(cospi(32), f[20], cospi(32), f[27], B),
        f[28],
        f[29],
        f[30],
        f[31],
    ];
    std::array::from_fn(|x| {
        if x < 16 {
            clamp_signed(g[x] + g[31 - x], r)
        } else {
            clamp_signed(g[31 - x] - g[x], r)
        }
    })
}

fn inverse_dct64_1d(input: &[i32], range: u8) -> Vec<i32> {
    dct64::inverse_dct64(input.try_into().expect("length checked"), range).to_vec()
}

fn inverse_transform_32x32_dct(dequant: &[i32], bit_depth: u8) -> Result<Vec<i32>, DecoderError> {
    if dequant.len() != TxSize::Tx32x32.sample_count() {
        return Err(DecoderError::InvalidParam(
            "AV1 32x32 DCT coefficient count does not match transform size".to_string(),
        ));
    }
    const SIDE: usize = 32;
    let row_range = bit_depth + 8;
    let mut intermediate = vec![0i32; SIDE * SIDE];
    for row in 0..SIDE {
        let input =
            std::array::from_fn(|column| clamp_signed(dequant[row * SIDE + column], bit_depth + 8));
        let transformed = inverse_dct32(input, row_range);
        for column in 0..SIDE {
            intermediate[row * SIDE + column] = round2_signed(transformed[column], 2);
        }
    }

    let residual_limit = 1i32 << (bit_depth + 7);
    let mut output = vec![0i32; SIDE * SIDE];
    for column in 0..SIDE {
        let input = std::array::from_fn(|row| clamp_signed(intermediate[row * SIDE + column], 16));
        let transformed = inverse_dct32(input, 16);
        for row in 0..SIDE {
            output[row * SIDE + column] =
                round2_signed(transformed[row], 4).clamp(-residual_limit, residual_limit - 1);
        }
    }
    Ok(output)
}

fn inverse_transform_64x64_dct(dequant: &[i32], bit_depth: u8) -> Result<Vec<i32>, DecoderError> {
    if has_non_zero_outside_tx64_coded_top_left(dequant) {
        return Err(DecoderError::InvalidParam(
            "AV1 64x64 DCT coefficients outside the coded top-left 32x32 area must be zero"
                .to_string(),
        ));
    }
    if dequant.len() != TxSize::Tx64x64.sample_count() {
        return Err(DecoderError::InvalidParam(
            "AV1 64x64 DCT coefficient count does not match transform size".to_string(),
        ));
    }
    Ok(inverse_dct64_fixed_basis(dequant, bit_depth))
}

fn inverse_dct64_fixed_basis(dequant: &[i32], bit_depth: u8) -> Vec<i32> {
    const SIDE: usize = 64;
    debug_assert_eq!(dequant.len(), SIDE * SIDE);

    if dequant[1..].iter().all(|coefficient| *coefficient == 0) {
        return inverse_dct64_dc_only(dequant[0], bit_depth);
    }

    let basis = inverse_dct64_basis_table();
    inverse_square_dct_fixed_basis::<SIDE>(&basis, dequant, bit_depth)
}

fn inverse_dct64_dc_only(dc: i32, bit_depth: u8) -> Vec<i32> {
    const COS_BIT: u8 = 12;
    let row = round_shift_i64(i64::from(dc) * i64::from(cospi(32)), COS_BIT) as i32;
    let row = round2_signed(row, 2);
    let column = round_shift_i64(i64::from(row) * i64::from(cospi(32)), COS_BIT) as i32;
    let residual_limit = 1i32 << (bit_depth + 7);
    let value = round2_signed(column, 4).clamp(-residual_limit, residual_limit - 1);
    vec![value; TxSize::Tx64x64.sample_count()]
}

fn inverse_square_dct_fixed_basis<const SIDE: usize>(
    basis: &[i32],
    dequant: &[i32],
    bit_depth: u8,
) -> Vec<i32> {
    const BASIS_BITS: u8 = 12;
    debug_assert_eq!(basis.len(), SIDE * SIDE);
    debug_assert_eq!(dequant.len(), SIDE * SIDE);

    let mut temp = vec![0i64; SIDE * SIDE];
    let mut out = vec![0i32; SIDE * SIDE];
    let residual_limit = 1i32 << (bit_depth + 7);

    for row in 0..SIDE {
        for x in 0..SIDE {
            let mut sum = 0i64;
            for u in 0..SIDE {
                sum += i64::from(basis[x * SIDE + u]) * i64::from(dequant[row * SIDE + u]);
            }
            temp[row * SIDE + x] = sum;
        }
    }

    for y in 0..SIDE {
        for x in 0..SIDE {
            let mut sum = 0i64;
            for v in 0..SIDE {
                sum += i64::from(basis[y * SIDE + v]) * temp[v * SIDE + x];
            }
            out[y * SIDE + x] = round_shift_i64(sum, BASIS_BITS * 2)
                .clamp(i64::from(-residual_limit), i64::from(residual_limit - 1))
                as i32;
        }
    }

    out
}

fn inverse_dct64_basis_table() -> [i32; 64 * 64] {
    std::array::from_fn(|index| {
        let sample = index / 64;
        let coeff = index % 64;
        inverse_dct64_basis(sample, coeff)
    })
}

fn inverse_dct64_basis(sample: usize, coeff: usize) -> i32 {
    if coeff == 0 {
        return 512;
    }
    round_shift_i64(
        i64::from(cospi_unit_128((2 * sample + 1) * coeff)) * NEW_SQRT2,
        NEW_SQRT2_BITS + 3,
    ) as i32
}

fn cospi_unit_128(unit: usize) -> i32 {
    let unit = unit % 256;
    let unit = if unit > 128 { 256 - unit } else { unit };
    if unit == 64 {
        return 0;
    }
    if unit <= 64 {
        cospi(unit)
    } else {
        -cospi(128 - unit)
    }
}

fn validate_transform_bit_depth(bit_depth: u8) -> Result<(), DecoderError> {
    if matches!(bit_depth, 8 | 10 | 12) {
        Ok(())
    } else {
        Err(DecoderError::InvalidParam(format!(
            "AV1 {bit_depth}-bit transform is not supported"
        )))
    }
}

fn has_non_zero_outside_tx64_coded_top_left(coefficients: &[i32]) -> bool {
    coefficients
        .iter()
        .enumerate()
        .any(|(index, coefficient)| *coefficient != 0 && (index / 64 >= 32 || index % 64 >= 32))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StagedTransform {
    Dct,
    Adst,
    Identity,
}

fn inverse_transform_4x4(tx_type: TxType, dequant: &[i32], bit_depth: u8) -> Vec<i32> {
    let (vertical, horizontal) = staged_transform_pair(tx_type);
    let row_range = bit_depth + 8;
    let mut intermediate = [0i32; 16];
    for row in 0..4 {
        let input =
            std::array::from_fn(|column| clamp_signed(dequant[row * 4 + column], bit_depth + 8));
        let output = inverse_staged_4(horizontal, input, row_range);
        intermediate[row * 4..row * 4 + 4].copy_from_slice(&output);
    }

    let residual_limit = 1i32 << (bit_depth + 7);
    let mut output = vec![0i32; 16];
    for column in 0..4 {
        let input = std::array::from_fn(|row| clamp_signed(intermediate[row * 4 + column], 16));
        let transformed = inverse_staged_4(vertical, input, 16);
        for row in 0..4 {
            output[row * 4 + column] =
                round2_signed(transformed[row], 4).clamp(-residual_limit, residual_limit - 1);
        }
    }
    output
}

fn inverse_staged_4(transform: StagedTransform, input: [i32; 4], range: u8) -> [i32; 4] {
    match transform {
        StagedTransform::Dct => inverse_dct4(input, range),
        StagedTransform::Adst => inverse_adst4(input),
        StagedTransform::Identity => {
            input.map(|value| round_shift_i64(i64::from(value) * NEW_SQRT2, NEW_SQRT2_BITS) as i32)
        }
    }
}

fn inverse_transform_8x8(tx_type: TxType, dequant: &[i32], bit_depth: u8) -> Vec<i32> {
    let (vertical, horizontal) = staged_transform_pair(tx_type);
    let row_range = bit_depth + 8;
    let mut intermediate = [0i32; 64];
    for row in 0..8 {
        let input =
            std::array::from_fn(|column| clamp_signed(dequant[row * 8 + column], bit_depth + 8));
        let transformed = inverse_staged_8(horizontal, input, row_range);
        for column in 0..8 {
            intermediate[row * 8 + column] = round2_signed(transformed[column], 1);
        }
    }

    let residual_limit = 1i32 << (bit_depth + 7);
    let mut output = vec![0i32; 64];
    for column in 0..8 {
        let input = std::array::from_fn(|row| clamp_signed(intermediate[row * 8 + column], 16));
        let transformed = inverse_staged_8(vertical, input, 16);
        for row in 0..8 {
            output[row * 8 + column] =
                round2_signed(transformed[row], 4).clamp(-residual_limit, residual_limit - 1);
        }
    }
    output
}

fn staged_transform_pair(tx_type: TxType) -> (StagedTransform, StagedTransform) {
    match tx_type {
        TxType::DctDct => (StagedTransform::Dct, StagedTransform::Dct),
        TxType::AdstDct => (StagedTransform::Adst, StagedTransform::Dct),
        TxType::DctAdst => (StagedTransform::Dct, StagedTransform::Adst),
        TxType::AdstAdst => (StagedTransform::Adst, StagedTransform::Adst),
        TxType::Identity => (StagedTransform::Identity, StagedTransform::Identity),
        TxType::VerticalDct => (StagedTransform::Dct, StagedTransform::Identity),
        TxType::HorizontalDct => (StagedTransform::Identity, StagedTransform::Dct),
    }
}

fn inverse_staged_8(transform: StagedTransform, input: [i32; 8], range: u8) -> [i32; 8] {
    match transform {
        StagedTransform::Dct => inverse_dct8(input, range),
        StagedTransform::Adst => inverse_adst8(input, range),
        StagedTransform::Identity => input.map(|value| value.saturating_mul(2)),
    }
}

fn inverse_dct8(input: [i32; 8], range: u8) -> [i32; 8] {
    const COS_BIT: u8 = 12;
    let s2 = [
        input[0],
        input[4],
        input[2],
        input[6],
        half_btf(cospi(56), input[1], -cospi(8), input[7], COS_BIT),
        half_btf(cospi(24), input[5], -cospi(40), input[3], COS_BIT),
        half_btf(cospi(40), input[5], cospi(24), input[3], COS_BIT),
        half_btf(cospi(8), input[1], cospi(56), input[7], COS_BIT),
    ];
    let s3 = [
        half_btf(cospi(32), s2[0], cospi(32), s2[1], COS_BIT),
        half_btf(cospi(32), s2[0], -cospi(32), s2[1], COS_BIT),
        half_btf(cospi(48), s2[2], -cospi(16), s2[3], COS_BIT),
        half_btf(cospi(16), s2[2], cospi(48), s2[3], COS_BIT),
        clamp_signed(s2[4] + s2[5], range),
        clamp_signed(s2[4] - s2[5], range),
        clamp_signed(-s2[6] + s2[7], range),
        clamp_signed(s2[6] + s2[7], range),
    ];
    let s4 = [
        clamp_signed(s3[0] + s3[3], range),
        clamp_signed(s3[1] + s3[2], range),
        clamp_signed(s3[1] - s3[2], range),
        clamp_signed(s3[0] - s3[3], range),
        s3[4],
        half_btf(-cospi(32), s3[5], cospi(32), s3[6], COS_BIT),
        half_btf(cospi(32), s3[5], cospi(32), s3[6], COS_BIT),
        s3[7],
    ];
    [
        clamp_signed(s4[0] + s4[7], range),
        clamp_signed(s4[1] + s4[6], range),
        clamp_signed(s4[2] + s4[5], range),
        clamp_signed(s4[3] + s4[4], range),
        clamp_signed(s4[3] - s4[4], range),
        clamp_signed(s4[2] - s4[5], range),
        clamp_signed(s4[1] - s4[6], range),
        clamp_signed(s4[0] - s4[7], range),
    ]
}

fn inverse_adst8(input: [i32; 8], range: u8) -> [i32; 8] {
    const COS_BIT: u8 = 12;
    let r = [
        input[7], input[0], input[5], input[2], input[3], input[4], input[1], input[6],
    ];
    let s2 = [
        half_btf(cospi(4), r[0], cospi(60), r[1], COS_BIT),
        half_btf(cospi(60), r[0], -cospi(4), r[1], COS_BIT),
        half_btf(cospi(20), r[2], cospi(44), r[3], COS_BIT),
        half_btf(cospi(44), r[2], -cospi(20), r[3], COS_BIT),
        half_btf(cospi(36), r[4], cospi(28), r[5], COS_BIT),
        half_btf(cospi(28), r[4], -cospi(36), r[5], COS_BIT),
        half_btf(cospi(52), r[6], cospi(12), r[7], COS_BIT),
        half_btf(cospi(12), r[6], -cospi(52), r[7], COS_BIT),
    ];
    let s3 = [
        clamp_signed(s2[0] + s2[4], range),
        clamp_signed(s2[1] + s2[5], range),
        clamp_signed(s2[2] + s2[6], range),
        clamp_signed(s2[3] + s2[7], range),
        clamp_signed(s2[0] - s2[4], range),
        clamp_signed(s2[1] - s2[5], range),
        clamp_signed(s2[2] - s2[6], range),
        clamp_signed(s2[3] - s2[7], range),
    ];
    let s4 = [
        s3[0],
        s3[1],
        s3[2],
        s3[3],
        half_btf(cospi(16), s3[4], cospi(48), s3[5], COS_BIT),
        half_btf(cospi(48), s3[4], -cospi(16), s3[5], COS_BIT),
        half_btf(-cospi(48), s3[6], cospi(16), s3[7], COS_BIT),
        half_btf(cospi(16), s3[6], cospi(48), s3[7], COS_BIT),
    ];
    let s5 = [
        clamp_signed(s4[0] + s4[2], range),
        clamp_signed(s4[1] + s4[3], range),
        clamp_signed(s4[0] - s4[2], range),
        clamp_signed(s4[1] - s4[3], range),
        clamp_signed(s4[4] + s4[6], range),
        clamp_signed(s4[5] + s4[7], range),
        clamp_signed(s4[4] - s4[6], range),
        clamp_signed(s4[5] - s4[7], range),
    ];
    let s6 = [
        s5[0],
        s5[1],
        half_btf(cospi(32), s5[2], cospi(32), s5[3], COS_BIT),
        half_btf(cospi(32), s5[2], -cospi(32), s5[3], COS_BIT),
        s5[4],
        s5[5],
        half_btf(cospi(32), s5[6], cospi(32), s5[7], COS_BIT),
        half_btf(cospi(32), s5[6], -cospi(32), s5[7], COS_BIT),
    ];
    [s6[0], -s6[4], s6[6], -s6[2], s6[3], -s6[7], s6[5], -s6[1]]
}

fn cospi(index: usize) -> i32 {
    const VALUES: [i32; 64] = [
        4096, 4095, 4091, 4085, 4076, 4065, 4052, 4036, 4017, 3996, 3973, 3948, 3920, 3889, 3857,
        3822, 3784, 3745, 3703, 3659, 3612, 3564, 3513, 3461, 3406, 3349, 3290, 3229, 3166, 3102,
        3035, 2967, 2896, 2824, 2751, 2675, 2598, 2520, 2440, 2359, 2276, 2191, 2106, 2019, 1931,
        1842, 1751, 1660, 1567, 1474, 1380, 1285, 1189, 1092, 995, 897, 799, 700, 601, 501, 401,
        301, 201, 101,
    ];
    VALUES[index]
}

fn inverse_transform_16x16(tx_type: TxType, dequant: &[i32], bit_depth: u8) -> Vec<i32> {
    let (vertical, horizontal) = staged_transform_pair(tx_type);
    let mut temp = vec![0i32; 256];
    for row in 0..16 {
        let input = std::array::from_fn(|x| clamp_signed(dequant[row * 16 + x], bit_depth + 8));
        let values = inverse_staged_16(horizontal, input, 16);
        for x in 0..16 {
            temp[row * 16 + x] = round2_signed(values[x], 2);
        }
    }
    let limit = 1i32 << (bit_depth + 7);
    let mut out = vec![0i32; 256];
    for x in 0..16 {
        let input = std::array::from_fn(|y| clamp_signed(temp[y * 16 + x], 16));
        let values = inverse_staged_16(vertical, input, 16);
        for y in 0..16 {
            out[y * 16 + x] = round2_signed(values[y], 4).clamp(-limit, limit - 1);
        }
    }
    out
}

fn inverse_staged_16(transform: StagedTransform, input: [i32; 16], range: u8) -> [i32; 16] {
    match transform {
        StagedTransform::Dct => inverse_dct16(input, range),
        StagedTransform::Identity => {
            input.map(|v| round_shift_i64(i64::from(v) * NEW_SQRT2 * 2, NEW_SQRT2_BITS) as i32)
        }
        StagedTransform::Adst => inverse_adst16(input, range),
    }
}

fn inverse_adst16(i: [i32; 16], r: u8) -> [i32; 16] {
    const B: u8 = 12;
    let p = [
        i[15], i[0], i[13], i[2], i[11], i[4], i[9], i[6], i[7], i[8], i[5], i[10], i[3], i[12],
        i[1], i[14],
    ];
    let a = [
        half_btf(cospi(2), p[0], cospi(62), p[1], B),
        half_btf(cospi(62), p[0], -cospi(2), p[1], B),
        half_btf(cospi(10), p[2], cospi(54), p[3], B),
        half_btf(cospi(54), p[2], -cospi(10), p[3], B),
        half_btf(cospi(18), p[4], cospi(46), p[5], B),
        half_btf(cospi(46), p[4], -cospi(18), p[5], B),
        half_btf(cospi(26), p[6], cospi(38), p[7], B),
        half_btf(cospi(38), p[6], -cospi(26), p[7], B),
        half_btf(cospi(34), p[8], cospi(30), p[9], B),
        half_btf(cospi(30), p[8], -cospi(34), p[9], B),
        half_btf(cospi(42), p[10], cospi(22), p[11], B),
        half_btf(cospi(22), p[10], -cospi(42), p[11], B),
        half_btf(cospi(50), p[12], cospi(14), p[13], B),
        half_btf(cospi(14), p[12], -cospi(50), p[13], B),
        half_btf(cospi(58), p[14], cospi(6), p[15], B),
        half_btf(cospi(6), p[14], -cospi(58), p[15], B),
    ];
    let b: [i32; 16] = std::array::from_fn(|x| {
        if x < 8 {
            clamp_signed(a[x] + a[x + 8], r)
        } else {
            clamp_signed(a[x - 8] - a[x], r)
        }
    });
    let c = [
        b[0],
        b[1],
        b[2],
        b[3],
        b[4],
        b[5],
        b[6],
        b[7],
        half_btf(cospi(8), b[8], cospi(56), b[9], B),
        half_btf(cospi(56), b[8], -cospi(8), b[9], B),
        half_btf(cospi(40), b[10], cospi(24), b[11], B),
        half_btf(cospi(24), b[10], -cospi(40), b[11], B),
        half_btf(-cospi(56), b[12], cospi(8), b[13], B),
        half_btf(cospi(8), b[12], cospi(56), b[13], B),
        half_btf(-cospi(24), b[14], cospi(40), b[15], B),
        half_btf(cospi(40), b[14], cospi(24), b[15], B),
    ];
    let d = [
        clamp_signed(c[0] + c[4], r),
        clamp_signed(c[1] + c[5], r),
        clamp_signed(c[2] + c[6], r),
        clamp_signed(c[3] + c[7], r),
        clamp_signed(c[0] - c[4], r),
        clamp_signed(c[1] - c[5], r),
        clamp_signed(c[2] - c[6], r),
        clamp_signed(c[3] - c[7], r),
        clamp_signed(c[8] + c[12], r),
        clamp_signed(c[9] + c[13], r),
        clamp_signed(c[10] + c[14], r),
        clamp_signed(c[11] + c[15], r),
        clamp_signed(c[8] - c[12], r),
        clamp_signed(c[9] - c[13], r),
        clamp_signed(c[10] - c[14], r),
        clamp_signed(c[11] - c[15], r),
    ];
    let e = [
        d[0],
        d[1],
        d[2],
        d[3],
        half_btf(cospi(16), d[4], cospi(48), d[5], B),
        half_btf(cospi(48), d[4], -cospi(16), d[5], B),
        half_btf(-cospi(48), d[6], cospi(16), d[7], B),
        half_btf(cospi(16), d[6], cospi(48), d[7], B),
        d[8],
        d[9],
        d[10],
        d[11],
        half_btf(cospi(16), d[12], cospi(48), d[13], B),
        half_btf(cospi(48), d[12], -cospi(16), d[13], B),
        half_btf(-cospi(48), d[14], cospi(16), d[15], B),
        half_btf(cospi(16), d[14], cospi(48), d[15], B),
    ];
    let f = [
        clamp_signed(e[0] + e[2], r),
        clamp_signed(e[1] + e[3], r),
        clamp_signed(e[0] - e[2], r),
        clamp_signed(e[1] - e[3], r),
        clamp_signed(e[4] + e[6], r),
        clamp_signed(e[5] + e[7], r),
        clamp_signed(e[4] - e[6], r),
        clamp_signed(e[5] - e[7], r),
        clamp_signed(e[8] + e[10], r),
        clamp_signed(e[9] + e[11], r),
        clamp_signed(e[8] - e[10], r),
        clamp_signed(e[9] - e[11], r),
        clamp_signed(e[12] + e[14], r),
        clamp_signed(e[13] + e[15], r),
        clamp_signed(e[12] - e[14], r),
        clamp_signed(e[13] - e[15], r),
    ];
    let g = [
        f[0],
        f[1],
        half_btf(cospi(32), f[2], cospi(32), f[3], B),
        half_btf(cospi(32), f[2], -cospi(32), f[3], B),
        f[4],
        f[5],
        half_btf(cospi(32), f[6], cospi(32), f[7], B),
        half_btf(cospi(32), f[6], -cospi(32), f[7], B),
        f[8],
        f[9],
        half_btf(cospi(32), f[10], cospi(32), f[11], B),
        half_btf(cospi(32), f[10], -cospi(32), f[11], B),
        f[12],
        f[13],
        half_btf(cospi(32), f[14], cospi(32), f[15], B),
        half_btf(cospi(32), f[14], -cospi(32), f[15], B),
    ];
    [
        g[0], -g[8], g[12], -g[4], g[6], -g[14], g[10], -g[2], g[3], -g[11], g[15], -g[7], g[5],
        -g[13], g[9], -g[1],
    ]
}

fn inverse_dct16(i: [i32; 16], r: u8) -> [i32; 16] {
    const B: u8 = 12;
    let p = [
        i[0], i[8], i[4], i[12], i[2], i[10], i[6], i[14], i[1], i[9], i[5], i[13], i[3], i[11],
        i[7], i[15],
    ];
    let a = [
        p[0],
        p[1],
        p[2],
        p[3],
        p[4],
        p[5],
        p[6],
        p[7],
        half_btf(cospi(60), p[8], -cospi(4), p[15], B),
        half_btf(cospi(28), p[9], -cospi(36), p[14], B),
        half_btf(cospi(44), p[10], -cospi(20), p[13], B),
        half_btf(cospi(12), p[11], -cospi(52), p[12], B),
        half_btf(cospi(52), p[11], cospi(12), p[12], B),
        half_btf(cospi(20), p[10], cospi(44), p[13], B),
        half_btf(cospi(36), p[9], cospi(28), p[14], B),
        half_btf(cospi(4), p[8], cospi(60), p[15], B),
    ];
    let b = [
        a[0],
        a[1],
        a[2],
        a[3],
        half_btf(cospi(56), a[4], -cospi(8), a[7], B),
        half_btf(cospi(24), a[5], -cospi(40), a[6], B),
        half_btf(cospi(40), a[5], cospi(24), a[6], B),
        half_btf(cospi(8), a[4], cospi(56), a[7], B),
        clamp_signed(a[8] + a[9], r),
        clamp_signed(a[8] - a[9], r),
        clamp_signed(-a[10] + a[11], r),
        clamp_signed(a[10] + a[11], r),
        clamp_signed(a[12] + a[13], r),
        clamp_signed(a[12] - a[13], r),
        clamp_signed(-a[14] + a[15], r),
        clamp_signed(a[14] + a[15], r),
    ];
    let c = [
        half_btf(cospi(32), b[0], cospi(32), b[1], B),
        half_btf(cospi(32), b[0], -cospi(32), b[1], B),
        half_btf(cospi(48), b[2], -cospi(16), b[3], B),
        half_btf(cospi(16), b[2], cospi(48), b[3], B),
        clamp_signed(b[4] + b[5], r),
        clamp_signed(b[4] - b[5], r),
        clamp_signed(-b[6] + b[7], r),
        clamp_signed(b[6] + b[7], r),
        b[8],
        half_btf(-cospi(16), b[9], cospi(48), b[14], B),
        half_btf(-cospi(48), b[10], -cospi(16), b[13], B),
        b[11],
        b[12],
        half_btf(-cospi(16), b[10], cospi(48), b[13], B),
        half_btf(cospi(48), b[9], cospi(16), b[14], B),
        b[15],
    ];
    let d = [
        clamp_signed(c[0] + c[3], r),
        clamp_signed(c[1] + c[2], r),
        clamp_signed(c[1] - c[2], r),
        clamp_signed(c[0] - c[3], r),
        c[4],
        half_btf(-cospi(32), c[5], cospi(32), c[6], B),
        half_btf(cospi(32), c[5], cospi(32), c[6], B),
        c[7],
        clamp_signed(c[8] + c[11], r),
        clamp_signed(c[9] + c[10], r),
        clamp_signed(c[9] - c[10], r),
        clamp_signed(c[8] - c[11], r),
        clamp_signed(-c[12] + c[15], r),
        clamp_signed(-c[13] + c[14], r),
        clamp_signed(c[13] + c[14], r),
        clamp_signed(c[12] + c[15], r),
    ];
    let e = [
        clamp_signed(d[0] + d[7], r),
        clamp_signed(d[1] + d[6], r),
        clamp_signed(d[2] + d[5], r),
        clamp_signed(d[3] + d[4], r),
        clamp_signed(d[3] - d[4], r),
        clamp_signed(d[2] - d[5], r),
        clamp_signed(d[1] - d[6], r),
        clamp_signed(d[0] - d[7], r),
        d[8],
        d[9],
        half_btf(-cospi(32), d[10], cospi(32), d[13], B),
        half_btf(-cospi(32), d[11], cospi(32), d[12], B),
        half_btf(cospi(32), d[11], cospi(32), d[12], B),
        half_btf(cospi(32), d[10], cospi(32), d[13], B),
        d[14],
        d[15],
    ];
    std::array::from_fn(|x| {
        if x < 8 {
            clamp_signed(e[x] + e[15 - x], r)
        } else {
            clamp_signed(e[15 - x] - e[x], r)
        }
    })
}

fn inverse_dct4(input: [i32; 4], range: u8) -> [i32; 4] {
    const COSPI_16: i32 = 3784;
    const COSPI_32: i32 = 2896;
    const COSPI_48: i32 = 1567;
    const COS_BIT: u8 = 12;

    let stage2 = [
        half_btf(COSPI_32, input[0], COSPI_32, input[2], COS_BIT),
        half_btf(COSPI_32, input[0], -COSPI_32, input[2], COS_BIT),
        half_btf(COSPI_48, input[1], -COSPI_16, input[3], COS_BIT),
        half_btf(COSPI_16, input[1], COSPI_48, input[3], COS_BIT),
    ];
    [
        clamp_signed(stage2[0] + stage2[3], range),
        clamp_signed(stage2[1] + stage2[2], range),
        clamp_signed(stage2[1] - stage2[2], range),
        clamp_signed(stage2[0] - stage2[3], range),
    ]
}

fn inverse_adst4(input: [i32; 4]) -> [i32; 4] {
    const SINPI_1_9: i64 = 1321;
    const SINPI_2_9: i64 = 2482;
    const SINPI_3_9: i64 = 3344;
    const SINPI_4_9: i64 = 3803;
    const SIN_BIT: u8 = 12;

    if input.iter().all(|value| *value == 0) {
        return [0; 4];
    }
    let x0 = i64::from(input[0]);
    let x1 = i64::from(input[1]);
    let x2 = i64::from(input[2]);
    let x3 = i64::from(input[3]);
    let mut s0 = SINPI_1_9 * x0 + SINPI_4_9 * x2 + SINPI_2_9 * x3;
    let mut s1 = SINPI_2_9 * x0 - SINPI_1_9 * x2 - SINPI_4_9 * x3;
    let s3 = SINPI_3_9 * x1;
    let s2 = SINPI_3_9 * (x0 - x2 + x3);
    s0 += s3;
    s1 += s3;
    [
        round_shift_i64(s0, SIN_BIT) as i32,
        round_shift_i64(s1, SIN_BIT) as i32,
        round_shift_i64(s2, SIN_BIT) as i32,
        round_shift_i64(s0 + s1 - 3 * s3, SIN_BIT) as i32,
    ]
}

fn half_btf(weight0: i32, input0: i32, weight1: i32, input1: i32, bits: u8) -> i32 {
    round_shift_i64(
        i64::from(weight0) * i64::from(input0) + i64::from(weight1) * i64::from(input1),
        bits,
    ) as i32
}

fn clamp_signed(value: i32, bits: u8) -> i32 {
    let limit = 1i32 << (bits - 1);
    value.clamp(-limit, limit - 1)
}

fn round_shift_i64(value: i64, bits: u8) -> i64 {
    if bits == 0 {
        value
    } else {
        (value + (1i64 << (bits - 1))) >> bits
    }
}

const NEW_SQRT2: i64 = 5793;
const NEW_INV_SQRT2: i64 = 2896;
const NEW_SQRT2_BITS: u8 = 12;

fn round2_signed(value: i32, bits: u8) -> i32 {
    round_shift_i64(i64::from(value), bits).clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
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
    fn zig_zag_scan_orders_aom_rectangular_transform() {
        assert_eq!(
            zig_zag_scan(TxSize::Tx8x4),
            vec![
                0, 1, 4, 2, 5, 8, 3, 6, 9, 12, 7, 10, 13, 16, 11, 14, 17, 20, 15, 18, 21, 24, 19,
                22, 25, 28, 23, 26, 29, 27, 30, 31,
            ]
        );
    }

    #[test]
    fn tx64_scan_codes_top_left_32_square_only() {
        let scan = zig_zag_scan(TxSize::Tx64x64);

        assert_eq!(scan.len(), TxSize::Tx32x32.sample_count());
        assert_eq!(&scan[..6], &[0, 64, 1, 2, 65, 128]);
        assert_eq!(
            scan.last().copied(),
            Some(31 * TxSize::Tx64x64.width() + 31)
        );
        assert!(
            scan.iter()
                .all(|position| position / TxSize::Tx64x64.width() < 32
                    && position % TxSize::Tx64x64.width() < 32)
        );
        assert!(!scan.contains(&(32 * TxSize::Tx64x64.width())));
        assert!(!scan.contains(&32));
    }

    #[test]
    fn coefficients_from_scan_places_values_in_raster_slots() {
        let coefficients =
            coefficients_from_scan(TxSize::Tx4x4, TxType::DctDct, &[1, 2, 3]).unwrap();

        assert_eq!(coefficients[0], 1);
        assert_eq!(coefficients[4], 2);
        assert_eq!(coefficients[1], 3);
        assert_eq!(coefficients.iter().filter(|value| **value != 0).count(), 3);
    }

    #[test]
    fn tx64_coefficients_from_scan_rejects_beyond_coded_top_left_32_square() {
        let coded_count = TxSize::Tx32x32.sample_count();
        let scanned_coefficients = vec![1; coded_count];
        let coefficients =
            coefficients_from_scan(TxSize::Tx64x64, TxType::DctDct, &scanned_coefficients).unwrap();

        assert_eq!(coefficients.len(), TxSize::Tx64x64.sample_count());
        assert_eq!(
            coefficients
                .iter()
                .filter(|coefficient| **coefficient != 0)
                .count(),
            coded_count
        );
        assert_eq!(coefficients[31 * TxSize::Tx64x64.width() + 31], 1);
        assert_eq!(coefficients[32 * TxSize::Tx64x64.width()], 0);
        assert_eq!(coefficients[32], 0);

        let too_many_coefficients = vec![1; coded_count + 1];
        assert!(matches!(
            coefficients_from_scan(TxSize::Tx64x64, TxType::DctDct, &too_many_coefficients),
            Err(DecoderError::InvalidParam(_))
        ));
    }

    #[test]
    fn tx64_inverse_transform_rejects_non_zero_coefficients_outside_coded_top_left() {
        let mut coefficients = vec![0; TxSize::Tx64x64.sample_count()];
        coefficients[32] = 1;
        assert!(matches!(
            inverse_transform(TxType::DctDct, TxSize::Tx64x64, &coefficients, 8),
            Err(DecoderError::InvalidParam(_))
        ));

        coefficients[32] = 0;
        coefficients[32 * TxSize::Tx64x64.width()] = 1;
        assert!(matches!(
            inverse_transform(TxType::DctDct, TxSize::Tx64x64, &coefficients, 8),
            Err(DecoderError::InvalidParam(_))
        ));
    }

    #[test]
    fn directional_dct_scans_match_aom_mrow_and_mcol_orders() {
        assert_eq!(
            &coefficient_scan(TxSize::Tx4x4, TxType::VerticalDct)[..8],
            &[0, 4, 8, 12, 1, 5, 9, 13]
        );
        assert_eq!(
            &coefficient_scan(TxSize::Tx4x4, TxType::HorizontalDct)[..8],
            &[0, 1, 2, 3, 4, 5, 6, 7]
        );
        assert_eq!(
            &coefficient_scan(TxSize::Tx8x8, TxType::VerticalDct)[..8],
            &[0, 8, 16, 24, 32, 40, 48, 56]
        );
        assert_eq!(
            &coefficient_scan(TxSize::Tx8x8, TxType::HorizontalDct)[..8],
            &[0, 1, 2, 3, 4, 5, 6, 7]
        );
    }

    #[test]
    fn remaps_one_dimensional_and_identity_square_coefficients() {
        let mut coefficients = vec![0; TxSize::Tx8x8.sample_count()];
        for (index, position) in coefficient_scan(TxSize::Tx8x8, TxType::HorizontalDct)
            .into_iter()
            .enumerate()
        {
            coefficients[position] = index as i32;
        }
        remap_coefficients_for_inverse_storage(
            TxSize::Tx8x8,
            TxType::HorizontalDct,
            false,
            &mut coefficients,
        );
        assert_eq!(&coefficients[..8], &[0, 8, 16, 24, 32, 40, 48, 56]);

        let mut coefficients = vec![0; TxSize::Tx8x8.sample_count()];
        for (index, position) in coefficient_scan(TxSize::Tx8x8, TxType::VerticalDct)
            .into_iter()
            .enumerate()
        {
            coefficients[position] = index as i32;
        }
        remap_coefficients_for_inverse_storage(
            TxSize::Tx8x8,
            TxType::VerticalDct,
            false,
            &mut coefficients,
        );
        assert_eq!(&coefficients[..8], &[0, 1, 2, 3, 4, 5, 6, 7]);

        let mut identity = (0..TxSize::Tx4x4.sample_count() as i32).collect::<Vec<_>>();
        remap_coefficients_for_inverse_storage(
            TxSize::Tx4x4,
            TxType::Identity,
            false,
            &mut identity,
        );
        assert_eq!(
            identity,
            vec![0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]
        );

        let mut two_dimensional = (0..TxSize::Tx4x4.sample_count() as i32).collect::<Vec<_>>();
        remap_coefficients_for_inverse_storage(
            TxSize::Tx4x4,
            TxType::DctDct,
            true,
            &mut two_dimensional,
        );
        assert_eq!(two_dimensional, (0..16).collect::<Vec<_>>());

        remap_coefficients_for_inverse_storage(
            TxSize::Tx4x4,
            TxType::DctDct,
            false,
            &mut two_dimensional,
        );
        assert_eq!(
            two_dimensional,
            vec![0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]
        );
    }

    #[test]
    fn normative_directional_scans_are_permutations_for_supported_square_sizes() {
        for tx_size in [TxSize::Tx4x4, TxSize::Tx8x8, TxSize::Tx16x16] {
            for scan in [
                normative_square_row_scan(tx_size),
                normative_square_col_scan(tx_size),
            ] {
                assert_eq!(scan.len(), tx_size.sample_count());
                let mut sorted = scan;
                sorted.sort_unstable();
                assert_eq!(sorted, (0..tx_size.sample_count()).collect::<Vec<_>>());
            }
        }
    }

    #[test]
    fn small_dct_dc_only_outputs_constant_residuals_for_supported_bit_depths() {
        for (tx_size, dc) in [
            (TxSize::Tx4x4, 16),
            (TxSize::Tx8x8, 32),
            (TxSize::Tx16x16, 64),
            (TxSize::Tx8x4, 32),
        ] {
            let mut coefficients = vec![0; tx_size.sample_count()];
            coefficients[0] = dc;
            for bit_depth in [8, 10, 12] {
                assert_eq!(
                    inverse_transform(TxType::DctDct, tx_size, &coefficients, bit_depth).unwrap(),
                    vec![1; tx_size.sample_count()],
                    "{tx_size:?} {bit_depth}-bit"
                );
            }
        }
    }

    #[test]
    fn tx4x8_adst_dct_matches_rectangular_inverse_reference() {
        let mut coefficients = vec![0; TxSize::Tx4x8.sample_count()];
        coefficients[1] = 176;
        coefficients[3] = -176;
        coefficients[8] = -176;
        coefficients[9] = -176;

        assert_eq!(
            inverse_transform(TxType::AdstDct, TxSize::Tx4x8, &coefficients, 8).unwrap(),
            vec![
                -5, -3, -1, 1, -8, -4, 3, 7, -3, 3, 11, 18, -1, 6, 15, 21, -6, -1, 7, 12, -11, -8,
                -3, 0, -8, -7, -5, -4, -1, -1, -1, -1,
            ]
        );
    }

    #[test]
    fn transform_round_shift_matches_aom_negative_ties() {
        assert_eq!(round_shift_i64(-8, 4), 0);
        assert_eq!(round_shift_i64(-24, 4), -1);
        assert_eq!(round2_signed(-8, 4), 0);
    }

    #[test]
    fn tx4x4_horizontal_dct_matches_aom_wml2viewer_block() {
        // AOM stores transform coefficients column-major. Transpose the
        // traced input into the row-major storage consumed by this module.
        let coefficients = vec![
            0, 0, 0, 0, 0, 0, 0, 0, 2112, -704, -1056, -528, 1936, 176, 0, 0,
        ];
        let prediction = vec![0, 0, 0, 0, 0, 0, 0, 2, 0, 3, 29, 66, 43, 80, 95, 91];
        let residual =
            inverse_transform(TxType::HorizontalDct, TxSize::Tx4x4, &coefficients, 8).unwrap();
        assert_eq!(
            add_residual_to_prediction(&prediction, &residual, 8).unwrap(),
            vec![0, 0, 0, 0, 0, 0, 0, 2, 0, 220, 208, 207, 178, 207, 210, 198]
        );
    }

    #[test]
    fn tx4x4_adst_adst_matches_aom_wml2viewer_block() {
        let mut coefficients = vec![
            420, 704, -704, -176, -704, -1056, 528, 880, 176, -528, 0, 528, 528, 352, -352, -176,
        ];
        remap_coefficients_for_inverse_storage(
            TxSize::Tx4x4,
            TxType::AdstAdst,
            false,
            &mut coefficients,
        );
        let prediction = vec![
            1, 1, 1, 1, 98, 98, 98, 98, 157, 157, 157, 157, 176, 176, 176, 176,
        ];
        let residual =
            inverse_transform(TxType::AdstAdst, TxSize::Tx4x4, &coefficients, 8).unwrap();
        assert_eq!(
            add_residual_to_prediction(&prediction, &residual, 8).unwrap(),
            vec![
                0, 0, 2, 0, 0, 0, 247, 236, 213, 127, 197, 227, 204, 156, 128, 147
            ]
        );
    }

    #[test]
    fn tx4x4_dct_dct_matches_aom_wml2viewer_block() {
        let mut coefficients = vec![
            0, -1056, -528, 0, 176, 1056, 176, 176, 0, -528, -352, -352, 0, 176, 352, 176,
        ];
        remap_coefficients_for_inverse_storage(
            TxSize::Tx4x4,
            TxType::DctDct,
            false,
            &mut coefficients,
        );
        let prediction = vec![
            175, 185, 204, 185, 95, 139, 168, 182, 38, 48, 79, 123, 36, 37, 38, 43,
        ];
        let residual = inverse_transform(TxType::DctDct, TxSize::Tx4x4, &coefficients, 8).unwrap();
        assert_eq!(
            add_residual_to_prediction(&prediction, &residual, 8).unwrap(),
            vec![
                163, 167, 180, 1, 116, 147, 125, 191, 60, 80, 105, 181, 34, 27, 68, 132
            ]
        );
    }

    #[test]
    fn tx8x8_dct_dct_matches_aom_wml2viewer_palette_block() {
        let mut coefficients = vec![0; TxSize::Tx8x8.sample_count()];
        coefficients[16] = 176;
        coefficients[47] = 176;
        coefficients[54] = 176;
        coefficients[55] = -176;
        remap_coefficients_for_inverse_storage(
            TxSize::Tx8x8,
            TxType::DctDct,
            false,
            &mut coefficients,
        );
        let prediction = vec![
            0, 0, 0, 0, 0, 0, 0, 0, 207, 207, 0, 0, 0, 0, 0, 0, 176, 176, 176, 207, 207, 0, 0, 0,
            114, 114, 176, 207, 207, 0, 0, 0, 39, 78, 142, 142, 142, 207, 207, 0, 39, 39, 39, 78,
            114, 176, 207, 142, 39, 39, 39, 39, 78, 114, 114, 142, 39, 39, 39, 39, 39, 0, 78, 78,
        ];
        let residual = inverse_transform(TxType::DctDct, TxSize::Tx8x8, &coefficients, 8).unwrap();
        assert_eq!(
            add_residual_to_prediction(&prediction, &residual, 8).unwrap(),
            vec![
                5, 0, 0, 0, 0, 0, 2, 3, 208, 213, 0, 0, 0, 0, 0, 5, 182, 173, 176, 207, 200, 0, 6,
                1, 116, 118, 177, 198, 207, 3, 0, 8, 43, 81, 135, 146, 137, 198, 221, 0, 44, 36,
                46, 67, 111, 184, 195, 152, 41, 45, 31, 41, 75, 104, 126, 141, 43, 39, 40, 33, 35,
                2, 76, 83,
            ]
        );
    }

    #[test]
    fn tx32x16_dct_dc_matches_aom_wml2viewer_chroma_block() {
        let mut quantized = vec![0; TxSize::Tx32x16.sample_count()];
        quantized[0] = -1;
        let coefficients = dequantize_coefficients(
            &quantized,
            PlaneQuant { dc: 140, ac: 176 },
            8,
            TxSize::Tx32x16.dq_denom(),
        );
        assert_eq!(coefficients[0], -70);
        assert_eq!(
            inverse_transform(TxType::DctDct, TxSize::Tx32x16, &coefficients, 8).unwrap(),
            vec![-1; TxSize::Tx32x16.sample_count()]
        );
    }

    #[test]
    fn tx32x32_dct_dc_matches_aom_wml2viewer_luma_block() {
        let mut coefficients = vec![0; TxSize::Tx32x32.sample_count()];
        coefficients[0] = -140;
        assert_eq!(
            inverse_transform(TxType::DctDct, TxSize::Tx32x32, &coefficients, 8).unwrap(),
            vec![-1; TxSize::Tx32x32.sample_count()]
        );
    }

    #[test]
    fn inverse_transform_rejects_unsupported_bit_depth_before_transform_math() {
        let mut coefficients = vec![0; TxSize::Tx4x4.sample_count()];
        coefficients[0] = 16;

        for bit_depth in [0, 7, 9, 11, 13, 16] {
            assert!(
                matches!(
                    inverse_transform(TxType::DctDct, TxSize::Tx4x4, &coefficients, bit_depth),
                    Err(DecoderError::InvalidParam(_))
                ),
                "{bit_depth}"
            );
        }
    }

    #[test]
    fn staged_4_point_vectors_match_aom_integer_rounding() {
        let input = [16, -8, 4, 2];

        assert_eq!(inverse_dct4(input, 16), [7, 3, 13, 21]);
        assert_eq!(inverse_adst4(input), [4, 0, 11, 23]);
    }

    fn inverse_transform_4x4_staged_reference(
        tx_type: TxType,
        dequant: &[i32],
        bit_depth: u8,
    ) -> Vec<i32> {
        let (vertical, horizontal) = staged_transform_pair(tx_type);
        let row_range = bit_depth + 8;
        let mut intermediate = [0i32; 16];
        for row in 0..4 {
            let input = std::array::from_fn(|column| {
                clamp_signed(dequant[row * 4 + column], bit_depth + 8)
            });
            let output = inverse_staged_4(horizontal, input, row_range);
            intermediate[row * 4..row * 4 + 4].copy_from_slice(&output);
        }

        let residual_limit = 1i32 << (bit_depth + 7);
        let mut output = vec![0i32; 16];
        for column in 0..4 {
            let input = std::array::from_fn(|row| clamp_signed(intermediate[row * 4 + column], 16));
            let transformed = inverse_staged_4(vertical, input, 16);
            for row in 0..4 {
                output[row * 4 + column] =
                    round2_signed(transformed[row], 4).clamp(-residual_limit, residual_limit - 1);
            }
        }
        output
    }

    #[test]
    fn tx4_dispatch_matches_staged_core_for_all_supported_types() {
        let mut coefficients = vec![0; TxSize::Tx4x4.sample_count()];
        coefficients[0] = 32;
        coefficients[1] = -8;
        coefficients[TxSize::Tx4x4.width()] = 4;
        coefficients[TxSize::Tx4x4.width() + 1] = -2;

        for bit_depth in [8, 10, 12] {
            for tx_type in [
                TxType::DctDct,
                TxType::AdstDct,
                TxType::DctAdst,
                TxType::AdstAdst,
                TxType::Identity,
                TxType::VerticalDct,
                TxType::HorizontalDct,
            ] {
                assert_eq!(
                    inverse_transform(tx_type, TxSize::Tx4x4, &coefficients, bit_depth).unwrap(),
                    inverse_transform_4x4_staged_reference(tx_type, &coefficients, bit_depth),
                    "{tx_type:?} {bit_depth}-bit"
                );
            }
        }
    }

    #[test]
    fn staged_8_point_vectors_match_aom_integer_rounding() {
        assert_eq!(inverse_dct8([32, 0, 0, 0, 0, 0, 0, 0], 16), [23; 8]);
        assert_eq!(
            inverse_adst8([32, 0, 0, 0, 0, 0, 0, 0], 16),
            [3, 9, 16, 21, 25, 28, 31, 32]
        );
    }

    fn inverse_transform_8x8_staged_reference(
        tx_type: TxType,
        dequant: &[i32],
        bit_depth: u8,
    ) -> Vec<i32> {
        let (vertical, horizontal) = staged_transform_pair(tx_type);
        let row_range = bit_depth + 8;
        let mut intermediate = [0i32; 64];
        for row in 0..8 {
            let input = std::array::from_fn(|column| {
                clamp_signed(dequant[row * 8 + column], bit_depth + 8)
            });
            let transformed = inverse_staged_8(horizontal, input, row_range);
            for column in 0..8 {
                intermediate[row * 8 + column] = round2_signed(transformed[column], 1);
            }
        }

        let residual_limit = 1i32 << (bit_depth + 7);
        let mut output = vec![0i32; 64];
        for column in 0..8 {
            let input = std::array::from_fn(|row| clamp_signed(intermediate[row * 8 + column], 16));
            let transformed = inverse_staged_8(vertical, input, 16);
            for row in 0..8 {
                output[row * 8 + column] =
                    round2_signed(transformed[row], 4).clamp(-residual_limit, residual_limit - 1);
            }
        }
        output
    }

    #[test]
    fn tx8_dispatch_matches_staged_core_for_all_supported_types() {
        let mut coefficients = vec![0; TxSize::Tx8x8.sample_count()];
        coefficients[0] = 48;
        coefficients[1] = -12;
        coefficients[TxSize::Tx8x8.width()] = 6;
        coefficients[TxSize::Tx8x8.width() + 1] = -3;

        for bit_depth in [8, 10, 12] {
            for tx_type in [
                TxType::DctDct,
                TxType::AdstDct,
                TxType::DctAdst,
                TxType::AdstAdst,
                TxType::Identity,
                TxType::VerticalDct,
                TxType::HorizontalDct,
            ] {
                assert_eq!(
                    inverse_transform(tx_type, TxSize::Tx8x8, &coefficients, bit_depth).unwrap(),
                    inverse_transform_8x8_staged_reference(tx_type, &coefficients, bit_depth),
                    "{tx_type:?} {bit_depth}-bit"
                );
            }
        }
    }

    #[test]
    fn staged_16_point_dct_matches_aom_integer_rounding() {
        assert_eq!(
            inverse_dct16([64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 16),
            [45; 16]
        );
    }

    #[test]
    fn large_dct_dc_outputs_follow_size_specific_rounding() {
        let mut tx32 = vec![0; TxSize::Tx32x32.sample_count()];
        tx32[0] = 32;
        assert_eq!(
            inverse_transform(TxType::DctDct, TxSize::Tx32x32, &tx32, 8).unwrap(),
            vec![0; TxSize::Tx32x32.sample_count()]
        );

        let mut tx64 = vec![0; TxSize::Tx64x64.sample_count()];
        tx64[0] = 64;
        assert_eq!(
            inverse_transform(TxType::DctDct, TxSize::Tx64x64, &tx64, 8).unwrap(),
            vec![1; TxSize::Tx64x64.sample_count()]
        );

        tx64[0] = -192;
        assert_eq!(
            inverse_transform(TxType::DctDct, TxSize::Tx64x64, &tx64, 8).unwrap(),
            vec![-1; TxSize::Tx64x64.sample_count()]
        );

        tx64[0] = -105;
        assert_eq!(
            inverse_transform(TxType::DctDct, TxSize::Tx64x64, &tx64, 8).unwrap(),
            vec![-1; TxSize::Tx64x64.sample_count()]
        );
    }

    #[test]
    fn tx16x64_dct_uses_staged_64_point_rounding() {
        let mut coefficients = vec![0; TxSize::Tx16x64.sample_count()];
        coefficients[1] = 88;
        let residual =
            inverse_transform(TxType::DctDct, TxSize::Tx16x64, &coefficients, 8).unwrap();

        assert_eq!(&residual[..16], &[1; 16]);
        assert!(residual.iter().any(|sample| *sample != 0));
    }

    #[test]
    fn all_zero_transforms_return_zero_without_transform_dispatch() {
        for tx_size in [
            TxSize::Tx4x4,
            TxSize::Tx8x8,
            TxSize::Tx16x16,
            TxSize::Tx32x32,
            TxSize::Tx64x64,
        ] {
            let coefficients = vec![0; tx_size.sample_count()];
            for bit_depth in [8, 10, 12] {
                for tx_type in [
                    TxType::DctDct,
                    TxType::AdstDct,
                    TxType::DctAdst,
                    TxType::AdstAdst,
                    TxType::Identity,
                    TxType::VerticalDct,
                    TxType::HorizontalDct,
                ] {
                    assert_eq!(
                        inverse_transform(tx_type, tx_size, &coefficients, bit_depth).unwrap(),
                        vec![0; tx_size.sample_count()],
                        "{tx_type:?} {tx_size:?} {bit_depth}-bit"
                    );
                }
            }
        }
    }

    #[test]
    fn tx32_dct_dispatch_outputs_bounded_nonzero_residuals_for_sparse_coefficients() {
        let residual_limit = 1 << 15;
        let mut coefficients = vec![0; TxSize::Tx32x32.sample_count()];
        coefficients[0] = 64;
        coefficients[1] = -32;
        coefficients[TxSize::Tx32x32.width()] = 16;

        let residual =
            inverse_transform(TxType::DctDct, TxSize::Tx32x32, &coefficients, 8).unwrap();

        assert_eq!(residual.len(), TxSize::Tx32x32.sample_count());
        assert!(
            residual
                .iter()
                .all(|value| (-residual_limit..residual_limit).contains(value))
        );
        assert!(residual.iter().any(|value| *value != 0));
    }

    #[test]
    fn tx64_dct_dispatch_outputs_clamped_residuals_for_coded_top_left_sparse_coefficients() {
        let residual_limit = 1 << 15;
        let mut coefficients = vec![0; TxSize::Tx64x64.sample_count()];
        coefficients[0] = 64;
        coefficients[1] = -32;
        coefficients[TxSize::Tx64x64.width()] = 16;
        coefficients[31 * TxSize::Tx64x64.width() + 31] = -8;

        let residual =
            inverse_transform(TxType::DctDct, TxSize::Tx64x64, &coefficients, 8).unwrap();

        assert_eq!(residual.len(), TxSize::Tx64x64.sample_count());
        assert!(
            residual
                .iter()
                .all(|value| (-residual_limit..residual_limit).contains(value))
        );
        assert_eq!(&residual[..8], &[1, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(
            (0..8)
                .map(|y| residual[y * TxSize::Tx64x64.width()])
                .collect::<Vec<_>>(),
            vec![1, 1, 1, 1, 0, 1, 1, 1]
        );
        assert_eq!(residual[63], 2);
        assert_eq!(residual[31 * TxSize::Tx64x64.width() + 31], 1);
        assert_eq!(residual[TxSize::Tx64x64.sample_count() - 1], 1);
    }

    #[test]
    fn tx64_dct_fixed_basis_matches_pinned_sparse_reference() {
        let mut coefficients = vec![0; TxSize::Tx64x64.sample_count()];
        coefficients[0] = 64;
        coefficients[1] = -32;
        coefficients[TxSize::Tx64x64.width()] = 16;
        coefficients[31 * TxSize::Tx64x64.width() + 31] = -8;
        let residual = inverse_dct64_fixed_basis(&coefficients, 8);

        assert_eq!(residual.len(), TxSize::Tx64x64.sample_count());
        assert_eq!(&residual[..8], &[1, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(
            (0..8)
                .map(|y| residual[y * TxSize::Tx64x64.width()])
                .collect::<Vec<_>>(),
            vec![1, 1, 1, 1, 0, 1, 1, 1]
        );
        assert_eq!(residual[63], 2);
        assert_eq!(residual[31 * TxSize::Tx64x64.width() + 31], 1);
        assert_eq!(residual[TxSize::Tx64x64.sample_count() - 1], 1);
    }

    #[test]
    fn tx64_dct_fixed_basis_matches_dc_reference() {
        let mut coefficients = vec![0; TxSize::Tx64x64.sample_count()];
        coefficients[0] = 64;
        assert_eq!(
            inverse_dct64_fixed_basis(&coefficients, 8),
            vec![1; TxSize::Tx64x64.sample_count()]
        );

        coefficients[0] = -192;
        assert_eq!(
            inverse_dct64_fixed_basis(&coefficients, 8),
            vec![-1; TxSize::Tx64x64.sample_count()]
        );
    }

    #[test]
    fn tx64_dct_dispatch_matches_fixed_basis_core() {
        let mut tx64_coefficients = vec![0; TxSize::Tx64x64.sample_count()];
        tx64_coefficients[0] = 64;
        tx64_coefficients[1] = -32;
        tx64_coefficients[TxSize::Tx64x64.width()] = 16;
        tx64_coefficients[31 * TxSize::Tx64x64.width() + 31] = -8;

        for bit_depth in [8, 10, 12] {
            assert_eq!(
                inverse_transform(
                    TxType::DctDct,
                    TxSize::Tx64x64,
                    &tx64_coefficients,
                    bit_depth
                )
                .unwrap(),
                inverse_dct64_fixed_basis(&tx64_coefficients, bit_depth),
                "Tx64x64 {bit_depth}-bit"
            );
        }
    }

    #[test]
    fn tx64_dct_fixed_basis_table_matches_expected_endpoints() {
        assert_eq!(inverse_dct64_basis(0, 0), 512);
        assert_eq!(inverse_dct64_basis(63, 0), 512);
        assert_eq!(inverse_dct64_basis(0, 1), 724);
        assert_eq!(inverse_dct64_basis(63, 1), -724);
        assert_eq!(inverse_dct64_basis(0, 32), 512);
        assert_eq!(inverse_dct64_basis(1, 32), -512);
    }

    #[test]
    fn cospi_unit_128_matches_quadrant_symmetry() {
        assert_eq!(cospi_unit_128(0), cospi(0));
        assert_eq!(cospi_unit_128(64), 0);
        assert_eq!(cospi_unit_128(96), -cospi(32));
        assert_eq!(cospi_unit_128(160), -cospi(32));
        assert_eq!(cospi_unit_128(224), cospi(32));
    }

    #[test]
    fn size_specific_large_dct_paths_validate_coefficient_counts() {
        let tx32 = vec![0; TxSize::Tx32x32.sample_count()];
        let tx64 = vec![0; TxSize::Tx64x64.sample_count()];

        assert!(matches!(
            inverse_transform_32x32_dct(&tx32[..TxSize::Tx32x32.sample_count() - 1], 8),
            Err(DecoderError::InvalidParam(_))
        ));
        assert!(matches!(
            inverse_transform_64x64_dct(&tx64[..TxSize::Tx64x64.sample_count() - 1], 8),
            Err(DecoderError::InvalidParam(_))
        ));
    }

    #[test]
    fn tx32_and_tx64_non_dct_large_transforms_reject_non_zero_coefficients() {
        for tx_size in [TxSize::Tx32x32, TxSize::Tx64x64] {
            for tx_type in [
                TxType::AdstDct,
                TxType::DctAdst,
                TxType::AdstAdst,
                TxType::Identity,
                TxType::VerticalDct,
                TxType::HorizontalDct,
            ] {
                let mut coefficients = vec![0; tx_size.sample_count()];
                coefficients[0] = 64;

                assert!(
                    matches!(
                        inverse_transform(tx_type, tx_size, &coefficients, 8),
                        Err(DecoderError::Unsupported(_))
                    ),
                    "{tx_type:?} {tx_size:?}"
                );
            }
        }
    }

    #[test]
    fn larger_identity_transform_zero_coefficients_use_zero_fast_path() {
        for tx_size in [TxSize::Tx32x32, TxSize::Tx64x64] {
            let coefficients = vec![0; tx_size.sample_count()];

            let residual = inverse_transform(TxType::Identity, tx_size, &coefficients, 8).unwrap();

            assert_eq!(residual, coefficients, "{tx_size:?}");
        }
    }

    #[test]
    fn larger_dct_dc_only_clips_to_bit_depth_residual_range() {
        for tx_size in [TxSize::Tx32x32, TxSize::Tx64x64] {
            let mut coefficients = vec![0; tx_size.sample_count()];

            coefficients[0] = i32::MAX;
            assert_eq!(
                inverse_transform(TxType::DctDct, tx_size, &coefficients, 8)
                    .unwrap()
                    .into_iter()
                    .max(),
                Some(if tx_size == TxSize::Tx32x32 {
                    1 << 8
                } else {
                    (1 << 15) - 1
                }),
                "{tx_size:?} 8-bit"
            );
            assert_eq!(
                inverse_transform(TxType::DctDct, tx_size, &coefficients, 10)
                    .unwrap()
                    .into_iter()
                    .max(),
                Some(if tx_size == TxSize::Tx32x32 {
                    1 << 10
                } else {
                    (1 << 17) - 1
                }),
                "{tx_size:?} 10-bit"
            );
            assert_eq!(
                inverse_transform(TxType::DctDct, tx_size, &coefficients, 12)
                    .unwrap()
                    .into_iter()
                    .max(),
                Some(if tx_size == TxSize::Tx32x32 {
                    1448
                } else {
                    (1 << 19) - 1
                }),
                "{tx_size:?} 12-bit"
            );

            coefficients[0] = i32::MIN;
            assert_eq!(
                inverse_transform(TxType::DctDct, tx_size, &coefficients, 8)
                    .unwrap()
                    .into_iter()
                    .min(),
                Some(if tx_size == TxSize::Tx32x32 {
                    -(1 << 8)
                } else {
                    -(1 << 15)
                }),
                "{tx_size:?} negative 8-bit"
            );
        }
    }

    #[test]
    fn staged_16_point_iadst_matches_aom_reference_vectors() {
        assert_eq!(
            inverse_adst16([64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 20),
            [
                3, 10, 15, 22, 27, 33, 37, 43, 47, 52, 54, 58, 60, 62, 63, 64
            ]
        );
        assert_eq!(
            inverse_adst16([64, -17, 9, 0, 3, -5, 0, 2, -1, 0, 4, -3, 0, 1, -2, 6], 20,),
            [9, 4, 16, 12, 26, 13, 22, 18, 31, 42, 61, 43, 67, 74, 83, 97]
        );
    }

    #[test]
    fn staged_16_point_identity_matches_aom_reference_vectors() {
        assert_eq!(
            inverse_staged_16(
                StagedTransform::Identity,
                [64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                20,
            ),
            [181, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            inverse_staged_16(
                StagedTransform::Identity,
                [64, -17, 9, 0, 3, -5, 0, 2, -1, 0, 4, -3, 0, 1, -2, 6],
                20,
            ),
            [181, -48, 25, 0, 8, -14, 0, 6, -3, 0, 11, -8, 0, 3, -6, 17]
        );
    }

    #[test]
    fn tx16_routes_all_supported_types_through_integer_stages() {
        let mut coefficients = vec![0; TxSize::Tx16x16.sample_count()];
        coefficients[0] = 64;
        coefficients[1] = -17;
        coefficients[16] = 23;

        for tx_type in [
            TxType::DctDct,
            TxType::AdstDct,
            TxType::DctAdst,
            TxType::AdstAdst,
            TxType::Identity,
            TxType::VerticalDct,
            TxType::HorizontalDct,
        ] {
            let residual = inverse_transform(tx_type, TxSize::Tx16x16, &coefficients, 8).unwrap();
            assert_eq!(residual.len(), 256);
            assert!(residual.iter().all(|value| value.abs() < (1 << 15)));
        }
    }

    fn inverse_transform_16x16_staged_reference(
        tx_type: TxType,
        dequant: &[i32],
        bit_depth: u8,
    ) -> Vec<i32> {
        let (vertical, horizontal) = staged_transform_pair(tx_type);
        let mut temp = vec![0i32; 256];
        for row in 0..16 {
            let input = std::array::from_fn(|x| clamp_signed(dequant[row * 16 + x], bit_depth + 8));
            let values = inverse_staged_16(horizontal, input, 16);
            for x in 0..16 {
                temp[row * 16 + x] = round2_signed(values[x], 2);
            }
        }

        let limit = 1i32 << (bit_depth + 7);
        let mut out = vec![0i32; 256];
        for x in 0..16 {
            let input = std::array::from_fn(|y| clamp_signed(temp[y * 16 + x], 16));
            let values = inverse_staged_16(vertical, input, 16);
            for y in 0..16 {
                out[y * 16 + x] = round2_signed(values[y], 4).clamp(-limit, limit - 1);
            }
        }
        out
    }

    #[test]
    fn tx16_dispatch_matches_staged_core_for_all_supported_types() {
        let mut coefficients = vec![0; TxSize::Tx16x16.sample_count()];
        coefficients[0] = 64;
        coefficients[1] = -16;
        coefficients[TxSize::Tx16x16.width()] = 8;
        coefficients[TxSize::Tx16x16.width() + 1] = -4;

        for bit_depth in [8, 10, 12] {
            for tx_type in [
                TxType::DctDct,
                TxType::AdstDct,
                TxType::DctAdst,
                TxType::AdstAdst,
                TxType::Identity,
                TxType::VerticalDct,
                TxType::HorizontalDct,
            ] {
                assert_eq!(
                    inverse_transform(tx_type, TxSize::Tx16x16, &coefficients, bit_depth).unwrap(),
                    inverse_transform_16x16_staged_reference(tx_type, &coefficients, bit_depth),
                    "{tx_type:?} {bit_depth}-bit"
                );
            }
        }
    }

    #[test]
    fn enabled_transform_size_reference_anchors_match_fixed_vectors() {
        let cases = [
            (TxSize::Tx4x4, TxType::DctDct, [2, 2, 1, 2, 2]),
            (TxSize::Tx4x4, TxType::AdstDct, [1, 1, 1, 2, 3]),
            (TxSize::Tx4x4, TxType::DctAdst, [1, 1, 0, 1, 3]),
            (TxSize::Tx4x4, TxType::AdstAdst, [0, 1, 0, 1, 4]),
            (TxSize::Tx4x4, TxType::Identity, [8, -2, 1, 0, 0]),
            (TxSize::Tx4x4, TxType::VerticalDct, [5, -1, 4, -1, 0]),
            (TxSize::Tx4x4, TxType::HorizontalDct, [3, 3, 1, 1, 0]),
            (TxSize::Tx8x8, TxType::DctDct, [1, 1, 1, 1, 1]),
            (TxSize::Tx8x8, TxType::AdstDct, [0, 0, 0, 0, 2]),
            (TxSize::Tx8x8, TxType::DctAdst, [0, 0, 0, 0, 2]),
            (TxSize::Tx8x8, TxType::AdstAdst, [0, 0, 0, 0, 2]),
            (TxSize::Tx8x8, TxType::Identity, [8, -2, 1, 0, 0]),
            (TxSize::Tx8x8, TxType::VerticalDct, [3, -1, 3, -1, 0]),
            (TxSize::Tx8x8, TxType::HorizontalDct, [2, 2, 0, 0, 0]),
            (TxSize::Tx16x16, TxType::DctDct, [0, 0, 0, 0, 1]),
            (TxSize::Tx16x16, TxType::AdstDct, [0, 0, 0, 0, 1]),
            (TxSize::Tx16x16, TxType::DctAdst, [0, 0, 0, 0, 1]),
            (TxSize::Tx16x16, TxType::AdstAdst, [0, 0, 0, 0, 1]),
            (TxSize::Tx16x16, TxType::Identity, [8, -2, 1, 0, 0]),
            (TxSize::Tx16x16, TxType::VerticalDct, [2, 0, 2, 0, 0]),
            (TxSize::Tx16x16, TxType::HorizontalDct, [1, 1, 0, 0, 0]),
            (TxSize::Tx32x32, TxType::DctDct, [0, 0, 0, 0, 1]),
            (TxSize::Tx64x64, TxType::DctDct, [1, 1, 1, 1, 1]),
        ];

        for (tx_size, tx_type, expected) in cases {
            let mut coefficients = vec![0; tx_size.sample_count()];
            coefficients[0] = 64;
            coefficients[1] = -16;
            coefficients[tx_size.width()] = 8;
            if tx_size.width() >= 32 {
                coefficients[31 * tx_size.width() + 31] = -4;
            }
            let residual = inverse_transform(tx_type, tx_size, &coefficients, 8).unwrap();
            let positions = [
                0,
                1,
                tx_size.width(),
                tx_size.width() + 1,
                residual.len() - 1,
            ];
            assert_eq!(
                positions.map(|position| residual[position]),
                expected,
                "{tx_size:?} {tx_type:?}"
            );
        }
    }

    #[test]
    fn identity_transform_rounds_and_clips() {
        let residual = inverse_transform(TxType::Identity, TxSize::Tx4x4, &[64; 16], 8).unwrap();

        assert_eq!(residual, vec![8; 16]);
    }

    #[test]
    fn lossless_wht_dc_basis_is_spread_over_the_block() {
        let mut coefficients = [0i32; 16];
        coefficients[0] = 16;

        assert_eq!(inverse_lossless_transform_4x4(&coefficients), vec![1; 16]);
    }

    #[test]
    fn lossless_wht_matches_aom_sparse_reference_vector() {
        let coefficients = [
            16, -8, 4, -12, 8, 0, -4, 12, -16, 20, 0, -8, 4, -20, 24, -28,
        ];

        assert_eq!(
            inverse_lossless_transform_4x4(&coefficients),
            vec![0, 2, -2, 0, 1, 1, -1, 2, -2, 6, -1, -3, 5, -1, 9, -4]
        );
    }

    #[test]
    fn vertical_dct_only_mixes_columns() {
        let mut coeffs = vec![0; TxSize::Tx4x4.sample_count()];
        coeffs[0] = 16;
        coeffs[1] = 32;

        let residual = inverse_transform(TxType::VerticalDct, TxSize::Tx4x4, &coeffs, 8).unwrap();

        assert_eq!(
            residual,
            vec![1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0]
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
            vec![1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0]
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
        assert_eq!(plane.samples, vec![132; 16]);
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
