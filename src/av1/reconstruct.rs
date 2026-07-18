use super::decode::{FrameBuffers, PlaneBuffer};
use super::sequence::ColorConfig;
use crate::{DecoderError, ImageBuffer, Rgba16ImageBuffer};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedIntraEdges {
    pub above: Vec<u16>,
    pub left: Vec<u16>,
    pub above_left: u16,
    pub above_available: bool,
    pub left_available: bool,
}

pub fn add_residual_to_prediction(
    prediction: &[u16],
    residual: &[i32],
    bit_depth: u8,
) -> Result<Vec<u16>, DecoderError> {
    if prediction.len() != residual.len() {
        return Err(DecoderError::InvalidParam(
            "AV1 prediction and residual sizes differ".to_string(),
        ));
    }
    let max_value = (1i32 << bit_depth) - 1;
    Ok(prediction
        .iter()
        .zip(residual)
        .map(|(pred, res)| (i32::from(*pred) + *res).clamp(0, max_value) as u16)
        .collect())
}

pub fn write_plane_block(
    plane: &mut PlaneBuffer,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    samples: &[u16],
) -> Result<(), DecoderError> {
    if samples.len() != width * height {
        return Err(DecoderError::InvalidParam(
            "AV1 block sample count does not match dimensions".to_string(),
        ));
    }
    if x >= plane.layout.width || y >= plane.layout.height {
        return Ok(());
    }

    let clipped_width = width.min(plane.layout.width - x);
    let clipped_height = height.min(plane.layout.height - y);
    for row in 0..clipped_height {
        let dst_start = (y + row) * plane.layout.width + x;
        let src_start = row * width;
        plane.samples[dst_start..dst_start + clipped_width]
            .copy_from_slice(&samples[src_start..src_start + clipped_width]);
    }
    Ok(())
}

pub fn read_intra_edges(
    plane: &PlaneBuffer,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    bit_depth: u8,
) -> OwnedIntraEdges {
    read_intra_edges_with_extension_availability(
        plane, x, y, width, height, bit_depth, width, height,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "public compatibility API keeps the established edge-read signature"
)]
pub fn read_intra_edges_with_extension_availability(
    plane: &PlaneBuffer,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    bit_depth: u8,
    top_right_available: usize,
    bottom_left_available: usize,
) -> OwnedIntraEdges {
    let directional_edge_len = width
        .checked_add(height)
        .expect("AV1 intra edge length overflows");
    let mut above = vec![0; directional_edge_len];
    let mut left = vec![0; directional_edge_len];
    let (above_available, left_available, above_left) = read_intra_edges_into(
        plane,
        x,
        y,
        width,
        height,
        bit_depth,
        top_right_available,
        bottom_left_available,
        &mut above,
        &mut left,
    )
    .expect("owned AV1 intra edge buffers have the requested length");

    OwnedIntraEdges {
        above,
        left,
        above_left,
        above_available,
        left_available,
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "internal edge scratch follows the established AV1 prediction inputs"
)]
pub(crate) fn read_intra_edges_into(
    plane: &PlaneBuffer,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    bit_depth: u8,
    top_right_available: usize,
    bottom_left_available: usize,
    above: &mut [u16],
    left: &mut [u16],
) -> Result<(bool, bool, u16), DecoderError> {
    let mid = 1u16 << (bit_depth - 1);
    let directional_edge_len = width
        .checked_add(height)
        .ok_or_else(|| DecoderError::InvalidParam("AV1 intra edge length overflows".to_string()))?;
    if above.len() < directional_edge_len || left.len() < directional_edge_len {
        return Err(DecoderError::InvalidParam(
            "AV1 intra edge scratch is shorter than the prediction block".to_string(),
        ));
    }
    let above_available = y > 0 && plane.layout.width > 0;
    let left_available = x > 0 && plane.layout.height > 0;

    for dx in 0..directional_edge_len {
        above[dx] = if !above_available {
            mid - 1
        } else {
            let extension_end = width.saturating_add(top_right_available);
            let edge_dx = if dx >= extension_end {
                extension_end.saturating_sub(1)
            } else {
                dx
            };
            let sample_x = (x + edge_dx).min(plane.layout.width - 1);
            plane.samples[(y - 1) * plane.layout.width + sample_x]
        };
    }

    for dy in 0..directional_edge_len {
        left[dy] = if !left_available {
            mid + 1
        } else {
            let extension_end = height.saturating_add(bottom_left_available);
            let edge_dy = if dy >= extension_end {
                extension_end.saturating_sub(1)
            } else {
                dy
            };
            let sample_y = (y + edge_dy).min(plane.layout.height - 1);
            plane.samples[sample_y * plane.layout.width + x - 1]
        };
    }

    let above_left = if x == 0 || y == 0 || plane.layout.width == 0 {
        mid
    } else {
        plane.samples[(y - 1) * plane.layout.width + x - 1]
    };
    Ok((above_available, left_available, above_left))
}

pub fn frame_buffers_to_identity_rgba_8(
    buffers: &FrameBuffers,
) -> Result<ImageBuffer, DecoderError> {
    frame_buffers_to_rgba_8(
        buffers,
        &ColorConfig {
            high_bitdepth: false,
            twelve_bit: false,
            bit_depth: 8,
            monochrome: false,
            color_description: Some(super::sequence::ColorDescription {
                color_primaries: 1,
                transfer_characteristics: 13,
                matrix_coefficients: 0,
            }),
            color_range: super::sequence::ColorRange::Full,
            subsampling_x: false,
            subsampling_y: false,
            chroma_sample_position: None,
            separate_uv_delta_q: false,
        },
    )
}

pub fn frame_buffers_to_rgba_8(
    buffers: &FrameBuffers,
    color_config: &ColorConfig,
) -> Result<ImageBuffer, DecoderError> {
    let rgba16 = frame_buffers_to_rgba_16(buffers, color_config)?;
    let rgba = rgba16
        .rgba
        .iter()
        .map(|sample| ((u32::from(*sample) * 255 + 32767) / 65535) as u8)
        .collect();
    Ok(ImageBuffer {
        width: rgba16.width,
        height: rgba16.height,
        rgba,
    })
}

pub fn frame_buffers_to_rgba_16(
    buffers: &FrameBuffers,
    color_config: &ColorConfig,
) -> Result<Rgba16ImageBuffer, DecoderError> {
    validate_rgba_conversion(buffers)?;
    let hdr_transfer = transfer_characteristics(color_config)?;
    if color_config.monochrome {
        let luma = buffers.planes.first().ok_or_else(|| {
            DecoderError::Bitstream("AV1 monochrome luma plane is missing".to_string())
        })?;
        let max_source = (1u32 << color_config.bit_depth) - 1;
        let mut rgba = vec![0u16; buffers.width * buffers.height * 4];
        for index in 0..buffers.width * buffers.height {
            let x = index % buffers.width;
            let y = index / buffers.width;
            let source_x = (x >> usize::from(luma.layout.subsampling_x))
                .min(luma.layout.width.saturating_sub(1));
            let source_y = (y >> usize::from(luma.layout.subsampling_y))
                .min(luma.layout.height.saturating_sub(1));
            let value = scale_sample_to_u16(
                luma.samples[source_y * luma.layout.width + source_x],
                max_source,
            );
            let out = index * 4;
            rgba[out..out + 3].fill(value);
            rgba[out + 3] = alpha_sample(buffers.planes.get(3), x, y, max_source);
        }
        if let Some(transfer) = hdr_transfer {
            apply_transfer_function(&mut rgba, transfer);
        }
        return Ok(Rgba16ImageBuffer {
            width: buffers.width,
            height: buffers.height,
            rgba,
        });
    }
    let matrix_coefficients = color_config
        .color_description
        .map(|description| description.matrix_coefficients)
        .unwrap_or(2);
    let mut rgba = vec![0u16; buffers.width * buffers.height * 4];

    if matrix_coefficients == 0 {
        let max_source = (1u32 << color_config.bit_depth) - 1;
        let plane_g = &buffers.planes[0].samples;
        let plane_b = &buffers.planes[1].samples;
        let plane_r = &buffers.planes[2].samples;
        for index in 0..buffers.width * buffers.height {
            let x = index % buffers.width;
            let y = index / buffers.width;
            let out = index * 4;
            rgba[out] = scale_sample_to_u16(plane_r[index], max_source);
            rgba[out + 1] = scale_sample_to_u16(plane_g[index], max_source);
            rgba[out + 2] = scale_sample_to_u16(plane_b[index], max_source);
            rgba[out + 3] = alpha_sample(buffers.planes.get(3), x, y, max_source);
        }
    } else {
        let matrix = MatrixCoefficients::from_av1(matrix_coefficients)?;
        let range = SampleRange::new(color_config.bit_depth, color_config.color_range)?;
        let plane_y = &buffers.planes[0];
        let plane_u = buffers.planes.get(1);
        let plane_v = buffers.planes.get(2);
        let chroma_mid = 1u16 << color_config.bit_depth.saturating_sub(1);
        let alpha_max_source = (1u32 << color_config.bit_depth) - 1;
        for index in 0..buffers.width * buffers.height {
            let x = index % buffers.width;
            let y = index / buffers.width;
            let rgb = yuv_to_rgb_u16(
                sample_plane(plane_y, x, y),
                plane_u
                    .map(|plane| sample_chroma_plane(plane, x, y))
                    .unwrap_or(chroma_mid),
                plane_v
                    .map(|plane| sample_chroma_plane(plane, x, y))
                    .unwrap_or(chroma_mid),
                range,
                matrix,
            );
            let out = index * 4;
            rgba[out] = rgb[0];
            rgba[out + 1] = rgb[1];
            rgba[out + 2] = rgb[2];
            rgba[out + 3] = alpha_sample(buffers.planes.get(3), x, y, alpha_max_source);
        }
    }

    if let Some(transfer) = hdr_transfer {
        apply_transfer_function(&mut rgba, transfer);
    }

    Ok(Rgba16ImageBuffer {
        width: buffers.width,
        height: buffers.height,
        rgba,
    })
}

fn sample_plane(plane: &super::decode::PlaneBuffer, x: usize, y: usize) -> u16 {
    let source_x = (x >> usize::from(plane.layout.subsampling_x)).min(plane.layout.width - 1);
    let source_y = (y >> usize::from(plane.layout.subsampling_y)).min(plane.layout.height - 1);
    plane.samples[source_y * plane.layout.width + source_x]
}

fn alpha_sample(
    plane: Option<&super::decode::PlaneBuffer>,
    x: usize,
    y: usize,
    max_source: u32,
) -> u16 {
    plane
        .map(|plane| scale_sample_to_u16(sample_plane(plane, x, y), max_source))
        .unwrap_or(u16::MAX)
}

fn sample_chroma_plane(plane: &super::decode::PlaneBuffer, x: usize, y: usize) -> u16 {
    let subsampling_x = usize::from(plane.layout.subsampling_x);
    let subsampling_y = usize::from(plane.layout.subsampling_y);
    if subsampling_x == 0 && subsampling_y == 0 {
        return sample_plane(plane, x, y);
    }
    let source_x = (x >> subsampling_x).min(plane.layout.width - 1);
    let source_y = (y >> subsampling_y).min(plane.layout.height - 1);
    let next_x = (source_x + 1).min(plane.layout.width - 1);
    let next_y = (source_y + 1).min(plane.layout.height - 1);
    // The AV1 still-image samples in the supported path use colocated
    // horizontal chroma.  Keep the horizontal sample nearest while applying
    // vertical interpolation for 4:2:0, which matches the reference edge
    // behavior for odd-height images.
    let fraction_x = 0;
    let fraction_y = if subsampling_y == 0 {
        0
    } else {
        (y & 1) as u32
    };
    let top_left = u32::from(plane.samples[source_y * plane.layout.width + source_x]);
    let top_right = u32::from(plane.samples[source_y * plane.layout.width + next_x]);
    let bottom_left = u32::from(plane.samples[next_y * plane.layout.width + source_x]);
    let bottom_right = u32::from(plane.samples[next_y * plane.layout.width + next_x]);
    let top = top_left * (2 - fraction_x) + top_right * fraction_x;
    let bottom = bottom_left * (2 - fraction_x) + bottom_right * fraction_x;
    ((top * (2 - fraction_y) + bottom * fraction_y + 2) / 4) as u16
}

fn validate_rgba_conversion(buffers: &FrameBuffers) -> Result<(), DecoderError> {
    let _ = buffers;
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum TransferFunction {
    Gamma22,
    Gamma28,
    Linear,
    Pq,
    Hlg,
}

fn transfer_characteristics(
    color_config: &ColorConfig,
) -> Result<Option<TransferFunction>, DecoderError> {
    let Some(description) = color_config.color_description else {
        return Ok(None);
    };
    match description.transfer_characteristics {
        // These curves are already display-referred for the existing RGBA
        // API. BT.2020 10/12-bit uses the BT.709 OETF.
        1 | 6 | 7 | 13 | 14 | 15 => Ok(None),
        4 => Ok(Some(TransferFunction::Gamma22)),
        5 => Ok(Some(TransferFunction::Gamma28)),
        8 => Ok(Some(TransferFunction::Linear)),
        16 => Ok(Some(TransferFunction::Pq)),
        18 => Ok(Some(TransferFunction::Hlg)),
        transfer => Err(DecoderError::Unsupported(format!(
            "AV1 transfer characteristics {transfer} RGBA conversion is not supported yet"
        ))),
    }
}

fn apply_transfer_function(rgba: &mut [u16], transfer: TransferFunction) {
    for pixel in rgba.chunks_exact_mut(4) {
        pixel[0] = transfer_to_sdr(pixel[0], transfer);
        pixel[1] = transfer_to_sdr(pixel[1], transfer);
        pixel[2] = transfer_to_sdr(pixel[2], transfer);
    }
}

fn transfer_to_sdr(sample: u16, transfer: TransferFunction) -> u16 {
    let encoded = f64::from(sample) / f64::from(u16::MAX);
    match transfer {
        TransferFunction::Gamma22 => linear_to_srgb(encoded.powf(2.2)),
        TransferFunction::Gamma28 => linear_to_srgb(encoded.powf(2.8)),
        TransferFunction::Linear => linear_to_srgb(encoded),
        TransferFunction::Pq | TransferFunction::Hlg => hdr_to_sdr(encoded, transfer),
    }
}

fn hdr_to_sdr(encoded: f64, transfer: TransferFunction) -> u16 {
    // Convert PQ to a 100-nit SDR reference and HLG to its relative scene
    // value, then use a bounded Reinhard-style shoulder before sRGB encoding.
    // This keeps the existing RGBA8/16 API display-safe without pretending to
    // perform display-specific HDR gamut or tone calibration.
    const HDR_REFERENCE_WHITE: f64 = 4.0;
    let linear = match transfer {
        TransferFunction::Pq => pq_to_linear(encoded) * 100.0,
        TransferFunction::Hlg => hlg_to_linear(encoded),
        TransferFunction::Gamma22 | TransferFunction::Gamma28 | TransferFunction::Linear => {
            unreachable!("SDR transfer functions are handled before HDR tone mapping")
        }
    };
    let mapped = (linear / (1.0 + linear)) * ((HDR_REFERENCE_WHITE + 1.0) / HDR_REFERENCE_WHITE);
    linear_to_srgb(mapped.clamp(0.0, 1.0))
}

fn pq_to_linear(encoded: f64) -> f64 {
    const M1: f64 = 0.1593017578125;
    const M2: f64 = 78.84375;
    const C1: f64 = 0.8359375;
    const C2: f64 = 18.8515625;
    const C3: f64 = 18.6875;
    if encoded <= 0.0 {
        return 0.0;
    }
    let powered = encoded.powf(1.0 / M2);
    let numerator = (powered - C1).max(0.0);
    let denominator = (C2 - C3 * powered).max(f64::MIN_POSITIVE);
    (numerator / denominator).powf(1.0 / M1)
}

fn hlg_to_linear(encoded: f64) -> f64 {
    const BETA: f64 = 0.28466892;
    const GAMMA: f64 = 1.2;
    if encoded <= 0.0 {
        return 0.0;
    }
    if encoded <= 0.5 {
        ((encoded * encoded) / 3.0).powf(GAMMA)
    } else {
        let relative = (((encoded - 0.55991073) / 0.17883277).exp() + BETA) / 12.0;
        relative.powf(GAMMA).min(1.0)
    }
}

fn linear_to_srgb(linear: f64) -> u16 {
    let encoded = if linear <= 0.0031308 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    };
    (encoded.clamp(0.0, 1.0) * f64::from(u16::MAX)).round() as u16
}

#[derive(Debug, Clone, Copy)]
enum MatrixCoefficients {
    Yuv { kr: f64, kb: f64 },
    YcGco,
}

impl MatrixCoefficients {
    fn from_av1(matrix_coefficients: u8) -> Result<Self, DecoderError> {
        match matrix_coefficients {
            1 => Ok(Self::Yuv {
                kr: 0.2126,
                kb: 0.0722,
            }),
            2 => Ok(Self::Yuv {
                kr: 0.2126,
                kb: 0.0722,
            }),
            4 => Ok(Self::Yuv { kr: 0.30, kb: 0.11 }),
            5 | 6 => Ok(Self::Yuv {
                kr: 0.299,
                kb: 0.114,
            }),
            7 => Ok(Self::Yuv {
                kr: 0.212,
                kb: 0.087,
            }),
            8 => Ok(Self::YcGco),
            9 => Ok(Self::Yuv {
                kr: 0.2627,
                kb: 0.0593,
            }),
            _ => Err(DecoderError::Unsupported(format!(
                "AV1 matrix coefficients {matrix_coefficients} RGBA conversion is not supported yet"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct SampleRange {
    y_offset: f64,
    y_scale: f64,
    chroma_offset: f64,
    chroma_scale: f64,
}

impl SampleRange {
    fn new(bit_depth: u8, color_range: super::sequence::ColorRange) -> Result<Self, DecoderError> {
        if !(8..=16).contains(&bit_depth) {
            return Err(DecoderError::Unsupported(format!(
                "AV1 {bit_depth}-bit RGBA conversion is not supported yet"
            )));
        }
        let max_value = ((1u32 << bit_depth) - 1) as f64;
        let range_shift = f64::from(1u32 << bit_depth.saturating_sub(8));
        Ok(match color_range {
            super::sequence::ColorRange::Full => Self {
                y_offset: 0.0,
                y_scale: max_value,
                chroma_offset: f64::from(1u32 << bit_depth.saturating_sub(1)),
                chroma_scale: max_value,
            },
            super::sequence::ColorRange::Studio => Self {
                y_offset: 16.0 * range_shift,
                y_scale: 219.0 * range_shift,
                chroma_offset: 128.0 * range_shift,
                chroma_scale: 224.0 * range_shift,
            },
        })
    }
}

fn yuv_to_rgb_u16(
    y_sample: u16,
    u_sample: u16,
    v_sample: u16,
    range: SampleRange,
    matrix: MatrixCoefficients,
) -> [u16; 3] {
    let y = ((f64::from(y_sample) - range.y_offset) / range.y_scale).clamp(0.0, 1.0);
    let cb = (f64::from(u_sample) - range.chroma_offset) / range.chroma_scale;
    let cr = (f64::from(v_sample) - range.chroma_offset) / range.chroma_scale;
    let (r, g, b) = match matrix {
        MatrixCoefficients::Yuv { kr, kb } => {
            let r = y + 2.0 * (1.0 - kr) * cr;
            let b = y + 2.0 * (1.0 - kb) * cb;
            let g = (y - kr * r - kb * b) / (1.0 - kr - kb);
            (r, g, b)
        }
        MatrixCoefficients::YcGco => {
            // YCgCo stores Cg and Co around the chroma midpoint.  Its
            // inverse lifting transform is reversible in the normalized
            // domain: G=Y+Cg, R=Y-Cg+Co, B=Y-Cg-Co.
            let g = y + cb;
            let r = y - cb + cr;
            let b = y - cb - cr;
            (r, g, b)
        }
    };
    [
        normalized_to_u16(r),
        normalized_to_u16(g),
        normalized_to_u16(b),
    ]
}

fn normalized_to_u16(value: f64) -> u16 {
    (value.clamp(0.0, 1.0) * f64::from(u16::MAX)).round() as u16
}

fn scale_sample_to_u16(sample: u16, max_source: u32) -> u16 {
    if max_source == u32::from(u16::MAX) {
        sample
    } else {
        ((u32::from(sample) * u32::from(u16::MAX) + (max_source / 2)) / max_source) as u16
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av1::decode::{PlaneBuffer, PlaneLayout};

    #[test]
    fn add_residual_clips_to_bit_depth() {
        let out = add_residual_to_prediction(&[10, 250, 128], &[-20, 20, 0], 8).unwrap();

        assert_eq!(out, vec![0, 255, 128]);
    }

    #[test]
    fn write_plane_block_clips_at_frame_edge() {
        let layout = PlaneLayout {
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

        write_plane_block(&mut plane, 2, 3, 3, 2, &[1, 2, 3, 4, 5, 6]).unwrap();

        assert_eq!(&plane.samples[14..16], &[1, 2]);
    }

    #[test]
    fn read_intra_edges_tracks_availability_and_frame_edge_defaults() {
        let layout = PlaneLayout {
            plane: 0,
            width: 4,
            height: 4,
            subsampling_x: 0,
            subsampling_y: 0,
            sample_count: 16,
        };
        let plane = PlaneBuffer {
            layout,
            samples: (0..16).collect(),
        };

        let inner = read_intra_edges(&plane, 1, 2, 2, 2, 8);

        assert_eq!(inner.above, vec![5, 6, 7, 7]);
        assert_eq!(inner.left, vec![8, 12, 12, 12]);
        assert_eq!(inner.above_left, 4);
        assert!(inner.above_available);
        assert!(inner.left_available);

        let frame_edge = read_intra_edges(&plane, 0, 0, 2, 2, 8);

        assert_eq!(frame_edge.above, vec![127, 127, 127, 127]);
        assert_eq!(frame_edge.left, vec![129, 129, 129, 129]);
        assert_eq!(frame_edge.above_left, 128);
        assert!(!frame_edge.above_available);
        assert!(!frame_edge.left_available);
    }

    #[test]
    fn read_intra_edges_can_mask_partition_unavailable_extensions() {
        let layout = PlaneLayout {
            plane: 0,
            width: 6,
            height: 6,
            subsampling_x: 0,
            subsampling_y: 0,
            sample_count: 36,
        };
        let plane = PlaneBuffer {
            layout,
            samples: (0..36).collect(),
        };

        let edges = read_intra_edges_with_extension_availability(&plane, 1, 2, 2, 2, 8, 0, 0);

        assert_eq!(edges.above, vec![7, 8, 8, 8]);
        assert_eq!(edges.left, vec![12, 18, 18, 18]);
        assert_eq!(edges.above_left, 6);
        assert!(edges.above_available);
        assert!(edges.left_available);

        let unmasked = read_intra_edges_with_extension_availability(&plane, 1, 2, 2, 2, 8, 2, 2);

        assert_eq!(unmasked.above, vec![7, 8, 9, 10]);
        assert_eq!(unmasked.left, vec![12, 18, 24, 30]);

        let partial = read_intra_edges_with_extension_availability(&plane, 1, 2, 2, 2, 8, 1, 1);

        assert_eq!(partial.above, vec![7, 8, 9, 9]);
        assert_eq!(partial.left, vec![12, 18, 24, 24]);
    }

    #[test]
    fn identity_rgba_uses_av1_gbr_identity_plane_order() {
        let layout = PlaneLayout {
            plane: 0,
            width: 2,
            height: 1,
            subsampling_x: 0,
            subsampling_y: 0,
            sample_count: 2,
        };
        let buffers = FrameBuffers {
            width: 2,
            height: 1,
            planes: vec![
                PlaneBuffer {
                    layout,
                    samples: vec![30, 40],
                },
                PlaneBuffer {
                    layout: PlaneLayout { plane: 1, ..layout },
                    samples: vec![50, 60],
                },
                PlaneBuffer {
                    layout: PlaneLayout { plane: 2, ..layout },
                    samples: vec![10, 20],
                },
            ],
        };

        let image = frame_buffers_to_identity_rgba_8(&buffers).unwrap();

        assert_eq!(image.rgba, vec![10, 30, 50, 255, 20, 40, 60, 255]);
    }

    #[test]
    fn rgba_conversion_supports_full_range_bt709_matrix() {
        let layout = PlaneLayout {
            plane: 0,
            width: 1,
            height: 1,
            subsampling_x: 0,
            subsampling_y: 0,
            sample_count: 1,
        };
        let buffers = FrameBuffers {
            width: 1,
            height: 1,
            planes: vec![
                PlaneBuffer {
                    layout,
                    samples: vec![0],
                },
                PlaneBuffer {
                    layout: PlaneLayout { plane: 1, ..layout },
                    samples: vec![128],
                },
                PlaneBuffer {
                    layout: PlaneLayout { plane: 2, ..layout },
                    samples: vec![128],
                },
            ],
        };
        let color_config = ColorConfig {
            high_bitdepth: false,
            twelve_bit: false,
            bit_depth: 8,
            monochrome: false,
            color_description: Some(super::super::sequence::ColorDescription {
                color_primaries: 1,
                transfer_characteristics: 13,
                matrix_coefficients: 1,
            }),
            color_range: super::super::sequence::ColorRange::Full,
            subsampling_x: false,
            subsampling_y: false,
            chroma_sample_position: None,
            separate_uv_delta_q: false,
        };

        let image = frame_buffers_to_rgba_8(&buffers, &color_config).unwrap();

        assert_eq!(image.rgba, vec![0, 0, 0, 255]);
    }

    #[test]
    fn rgba_conversion_supports_ycgco_matrix() {
        let layout = PlaneLayout {
            plane: 0,
            width: 1,
            height: 1,
            subsampling_x: 0,
            subsampling_y: 0,
            sample_count: 1,
        };
        let buffers = FrameBuffers {
            width: 1,
            height: 1,
            planes: vec![
                PlaneBuffer {
                    layout,
                    samples: vec![128], // Y = 0.5
                },
                PlaneBuffer {
                    layout: PlaneLayout { plane: 1, ..layout },
                    samples: vec![128], // Cg = 0
                },
                PlaneBuffer {
                    layout: PlaneLayout { plane: 2, ..layout },
                    samples: vec![192], // Co = 0.25
                },
            ],
        };
        let color_config = ColorConfig {
            high_bitdepth: false,
            twelve_bit: false,
            bit_depth: 8,
            monochrome: false,
            color_description: Some(super::super::sequence::ColorDescription {
                color_primaries: 1,
                transfer_characteristics: 13,
                matrix_coefficients: 8,
            }),
            color_range: super::super::sequence::ColorRange::Full,
            subsampling_x: false,
            subsampling_y: false,
            chroma_sample_position: None,
            separate_uv_delta_q: false,
        };
        let image = frame_buffers_to_rgba_16(&buffers, &color_config).unwrap();
        assert!(image.rgba[0] > 48_000);
        assert!((image.rgba[1] as i32 - 32_896).abs() < 512);
        assert!(image.rgba[2] < 18_000);
    }

    #[test]
    fn rgba_conversion_supports_studio_range_bt601_matrix() {
        let layout = PlaneLayout {
            plane: 0,
            width: 3,
            height: 1,
            subsampling_x: 0,
            subsampling_y: 0,
            sample_count: 3,
        };
        let buffers = FrameBuffers {
            width: 3,
            height: 1,
            planes: vec![
                PlaneBuffer {
                    layout,
                    samples: vec![16, 235, 81],
                },
                PlaneBuffer {
                    layout: PlaneLayout { plane: 1, ..layout },
                    samples: vec![128, 128, 90],
                },
                PlaneBuffer {
                    layout: PlaneLayout { plane: 2, ..layout },
                    samples: vec![128, 128, 240],
                },
            ],
        };
        let color_config = ColorConfig {
            high_bitdepth: false,
            twelve_bit: false,
            bit_depth: 8,
            monochrome: false,
            color_description: Some(super::super::sequence::ColorDescription {
                color_primaries: 6,
                transfer_characteristics: 6,
                matrix_coefficients: 6,
            }),
            color_range: super::super::sequence::ColorRange::Studio,
            subsampling_x: false,
            subsampling_y: false,
            chroma_sample_position: None,
            separate_uv_delta_q: false,
        };

        let image = frame_buffers_to_rgba_8(&buffers, &color_config).unwrap();

        assert_eq!(&image.rgba[..8], &[0, 0, 0, 255, 255, 255, 255, 255]);
        assert_rgb_close(&image.rgba[8..12], &[255, 0, 0, 255], 1);
    }

    #[test]
    fn rgba_conversion_tone_maps_pq_transfer() {
        let layout = PlaneLayout {
            plane: 0,
            width: 2,
            height: 1,
            subsampling_x: 0,
            subsampling_y: 0,
            sample_count: 2,
        };
        let buffers = FrameBuffers {
            width: 2,
            height: 1,
            planes: vec![
                PlaneBuffer {
                    layout,
                    samples: vec![0, 512],
                },
                PlaneBuffer {
                    layout: PlaneLayout { plane: 1, ..layout },
                    samples: vec![512, 512],
                },
                PlaneBuffer {
                    layout: PlaneLayout { plane: 2, ..layout },
                    samples: vec![512, 512],
                },
            ],
        };
        let color_config = ColorConfig {
            high_bitdepth: true,
            twelve_bit: false,
            bit_depth: 10,
            monochrome: false,
            color_description: Some(super::super::sequence::ColorDescription {
                color_primaries: 9,
                transfer_characteristics: 16,
                matrix_coefficients: 9,
            }),
            color_range: super::super::sequence::ColorRange::Full,
            subsampling_x: false,
            subsampling_y: false,
            chroma_sample_position: None,
            separate_uv_delta_q: false,
        };

        let image = frame_buffers_to_rgba_16(&buffers, &color_config).unwrap();

        assert_eq!(&image.rgba[..4], &[0, 0, 0, u16::MAX]);
        assert!(image.rgba[4] > 0);
        assert_eq!(image.rgba[7], u16::MAX);
    }

    #[test]
    fn rgba_conversion_tone_maps_hlg_transfer() {
        let layout = PlaneLayout {
            plane: 0,
            width: 1,
            height: 1,
            subsampling_x: 0,
            subsampling_y: 0,
            sample_count: 1,
        };
        let buffers = FrameBuffers {
            width: 1,
            height: 1,
            planes: vec![
                PlaneBuffer {
                    layout,
                    samples: vec![255],
                },
                PlaneBuffer {
                    layout: PlaneLayout { plane: 1, ..layout },
                    samples: vec![128],
                },
                PlaneBuffer {
                    layout: PlaneLayout { plane: 2, ..layout },
                    samples: vec![128],
                },
            ],
        };
        let color_config = ColorConfig {
            high_bitdepth: false,
            twelve_bit: false,
            bit_depth: 8,
            monochrome: false,
            color_description: Some(super::super::sequence::ColorDescription {
                color_primaries: 9,
                transfer_characteristics: 18,
                matrix_coefficients: 9,
            }),
            color_range: super::super::sequence::ColorRange::Full,
            subsampling_x: false,
            subsampling_y: false,
            chroma_sample_position: None,
            separate_uv_delta_q: false,
        };

        let image = frame_buffers_to_rgba_16(&buffers, &color_config).unwrap();

        assert!(image.rgba[0] > 0);
        assert!(image.rgba[0] <= u16::MAX);
        assert_eq!(image.rgba[3], u16::MAX);
    }

    #[test]
    fn hdr_transfer_curves_are_bounded_and_monotonic() {
        assert_eq!(pq_to_linear(0.0), 0.0);
        assert!((pq_to_linear(1.0) - 1.0).abs() < 1e-6);
        assert_eq!(hlg_to_linear(0.0), 0.0);

        let samples = [0.0, 0.25, 0.5, 0.75, 1.0];
        let mut previous_pq = 0.0;
        let mut previous_hlg = 0.0;
        for sample in samples {
            let pq = pq_to_linear(sample);
            let hlg = hlg_to_linear(sample);
            assert!(pq.is_finite() && (0.0..=1.0).contains(&pq));
            assert!(hlg.is_finite() && (0.0..=1.0).contains(&hlg));
            assert!(pq >= previous_pq);
            assert!(hlg >= previous_hlg);
            previous_pq = pq;
            previous_hlg = hlg;
        }

        let mut rgba = [u16::MAX, u16::MAX / 2, 0, 1234];
        apply_transfer_function(&mut rgba, TransferFunction::Pq);
        assert_eq!(rgba[3], 1234);
        assert!(rgba[..3].iter().all(|sample| *sample <= u16::MAX));
    }

    #[test]
    fn sdr_transfer_curves_decode_to_srgb() {
        let midpoint = u16::MAX / 2;
        let gamma22 = transfer_to_sdr(midpoint, TransferFunction::Gamma22);
        let gamma28 = transfer_to_sdr(midpoint, TransferFunction::Gamma28);
        let linear = transfer_to_sdr(midpoint, TransferFunction::Linear);

        assert_eq!(transfer_to_sdr(0, TransferFunction::Linear), 0);
        assert_eq!(
            transfer_to_sdr(u16::MAX, TransferFunction::Gamma22),
            u16::MAX
        );
        assert!(gamma28 < gamma22);
        assert!(gamma22 < linear);
        assert!(linear < u16::MAX);
    }

    fn assert_rgb_close(actual: &[u8], expected: &[u8], tolerance: u8) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                actual.abs_diff(*expected) <= tolerance,
                "{actual} differed from {expected} by more than {tolerance}"
            );
        }
    }
}
