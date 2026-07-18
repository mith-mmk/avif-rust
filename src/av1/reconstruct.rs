use super::decode::{FrameBuffers, PlaneBuffer};
use super::sequence::{ChromaSamplePosition, ColorConfig};
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
    frame_buffers_to_identity_rgba_8_fast(buffers)
}

fn frame_buffers_to_identity_rgba_8_fast(
    buffers: &FrameBuffers,
) -> Result<ImageBuffer, DecoderError> {
    validate_rgba_conversion(buffers)?;
    if buffers.planes.len() < 3
        || buffers.planes[..3]
            .iter()
            .any(|plane| plane.layout.subsampling_x != 0 || plane.layout.subsampling_y != 0)
    {
        return Err(DecoderError::Bitstream(
            "AV1 identity GBR planes must be full-resolution".to_string(),
        ));
    }
    let plane_g = &buffers.planes[0].samples;
    let plane_b = &buffers.planes[1].samples;
    let plane_r = &buffers.planes[2].samples;
    let mut rgba = vec![0u8; buffers.width * buffers.height * 4];
    for (index, pixel) in rgba.chunks_exact_mut(4).enumerate() {
        pixel[0] = plane_r[index] as u8;
        pixel[1] = plane_g[index] as u8;
        pixel[2] = plane_b[index] as u8;
        pixel[3] = u8::try_from(
            buffers
                .planes
                .get(3)
                .map(|plane| plane.samples[index])
                .unwrap_or(u16::MAX),
        )
        .unwrap_or(u8::MAX);
    }
    Ok(ImageBuffer {
        width: buffers.width,
        height: buffers.height,
        rgba,
    })
}

pub fn frame_buffers_to_rgba_8(
    buffers: &FrameBuffers,
    color_config: &ColorConfig,
) -> Result<ImageBuffer, DecoderError> {
    if color_config.bit_depth == 8
        && !color_config.monochrome
        && !color_config.subsampling_x
        && !color_config.subsampling_y
        && color_config
            .color_description
            .is_none_or(|description| description.transfer_characteristics == 13)
        && color_config
            .color_description
            .is_some_and(|description| description.matrix_coefficients == 0)
        && buffers
            .planes
            .iter()
            .all(|plane| plane.layout.subsampling_x == 0 && plane.layout.subsampling_y == 0)
    {
        return frame_buffers_to_identity_rgba_8_fast(buffers);
    }
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
        for y in 0..buffers.height {
            for x in 0..buffers.width {
                let index = y * buffers.width + x;
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
        if buffers.planes.get(3).is_none() {
            for (index, pixel) in rgba.chunks_exact_mut(4).enumerate() {
                pixel[0] = scale_sample_to_u16(plane_r[index], max_source);
                pixel[1] = scale_sample_to_u16(plane_g[index], max_source);
                pixel[2] = scale_sample_to_u16(plane_b[index], max_source);
                pixel[3] = u16::MAX;
            }
        } else {
            for y in 0..buffers.height {
                for x in 0..buffers.width {
                    let index = y * buffers.width + x;
                    let out = index * 4;
                    rgba[out] = scale_sample_to_u16(plane_r[index], max_source);
                    rgba[out + 1] = scale_sample_to_u16(plane_g[index], max_source);
                    rgba[out + 2] = scale_sample_to_u16(plane_b[index], max_source);
                    rgba[out + 3] = alpha_sample(buffers.planes.get(3), x, y, max_source);
                }
            }
        }
    } else {
        let color_primaries = color_config
            .color_description
            .map(|description| description.color_primaries)
            .unwrap_or(2);
        let matrix = MatrixCoefficients::from_av1(matrix_coefficients, color_primaries)?;
        let range = SampleRange::new(color_config.bit_depth, color_config.color_range)?;
        let plane_y = &buffers.planes[0];
        let plane_u = buffers.planes.get(1);
        let plane_v = buffers.planes.get(2);
        let chroma_mid = 1u16 << color_config.bit_depth.saturating_sub(1);
        let alpha_max_source = (1u32 << color_config.bit_depth) - 1;
        let fast_yuv_coefficients = match matrix {
            MatrixCoefficients::Yuv { kr, kb } => Some((kr as f32, kb as f32)),
            _ => None,
        };
        let direct_yuv444 = fast_yuv_coefficients.is_some()
            && buffers.planes.get(3).is_none()
            && plane_y.layout.subsampling_x == 0
            && plane_y.layout.subsampling_y == 0
            && plane_y.layout.width == buffers.width
            && plane_y.layout.height == buffers.height
            && plane_u.is_some_and(|plane| {
                plane.layout.subsampling_x == 0
                    && plane.layout.subsampling_y == 0
                    && plane.layout.width == buffers.width
                    && plane.layout.height == buffers.height
            })
            && plane_v.is_some_and(|plane| {
                plane.layout.subsampling_x == 0
                    && plane.layout.subsampling_y == 0
                    && plane.layout.width == buffers.width
                    && plane.layout.height == buffers.height
            });
        if direct_yuv444 {
            let plane_u = plane_u.expect("direct YUV444 path requires U plane");
            let plane_v = plane_v.expect("direct YUV444 path requires V plane");
            let (kr, kb) = fast_yuv_coefficients.expect("YUV matrix was checked above");
            for (index, pixel) in rgba.chunks_exact_mut(4).enumerate() {
                let rgb = yuv_to_rgb_u16_fast(
                    plane_y.samples[index],
                    plane_u.samples[index],
                    plane_v.samples[index],
                    range,
                    kr,
                    kb,
                );
                pixel[..3].copy_from_slice(&rgb);
                pixel[3] = u16::MAX;
            }
        } else {
            for y in 0..buffers.height {
                for x in 0..buffers.width {
                    let index = y * buffers.width + x;
                    let rgb = yuv_to_rgb_u16(
                        sample_plane(plane_y, x, y),
                        plane_u
                            .map(|plane| {
                                sample_chroma_plane(
                                    plane,
                                    x,
                                    y,
                                    color_config.chroma_sample_position,
                                )
                            })
                            .unwrap_or(chroma_mid),
                        plane_v
                            .map(|plane| {
                                sample_chroma_plane(
                                    plane,
                                    x,
                                    y,
                                    color_config.chroma_sample_position,
                                )
                            })
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

fn sample_chroma_plane(
    plane: &super::decode::PlaneBuffer,
    x: usize,
    y: usize,
    position: Option<ChromaSamplePosition>,
) -> u16 {
    let subsampling_x = usize::from(plane.layout.subsampling_x);
    let subsampling_y = usize::from(plane.layout.subsampling_y);
    if subsampling_x == 0 && subsampling_y == 0 {
        return sample_plane(plane, x, y);
    }
    let source_x = (x >> subsampling_x).min(plane.layout.width - 1);
    let source_y = (y >> subsampling_y).min(plane.layout.height - 1);
    let next_x = (source_x + 1).min(plane.layout.width - 1);
    let next_y = (source_y + 1).min(plane.layout.height - 1);
    let (fraction_x, fraction_y) = if subsampling_x == 1 && subsampling_y == 1 {
        match position.unwrap_or(ChromaSamplePosition::Unknown) {
            // Horizontally co-located and halfway between vertical samples.
            ChromaSamplePosition::Vertical => (0, (y & 1) as u32),
            // Co-located with the top-left luma sample of each 2x2 block.
            ChromaSamplePosition::Colocated => (0, 0),
            // Unknown is kept on the historical vertical path.  AV1 leaves
            // the source-side placement to the container/application when it
            // is unknown; this is also the fallback used by FFmpeg/libaom.
            ChromaSamplePosition::Unknown | ChromaSamplePosition::Reserved => (0, (y & 1) as u32),
        }
    } else {
        // AV1 4:2:2 uses horizontally co-located chroma in this path.
        (0, 0)
    };
    let top_left = u32::from(plane.samples[source_y * plane.layout.width + source_x]);
    let top_right = u32::from(plane.samples[source_y * plane.layout.width + next_x]);
    let bottom_left = u32::from(plane.samples[next_y * plane.layout.width + source_x]);
    let bottom_right = u32::from(plane.samples[next_y * plane.layout.width + next_x]);
    let top = top_left * (2 - fraction_x) + top_right * fraction_x;
    let bottom = bottom_left * (2 - fraction_x) + bottom_right * fraction_x;
    ((top * (2 - fraction_y) + bottom * fraction_y + 2) / 4) as u16
}

fn yuv_to_rgb_u16_fast(
    y_sample: u16,
    u_sample: u16,
    v_sample: u16,
    range: SampleRange,
    kr: f32,
    kb: f32,
) -> [u16; 3] {
    let y = ((f32::from(y_sample) - range.y_offset as f32) / range.y_scale as f32).clamp(0.0, 1.0);
    let cb = (f32::from(u_sample) - range.chroma_offset as f32) / range.chroma_scale as f32;
    let cr = (f32::from(v_sample) - range.chroma_offset as f32) / range.chroma_scale as f32;
    let r = y + 2.0 * (1.0 - kr) * cr;
    let b = y + 2.0 * (1.0 - kb) * cb;
    let g = (y - kr * r - kb * b) / (1.0 - kr - kb);
    [
        normalized_to_u16_fast(r),
        normalized_to_u16_fast(g),
        normalized_to_u16_fast(b),
    ]
}

fn validate_rgba_conversion(buffers: &FrameBuffers) -> Result<(), DecoderError> {
    let _ = buffers;
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum TransferFunction {
    Gamma22,
    Gamma28,
    Log,
    LogSqrt,
    Linear,
    Smpte428,
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
        // These display-referred curves preserve the encoded samples for the
        // existing RGBA API. BT.2020 10/12-bit uses the BT.709 OETF; the
        // IEC 61966-2-4 and BT.1361 variants are equivalent in this bounded,
        // non-negative display path.
        1 | 2 | 6 | 7 | 11 | 12 | 13 | 14 | 15 => Ok(None),
        4 => Ok(Some(TransferFunction::Gamma22)),
        5 => Ok(Some(TransferFunction::Gamma28)),
        9 => Ok(Some(TransferFunction::Log)),
        10 => Ok(Some(TransferFunction::LogSqrt)),
        8 => Ok(Some(TransferFunction::Linear)),
        16 => Ok(Some(TransferFunction::Pq)),
        17 => Ok(Some(TransferFunction::Smpte428)),
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
        TransferFunction::Log => linear_to_srgb(log_to_linear(encoded)),
        TransferFunction::LogSqrt => linear_to_srgb(log_sqrt_to_linear(encoded)),
        TransferFunction::Linear => linear_to_srgb(encoded),
        TransferFunction::Smpte428 => linear_to_srgb(smpte428_to_linear(encoded)),
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
        TransferFunction::Gamma22
        | TransferFunction::Gamma28
        | TransferFunction::Log
        | TransferFunction::LogSqrt
        | TransferFunction::Linear
        | TransferFunction::Smpte428 => {
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

fn log_to_linear(encoded: f64) -> f64 {
    if encoded <= 0.0 {
        0.01
    } else {
        10.0_f64.powf(2.0 * (encoded - 1.0))
    }
}

fn log_sqrt_to_linear(encoded: f64) -> f64 {
    if encoded <= 0.0 {
        10.0_f64.sqrt() / 1000.0
    } else {
        10.0_f64.powf(2.5 * (encoded - 1.0))
    }
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

fn smpte428_to_linear(encoded: f64) -> f64 {
    // H.273 / SMPTE ST 428-1: V = (48 * L_o / 52.37)^(1 / 2.6).
    (52.37 / 48.0) * encoded.clamp(0.0, 1.0).powf(2.6)
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
    Bt2020ConstantLuminance { kr: f64, kb: f64 },
    Smpte2085,
    YcGco,
    Ictcp,
}

impl MatrixCoefficients {
    fn from_av1(matrix_coefficients: u8, color_primaries: u8) -> Result<Self, DecoderError> {
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
            10 => Ok(Self::Bt2020ConstantLuminance {
                kr: 0.2627,
                kb: 0.0593,
            }),
            11 => Ok(Self::Smpte2085),
            12 => {
                let (kr, kb) = derived_luma_coefficients(color_primaries)?;
                Ok(Self::Yuv { kr, kb })
            }
            13 => {
                let (kr, kb) = derived_luma_coefficients(color_primaries)?;
                Ok(Self::Bt2020ConstantLuminance { kr, kb })
            }
            14 => Ok(Self::Ictcp),
            _ => Err(DecoderError::Unsupported(format!(
                "AV1 matrix coefficients {matrix_coefficients} RGBA conversion is not supported yet"
            ))),
        }
    }
}

fn derived_luma_coefficients(color_primaries: u8) -> Result<(f64, f64), DecoderError> {
    let (red, green, blue, white) = match color_primaries {
        1 => (
            (0.640, 0.330),
            (0.300, 0.600),
            (0.150, 0.060),
            (0.3127, 0.3290),
        ),
        4 => (
            (0.670, 0.330),
            (0.210, 0.710),
            (0.140, 0.080),
            (0.310, 0.316),
        ),
        5 => (
            (0.640, 0.330),
            (0.290, 0.600),
            (0.150, 0.060),
            (0.3127, 0.3290),
        ),
        6 | 7 => (
            (0.630, 0.340),
            (0.310, 0.595),
            (0.155, 0.070),
            (0.3127, 0.3290),
        ),
        8 => (
            (0.681, 0.319),
            (0.243, 0.692),
            (0.145, 0.049),
            (0.310, 0.316),
        ),
        9 => (
            (0.708, 0.292),
            (0.170, 0.797),
            (0.131, 0.046),
            (0.3127, 0.3290),
        ),
        10 => ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (1.0 / 3.0, 1.0 / 3.0)),
        11 => (
            (0.680, 0.320),
            (0.265, 0.690),
            (0.150, 0.060),
            (0.314, 0.351),
        ),
        12 => (
            (0.680, 0.320),
            (0.265, 0.690),
            (0.150, 0.060),
            (0.3127, 0.3290),
        ),
        22 => (
            (0.630, 0.340),
            (0.310, 0.595),
            (0.155, 0.070),
            (0.3127, 0.3290),
        ),
        _ => {
            return Err(DecoderError::Unsupported(format!(
                "AV1 colour primaries {color_primaries} are required for chromaticity-derived matrix"
            )));
        }
    };
    let to_xyz = |(x, y): (f64, f64)| [x / y, 1.0, (1.0 - x - y) / y];
    let [xr, _, zr] = to_xyz(red);
    let [xg, _, zg] = to_xyz(green);
    let [xb, _, zb] = to_xyz(blue);
    let xw = white.0 / white.1;
    let zw = (1.0 - white.0 - white.1) / white.1;
    let ax = xr - xb;
    let bx = xg - xb;
    let az = zr - zb;
    let bz = zg - zb;
    let det = ax * bz - bx * az;
    if det.abs() < f64::EPSILON {
        return Err(DecoderError::Unsupported(
            "AV1 chromaticity-derived matrix has singular primaries".to_string(),
        ));
    }
    let rhs_x = xw - xb;
    let rhs_z = zw - zb;
    let kr = (rhs_x * bz - bx * rhs_z) / det;
    let kg = (ax * rhs_z - rhs_x * az) / det;
    let kb = 1.0 - kr - kg;
    if ![kr, kg, kb].iter().all(|value| value.is_finite()) {
        return Err(DecoderError::Unsupported(
            "AV1 chromaticity-derived matrix coefficients are not finite".to_string(),
        ));
    }
    Ok((kr, kb))
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
        MatrixCoefficients::Bt2020ConstantLuminance { kr, kb } => {
            // BT.2020 constant-luminance stores each chroma component with a
            // different scale on either side of the luma axis.  The sign of
            // the component selects the inverse branch (H.273 equations
            // 69/70), rather than the fixed scale used by matrix 9.
            let r = y + if cr < 0.0 {
                2.0 * (1.0 - kr) * cr
            } else {
                2.0 * kr * cr
            };
            let b = y + if cb < 0.0 {
                2.0 * (1.0 - kb) * cb
            } else {
                2.0 * kb * cb
            };
            let g = (y - kr * r - kb * b) / (1.0 - kr - kb);
            (r, g, b)
        }
        MatrixCoefficients::Smpte2085 => {
            // SMPTE ST 2085 / H.273 equations 76--78 encode Y'DzDx.
            // The chroma samples are centered at zero and scaled by two.
            let g = y;
            let b = (2.0 * cb + y) / 0.986566;
            let r = 2.0 * cr + 0.991902 * y;
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
        MatrixCoefficients::Ictcp => {
            // BT.2100 ICtCp stores PQ-coded LMS' values transformed into
            // I/Ct/Cp.  The inverse matrices below are the exact inverse of
            // the integer matrices from BT.2100 (with the 4096 scale
            // cancelled), so the result remains PQ-coded until the common
            // transfer-characteristics stage converts it to display RGB.
            let l = y + 0.008609037037932756 * cb + 0.11102962500302596 * cr;
            let m = y - 0.008609037037932756 * cb - 0.11102962500302596 * cr;
            let s = y + 0.5600313357106791 * cb - 0.32062717498731885 * cr;
            let r = 3.4366066943330784 * l - 2.50645211865627 * m + 0.06984542432319148 * s;
            let g = -0.7913295555989287 * l + 1.9836004517922907 * m - 0.192270896193362 * s;
            let b = -0.025949899690592672 * l - 0.09891371471172644 * m + 1.1248636144023192 * s;
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

fn normalized_to_u16_fast(value: f32) -> u16 {
    (value.clamp(0.0, 1.0) * f32::from(u16::MAX)).round() as u16
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
    fn chroma_sample_position_selects_420_interpolation() {
        let plane = PlaneBuffer {
            layout: PlaneLayout {
                plane: 1,
                width: 2,
                height: 2,
                subsampling_x: 1,
                subsampling_y: 1,
                sample_count: 4,
            },
            samples: vec![10, 20, 30, 40],
        };

        assert_eq!(
            sample_chroma_plane(&plane, 1, 1, Some(ChromaSamplePosition::Vertical)),
            20
        );
        assert_eq!(
            sample_chroma_plane(&plane, 1, 1, Some(ChromaSamplePosition::Colocated)),
            10
        );
        assert_eq!(
            sample_chroma_plane(&plane, 1, 1, Some(ChromaSamplePosition::Unknown)),
            20
        );
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
    fn rgba_conversion_supports_bt2020_constant_luminance_matrix() {
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
                    samples: vec![128], // Y' ~= 0.5
                },
                PlaneBuffer {
                    layout: PlaneLayout { plane: 1, ..layout },
                    samples: vec![192], // positive Cb branch
                },
                PlaneBuffer {
                    layout: PlaneLayout { plane: 2, ..layout },
                    samples: vec![64], // negative Cr branch
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
                transfer_characteristics: 13,
                matrix_coefficients: 10,
            }),
            color_range: super::super::sequence::ColorRange::Full,
            subsampling_x: false,
            subsampling_y: false,
            chroma_sample_position: None,
            separate_uv_delta_q: false,
        };

        let image = frame_buffers_to_rgba_16(&buffers, &color_config).unwrap();

        assert!(image.rgba[0] > 8_000 && image.rgba[0] < 10_000);
        assert!(image.rgba[1] > 38_000 && image.rgba[1] < 44_000);
        assert!(image.rgba[2] > 33_000 && image.rgba[2] < 36_000);
        assert_eq!(image.rgba[3], u16::MAX);
    }

    #[test]
    fn rgba_conversion_supports_smpte2085_matrix() {
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
                    samples: vec![128], // Y' ~= 0.5
                },
                PlaneBuffer {
                    layout: PlaneLayout { plane: 1, ..layout },
                    samples: vec![154], // D'z ~= 0.1
                },
                PlaneBuffer {
                    layout: PlaneLayout { plane: 2, ..layout },
                    samples: vec![77], // D'x ~= -0.2
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
                transfer_characteristics: 13,
                matrix_coefficients: 11,
            }),
            color_range: super::super::sequence::ColorRange::Full,
            subsampling_x: false,
            subsampling_y: false,
            chroma_sample_position: None,
            separate_uv_delta_q: false,
        };

        let image = frame_buffers_to_rgba_16(&buffers, &color_config).unwrap();

        assert!(image.rgba[0] > 5_000 && image.rgba[0] < 8_000);
        assert!(image.rgba[1] > 31_000 && image.rgba[1] < 34_000);
        assert!(image.rgba[2] > 45_000 && image.rgba[2] < 49_000);
        assert_eq!(image.rgba[3], u16::MAX);
    }

    #[test]
    fn fast_yuv444_conversion_stays_within_one_code_value_of_scalar_path() {
        let range = SampleRange::new(8, super::super::sequence::ColorRange::Studio).unwrap();
        let matrix = MatrixCoefficients::Yuv {
            kr: 0.2627,
            kb: 0.0593,
        };
        let fast = yuv_to_rgb_u16_fast(128, 96, 160, range, 0.2627, 0.0593);
        let scalar = yuv_to_rgb_u16(128, 96, 160, range, matrix);
        assert!(
            fast.iter()
                .zip(scalar)
                .all(|(fast, scalar)| fast.abs_diff(scalar) <= 1)
        );
    }

    #[test]
    fn chromaticity_derived_matrices_match_bt2020_coefficients() {
        let (kr, kb) = derived_luma_coefficients(9).unwrap();
        assert!((kr - 0.2627).abs() < 0.0001);
        assert!((kb - 0.0593).abs() < 0.0001);
        let dci = derived_luma_coefficients(11).unwrap();
        let d65 = derived_luma_coefficients(12).unwrap();
        assert!((dci.0 - d65.0).abs() > 0.0001);
        assert!(matches!(
            MatrixCoefficients::from_av1(12, 9),
            Ok(MatrixCoefficients::Yuv { .. })
        ));
        assert!(matches!(
            MatrixCoefficients::from_av1(13, 9),
            Ok(MatrixCoefficients::Bt2020ConstantLuminance { .. })
        ));
        assert!(matches!(
            MatrixCoefficients::from_av1(14, 9),
            Ok(MatrixCoefficients::Ictcp)
        ));
    }

    #[test]
    fn chromaticity_derived_matrices_reject_unspecified_primaries() {
        let error = MatrixCoefficients::from_av1(12, 2).unwrap_err();
        assert!(
            matches!(error, DecoderError::Unsupported(message) if message.contains("colour primaries"))
        );
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
        let log = transfer_to_sdr(midpoint, TransferFunction::Log);
        let log_sqrt = transfer_to_sdr(midpoint, TransferFunction::LogSqrt);
        let linear = transfer_to_sdr(midpoint, TransferFunction::Linear);
        let smpte428 = transfer_to_sdr(midpoint, TransferFunction::Smpte428);

        assert_eq!(transfer_to_sdr(0, TransferFunction::Linear), 0);
        assert_eq!(
            transfer_to_sdr(u16::MAX, TransferFunction::Gamma22),
            u16::MAX
        );
        assert!(gamma28 < gamma22);
        assert!(log < linear);
        assert!(log_sqrt < linear);
        assert!(gamma22 < linear);
        assert!(smpte428 < linear);
        assert!(linear < u16::MAX);
        assert!((smpte428_to_linear(1.0) - 52.37 / 48.0).abs() < 1e-12);
        assert!((log_to_linear(0.0) - 0.01).abs() < 1e-12);
        assert!((log_sqrt_to_linear(0.0) - 10.0_f64.sqrt() / 1000.0).abs() < 1e-12);
    }

    #[test]
    fn unspecified_cicp_transfer_keeps_existing_rgba_encoding() {
        let color_config = ColorConfig {
            high_bitdepth: false,
            twelve_bit: false,
            bit_depth: 8,
            monochrome: false,
            color_description: Some(super::super::sequence::ColorDescription {
                color_primaries: 2,
                transfer_characteristics: 2,
                matrix_coefficients: 2,
            }),
            color_range: super::super::sequence::ColorRange::Full,
            subsampling_x: false,
            subsampling_y: false,
            chroma_sample_position: None,
            separate_uv_delta_q: false,
        };

        assert_eq!(transfer_characteristics(&color_config), Ok(None));
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
