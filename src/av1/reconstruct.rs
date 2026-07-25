use super::decode::{FrameBuffers, PlaneBuffer};
use super::sequence::{ChromaSamplePosition, ColorConfig, ColorRange};
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
    let mut output = vec![0; prediction.len()];
    add_residual_to_prediction_into(prediction, residual, bit_depth, &mut output)?;
    Ok(output)
}

pub fn add_residual_to_prediction_into(
    prediction: &[u16],
    residual: &[i32],
    bit_depth: u8,
    output: &mut [u16],
) -> Result<(), DecoderError> {
    if prediction.len() != residual.len() {
        return Err(DecoderError::InvalidParam(
            "AV1 prediction and residual sizes differ".to_string(),
        ));
    }
    if output.len() != prediction.len() {
        return Err(DecoderError::InvalidParam(
            "AV1 reconstruction output size differs from prediction".to_string(),
        ));
    }
    let max_value = (1i32 << bit_depth) - 1;
    for ((output, pred), res) in output.iter_mut().zip(prediction).zip(residual) {
        *output = (i32::from(*pred) + *res).clamp(0, max_value) as u16;
    }
    Ok(())
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

#[cfg(not(target_family = "wasm"))]
const PARALLEL_RGBA_MIN_PIXELS: usize = 256 * 1024;
#[cfg(not(target_family = "wasm"))]
const MAX_RGBA_WORKERS: usize = 8;

fn for_each_rgba_row_chunk<T, F>(rgba: &mut [T], width: usize, height: usize, function: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    if width == 0 || height == 0 {
        return;
    }
    #[cfg(not(target_family = "wasm"))]
    let workers = if width.saturating_mul(height) < PARALLEL_RGBA_MIN_PIXELS {
        1
    } else {
        std::thread::available_parallelism()
            .map_or(1, std::num::NonZeroUsize::get)
            .min(MAX_RGBA_WORKERS)
            .min(height)
    };
    #[cfg(target_family = "wasm")]
    let workers = 1;
    if workers <= 1 {
        function(0, rgba);
        return;
    }

    #[cfg(not(target_family = "wasm"))]
    let rows_per_worker = height.div_ceil(workers);
    #[cfg(not(target_family = "wasm"))]
    let samples_per_chunk = rows_per_worker.saturating_mul(width).saturating_mul(4);
    #[cfg(not(target_family = "wasm"))]
    std::thread::scope(|scope| {
        for (chunk_index, chunk) in rgba.chunks_mut(samples_per_chunk).enumerate() {
            let first_row = chunk_index * rows_per_worker;
            let function = &function;
            scope.spawn(move || function(first_row, chunk));
        }
    });
    #[cfg(target_family = "wasm")]
    unreachable!("Wasm uses the sequential RGBA row path");
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
    if let Some(alpha) = buffers.planes.get(3).map(|plane| plane.samples.as_slice()) {
        for_each_rgba_row_chunk(
            &mut rgba,
            buffers.width,
            buffers.height,
            |first_row, chunk| {
                for (local_index, pixel) in chunk.chunks_exact_mut(4).enumerate() {
                    let index = first_row * buffers.width + local_index;
                    pixel[0] = plane_r[index] as u8;
                    pixel[1] = plane_g[index] as u8;
                    pixel[2] = plane_b[index] as u8;
                    pixel[3] = u8::try_from(alpha[index]).unwrap_or(u8::MAX);
                }
            },
        );
    } else {
        for_each_rgba_row_chunk(
            &mut rgba,
            buffers.width,
            buffers.height,
            |first_row, chunk| {
                for (local_index, pixel) in chunk.chunks_exact_mut(4).enumerate() {
                    let index = first_row * buffers.width + local_index;
                    pixel[0] = plane_r[index] as u8;
                    pixel[1] = plane_g[index] as u8;
                    pixel[2] = plane_b[index] as u8;
                    pixel[3] = u8::MAX;
                }
            },
        );
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
    if color_config.bit_depth > 8 && transfer_characteristics(color_config)?.is_none() {
        if let Some(image) = frame_buffers_to_rgba_8_high_bit_sdr(buffers, color_config)? {
            return Ok(image);
        }
    }
    if color_config.bit_depth == 8 && transfer_characteristics(color_config)?.is_none() {
        if color_config.monochrome {
            return frame_buffers_to_rgba_8_monochrome_sdr(buffers);
        }
        return frame_buffers_to_rgba_8_sdr(buffers, color_config);
    }
    if color_config.bit_depth == 8
        && !color_config.monochrome
        && !color_config.subsampling_x
        && !color_config.subsampling_y
        && color_config
            .color_description
            .is_none_or(|description| description.transfer_characteristics == 13)
        && color_config
            .color_description
            .is_some_and(|description| matches!(description.matrix_coefficients, 0 | 3))
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

fn frame_buffers_to_rgba_8_high_bit_sdr(
    buffers: &FrameBuffers,
    color_config: &ColorConfig,
) -> Result<Option<ImageBuffer>, DecoderError> {
    validate_rgba_conversion(buffers)?;
    let max_source = (1u32 << color_config.bit_depth) - 1;
    let mut rgba = vec![0u8; buffers.width * buffers.height * 4];

    if color_config.monochrome {
        let luma = buffers.planes.first().ok_or_else(|| {
            DecoderError::Bitstream("AV1 monochrome luma plane is missing".to_string())
        })?;
        let alpha = buffers.planes.get(3);
        for_each_rgba_row_chunk(
            &mut rgba,
            buffers.width,
            buffers.height,
            |first_row, chunk| {
                for (row_offset, row) in chunk.chunks_exact_mut(buffers.width * 4).enumerate() {
                    let y = first_row + row_offset;
                    for (x, pixel) in row.chunks_exact_mut(4).enumerate() {
                        let value =
                            u16_to_u8(scale_sample_to_u16(sample_plane(luma, x, y), max_source));
                        pixel[..3].fill(value);
                        pixel[3] = u16_to_u8(alpha_sample(alpha, x, y, max_source));
                    }
                }
            },
        );
        return Ok(Some(ImageBuffer {
            width: buffers.width,
            height: buffers.height,
            rgba,
        }));
    }

    let matrix_coefficients = matrix_coefficients_for_conversion(color_config);
    if matches!(matrix_coefficients, 0 | 3) {
        if buffers.planes.len() < 3
            || buffers.planes[..3]
                .iter()
                .any(|plane| plane.layout.subsampling_x != 0 || plane.layout.subsampling_y != 0)
        {
            return Ok(None);
        }
        let plane_g = &buffers.planes[0].samples;
        let plane_b = &buffers.planes[1].samples;
        let plane_r = &buffers.planes[2].samples;
        let alpha = buffers.planes.get(3);
        let scale_table = (max_source <= 4095).then(|| build_rgba8_scale_table(max_source));
        for_each_rgba_row_chunk(
            &mut rgba,
            buffers.width,
            buffers.height,
            |first_row, chunk| {
                for (row_offset, row) in chunk.chunks_exact_mut(buffers.width * 4).enumerate() {
                    let y = first_row + row_offset;
                    for (x, pixel) in row.chunks_exact_mut(4).enumerate() {
                        let index = y * buffers.width + x;
                        pixel[0] =
                            scale_sample_to_rgba8(plane_r[index], max_source, scale_table.as_ref());
                        pixel[1] =
                            scale_sample_to_rgba8(plane_g[index], max_source, scale_table.as_ref());
                        pixel[2] =
                            scale_sample_to_rgba8(plane_b[index], max_source, scale_table.as_ref());
                        pixel[3] = scale_sample_to_rgba8(
                            alpha
                                .map(|plane| plane.samples[index])
                                .unwrap_or(u16::try_from(max_source).unwrap_or(u16::MAX)),
                            max_source,
                            scale_table.as_ref(),
                        );
                    }
                }
            },
        );
        return Ok(Some(ImageBuffer {
            width: buffers.width,
            height: buffers.height,
            rgba,
        }));
    }

    let color_primaries = color_config
        .color_description
        .map(|description| description.color_primaries)
        .unwrap_or(2);
    let matrix = MatrixCoefficients::from_av1(matrix_coefficients, color_primaries)?;
    let MatrixCoefficients::Yuv { kr, kb } = matrix else {
        return Ok(None);
    };
    let range = SampleRange::new(color_config.bit_depth, color_config.color_range)?;
    let fast_range = range.as_fast();
    let plane_y = buffers
        .planes
        .first()
        .ok_or_else(|| DecoderError::Bitstream("AV1 luma plane is missing".to_string()))?;
    let plane_u = buffers.planes.get(1);
    let plane_v = buffers.planes.get(2);
    let chroma_mid = 1u16 << color_config.bit_depth.saturating_sub(1);
    let direct_yuv444 = plane_y.layout.subsampling_x == 0
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
        })
        && buffers.planes.get(3).is_none_or(|plane| {
            plane.layout.subsampling_x == 0
                && plane.layout.subsampling_y == 0
                && plane.layout.width == buffers.width
                && plane.layout.height == buffers.height
        });
    if direct_yuv444 {
        let plane_u = plane_u.expect("direct YUV444 path requires U plane");
        let plane_v = plane_v.expect("direct YUV444 path requires V plane");
        let MatrixCoefficients::Yuv { kr, kb } = matrix else {
            return Ok(None);
        };
        let alpha = buffers.planes.get(3);
        for_each_rgba_row_chunk(
            &mut rgba,
            buffers.width,
            buffers.height,
            |first_row, chunk| {
                for (local_index, pixel) in chunk.chunks_exact_mut(4).enumerate() {
                    let index = first_row * buffers.width + local_index;
                    let rgb = yuv_to_rgb_u16_fast(
                        plane_y.samples[index],
                        plane_u.samples[index],
                        plane_v.samples[index],
                        fast_range,
                        kr as f32,
                        kb as f32,
                    );
                    pixel[0] = u16_to_u8(rgb[0]);
                    pixel[1] = u16_to_u8(rgb[1]);
                    pixel[2] = u16_to_u8(rgb[2]);
                    pixel[3] = alpha
                        .map(|plane| {
                            u16_to_u8(scale_sample_to_u16(plane.samples[index], max_source))
                        })
                        .unwrap_or(u8::MAX);
                }
            },
        );
        return Ok(Some(ImageBuffer {
            width: buffers.width,
            height: buffers.height,
            rgba,
        }));
    }
    for_each_rgba_row_chunk(
        &mut rgba,
        buffers.width,
        buffers.height,
        |first_row, chunk| {
            for (row_offset, row) in chunk.chunks_exact_mut(buffers.width * 4).enumerate() {
                let y = first_row + row_offset;
                for (x, pixel) in row.chunks_exact_mut(4).enumerate() {
                    let y_sample = sample_plane(plane_y, x, y);
                    let u_sample = plane_u
                        .map(|plane| {
                            sample_chroma_plane(plane, x, y, color_config.chroma_sample_position)
                        })
                        .unwrap_or(chroma_mid);
                    let v_sample = plane_v
                        .map(|plane| {
                            sample_chroma_plane(plane, x, y, color_config.chroma_sample_position)
                        })
                        .unwrap_or(chroma_mid);
                    let rgb = yuv_to_rgb_u16_fast(
                        y_sample, u_sample, v_sample, fast_range, kr as f32, kb as f32,
                    );
                    pixel[0] = u16_to_u8(rgb[0]);
                    pixel[1] = u16_to_u8(rgb[1]);
                    pixel[2] = u16_to_u8(rgb[2]);
                    pixel[3] = u16_to_u8(alpha_sample(buffers.planes.get(3), x, y, max_source));
                }
            }
        },
    );
    Ok(Some(ImageBuffer {
        width: buffers.width,
        height: buffers.height,
        rgba,
    }))
}

fn frame_buffers_to_rgba_8_sdr(
    buffers: &FrameBuffers,
    color_config: &ColorConfig,
) -> Result<ImageBuffer, DecoderError> {
    validate_rgba_conversion(buffers)?;
    let matrix_coefficients = matrix_coefficients_for_conversion(color_config);
    if matches!(matrix_coefficients, 0 | 3)
        && buffers
            .planes
            .iter()
            .all(|plane| plane.layout.subsampling_x == 0 && plane.layout.subsampling_y == 0)
    {
        return frame_buffers_to_identity_rgba_8_fast(buffers);
    }

    let color_primaries = color_config
        .color_description
        .map(|description| description.color_primaries)
        .unwrap_or(2);
    let matrix = MatrixCoefficients::from_av1(matrix_coefficients, color_primaries)?;
    let range = SampleRange::new(8, color_config.color_range)?;
    let fast_range = range.as_fast();
    let plane_y = buffers
        .planes
        .first()
        .ok_or_else(|| DecoderError::Bitstream("AV1 luma plane is missing".to_string()))?;
    let plane_u = buffers.planes.get(1);
    let plane_v = buffers.planes.get(2);
    let fast_yuv_coefficients = match matrix {
        MatrixCoefficients::Yuv { kr, kb } => Some((kr as f32, kb as f32)),
        _ => None,
    };
    let fast_ycgco = matches!(matrix, MatrixCoefficients::YcGco);
    let direct_yuv444 = matches!(matrix, MatrixCoefficients::Yuv { .. })
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
        })
        && buffers.planes.get(3).is_none_or(|plane| {
            plane.layout.subsampling_x == 0
                && plane.layout.subsampling_y == 0
                && plane.layout.width == buffers.width
                && plane.layout.height == buffers.height
        });
    let direct_yuv420 = fast_yuv_coefficients.is_some()
        && buffers.planes.get(3).is_none()
        && plane_y.layout.subsampling_x == 0
        && plane_y.layout.subsampling_y == 0
        && plane_y.layout.width == buffers.width
        && plane_y.layout.height == buffers.height
        && plane_u.is_some_and(|plane| {
            plane.layout.subsampling_x == 1 && plane.layout.subsampling_y == 1
        })
        && plane_v.is_some_and(|plane| {
            plane.layout.subsampling_x == 1 && plane.layout.subsampling_y == 1
        });
    let mut rgba = vec![0u8; buffers.width * buffers.height * 4];
    if direct_yuv420 {
        let plane_u = plane_u.expect("direct YUV420 path requires U plane");
        let plane_v = plane_v.expect("direct YUV420 path requires V plane");
        let (kr, kb) = fast_yuv_coefficients.expect("YUV matrix was checked above");
        convert_yuv420_to_rgba8(
            &mut rgba,
            buffers.width,
            buffers.height,
            plane_y,
            plane_u,
            plane_v,
            fast_range,
            kr,
            kb,
            color_config.chroma_sample_position,
        );
        return Ok(ImageBuffer {
            width: buffers.width,
            height: buffers.height,
            rgba,
        });
    }
    if direct_yuv444 {
        let plane_u = plane_u.expect("direct YUV444 path requires U plane");
        let plane_v = plane_v.expect("direct YUV444 path requires V plane");
        let MatrixCoefficients::Yuv { kr, kb } = matrix else {
            unreachable!("direct YUV444 path requires a YUV matrix");
        };
        let alpha = buffers.planes.get(3);
        for_each_rgba_row_chunk(
            &mut rgba,
            buffers.width,
            buffers.height,
            |first_row, chunk| {
                for (local_index, pixel) in chunk.chunks_exact_mut(4).enumerate() {
                    let index = first_row * buffers.width + local_index;
                    let rgb = yuv_to_rgb_u16_fast(
                        plane_y.samples[index],
                        plane_u.samples[index],
                        plane_v.samples[index],
                        fast_range,
                        kr as f32,
                        kb as f32,
                    );
                    pixel[0] = u16_to_u8(rgb[0]);
                    pixel[1] = u16_to_u8(rgb[1]);
                    pixel[2] = u16_to_u8(rgb[2]);
                    pixel[3] = alpha
                        .map(|plane| plane.samples[index] as u8)
                        .unwrap_or(u8::MAX);
                }
            },
        );
        return Ok(ImageBuffer {
            width: buffers.width,
            height: buffers.height,
            rgba,
        });
    }
    for_each_rgba_row_chunk(
        &mut rgba,
        buffers.width,
        buffers.height,
        |first_row, chunk| {
            for (row_offset, row) in chunk.chunks_exact_mut(buffers.width * 4).enumerate() {
                let y = first_row + row_offset;
                for (x, pixel) in row.chunks_exact_mut(4).enumerate() {
                    let y_sample = sample_plane(plane_y, x, y);
                    let u_sample = plane_u
                        .map(|plane| {
                            sample_chroma_plane(plane, x, y, color_config.chroma_sample_position)
                        })
                        .unwrap_or(128);
                    let v_sample = plane_v
                        .map(|plane| {
                            sample_chroma_plane(plane, x, y, color_config.chroma_sample_position)
                        })
                        .unwrap_or(128);
                    let rgb = if let Some((kr, kb)) = fast_yuv_coefficients {
                        // Sampling remains normative, but the YUV matrix itself can
                        // use the bounded f32 path for subsampled planes too.
                        yuv_to_rgb_u16_fast(y_sample, u_sample, v_sample, fast_range, kr, kb)
                    } else if fast_ycgco {
                        ycgco_to_rgb_u16_fast(y_sample, u_sample, v_sample, fast_range)
                    } else {
                        yuv_to_rgb_u16(y_sample, u_sample, v_sample, range, matrix)
                    };
                    pixel[0] = u16_to_u8(rgb[0]);
                    pixel[1] = u16_to_u8(rgb[1]);
                    pixel[2] = u16_to_u8(rgb[2]);
                    pixel[3] = alpha_sample_u8(buffers.planes.get(3), x, y);
                }
            }
        },
    );
    Ok(ImageBuffer {
        width: buffers.width,
        height: buffers.height,
        rgba,
    })
}

fn frame_buffers_to_rgba_8_monochrome_sdr(
    buffers: &FrameBuffers,
) -> Result<ImageBuffer, DecoderError> {
    validate_rgba_conversion(buffers)?;
    let luma = buffers.planes.first().ok_or_else(|| {
        DecoderError::Bitstream("AV1 monochrome luma plane is missing".to_string())
    })?;
    let mut rgba = vec![0u8; buffers.width * buffers.height * 4];
    for_each_rgba_row_chunk(
        &mut rgba,
        buffers.width,
        buffers.height,
        |first_row, chunk| {
            for (row_offset, row) in chunk.chunks_exact_mut(buffers.width * 4).enumerate() {
                let y = first_row + row_offset;
                for (x, pixel) in row.chunks_exact_mut(4).enumerate() {
                    let value = sample_plane(luma, x, y) as u8;
                    let alpha = buffers
                        .planes
                        .get(3)
                        .map(|plane| sample_plane(plane, x, y) as u8)
                        .unwrap_or(u8::MAX);
                    pixel[0] = value;
                    pixel[1] = value;
                    pixel[2] = value;
                    pixel[3] = alpha;
                }
            }
        },
    );
    Ok(ImageBuffer {
        width: buffers.width,
        height: buffers.height,
        rgba,
    })
}

#[inline]
fn u16_to_u8(sample: u16) -> u8 {
    ((u32::from(sample) * 255 + 32_767) / 65_535) as u8
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
        let alpha = buffers.planes.get(3);
        for_each_rgba_row_chunk(
            &mut rgba,
            buffers.width,
            buffers.height,
            |first_row, chunk| {
                for (row_offset, row) in chunk.chunks_exact_mut(buffers.width * 4).enumerate() {
                    let y = first_row + row_offset;
                    for (x, pixel) in row.chunks_exact_mut(4).enumerate() {
                        let source_x = (x >> usize::from(luma.layout.subsampling_x))
                            .min(luma.layout.width.saturating_sub(1));
                        let source_y = (y >> usize::from(luma.layout.subsampling_y))
                            .min(luma.layout.height.saturating_sub(1));
                        let value = scale_sample_to_u16(
                            luma.samples[source_y * luma.layout.width + source_x],
                            max_source,
                        );
                        pixel[..3].fill(value);
                        pixel[3] = alpha_sample(alpha, x, y, max_source);
                    }
                }
            },
        );
        if let Some(transfer) = hdr_transfer {
            apply_transfer_function_rows(&mut rgba, buffers.width, buffers.height, transfer);
        }
        return Ok(Rgba16ImageBuffer {
            width: buffers.width,
            height: buffers.height,
            rgba,
        });
    }
    let matrix_coefficients = matrix_coefficients_for_conversion(color_config);
    let mut rgba = vec![0u16; buffers.width * buffers.height * 4];

    if matches!(matrix_coefficients, 0 | 3) {
        let max_source = (1u32 << color_config.bit_depth) - 1;
        let plane_g = &buffers.planes[0].samples;
        let plane_b = &buffers.planes[1].samples;
        let plane_r = &buffers.planes[2].samples;
        let alpha = buffers.planes.get(3);
        for_each_rgba_row_chunk(
            &mut rgba,
            buffers.width,
            buffers.height,
            |first_row, chunk| {
                for (row_offset, row) in chunk.chunks_exact_mut(buffers.width * 4).enumerate() {
                    let y = first_row + row_offset;
                    for (x, pixel) in row.chunks_exact_mut(4).enumerate() {
                        let index = y * buffers.width + x;
                        pixel[0] = scale_sample_to_u16(plane_r[index], max_source);
                        pixel[1] = scale_sample_to_u16(plane_g[index], max_source);
                        pixel[2] = scale_sample_to_u16(plane_b[index], max_source);
                        pixel[3] = alpha_sample(alpha, x, y, max_source);
                    }
                }
            },
        );
    } else {
        let color_primaries = color_config
            .color_description
            .map(|description| description.color_primaries)
            .unwrap_or(2);
        let matrix = MatrixCoefficients::from_av1(matrix_coefficients, color_primaries)?;
        let range = SampleRange::new(color_config.bit_depth, color_config.color_range)?;
        let fast_range = range.as_fast();
        let plane_y = &buffers.planes[0];
        let plane_u = buffers.planes.get(1);
        let plane_v = buffers.planes.get(2);
        let chroma_mid = 1u16 << color_config.bit_depth.saturating_sub(1);
        let alpha_max_source = (1u32 << color_config.bit_depth) - 1;
        let fast_yuv_coefficients = match matrix {
            MatrixCoefficients::Yuv { kr, kb } => Some((kr as f32, kb as f32)),
            _ => None,
        };
        let fast_ycgco = matches!(matrix, MatrixCoefficients::YcGco);
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
        let direct_yuv420 = fast_yuv_coefficients.is_some()
            && buffers.planes.get(3).is_none()
            && plane_y.layout.subsampling_x == 0
            && plane_y.layout.subsampling_y == 0
            && plane_y.layout.width == buffers.width
            && plane_y.layout.height == buffers.height
            && plane_u.is_some_and(|plane| {
                plane.layout.subsampling_x == 1 && plane.layout.subsampling_y == 1
            })
            && plane_v.is_some_and(|plane| {
                plane.layout.subsampling_x == 1 && plane.layout.subsampling_y == 1
            });
        if direct_yuv420 {
            let plane_u = plane_u.expect("direct YUV420 path requires U plane");
            let plane_v = plane_v.expect("direct YUV420 path requires V plane");
            let (kr, kb) = fast_yuv_coefficients.expect("YUV matrix was checked above");
            convert_yuv420_to_rgba16(
                &mut rgba,
                buffers.width,
                buffers.height,
                plane_y,
                plane_u,
                plane_v,
                fast_range,
                kr,
                kb,
                color_config.chroma_sample_position,
            );
        } else if direct_yuv444 {
            let plane_u = plane_u.expect("direct YUV444 path requires U plane");
            let plane_v = plane_v.expect("direct YUV444 path requires V plane");
            let (kr, kb) = fast_yuv_coefficients.expect("YUV matrix was checked above");
            for_each_rgba_row_chunk(
                &mut rgba,
                buffers.width,
                buffers.height,
                |first_row, chunk| {
                    for (local_index, pixel) in chunk.chunks_exact_mut(4).enumerate() {
                        let index = first_row * buffers.width + local_index;
                        let rgb = yuv_to_rgb_u16_fast(
                            plane_y.samples[index],
                            plane_u.samples[index],
                            plane_v.samples[index],
                            fast_range,
                            kr,
                            kb,
                        );
                        pixel[..3].copy_from_slice(&rgb);
                        pixel[3] = u16::MAX;
                    }
                },
            );
        } else {
            for_each_rgba_row_chunk(
                &mut rgba,
                buffers.width,
                buffers.height,
                |first_row, chunk| {
                    for (row_offset, row) in chunk.chunks_exact_mut(buffers.width * 4).enumerate() {
                        let y = first_row + row_offset;
                        for (x, pixel) in row.chunks_exact_mut(4).enumerate() {
                            let y_sample = sample_plane(plane_y, x, y);
                            let u_sample = plane_u
                                .map(|plane| {
                                    sample_chroma_plane(
                                        plane,
                                        x,
                                        y,
                                        color_config.chroma_sample_position,
                                    )
                                })
                                .unwrap_or(chroma_mid);
                            let v_sample = plane_v
                                .map(|plane| {
                                    sample_chroma_plane(
                                        plane,
                                        x,
                                        y,
                                        color_config.chroma_sample_position,
                                    )
                                })
                                .unwrap_or(chroma_mid);
                            let rgb = if let Some((kr, kb)) = fast_yuv_coefficients {
                                yuv_to_rgb_u16_fast(
                                    y_sample, u_sample, v_sample, fast_range, kr, kb,
                                )
                            } else if fast_ycgco {
                                ycgco_to_rgb_u16_fast(y_sample, u_sample, v_sample, fast_range)
                            } else {
                                yuv_to_rgb_u16(y_sample, u_sample, v_sample, range, matrix)
                            };
                            pixel[0] = rgb[0];
                            pixel[1] = rgb[1];
                            pixel[2] = rgb[2];
                            pixel[3] = alpha_sample(buffers.planes.get(3), x, y, alpha_max_source);
                        }
                    }
                },
            );
        }
    }

    if let Some(transfer) = hdr_transfer {
        apply_transfer_function_rows(&mut rgba, buffers.width, buffers.height, transfer);
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

fn convert_yuv420_to_rgba8(
    rgba: &mut [u8],
    width: usize,
    height: usize,
    plane_y: &super::decode::PlaneBuffer,
    plane_u: &super::decode::PlaneBuffer,
    plane_v: &super::decode::PlaneBuffer,
    range: FastSampleRange,
    kr: f32,
    kb: f32,
    position: Option<ChromaSamplePosition>,
) {
    convert_yuv420_to_rgba(
        rgba,
        width,
        height,
        plane_y,
        plane_u,
        plane_v,
        range,
        kr,
        kb,
        position,
        |pixel, rgb| {
            pixel[0] = u16_to_u8(rgb[0]);
            pixel[1] = u16_to_u8(rgb[1]);
            pixel[2] = u16_to_u8(rgb[2]);
            pixel[3] = u8::MAX;
        },
    );
}

fn convert_yuv420_to_rgba16(
    rgba: &mut [u16],
    width: usize,
    height: usize,
    plane_y: &super::decode::PlaneBuffer,
    plane_u: &super::decode::PlaneBuffer,
    plane_v: &super::decode::PlaneBuffer,
    range: FastSampleRange,
    kr: f32,
    kb: f32,
    position: Option<ChromaSamplePosition>,
) {
    convert_yuv420_to_rgba(
        rgba,
        width,
        height,
        plane_y,
        plane_u,
        plane_v,
        range,
        kr,
        kb,
        position,
        |pixel, rgb| {
            pixel[..3].copy_from_slice(&rgb);
            pixel[3] = u16::MAX;
        },
    );
}

fn convert_yuv420_to_rgba<T, F>(
    rgba: &mut [T],
    width: usize,
    height: usize,
    plane_y: &super::decode::PlaneBuffer,
    plane_u: &super::decode::PlaneBuffer,
    plane_v: &super::decode::PlaneBuffer,
    range: FastSampleRange,
    kr: f32,
    kb: f32,
    position: Option<ChromaSamplePosition>,
    write_pixel: F,
) where
    T: Send,
    F: Fn(&mut [T], [u16; 3]) + Sync,
{
    let chroma_width = plane_u.layout.width;
    let chroma_height = plane_u.layout.height;
    let interpolate_vertical = matches!(
        position.unwrap_or(ChromaSamplePosition::Unknown),
        ChromaSamplePosition::Vertical
            | ChromaSamplePosition::Unknown
            | ChromaSamplePosition::Reserved
    );
    for_each_rgba_row_chunk(rgba, width, height, |first_row, chunk| {
        for (row_offset, row) in chunk.chunks_exact_mut(width * 4).enumerate() {
            let y = first_row + row_offset;
            let luma_row = y * plane_y.layout.width;
            let chroma_y = (y >> 1).min(chroma_height - 1);
            let chroma_next_y = (chroma_y + 1).min(chroma_height - 1);
            let interpolate = interpolate_vertical && (y & 1) != 0;
            for (x, pixel) in row.chunks_exact_mut(4).enumerate() {
                let luma = plane_y.samples[luma_row + x];
                let chroma_x = (x >> 1).min(chroma_width - 1);
                let top = chroma_y * chroma_width + chroma_x;
                let bottom = chroma_next_y * chroma_width + chroma_x;
                let u = if interpolate {
                    ((u32::from(plane_u.samples[top]) + u32::from(plane_u.samples[bottom]) + 1) / 2)
                        as u16
                } else {
                    plane_u.samples[top]
                };
                let v = if interpolate {
                    ((u32::from(plane_v.samples[top]) + u32::from(plane_v.samples[bottom]) + 1) / 2)
                        as u16
                } else {
                    plane_v.samples[top]
                };
                write_pixel(pixel, yuv_to_rgb_u16_fast(luma, u, v, range, kr, kb));
            }
        }
    });
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

#[inline]
fn alpha_sample_u8(plane: Option<&super::decode::PlaneBuffer>, x: usize, y: usize) -> u8 {
    plane
        .map(|plane| sample_plane(plane, x, y) as u8)
        .unwrap_or(u8::MAX)
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
    if subsampling_x == 1 && subsampling_y == 1 {
        let top = plane.samples[source_y * plane.layout.width + source_x];
        if matches!(
            position.unwrap_or(ChromaSamplePosition::Unknown),
            ChromaSamplePosition::Vertical
                | ChromaSamplePosition::Unknown
                | ChromaSamplePosition::Reserved
        ) && (y & 1) != 0
        {
            let next_y = (source_y + 1).min(plane.layout.height - 1);
            let bottom = plane.samples[next_y * plane.layout.width + source_x];
            return ((u32::from(top) + u32::from(bottom) + 1) / 2) as u16;
        }
        return top;
    }
    // The current AV1 sample positions use only integer source coordinates:
    // 4:2:2/4:4:0 are co-located and 4:2:0 optionally averages the two
    // vertically adjacent samples. Keep this path to avoid the former
    // four-load blend.
    debug_assert!(subsampling_x != 0 || subsampling_y != 0);
    plane.samples[source_y * plane.layout.width + source_x]
}

fn yuv_to_rgb_u16_fast(
    y_sample: u16,
    u_sample: u16,
    v_sample: u16,
    range: FastSampleRange,
    kr: f32,
    kb: f32,
) -> [u16; 3] {
    let y = ((f32::from(y_sample) - range.y_offset) / range.y_scale).clamp(0.0, 1.0);
    let cb = (f32::from(u_sample) - range.chroma_offset) / range.chroma_scale;
    let cr = (f32::from(v_sample) - range.chroma_offset) / range.chroma_scale;
    let r = y + 2.0 * (1.0 - kr) * cr;
    let b = y + 2.0 * (1.0 - kb) * cb;
    let g = (y - kr * r - kb * b) / (1.0 - kr - kb);
    [
        normalized_to_u16_fast(r),
        normalized_to_u16_fast(g),
        normalized_to_u16_fast(b),
    ]
}

#[inline]
fn ycgco_to_rgb_u16_fast(
    y_sample: u16,
    cg_sample: u16,
    co_sample: u16,
    range: FastSampleRange,
) -> [u16; 3] {
    let y = ((f32::from(y_sample) - range.y_offset) / range.y_scale).clamp(0.0, 1.0);
    let cg = (f32::from(cg_sample) - range.chroma_offset) / range.chroma_scale;
    let co = (f32::from(co_sample) - range.chroma_offset) / range.chroma_scale;
    [
        normalized_to_u16_fast(y - cg + co),
        normalized_to_u16_fast(y + cg),
        normalized_to_u16_fast(y - cg - co),
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

fn apply_transfer_function_rows(
    rgba: &mut [u16],
    width: usize,
    height: usize,
    transfer: TransferFunction,
) {
    for_each_rgba_row_chunk(rgba, width, height, |_, chunk| {
        apply_transfer_function(chunk, transfer);
    });
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

fn matrix_coefficients_for_conversion(color_config: &ColorConfig) -> u8 {
    color_config
        .color_description
        .map(|description| description.matrix_coefficients)
        .unwrap_or_else(|| {
            // Full-range AVIFs without CICP are commonly JPEG-style YUV;
            // preserve the historical BT.709 fallback for studio-range AV1.
            if matches!(color_config.color_range, ColorRange::Full) {
                6
            } else {
                2
            }
        })
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
    let (red, green, blue, white) = primary_chromaticities(color_primaries)?;
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

fn primary_chromaticities(
    color_primaries: u8,
) -> Result<((f64, f64), (f64, f64), (f64, f64), (f64, f64)), DecoderError> {
    Ok(match color_primaries {
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
    })
}

/// Converts linear RGB samples between two AV1 CICP primary sets.
///
/// Gain-map composition uses this small matrix path when its alternate image
/// is authored in a different colour space. The supported primary table is
/// shared with chromaticity-derived YUV conversion above; unsupported or
/// singular primary sets fail closed rather than silently changing colours.
pub(crate) fn convert_linear_rgb_primaries(
    rgb: &mut [f64; 3],
    source_primaries: u8,
    destination_primaries: u8,
) -> Result<(), DecoderError> {
    if source_primaries == destination_primaries {
        return Ok(());
    }
    let source = rgb_to_xyz_matrix(source_primaries)?;
    let destination = invert_3x3(rgb_to_xyz_matrix(destination_primaries)?)?;
    let xyz = multiply_3x3_vector(source, *rgb);
    *rgb = multiply_3x3_vector(destination, xyz);
    if rgb.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(DecoderError::Unsupported(
            "AV1 RGB primary conversion produced a non-finite value".to_string(),
        ))
    }
}

fn rgb_to_xyz_matrix(color_primaries: u8) -> Result<[[f64; 3]; 3], DecoderError> {
    let (red, green, blue, white) = primary_chromaticities(color_primaries)?;
    let to_xyz = |(x, y): (f64, f64)| [x / y, 1.0, (1.0 - x - y) / y];
    let red = to_xyz(red);
    let green = to_xyz(green);
    let blue = to_xyz(blue);
    let xw = white.0 / white.1;
    let zw = (1.0 - white.0 - white.1) / white.1;
    let unscaled = [
        [red[0], green[0], blue[0]],
        [red[1], green[1], blue[1]],
        [red[2], green[2], blue[2]],
    ];
    let scale = multiply_3x3_vector(invert_3x3(unscaled)?, [xw, 1.0, zw]);
    Ok([
        [red[0] * scale[0], green[0] * scale[1], blue[0] * scale[2]],
        [red[1] * scale[0], green[1] * scale[1], blue[1] * scale[2]],
        [red[2] * scale[0], green[2] * scale[1], blue[2] * scale[2]],
    ])
}

fn multiply_3x3_vector(matrix: [[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    ]
}

fn invert_3x3(matrix: [[f64; 3]; 3]) -> Result<[[f64; 3]; 3], DecoderError> {
    let determinant = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
    if determinant.abs() < f64::EPSILON {
        return Err(DecoderError::Unsupported(
            "AV1 RGB primary matrix is singular".to_string(),
        ));
    }
    let inverse = [
        [
            matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1],
            matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2],
            matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1],
        ],
        [
            matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2],
            matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0],
            matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2],
        ],
        [
            matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0],
            matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1],
            matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0],
        ],
    ];
    Ok(inverse.map(|row| row.map(|value| value / determinant)))
}

#[derive(Debug, Clone, Copy)]
struct SampleRange {
    y_offset: f64,
    y_scale: f64,
    chroma_offset: f64,
    chroma_scale: f64,
}

#[derive(Debug, Clone, Copy)]
struct FastSampleRange {
    y_offset: f32,
    y_scale: f32,
    chroma_offset: f32,
    chroma_scale: f32,
}

impl SampleRange {
    #[inline]
    fn as_fast(self) -> FastSampleRange {
        FastSampleRange {
            y_offset: self.y_offset as f32,
            y_scale: self.y_scale as f32,
            chroma_offset: self.chroma_offset as f32,
            chroma_scale: self.chroma_scale as f32,
        }
    }
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
    } else if max_source == 255 {
        // The common 8-bit SDR path has an exact bit replication mapping;
        // avoid a 32-bit divide for every luma/chroma/alpha sample.
        (sample << 8) | sample
    } else {
        ((u32::from(sample) * u32::from(u16::MAX) + (max_source / 2)) / max_source) as u16
    }
}

fn build_rgba8_scale_table(max_source: u32) -> [u8; 4096] {
    let mut table = [0; 4096];
    for (sample, value) in table
        .iter_mut()
        .enumerate()
        .take(usize::try_from(max_source).expect("high-bit-depth scale table fits") + 1)
    {
        *value = u16_to_u8(scale_sample_to_u16(sample as u16, max_source));
    }
    table
}

#[inline]
fn scale_sample_to_rgba8(sample: u16, max_source: u32, table: Option<&[u8; 4096]>) -> u8 {
    table.map_or_else(
        || u16_to_u8(scale_sample_to_u16(sample, max_source)),
        |table| table[usize::from(sample)],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av1::decode::{PlaneBuffer, PlaneLayout};
    use crate::av1::{ColorDescription, ColorRange};

    #[test]
    fn add_residual_clips_to_bit_depth() {
        let out = add_residual_to_prediction(&[10, 250, 128], &[-20, 20, 0], 8).unwrap();

        assert_eq!(out, vec![0, 255, 128]);
    }

    #[test]
    fn add_residual_into_matches_allocating_wrapper() {
        let prediction = [10, 250, 128];
        let residual = [-20, 20, 0];
        let expected = add_residual_to_prediction(&prediction, &residual, 8).unwrap();
        let mut actual = [0; 3];

        add_residual_to_prediction_into(&prediction, &residual, 8, &mut actual).unwrap();

        assert_eq!(actual, expected.as_slice());
    }

    #[test]
    fn parallel_rgba_row_chunks_match_sequential_layout() {
        let width = 640;
        let height = 512;
        let mut actual = vec![0u16; width * height * 4];
        for_each_rgba_row_chunk(&mut actual, width, height, |first_row, chunk| {
            for (local_index, pixel) in chunk.chunks_exact_mut(4).enumerate() {
                let index = first_row * width + local_index;
                pixel[0] = (index & 0xffff) as u16;
                pixel[1] = (index >> 4) as u16;
                pixel[2] = (index >> 8) as u16;
                pixel[3] = u16::MAX;
            }
        });

        let mut expected = vec![0u16; width * height * 4];
        for (index, pixel) in expected.chunks_exact_mut(4).enumerate() {
            pixel[0] = (index & 0xffff) as u16;
            pixel[1] = (index >> 4) as u16;
            pixel[2] = (index >> 8) as u16;
            pixel[3] = u16::MAX;
        }
        assert_eq!(actual, expected);
    }

    #[test]
    fn parallel_transfer_rows_match_one_pass() {
        let width = 640;
        let height = 512;
        let source = (0..width * height)
            .flat_map(|index| {
                let sample = (index as u16).wrapping_mul(257);
                [sample, sample / 2, u16::MAX - sample, 1234]
            })
            .collect::<Vec<_>>();
        let mut expected = source.clone();
        apply_transfer_function(&mut expected, TransferFunction::Pq);
        let mut actual = source;
        apply_transfer_function_rows(&mut actual, width, height, TransferFunction::Pq);
        assert_eq!(actual, expected);
    }

    #[test]
    fn eight_bit_sample_scaling_uses_exact_u16_replication() {
        for sample in [0, 1, 17, 128, 254, 255] {
            assert_eq!(
                scale_sample_to_u16(sample, 255),
                (sample << 8) | sample,
                "8-bit sample {sample} should map by bit replication"
            );
        }
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
    fn identity_rgba_preserves_alpha_plane_in_fast_path() {
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
                PlaneBuffer {
                    layout: PlaneLayout { plane: 3, ..layout },
                    samples: vec![7, 230],
                },
            ],
        };

        let image = frame_buffers_to_identity_rgba_8(&buffers).unwrap();

        assert_eq!(image.rgba, vec![10, 30, 50, 7, 20, 40, 60, 230]);
    }

    #[test]
    fn matrix_three_gbr_uses_identity_plane_order_for_16_bit_output() {
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
        let color_config = ColorConfig {
            high_bitdepth: false,
            twelve_bit: false,
            bit_depth: 8,
            monochrome: false,
            color_description: Some(ColorDescription {
                color_primaries: 1,
                transfer_characteristics: 13,
                matrix_coefficients: 3,
            }),
            color_range: ColorRange::Full,
            subsampling_x: false,
            subsampling_y: false,
            chroma_sample_position: None,
            separate_uv_delta_q: false,
        };

        let image = frame_buffers_to_rgba_16(&buffers, &color_config).unwrap();

        assert_eq!(
            image.rgba,
            vec![2570, 7710, 12850, u16::MAX, 5140, 10280, 15420, u16::MAX]
        );
    }

    #[test]
    fn rgba8_high_bit_identity_gbr_preserves_channel_order_and_alpha() {
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
                    samples: vec![128, 512],
                },
                PlaneBuffer {
                    layout: PlaneLayout { plane: 1, ..layout },
                    samples: vec![256, 768],
                },
                PlaneBuffer {
                    layout: PlaneLayout { plane: 2, ..layout },
                    samples: vec![900, 64],
                },
                PlaneBuffer {
                    layout: PlaneLayout { plane: 3, ..layout },
                    samples: vec![1023, 128],
                },
            ],
        };
        let color_config = ColorConfig {
            high_bitdepth: true,
            twelve_bit: false,
            bit_depth: 10,
            monochrome: false,
            color_description: Some(ColorDescription {
                color_primaries: 1,
                transfer_characteristics: 13,
                matrix_coefficients: 3,
            }),
            color_range: ColorRange::Full,
            subsampling_x: false,
            subsampling_y: false,
            chroma_sample_position: None,
            separate_uv_delta_q: false,
        };

        let image = frame_buffers_to_rgba_8(&buffers, &color_config).unwrap();

        let scale = |sample| u16_to_u8(scale_sample_to_u16(sample, 1023));
        assert_eq!(
            image.rgba,
            vec![
                scale(900),
                scale(128),
                scale(256),
                255,
                scale(64),
                scale(512),
                scale(768),
                scale(128),
            ]
        );
    }

    #[test]
    fn high_bit_rgba8_scale_table_matches_scalar_mapping() {
        for max_source in [1023, 4095] {
            let table = build_rgba8_scale_table(max_source);
            for sample in [0, 1, max_source / 2, max_source] {
                let sample = sample as u16;
                assert_eq!(
                    scale_sample_to_rgba8(sample, max_source, Some(&table)),
                    u16_to_u8(scale_sample_to_u16(sample, max_source))
                );
            }
        }
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
        assert_eq!(
            sample_chroma_plane(&plane, 1, 1, Some(ChromaSamplePosition::Reserved)),
            20
        );

        for y in 0..4 {
            for x in 0..4 {
                let source_x = (x / 2).min(1);
                let source_y = (y / 2).min(1);
                let vertical_expected = if (y & 1) == 0 {
                    plane.samples[source_y * 2 + source_x]
                } else {
                    let next_y = (source_y + 1).min(1);
                    ((u32::from(plane.samples[source_y * 2 + source_x])
                        + u32::from(plane.samples[next_y * 2 + source_x])
                        + 1)
                        / 2) as u16
                };
                assert_eq!(
                    sample_chroma_plane(&plane, x, y, Some(ChromaSamplePosition::Vertical)),
                    vertical_expected
                );
                assert_eq!(
                    sample_chroma_plane(&plane, x, y, Some(ChromaSamplePosition::Colocated)),
                    plane.samples[(y / 2).min(1) * 2 + source_x]
                );
            }
        }

        let yuv422 = PlaneBuffer {
            layout: PlaneLayout {
                plane: 1,
                width: 2,
                height: 1,
                subsampling_x: 1,
                subsampling_y: 0,
                sample_count: 2,
            },
            samples: vec![7, 19],
        };
        assert_eq!(sample_chroma_plane(&yuv422, 0, 0, None), 7);
        assert_eq!(sample_chroma_plane(&yuv422, 1, 0, None), 7);
        assert_eq!(sample_chroma_plane(&yuv422, 2, 0, None), 19);
        assert_eq!(sample_chroma_plane(&yuv422, 3, 0, None), 19);
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
    fn rgba8_sdr_subsampled_path_matches_rgba16_conversion() {
        let luma_layout = PlaneLayout {
            plane: 0,
            width: 3,
            height: 3,
            subsampling_x: 0,
            subsampling_y: 0,
            sample_count: 9,
        };
        let chroma_layout = PlaneLayout {
            plane: 1,
            width: 2,
            height: 2,
            subsampling_x: 1,
            subsampling_y: 1,
            sample_count: 4,
        };
        let buffers = FrameBuffers {
            width: 3,
            height: 3,
            planes: vec![
                PlaneBuffer {
                    layout: luma_layout,
                    samples: vec![16, 64, 128, 32, 96, 160, 48, 112, 208],
                },
                PlaneBuffer {
                    layout: chroma_layout,
                    samples: vec![90, 128, 170, 210],
                },
                PlaneBuffer {
                    layout: PlaneLayout {
                        plane: 2,
                        ..chroma_layout
                    },
                    samples: vec![200, 160, 120, 80],
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
            color_range: super::super::sequence::ColorRange::Studio,
            subsampling_x: true,
            subsampling_y: true,
            chroma_sample_position: Some(ChromaSamplePosition::Vertical),
            separate_uv_delta_q: false,
        };

        let expected = frame_buffers_to_rgba_16(&buffers, &color_config)
            .unwrap()
            .rgba
            .into_iter()
            .map(u16_to_u8)
            .collect::<Vec<_>>();
        let actual = frame_buffers_to_rgba_8(&buffers, &color_config).unwrap();

        assert_eq!(actual.rgba, expected);
    }

    #[test]
    fn rgba8_yuv444_alpha_fast_path_matches_rgba16_conversion() {
        let layout = PlaneLayout {
            plane: 0,
            width: 2,
            height: 2,
            subsampling_x: 0,
            subsampling_y: 0,
            sample_count: 4,
        };
        let buffers = FrameBuffers {
            width: 2,
            height: 2,
            planes: vec![
                PlaneBuffer {
                    layout,
                    samples: vec![16, 64, 128, 220],
                },
                PlaneBuffer {
                    layout: PlaneLayout { plane: 1, ..layout },
                    samples: vec![90, 128, 170, 210],
                },
                PlaneBuffer {
                    layout: PlaneLayout { plane: 2, ..layout },
                    samples: vec![200, 160, 120, 80],
                },
                PlaneBuffer {
                    layout: PlaneLayout { plane: 3, ..layout },
                    samples: vec![0, 64, 160, 255],
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
            color_range: super::super::sequence::ColorRange::Studio,
            subsampling_x: false,
            subsampling_y: false,
            chroma_sample_position: None,
            separate_uv_delta_q: false,
        };

        let expected = frame_buffers_to_rgba_16(&buffers, &color_config)
            .unwrap()
            .rgba
            .into_iter()
            .map(u16_to_u8)
            .collect::<Vec<_>>();
        let actual = frame_buffers_to_rgba_8(&buffers, &color_config).unwrap();

        assert_eq!(actual.rgba, expected);
    }

    #[test]
    fn rgba8_monochrome_sdr_path_matches_rgba16_conversion() {
        let luma_layout = PlaneLayout {
            plane: 0,
            width: 2,
            height: 1,
            subsampling_x: 1,
            subsampling_y: 1,
            sample_count: 2,
        };
        let alpha_layout = PlaneLayout {
            plane: 3,
            width: 3,
            height: 2,
            subsampling_x: 0,
            subsampling_y: 0,
            sample_count: 6,
        };
        let buffers = FrameBuffers {
            width: 3,
            height: 2,
            planes: vec![
                PlaneBuffer {
                    layout: luma_layout,
                    samples: vec![32, 224],
                },
                PlaneBuffer {
                    layout: alpha_layout,
                    samples: vec![0, 64, 128, 160, 192, 255],
                },
            ],
        };
        let color_config = ColorConfig {
            high_bitdepth: false,
            twelve_bit: false,
            bit_depth: 8,
            monochrome: true,
            color_description: Some(super::super::sequence::ColorDescription {
                color_primaries: 1,
                transfer_characteristics: 13,
                matrix_coefficients: 0,
            }),
            color_range: super::super::sequence::ColorRange::Full,
            subsampling_x: true,
            subsampling_y: true,
            chroma_sample_position: None,
            separate_uv_delta_q: false,
        };

        let expected = frame_buffers_to_rgba_16(&buffers, &color_config)
            .unwrap()
            .rgba
            .into_iter()
            .map(u16_to_u8)
            .collect::<Vec<_>>();
        let actual = frame_buffers_to_rgba_8(&buffers, &color_config).unwrap();

        assert_eq!(actual.rgba, expected);
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
        let fast = yuv_to_rgb_u16_fast(128, 96, 160, range.as_fast(), 0.2627, 0.0593);
        let scalar = yuv_to_rgb_u16(128, 96, 160, range, matrix);
        assert!(
            fast.iter()
                .zip(scalar)
                .all(|(fast, scalar)| fast.abs_diff(scalar) <= 1)
        );
    }

    #[test]
    fn fast_ycgco_conversion_stays_within_one_code_value_of_scalar_path() {
        let range = SampleRange::new(8, super::super::sequence::ColorRange::Full).unwrap();
        let fast = ycgco_to_rgb_u16_fast(128, 128, 192, range.as_fast());
        let scalar = yuv_to_rgb_u16(128, 128, 192, range, MatrixCoefficients::YcGco);
        assert!(
            fast.iter()
                .zip(scalar)
                .all(|(fast, scalar)| fast.abs_diff(scalar) <= 1),
            "fast={fast:?} scalar={scalar:?}"
        );
    }

    #[test]
    fn fast_subsampled_yuv_conversion_stays_within_one_code_value_of_scalar_path() {
        let y_layout = PlaneLayout {
            plane: 0,
            width: 4,
            height: 4,
            subsampling_x: 0,
            subsampling_y: 0,
            sample_count: 16,
        };
        let uv_layout = PlaneLayout {
            plane: 1,
            width: 2,
            height: 2,
            subsampling_x: 1,
            subsampling_y: 1,
            sample_count: 4,
        };
        let buffers = FrameBuffers {
            width: 4,
            height: 4,
            planes: vec![
                PlaneBuffer {
                    layout: y_layout,
                    samples: (0..16).map(|index| 32 + index as u16 * 11).collect(),
                },
                PlaneBuffer {
                    layout: uv_layout,
                    samples: vec![96, 128, 160, 192],
                },
                PlaneBuffer {
                    layout: PlaneLayout {
                        plane: 2,
                        ..uv_layout
                    },
                    samples: vec![192, 160, 128, 96],
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
            color_range: ColorRange::Studio,
            subsampling_x: true,
            subsampling_y: true,
            chroma_sample_position: Some(ChromaSamplePosition::Unknown),
            separate_uv_delta_q: false,
        };
        let actual = frame_buffers_to_rgba_8(&buffers, &color_config).unwrap();
        let range = SampleRange::new(8, ColorRange::Studio).unwrap();
        let matrix = MatrixCoefficients::Yuv {
            kr: 0.2126,
            kb: 0.0722,
        };
        for y in 0..buffers.height {
            for x in 0..buffers.width {
                let index = (y * buffers.width + x) * 4;
                let expected = yuv_to_rgb_u16(
                    sample_plane(&buffers.planes[0], x, y),
                    sample_chroma_plane(
                        &buffers.planes[1],
                        x,
                        y,
                        color_config.chroma_sample_position,
                    ),
                    sample_chroma_plane(
                        &buffers.planes[2],
                        x,
                        y,
                        color_config.chroma_sample_position,
                    ),
                    range,
                    matrix,
                );
                for channel in 0..3 {
                    assert!(
                        actual.rgba[index + channel].abs_diff(u16_to_u8(expected[channel])) <= 1,
                        "pixel ({x},{y}) channel {channel} differs from scalar path"
                    );
                }
                assert_eq!(actual.rgba[index + 3], u8::MAX);
            }
        }
    }

    #[test]
    fn fast_high_bit_depth_yuv_conversion_stays_within_two_code_values() {
        let range = SampleRange::new(10, ColorRange::Studio).unwrap();
        let matrix = MatrixCoefficients::Yuv {
            kr: 0.2126,
            kb: 0.0722,
        };
        for (y, u, v) in [(64, 512, 960), (512, 448, 576), (900, 128, 384)] {
            let fast = yuv_to_rgb_u16_fast(y, u, v, range.as_fast(), 0.2126, 0.0722);
            let scalar = yuv_to_rgb_u16(y, u, v, range, matrix);
            assert!(
                fast.iter()
                    .zip(scalar)
                    .all(|(fast, scalar)| fast.abs_diff(scalar) <= 2),
                "fast={fast:?} scalar={scalar:?}"
            );
        }
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
    fn linear_rgb_primary_conversion_round_trips_bt709_and_bt2020() {
        let original = [0.73, 0.21, 0.91];
        let mut converted = original;
        convert_linear_rgb_primaries(&mut converted, 1, 9).unwrap();
        assert_ne!(converted, original);
        convert_linear_rgb_primaries(&mut converted, 9, 1).unwrap();
        for (actual, expected) in converted.into_iter().zip(original) {
            assert!((actual - expected).abs() < 1e-10, "{actual} != {expected}");
        }
    }

    #[test]
    fn unspecified_matrix_coefficients_use_range_fallback() {
        let studio = ColorConfig {
            high_bitdepth: false,
            twelve_bit: false,
            bit_depth: 8,
            monochrome: false,
            color_description: None,
            color_range: ColorRange::Studio,
            subsampling_x: true,
            subsampling_y: true,
            chroma_sample_position: None,
            separate_uv_delta_q: false,
        };
        let full = ColorConfig {
            color_range: ColorRange::Full,
            ..studio
        };
        assert_eq!(matrix_coefficients_for_conversion(&studio), 2);
        assert_eq!(matrix_coefficients_for_conversion(&full), 6);
        assert!(matches!(
            MatrixCoefficients::from_av1(matrix_coefficients_for_conversion(&studio), 2),
            Ok(MatrixCoefficients::Yuv { kr, kb })
                if (kr - 0.2126).abs() < f64::EPSILON && (kb - 0.0722).abs() < f64::EPSILON
        ));
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
