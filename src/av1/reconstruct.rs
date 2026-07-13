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
    let mid = 1u16 << (bit_depth - 1);
    let above_available = y > 0 && plane.layout.width > 0;
    let left_available = x > 0 && plane.layout.height > 0;
    let directional_edge_len = width + height;
    let mut above = Vec::with_capacity(directional_edge_len);
    let mut left = Vec::with_capacity(directional_edge_len);

    for dx in 0..directional_edge_len {
        if !above_available {
            above.push(mid - 1);
        } else {
            let extension_end = width.saturating_add(top_right_available);
            let edge_dx = if dx >= extension_end {
                extension_end.saturating_sub(1)
            } else {
                dx
            };
            let sample_x = (x + edge_dx).min(plane.layout.width - 1);
            above.push(plane.samples[(y - 1) * plane.layout.width + sample_x]);
        }
    }

    for dy in 0..directional_edge_len {
        if !left_available {
            left.push(mid + 1);
        } else {
            let extension_end = height.saturating_add(bottom_left_available);
            let edge_dy = if dy >= extension_end {
                extension_end.saturating_sub(1)
            } else {
                dy
            };
            let sample_y = (y + edge_dy).min(plane.layout.height - 1);
            left.push(plane.samples[sample_y * plane.layout.width + x - 1]);
        }
    }

    let above_left = if x == 0 || y == 0 || plane.layout.width == 0 {
        mid
    } else {
        plane.samples[(y - 1) * plane.layout.width + x - 1]
    };

    OwnedIntraEdges {
        above,
        left,
        above_left,
        above_available,
        left_available,
    }
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
    validate_sdr_transfer(color_config)?;
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
            let out = index * 4;
            rgba[out] = scale_sample_to_u16(plane_r[index], max_source);
            rgba[out + 1] = scale_sample_to_u16(plane_g[index], max_source);
            rgba[out + 2] = scale_sample_to_u16(plane_b[index], max_source);
            rgba[out + 3] = u16::MAX;
        }
    } else {
        let matrix = MatrixCoefficients::from_av1(matrix_coefficients)?;
        let range = SampleRange::new(color_config.bit_depth, color_config.color_range)?;
        let plane_y = &buffers.planes[0].samples;
        let plane_u = &buffers.planes[1].samples;
        let plane_v = &buffers.planes[2].samples;
        for index in 0..buffers.width * buffers.height {
            let rgb = yuv_to_rgb_u16(
                plane_y[index],
                plane_u[index],
                plane_v[index],
                range,
                matrix,
            );
            let out = index * 4;
            rgba[out] = rgb[0];
            rgba[out + 1] = rgb[1];
            rgba[out + 2] = rgb[2];
            rgba[out + 3] = u16::MAX;
        }
    }

    Ok(Rgba16ImageBuffer {
        width: buffers.width,
        height: buffers.height,
        rgba,
    })
}

fn validate_rgba_conversion(buffers: &FrameBuffers) -> Result<(), DecoderError> {
    if buffers.planes.len() < 3 {
        return Err(DecoderError::Unsupported(
            "AV1 monochrome RGBA conversion is not supported yet".to_string(),
        ));
    }
    if buffers
        .planes
        .iter()
        .take(3)
        .any(|plane| plane.layout.width != buffers.width || plane.layout.height != buffers.height)
    {
        return Err(DecoderError::Unsupported(
            "AV1 subsampled RGBA conversion is not supported yet".to_string(),
        ));
    }
    Ok(())
}

fn validate_sdr_transfer(color_config: &ColorConfig) -> Result<(), DecoderError> {
    let Some(description) = color_config.color_description else {
        return Ok(());
    };
    match description.transfer_characteristics {
        1 | 6 | 13 => Ok(()),
        16 | 18 => Err(DecoderError::Unsupported(format!(
            "AV1 transfer characteristics {} require unimplemented HDR colour management",
            description.transfer_characteristics
        ))),
        transfer => Err(DecoderError::Unsupported(format!(
            "AV1 transfer characteristics {transfer} RGBA conversion is not supported yet"
        ))),
    }
}

#[derive(Debug, Clone, Copy)]
struct MatrixCoefficients {
    kr: f64,
    kb: f64,
}

impl MatrixCoefficients {
    fn from_av1(matrix_coefficients: u8) -> Result<Self, DecoderError> {
        match matrix_coefficients {
            1 => Ok(Self {
                kr: 0.2126,
                kb: 0.0722,
            }),
            5 | 6 => Ok(Self {
                kr: 0.299,
                kb: 0.114,
            }),
            9 => Ok(Self {
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
    let r = y + 2.0 * (1.0 - matrix.kr) * cr;
    let b = y + 2.0 * (1.0 - matrix.kb) * cb;
    let g = (y - matrix.kr * r - matrix.kb * b) / (1.0 - matrix.kr - matrix.kb);
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
    fn rgba_conversion_rejects_hdr_transfer_characteristics() {
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
                    samples: vec![64],
                },
                PlaneBuffer {
                    layout: PlaneLayout { plane: 1, ..layout },
                    samples: vec![512],
                },
                PlaneBuffer {
                    layout: PlaneLayout { plane: 2, ..layout },
                    samples: vec![512],
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
            color_range: super::super::sequence::ColorRange::Studio,
            subsampling_x: false,
            subsampling_y: false,
            chroma_sample_position: None,
            separate_uv_delta_q: false,
        };

        let err = frame_buffers_to_rgba_16(&buffers, &color_config).unwrap_err();

        assert!(matches!(err, DecoderError::Unsupported(message) if message.contains("HDR")));
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
