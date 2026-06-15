use super::decode::{FrameBuffers, PlaneBuffer};
use super::sequence::ColorConfig;
use crate::{DecoderError, ImageBuffer};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedIntraEdges {
    pub above: Vec<u16>,
    pub left: Vec<u16>,
    pub above_left: u16,
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
    let mid = 1u16 << (bit_depth - 1);
    let mut above = Vec::with_capacity(width);
    let mut left = Vec::with_capacity(height);

    for dx in 0..width {
        if y == 0 || plane.layout.width == 0 {
            above.push(mid);
        } else {
            let sample_x = (x + dx).min(plane.layout.width - 1);
            above.push(plane.samples[(y - 1) * plane.layout.width + sample_x]);
        }
    }

    for dy in 0..height {
        if x == 0 || plane.layout.height == 0 {
            left.push(mid);
        } else {
            let sample_y = (y + dy).min(plane.layout.height - 1);
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
    if color_config.bit_depth != 8 {
        return Err(DecoderError::Unsupported(
            "AV1 RGBA conversion currently supports 8-bit output only".to_string(),
        ));
    }
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
    let matrix_coefficients = color_config
        .color_description
        .map(|description| description.matrix_coefficients)
        .unwrap_or(2);
    if matrix_coefficients != 0 {
        return Err(DecoderError::Unsupported(format!(
            "AV1 matrix coefficients {matrix_coefficients} RGBA conversion is not supported yet"
        )));
    }

    let mut rgba = vec![0u8; buffers.width * buffers.height * 4];
    let plane_g = &buffers.planes[0].samples;
    let plane_b = &buffers.planes[1].samples;
    let plane_r = &buffers.planes[2].samples;
    for index in 0..buffers.width * buffers.height {
        let out = index * 4;
        rgba[out] = plane_r[index].min(255) as u8;
        rgba[out + 1] = plane_g[index].min(255) as u8;
        rgba[out + 2] = plane_b[index].min(255) as u8;
        rgba[out + 3] = 255;
    }
    Ok(ImageBuffer {
        width: buffers.width,
        height: buffers.height,
        rgba,
    })
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
    fn read_intra_edges_uses_reconstructed_neighbors_and_midpoint_at_frame_edge() {
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

        assert_eq!(inner.above, vec![5, 6]);
        assert_eq!(inner.left, vec![8, 12]);
        assert_eq!(inner.above_left, 4);

        let frame_edge = read_intra_edges(&plane, 0, 0, 2, 2, 8);

        assert_eq!(frame_edge.above, vec![128, 128]);
        assert_eq!(frame_edge.left, vec![128, 128]);
        assert_eq!(frame_edge.above_left, 128);
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
    fn rgba_conversion_rejects_non_identity_matrix_for_now() {
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
                    samples: vec![0],
                },
                PlaneBuffer {
                    layout: PlaneLayout { plane: 2, ..layout },
                    samples: vec![0],
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

        let err = frame_buffers_to_rgba_8(&buffers, &color_config).unwrap_err();

        assert!(matches!(err, DecoderError::Unsupported(_)));
    }
}
