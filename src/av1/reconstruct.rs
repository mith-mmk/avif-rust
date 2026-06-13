use super::decode::{FrameBuffers, PlaneBuffer};
use crate::{DecoderError, ImageBuffer};

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

pub fn frame_buffers_to_identity_rgba_8(
    buffers: &FrameBuffers,
) -> Result<ImageBuffer, DecoderError> {
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

    let mut rgba = vec![0u8; buffers.width * buffers.height * 4];
    let plane_r = &buffers.planes[0].samples;
    let plane_g = &buffers.planes[1].samples;
    let plane_b = &buffers.planes[2].samples;
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
    fn identity_rgba_uses_first_three_planes() {
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
                    samples: vec![10, 20],
                },
                PlaneBuffer {
                    layout: PlaneLayout { plane: 1, ..layout },
                    samples: vec![30, 40],
                },
                PlaneBuffer {
                    layout: PlaneLayout { plane: 2, ..layout },
                    samples: vec![50, 60],
                },
            ],
        };

        let image = frame_buffers_to_identity_rgba_8(&buffers).unwrap();

        assert_eq!(image.rgba, vec![10, 30, 50, 255, 20, 40, 60, 255]);
    }
}
