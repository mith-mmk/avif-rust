use crate::container::{CleanAperture, ImageMirror, ImageRotation};
use crate::{DecoderError, ImageBuffer};

pub(super) fn apply_image_transforms(
    image: &mut ImageBuffer,
    aperture: Option<CleanAperture>,
    mirror: Option<ImageMirror>,
    rotation: Option<ImageRotation>,
) -> Result<(), DecoderError> {
    apply_clean_aperture(image, aperture)?;
    apply_mirror(image, mirror)?;
    apply_rotation(image, rotation)
}

pub(super) fn clean_aperture_rect(
    image_width: usize,
    image_height: usize,
    aperture: CleanAperture,
) -> Result<(usize, usize, usize, usize), DecoderError> {
    if aperture.width_d == 0
        || aperture.height_d == 0
        || aperture.horizontal_offset_d == 0
        || aperture.vertical_offset_d == 0
    {
        return Err(DecoderError::Bitstream(
            "AVIF clean aperture has a zero denominator".to_string(),
        ));
    }
    let width = (u64::from(aperture.width_n) / u64::from(aperture.width_d)) as usize;
    let height = (u64::from(aperture.height_n) / u64::from(aperture.height_d)) as usize;
    let center_x = image_width as i64 / 2;
    let center_y = image_height as i64 / 2;
    let offset_x_n = i32::from_be_bytes(aperture.horizontal_offset_n.to_be_bytes());
    let offset_y_n = i32::from_be_bytes(aperture.vertical_offset_n.to_be_bytes());
    let offset_x = i64::from(offset_x_n) / i64::from(aperture.horizontal_offset_d);
    let offset_y = i64::from(offset_y_n) / i64::from(aperture.vertical_offset_d);
    let start_x = center_x + offset_x - width as i64 / 2;
    let start_y = center_y + offset_y - height as i64 / 2;
    if width == 0
        || height == 0
        || start_x < 0
        || start_y < 0
        || start_x as usize + width > image_width
        || start_y as usize + height > image_height
    {
        return Err(DecoderError::Bitstream(
            "AVIF clean aperture is outside the decoded image".to_string(),
        ));
    }
    Ok((start_x as usize, start_y as usize, width, height))
}

fn apply_clean_aperture(
    image: &mut ImageBuffer,
    aperture: Option<CleanAperture>,
) -> Result<(), DecoderError> {
    let Some(aperture) = aperture else {
        return Ok(());
    };
    let (start_x, start_y, width, height) =
        clean_aperture_rect(image.width, image.height, aperture)?;
    let mut cropped = vec![0; width * height * 4];
    for row in 0..height {
        let src = ((start_y + row) * image.width + start_x) * 4;
        let dst = row * width * 4;
        cropped[dst..dst + width * 4].copy_from_slice(&image.rgba[src..src + width * 4]);
    }
    image.width = width;
    image.height = height;
    image.rgba = cropped;
    Ok(())
}

fn apply_mirror(image: &mut ImageBuffer, mirror: Option<ImageMirror>) -> Result<(), DecoderError> {
    let Some(mirror) = mirror else {
        return Ok(());
    };
    if mirror.axis > 1 {
        return Err(DecoderError::Bitstream(format!(
            "AVIF mirror axis {} is invalid",
            mirror.axis
        )));
    }
    let horizontal = mirror.axis == 0;
    let mut transformed = vec![0; image.rgba.len()];
    for y in 0..image.height {
        for x in 0..image.width {
            let (source_x, source_y) = if horizontal {
                (image.width - 1 - x, y)
            } else {
                (x, image.height - 1 - y)
            };
            let source = (source_y * image.width + source_x) * 4;
            let destination = (y * image.width + x) * 4;
            transformed[destination..destination + 4]
                .copy_from_slice(&image.rgba[source..source + 4]);
        }
    }
    image.rgba = transformed;
    Ok(())
}

fn apply_rotation(
    image: &mut ImageBuffer,
    rotation: Option<ImageRotation>,
) -> Result<(), DecoderError> {
    let Some(rotation) = rotation else {
        return Ok(());
    };
    if rotation.angle > 3 {
        return Err(DecoderError::Bitstream(format!(
            "AVIF rotation angle {} is invalid",
            rotation.angle
        )));
    }
    for _ in 0..rotation.angle {
        let mut transformed = vec![0; image.rgba.len()];
        let new_width = image.height;
        let new_height = image.width;
        for y in 0..image.height {
            for x in 0..image.width {
                // The AVIF `irot` angle is a counter-clockwise quarter-turn.
                // The output width/height are swapped for each turn.
                let destination_x = y;
                let destination_y = image.width - 1 - x;
                let source = (y * image.width + x) * 4;
                let destination = (destination_y * new_width + destination_x) * 4;
                transformed[destination..destination + 4]
                    .copy_from_slice(&image.rgba[source..source + 4]);
            }
        }
        image.width = new_width;
        image.height = new_height;
        image.rgba = transformed;
    }
    Ok(())
}
