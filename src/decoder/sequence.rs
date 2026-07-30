use super::{append_sequence_alpha_frames, decode_sequence_frames_from_info};
use crate::container::{AvifInfo, AvifSequence};
use crate::{DecoderError, ImageBuffer};

pub(super) fn decode_animation_images(
    info: &AvifInfo,
    sequence: &AvifSequence,
) -> Result<Vec<ImageBuffer>, DecoderError> {
    let mut frames = decode_sequence_frames_from_info(info)?;
    append_sequence_alpha_frames(info, sequence, &mut frames)?;
    let mut images = Vec::with_capacity(frames.len());
    for frame in frames {
        let mut image = frame.to_rgba8()?;
        super::composition::apply_image_transforms(
            &mut image,
            info.clean_aperture,
            info.mirror,
            info.rotation,
        )?;
        images.push(image);
    }
    let first = images.first().ok_or_else(|| {
        DecoderError::Bitstream("AVIS sequence has no decodable samples".to_string())
    })?;
    if images
        .iter()
        .any(|image| image.width != first.width || image.height != first.height)
    {
        return Err(DecoderError::Unsupported(
            "AVIS sequence samples have different output dimensions".to_string(),
        ));
    }
    Ok(images)
}
