use crate::ImageBuffer;
use crate::compat::{CallbackResponse, DrawCallback, InitOptions, NextOptions, ResponseCommand};
use crate::container::{CleanAperture, ImageMirror, ImageRotation};
use crate::decoder::frame::AvifSequenceDecoder;

type Error = Box<dyn std::error::Error>;

pub(super) fn emit_single(drawer: &mut dyn DrawCallback, image: &ImageBuffer) -> Result<(), Error> {
    if is_abort(drawer.init(
        image.width,
        image.height,
        Some(InitOptions {
            loop_count: 1,
            animation: false,
        }),
    )?) {
        return Ok(());
    }
    if is_abort(drawer.draw(0, 0, image.width, image.height, &image.rgba, None)?) {
        return Ok(());
    }
    drawer.terminate(None)?;
    Ok(())
}

pub(super) fn emit_animation(
    drawer: &mut dyn DrawCallback,
    decoder: &mut AvifSequenceDecoder,
    clean_aperture: Option<CleanAperture>,
    mirror: Option<ImageMirror>,
    rotation: Option<ImageRotation>,
) -> Result<(), Error> {
    let first = decoder
        .next_frame()?
        .ok_or_else(|| "AVIS sequence has no decodable samples".to_string())?;
    let mut first_image = first.frame.to_rgba8()?;
    super::composition::apply_image_transforms(&mut first_image, clean_aperture, mirror, rotation)?;
    let loop_count = match decoder.animation().repetition_count {
        crate::container::AvifRepetitionCount::Finite(value) => value,
        crate::container::AvifRepetitionCount::Infinite => 0,
        crate::container::AvifRepetitionCount::Unknown => 1,
    };
    if is_abort(drawer.init(
        first_image.width,
        first_image.height,
        Some(InitOptions {
            loop_count,
            animation: true,
        }),
    )?) {
        return Ok(());
    }
    if emit_frame(drawer, &first_image, first.timing.duration_ms)? {
        return Ok(());
    }
    while let Some(decoded) = decoder.next_frame()? {
        let mut image = decoded.frame.to_rgba8()?;
        super::composition::apply_image_transforms(&mut image, clean_aperture, mirror, rotation)?;
        if image.width != first_image.width || image.height != first_image.height {
            return Err("AVIS sequence samples have different output dimensions".into());
        }
        if emit_frame(drawer, &image, decoded.timing.duration_ms)? {
            return Ok(());
        }
    }
    drawer.terminate(None)?;
    Ok(())
}

fn emit_frame(
    drawer: &mut dyn DrawCallback,
    image: &crate::ImageBuffer,
    duration_ms: u64,
) -> Result<bool, Error> {
    if is_abort(drawer.next(Some(NextOptions::full_canvas(
        image.width,
        image.height,
        duration_ms,
    )))?) {
        return Ok(true);
    }
    Ok(is_abort(drawer.draw(
        0,
        0,
        image.width,
        image.height,
        &image.rgba,
        None,
    )?))
}

fn is_abort(response: Option<CallbackResponse>) -> bool {
    response.is_some_and(|response| response.response == ResponseCommand::Abort)
}
