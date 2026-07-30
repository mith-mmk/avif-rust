use crate::ImageBuffer;
use crate::compat::{DrawCallback, InitOptions, NextOptions};

type Error = Box<dyn std::error::Error>;

pub(super) fn emit_single(drawer: &mut dyn DrawCallback, image: &ImageBuffer) -> Result<(), Error> {
    drawer.init(
        image.width,
        image.height,
        Some(InitOptions {
            loop_count: 1,
            animation: false,
        }),
    )?;
    drawer.draw(0, 0, image.width, image.height, &image.rgba, None)?;
    drawer.terminate(None)?;
    Ok(())
}

pub(super) fn emit_animation(
    drawer: &mut dyn DrawCallback,
    images: &[ImageBuffer],
    durations_ms: &[u64],
) -> Result<(), Error> {
    let first = images
        .first()
        .ok_or_else(|| "AVIS sequence has no decodable samples".to_string())?;
    drawer.init(
        first.width,
        first.height,
        Some(InitOptions {
            loop_count: 1,
            animation: true,
        }),
    )?;
    for (index, image) in images.iter().enumerate() {
        let duration = durations_ms.get(index).copied().unwrap_or(0);
        drawer.next(Some(NextOptions::full_canvas(
            image.width,
            image.height,
            duration,
        )))?;
        drawer.draw(0, 0, image.width, image.height, &image.rgba, None)?;
    }
    drawer.terminate(None)?;
    Ok(())
}
