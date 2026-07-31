use super::composition;
use super::*;

pub(super) fn decode_still_image(
    headers: &Av1Headers,
    info: Option<&AvifInfo>,
) -> Result<ImageBuffer, DecoderError> {
    let frame = decode_still_frame(headers, info)?;
    let mut image = frame.to_rgba8()?;
    if let Some(info) = info {
        if !info.alpha_auxiliary_items.is_empty() {
            apply_alpha_auxiliary(&mut image, info)?;
        }
        composition::apply_image_transforms(
            &mut image,
            info.clean_aperture,
            info.mirror,
            info.rotation,
        )?;
    }
    Ok(image)
}

pub(super) fn decode_grid_frame(info: &AvifInfo) -> Result<DecodedFrame, DecoderError> {
    let grid = info
        .primary_grid
        .as_ref()
        .ok_or_else(|| DecoderError::Bitstream("AVIF grid metadata is missing".to_string()))?;
    let rows = usize::from(grid.rows);
    let columns = usize::from(grid.columns);
    let cell_count = rows
        .checked_mul(columns)
        .ok_or_else(|| DecoderError::Bitstream("grid cell count overflow".to_string()))?;
    if grid.cells.len() != cell_count {
        return Err(DecoderError::Bitstream(format!(
            "grid has {} cells, expected {cell_count}",
            grid.cells.len()
        )));
    }
    let width = usize::try_from(grid.output_width)
        .map_err(|_| DecoderError::InvalidParam("grid output width is too large".to_string()))?;
    let height = usize::try_from(grid.output_height)
        .map_err(|_| DecoderError::InvalidParam("grid output height is too large".to_string()))?;
    if width == 0 || height == 0 {
        return Err(DecoderError::Bitstream(
            "grid output dimensions must be non-zero".to_string(),
        ));
    }
    let mut column_widths = vec![0usize; columns];
    let mut row_heights = vec![0usize; rows];
    let mut decoded_cells = decode_grid_cells(info, &grid.cells)?;
    normalize_grid_cells(&mut decoded_cells)?;
    for (index, cell) in grid.cells.iter().enumerate() {
        let cell_width = usize::try_from(cell.width)
            .map_err(|_| DecoderError::InvalidParam("grid cell width is too large".to_string()))?;
        let cell_height = usize::try_from(cell.height)
            .map_err(|_| DecoderError::InvalidParam("grid cell height is too large".to_string()))?;
        if cell_width == 0 || cell_height == 0 {
            return Err(DecoderError::Bitstream(format!(
                "grid cell {} has zero dimensions",
                cell.item_id
            )));
        }
        let row = index / columns;
        let column = index % columns;
        if column_widths[column] != 0 && column_widths[column] != cell_width {
            return Err(DecoderError::Bitstream(
                "grid cells in one column have different widths".to_string(),
            ));
        }
        if row_heights[row] != 0 && row_heights[row] != cell_height {
            return Err(DecoderError::Bitstream(
                "grid cells in one row have different heights".to_string(),
            ));
        }
        column_widths[column] = cell_width;
        row_heights[row] = cell_height;
        let decoded = decoded_cells.get(index).ok_or_else(|| {
            DecoderError::Bitstream("grid cell decode result is missing".to_string())
        })?;
        if decoded.width != cell_width || decoded.height != cell_height {
            return Err(DecoderError::Bitstream(format!(
                "grid cell {} decoded as {}x{}, metadata declares {}x{}",
                cell.item_id, decoded.width, decoded.height, cell_width, cell_height
            )));
        }
    }
    if column_widths.iter().sum::<usize>() < width || row_heights.iter().sum::<usize>() < height {
        return Err(DecoderError::Bitstream(format!(
            "grid cell dimensions do not cover declared output {}x{} (columns {:?}, rows {:?})",
            width, height, column_widths, row_heights
        )));
    }
    let first = decoded_cells
        .first()
        .ok_or_else(|| DecoderError::Bitstream("grid has no cells".to_string()))?;
    let plane_count = first.buffers.planes.len();
    for (index, cell) in decoded_cells.iter().enumerate() {
        if cell.buffers.planes.len() != plane_count
            || cell.bit_depth != first.bit_depth
            || cell.color_config != first.color_config
        {
            return Err(DecoderError::Unsupported(format!(
                "grid cell {index} uses different AV1 plane/color configuration: planes={}, bit_depth={}, color_config={:?}; first planes={}, bit_depth={}, color_config={:?}",
                cell.buffers.planes.len(),
                cell.bit_depth,
                cell.color_config,
                plane_count,
                first.bit_depth,
                first.color_config
            )));
        }
    }
    let mut planes = Vec::with_capacity(plane_count);
    for source in &first.buffers.planes {
        let subsampling_x = source.layout.subsampling_x;
        let subsampling_y = source.layout.subsampling_y;
        let plane_width = width.div_ceil(1usize << subsampling_x);
        let plane_height = height.div_ceil(1usize << subsampling_y);
        let sample_count = plane_width.checked_mul(plane_height).ok_or_else(|| {
            DecoderError::InvalidParam("grid plane sample count overflows".to_string())
        })?;
        planes.push(PlaneBuffer {
            layout: PlaneLayout {
                plane: source.layout.plane,
                width: plane_width,
                height: plane_height,
                subsampling_x,
                subsampling_y,
                sample_count,
            },
            samples: vec![0; sample_count],
        });
    }
    let mut y_offset = 0usize;
    for row in 0..rows {
        let mut x_offset = 0usize;
        for column in 0..columns {
            let cell = &decoded_cells[row * columns + column];
            for (plane_index, source) in cell.buffers.planes.iter().enumerate() {
                let destination = planes.get_mut(plane_index).ok_or_else(|| {
                    DecoderError::Bitstream("grid cell plane count differs".to_string())
                })?;
                if source.layout.subsampling_x != destination.layout.subsampling_x
                    || source.layout.subsampling_y != destination.layout.subsampling_y
                {
                    return Err(DecoderError::Unsupported(
                        "grid cells use different chroma subsampling".to_string(),
                    ));
                }
                let scale_x = 1usize << source.layout.subsampling_x;
                let scale_y = 1usize << source.layout.subsampling_y;
                if !x_offset.is_multiple_of(scale_x) || !y_offset.is_multiple_of(scale_y) {
                    return Err(DecoderError::Unsupported(
                        "grid cell boundary is not aligned to chroma samples".to_string(),
                    ));
                }
                let destination_x = x_offset / scale_x;
                let destination_y = y_offset / scale_y;
                if destination_x >= destination.layout.width
                    || destination_y >= destination.layout.height
                {
                    continue;
                }
                let copy_width = source
                    .layout
                    .width
                    .min(destination.layout.width - destination_x);
                let copy_height = source
                    .layout
                    .height
                    .min(destination.layout.height - destination_y);
                for source_y in 0..copy_height {
                    let source_start = source_y * source.layout.width;
                    let destination_start =
                        (destination_y + source_y) * destination.layout.width + destination_x;
                    destination.samples[destination_start..destination_start + copy_width]
                        .copy_from_slice(&source.samples[source_start..source_start + copy_width]);
                }
            }
            x_offset += column_widths[column];
        }
        y_offset += row_heights[row];
    }
    let mut frame = DecodedFrame {
        width,
        height,
        render_width: width,
        render_height: height,
        bit_depth: first.bit_depth,
        color_config: first.color_config,
        color_information: info.color_information.clone(),
        alpha_premultiplied: info.alpha_premultiplied,
        buffers: FrameBuffers {
            width,
            height,
            planes,
        },
    };
    if let Some(alpha_grid) = info.alpha_grid.as_ref() {
        let (alpha_plane, alpha_bit_depth) = decode_alpha_grid_plane(alpha_grid)?;
        append_alpha_plane_buffer(&mut frame, alpha_plane, alpha_bit_depth)?;
    } else if !info.alpha_auxiliary_items.is_empty()
        && !info
            .primary_grid
            .as_ref()
            .is_some_and(|grid| grid_has_cell_alpha(&grid.payload))
    {
        let alpha_frame = decode_alpha_auxiliary_frame(info)?;
        append_alpha_plane(&mut frame, &alpha_frame)?;
    }
    apply_native_grid_geometry(&mut frame, info)?;
    Ok(frame)
}

fn decode_grid_cells(
    info: &AvifInfo,
    cells: &[GridCell],
) -> Result<Vec<DecodedFrame>, DecoderError> {
    decode_grid_items(cells, |cell| decode_grid_cell_frame(info, cell))
}

fn decode_grid_images(
    info: &AvifInfo,
    cells: &[GridCell],
) -> Result<Vec<ImageBuffer>, DecoderError> {
    decode_grid_items(cells, |cell| decode_grid_cell(info, cell))
}

#[cfg(not(target_family = "wasm"))]
const PARALLEL_GRID_MIN_PIXELS: usize = 256 * 1024;
#[cfg(not(target_family = "wasm"))]
const MAX_GRID_WORKERS: usize = 8;

fn decode_grid_items<T, F>(cells: &[GridCell], decode: F) -> Result<Vec<T>, DecoderError>
where
    T: Send,
    F: Fn(&GridCell) -> Result<T, DecoderError> + Sync,
{
    #[cfg(not(target_family = "wasm"))]
    {
        let total_pixels = cells.iter().fold(0usize, |total, cell| {
            total.saturating_add(
                usize::try_from(cell.width)
                    .unwrap_or(usize::MAX)
                    .saturating_mul(usize::try_from(cell.height).unwrap_or(usize::MAX)),
            )
        });
        if cells.len() > 1 && total_pixels >= PARALLEL_GRID_MIN_PIXELS {
            let worker_count = cells.len().min(MAX_GRID_WORKERS);
            let chunk_size = cells.len().div_ceil(worker_count);
            return std::thread::scope(|scope| {
                let handles = cells
                    .chunks(chunk_size)
                    .map(|chunk| {
                        scope.spawn(|| chunk.iter().map(&decode).collect::<Result<Vec<_>, _>>())
                    })
                    .collect::<Vec<_>>();
                let mut decoded = Vec::with_capacity(cells.len());
                for handle in handles {
                    let chunk = handle.join().map_err(|_| {
                        DecoderError::Bitstream(
                            "AVIF grid cell decoder thread panicked".to_string(),
                        )
                    })??;
                    decoded.extend(chunk);
                }
                Ok(decoded)
            });
        }
    }

    cells.iter().map(decode).collect()
}

fn apply_native_grid_geometry(
    frame: &mut DecodedFrame,
    info: &AvifInfo,
) -> Result<(), DecoderError> {
    if let Some(aperture) = info.clean_aperture {
        let (start_x, start_y, width, height) =
            composition::clean_aperture_rect(frame.width, frame.height, aperture)?;
        for plane in &mut frame.buffers.planes {
            let scale_x = 1usize << plane.layout.subsampling_x;
            let scale_y = 1usize << plane.layout.subsampling_y;
            if start_x % scale_x != 0 || start_y % scale_y != 0 {
                return Err(DecoderError::Unsupported(
                    "AVIF native-plane clean aperture is not aligned to chroma samples".to_string(),
                ));
            }
            let plane_start_x = start_x / scale_x;
            let plane_start_y = start_y / scale_y;
            let plane_width = width.div_ceil(scale_x);
            let plane_height = height.div_ceil(scale_y);
            if plane_start_x + plane_width > plane.layout.width
                || plane_start_y + plane_height > plane.layout.height
            {
                return Err(DecoderError::Bitstream(
                    "AVIF native-plane clean aperture exceeds a source plane".to_string(),
                ));
            }
            let mut cropped = vec![0; plane_width * plane_height];
            for row in 0..plane_height {
                let source_start = (plane_start_y + row) * plane.layout.width + plane_start_x;
                let destination_start = row * plane_width;
                cropped[destination_start..destination_start + plane_width]
                    .copy_from_slice(&plane.samples[source_start..source_start + plane_width]);
            }
            plane.layout.width = plane_width;
            plane.layout.height = plane_height;
            plane.layout.sample_count = cropped.len();
            plane.samples = cropped;
        }
        frame.width = width;
        frame.height = height;
        frame.render_width = width;
        frame.render_height = height;
        frame.buffers.width = width;
        frame.buffers.height = height;
    }
    if let Some(mirror) = info.mirror {
        apply_native_mirror(&mut frame.buffers, mirror)?;
    }
    if let Some(rotation) = info.rotation {
        apply_native_rotation(frame, rotation)?;
    }
    Ok(())
}

fn apply_native_mirror(
    buffers: &mut FrameBuffers,
    mirror: ImageMirror,
) -> Result<(), DecoderError> {
    if mirror.axis > 1 {
        return Err(DecoderError::Bitstream(format!(
            "AVIF mirror axis {} is invalid",
            mirror.axis
        )));
    }
    for plane in &mut buffers.planes {
        if mirror.axis == 0 {
            for row in plane.samples.chunks_exact_mut(plane.layout.width) {
                row.reverse();
            }
        } else {
            let width = plane.layout.width;
            let height = plane.layout.height;
            for row in 0..height / 2 {
                let opposite = height - 1 - row;
                for column in 0..width {
                    plane
                        .samples
                        .swap(row * width + column, opposite * width + column);
                }
            }
        }
    }
    Ok(())
}

fn apply_native_rotation(
    frame: &mut DecodedFrame,
    rotation: ImageRotation,
) -> Result<(), DecoderError> {
    if rotation.angle > 3 {
        return Err(DecoderError::Bitstream(format!(
            "AVIF rotation angle {} is invalid",
            rotation.angle
        )));
    }
    for _ in 0..rotation.angle {
        for plane in &mut frame.buffers.planes {
            let width = plane.layout.width;
            let height = plane.layout.height;
            let mut transformed = vec![0; plane.samples.len()];
            let new_width = height;
            for y in 0..height {
                for x in 0..width {
                    let destination_x = y;
                    let destination_y = width - 1 - x;
                    transformed[destination_y * new_width + destination_x] =
                        plane.samples[y * width + x];
                }
            }
            plane.layout.width = new_width;
            plane.layout.height = width;
            if plane.layout.plane != 0 {
                std::mem::swap(
                    &mut plane.layout.subsampling_x,
                    &mut plane.layout.subsampling_y,
                );
            }
            plane.layout.sample_count = transformed.len();
            plane.samples = transformed;
        }
        std::mem::swap(
            &mut frame.color_config.subsampling_x,
            &mut frame.color_config.subsampling_y,
        );
        std::mem::swap(&mut frame.width, &mut frame.height);
        std::mem::swap(&mut frame.render_width, &mut frame.render_height);
        std::mem::swap(&mut frame.buffers.width, &mut frame.buffers.height);
    }

    Ok(())
}

pub(super) fn decode_grid_image(info: &AvifInfo) -> Result<ImageBuffer, DecoderError> {
    let grid = info
        .primary_grid
        .as_ref()
        .ok_or_else(|| DecoderError::Bitstream("AVIF grid metadata is missing".to_string()))?;
    let rows = usize::from(grid.rows);
    let columns = usize::from(grid.columns);
    let cell_count = rows
        .checked_mul(columns)
        .ok_or_else(|| DecoderError::Bitstream("grid cell count overflow".to_string()))?;
    if grid.cells.len() != cell_count {
        return Err(DecoderError::Bitstream(format!(
            "grid has {} cells, expected {cell_count}",
            grid.cells.len()
        )));
    }
    let width = usize::try_from(grid.output_width)
        .map_err(|_| DecoderError::InvalidParam("grid output width is too large".to_string()))?;
    let height = usize::try_from(grid.output_height)
        .map_err(|_| DecoderError::InvalidParam("grid output height is too large".to_string()))?;
    if width == 0 || height == 0 {
        return Err(DecoderError::Bitstream(
            "grid output dimensions must be non-zero".to_string(),
        ));
    }
    let mut column_widths = vec![0usize; columns];
    let mut row_heights = vec![0usize; rows];
    let decoded_cells = decode_grid_images(info, &grid.cells)?;
    for (index, cell) in grid.cells.iter().enumerate() {
        let cell_width = usize::try_from(cell.width)
            .map_err(|_| DecoderError::InvalidParam("grid cell width is too large".to_string()))?;
        let cell_height = usize::try_from(cell.height)
            .map_err(|_| DecoderError::InvalidParam("grid cell height is too large".to_string()))?;
        if cell_width == 0 || cell_height == 0 {
            return Err(DecoderError::Bitstream(format!(
                "grid cell {} has zero dimensions",
                cell.item_id
            )));
        }
        let row = index / columns;
        let column = index % columns;
        if column_widths[column] != 0 && column_widths[column] != cell_width {
            return Err(DecoderError::Bitstream(
                "grid cells in one column have different widths".to_string(),
            ));
        }
        if row_heights[row] != 0 && row_heights[row] != cell_height {
            return Err(DecoderError::Bitstream(
                "grid cells in one row have different heights".to_string(),
            ));
        }
        column_widths[column] = cell_width;
        row_heights[row] = cell_height;
    }
    if column_widths.iter().sum::<usize>() < width || row_heights.iter().sum::<usize>() < height {
        return Err(DecoderError::Bitstream(format!(
            "grid cell dimensions do not cover declared output {}x{} (columns {:?}, rows {:?})",
            width, height, column_widths, row_heights
        )));
    }
    let mut image = compose_grid_images(grid, &decoded_cells, &column_widths, &row_heights)?;
    if let Some(alpha_grid) = info.alpha_grid.as_ref() {
        apply_alpha_grid(&mut image, alpha_grid)?;
    } else if !info.alpha_auxiliary_items.is_empty()
        && !info
            .primary_grid
            .as_ref()
            .is_some_and(|grid| grid_has_cell_alpha(&grid.payload))
    {
        apply_alpha_auxiliary(&mut image, info)?;
    }
    composition::apply_image_transforms(
        &mut image,
        info.clean_aperture,
        info.mirror,
        info.rotation,
    )?;
    Ok(image)
}

fn compose_grid_images(
    grid: &crate::container::GridImage,
    decoded_cells: &[ImageBuffer],
    column_widths: &[usize],
    row_heights: &[usize],
) -> Result<ImageBuffer, DecoderError> {
    let width = usize::try_from(grid.output_width)
        .map_err(|_| DecoderError::InvalidParam("grid output width is too large".to_string()))?;
    let height = usize::try_from(grid.output_height)
        .map_err(|_| DecoderError::InvalidParam("grid output height is too large".to_string()))?;
    let rows = usize::from(grid.rows);
    let columns = usize::from(grid.columns);
    if decoded_cells.len() != rows.saturating_mul(columns)
        || column_widths.len() != columns
        || row_heights.len() != rows
    {
        return Err(DecoderError::InvalidParam(
            "grid composition dimensions do not match cell count".to_string(),
        ));
    }
    let rgba_len = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| DecoderError::InvalidParam("grid output buffer is too large".to_string()))?;
    let mut image = ImageBuffer {
        width,
        height,
        rgba: vec![0; rgba_len],
    };
    let mut y_offset = 0;
    for row in 0..rows {
        let mut x_offset = 0;
        for column in 0..columns {
            let cell = &decoded_cells[row * columns + column];
            if cell.width != column_widths[column] || cell.height != row_heights[row] {
                return Err(DecoderError::Bitstream(
                    "grid cell image dimensions do not match metadata".to_string(),
                ));
            }
            if x_offset < width && y_offset < height {
                let copy_width = cell.width.min(width - x_offset);
                let copy_height = cell.height.min(height - y_offset);
                for y in 0..copy_height {
                    let destination = ((y_offset + y) * width + x_offset) * 4;
                    let source = y * cell.width * 4;
                    let row_len = copy_width * 4;
                    image.rgba[destination..destination + row_len]
                        .copy_from_slice(&cell.rgba[source..source + row_len]);
                }
            }
            x_offset += column_widths[column];
        }
        y_offset += row_heights[row];
    }
    Ok(image)
}

fn normalize_grid_cells(decoded_cells: &mut [DecodedFrame]) -> Result<(), DecoderError> {
    let Some(target) = decoded_cells
        .iter()
        .find(|cell| !cell.color_config.monochrome)
    else {
        return Ok(());
    };
    let target_config = target.color_config;
    let target_color_information = target.color_information.clone();
    let target_bit_depth = target.bit_depth;
    let target_has_alpha = decoded_cells.iter().any(|cell| {
        cell.buffers
            .planes
            .iter()
            .any(|plane| plane.layout.plane == 3)
    });
    for cell in decoded_cells {
        if cell.color_config.monochrome {
            let luma = cell
                .buffers
                .planes
                .iter()
                .find(|plane| plane.layout.plane == 0)
                .cloned()
                .ok_or_else(|| {
                    DecoderError::Bitstream(
                        "monochrome grid cell is missing its luma plane".to_string(),
                    )
                })?;
            let alpha = cell
                .buffers
                .planes
                .iter()
                .find(|plane| plane.layout.plane == 3)
                .cloned();
            let chroma_x = u8::from(target_config.subsampling_x);
            let chroma_y = u8::from(target_config.subsampling_y);
            let chroma_width = cell.width.div_ceil(1usize << chroma_x);
            let chroma_height = cell.height.div_ceil(1usize << chroma_y);
            let chroma_samples = chroma_width.checked_mul(chroma_height).ok_or_else(|| {
                DecoderError::InvalidParam("grid chroma plane size overflows".to_string())
            })?;
            let neutral = 1u16 << target_bit_depth.saturating_sub(1);
            let make_chroma = |plane: u8| PlaneBuffer {
                layout: PlaneLayout {
                    plane,
                    width: chroma_width,
                    height: chroma_height,
                    subsampling_x: chroma_x,
                    subsampling_y: chroma_y,
                    sample_count: chroma_samples,
                },
                samples: vec![neutral; chroma_samples],
            };
            let mut planes = vec![luma, make_chroma(1), make_chroma(2)];
            if let Some(alpha) = alpha {
                planes.push(alpha);
            }
            cell.buffers.planes = planes;
            cell.color_config = target_config;

            cell.color_information = target_color_information.clone();
        }
        if target_has_alpha
            && !cell
                .buffers
                .planes
                .iter()
                .any(|plane| plane.layout.plane == 3)
        {
            let alpha_samples = cell.width.checked_mul(cell.height).ok_or_else(|| {
                DecoderError::InvalidParam("grid alpha plane size overflows".to_string())
            })?;
            let opaque = ((1u32 << target_bit_depth) - 1) as u16;
            cell.buffers.planes.push(PlaneBuffer {
                layout: PlaneLayout {
                    plane: 3,
                    width: cell.width,
                    height: cell.height,
                    subsampling_x: 0,
                    subsampling_y: 0,
                    sample_count: alpha_samples,
                },
                samples: vec![opaque; alpha_samples],
            });
        }
    }
    Ok(())
}

#[cfg(test)]
pub(super) mod grid_composition_tests {
    use super::*;
    use crate::container::GridImage;

    fn cell(width: usize, height: usize, value: u8) -> ImageBuffer {
        ImageBuffer {
            width,
            height,
            rgba: vec![value; width * height * 4],
        }
    }

    #[test]
    fn compose_grid_places_cells_in_reference_order() {
        let grid = GridImage {
            item_id: 1,
            rows: 2,
            columns: 2,
            output_width: 3,
            output_height: 3,
            payload: Vec::new(),
            cells: Vec::new(),
        };
        let cells = vec![
            cell(2, 2, 10),
            cell(1, 2, 20),
            cell(2, 1, 30),
            cell(1, 1, 40),
        ];
        let image = compose_grid_images(&grid, &cells, &[2, 1], &[2, 1]).unwrap();
        assert_eq!((image.width, image.height), (3, 3));
        assert_eq!(image.rgba[0], 10);
        assert_eq!(image.rgba[8], 20);
        assert_eq!(image.rgba[2 * 3 * 4], 30);
        assert_eq!(image.rgba[(2 * 3 + 2) * 4], 40);
    }

    #[test]
    fn compose_grid_crops_oversized_edge_cells_to_output() {
        let grid = GridImage {
            item_id: 1,
            rows: 2,
            columns: 1,
            output_width: 2,
            output_height: 3,
            payload: Vec::new(),
            cells: Vec::new(),
        };
        let cells = vec![cell(2, 2, 10), cell(2, 2, 20)];
        let image = compose_grid_images(&grid, &cells, &[2], &[2, 2]).unwrap();
        assert_eq!((image.width, image.height), (2, 3));
        assert_eq!(image.rgba[0], 10);
        assert_eq!(image.rgba[(2 * 2) * 4], 20);
        assert_eq!(image.rgba[(2 * 2 + 1) * 4], 20);
    }

    #[test]
    fn irot_angle_one_rotates_counter_clockwise() {
        let mut image = ImageBuffer {
            width: 2,
            height: 1,
            rgba: vec![10, 0, 0, 255, 20, 0, 0, 255],
        };
        composition::apply_image_transforms(
            &mut image,
            None,
            None,
            Some(ImageRotation { angle: 1 }),
        )
        .unwrap();

        assert_eq!((image.width, image.height), (1, 2));
        assert_eq!(image.rgba, vec![20, 0, 0, 255, 10, 0, 0, 255]);
    }

    #[test]
    fn alpha_grid_rejects_mismatched_output_dimensions() {
        let mut image = cell(2, 1, 90);
        let grid = GridImage {
            item_id: 2,
            rows: 1,
            columns: 1,
            output_width: 1,
            output_height: 1,
            payload: Vec::new(),
            cells: Vec::new(),
        };
        let error = apply_alpha_grid(&mut image, &grid).unwrap_err();
        assert!(error.to_string().contains("dimensions do not match"));
    }

    pub(crate) fn native_frame(width: usize, height: usize, plane: PlaneBuffer) -> DecodedFrame {
        DecodedFrame {
            width,
            height,
            render_width: width,
            render_height: height,
            bit_depth: 8,
            color_config: ColorConfig {
                high_bitdepth: false,
                twelve_bit: false,
                bit_depth: 8,
                monochrome: true,
                color_description: None,
                color_range: ColorRange::Full,
                subsampling_x: false,
                subsampling_y: false,
                chroma_sample_position: None,
                separate_uv_delta_q: false,
            },
            color_information: None,
            alpha_premultiplied: false,
            buffers: FrameBuffers {
                width,
                height,
                planes: vec![plane],
            },
        }
    }

    pub(crate) fn native_plane(width: usize, height: usize, samples: Vec<u16>) -> PlaneBuffer {
        PlaneBuffer {
            layout: PlaneLayout {
                plane: 0,
                width,
                height,
                subsampling_x: 0,
                subsampling_y: 0,
                sample_count: width * height,
            },
            samples,
        }
    }

    #[test]
    fn native_grid_mirror_reverses_each_plane_without_reallocation() {
        let mut frame = native_frame(3, 2, native_plane(3, 2, vec![1, 2, 3, 4, 5, 6]));
        apply_native_mirror(&mut frame.buffers, ImageMirror { axis: 0 }).unwrap();
        assert_eq!(frame.buffers.planes[0].samples, vec![3, 2, 1, 6, 5, 4]);
    }

    #[test]
    fn native_grid_rotation_updates_frame_and_plane_dimensions() {
        let mut frame = native_frame(2, 3, native_plane(2, 3, vec![1, 2, 3, 4, 5, 6]));
        apply_native_rotation(&mut frame, ImageRotation { angle: 1 }).unwrap();
        assert_eq!((frame.width, frame.height), (3, 2));
        assert_eq!(frame.buffers.planes[0].samples, vec![2, 4, 6, 1, 3, 5]);
    }

    #[test]
    fn native_grid_rotation_swaps_422_subsampling_axes() {
        let mut frame = native_frame(4, 2, native_plane(4, 2, (10..18).collect()));
        frame.color_config.monochrome = false;
        frame.color_config.subsampling_x = true;
        frame.buffers.planes.push(PlaneBuffer {
            layout: PlaneLayout {
                plane: 1,
                width: 2,
                height: 2,
                subsampling_x: 1,
                subsampling_y: 0,
                sample_count: 4,
            },
            samples: (0..4).collect(),
        });

        apply_native_rotation(&mut frame, ImageRotation { angle: 1 }).unwrap();

        assert_eq!((frame.width, frame.height), (2, 4));
        assert_eq!(
            (
                frame.color_config.subsampling_x,
                frame.color_config.subsampling_y
            ),
            (false, true)
        );
        let chroma = &frame.buffers.planes[1];
        assert_eq!((chroma.layout.width, chroma.layout.height), (2, 2));
        assert_eq!(
            (chroma.layout.subsampling_x, chroma.layout.subsampling_y),
            (0, 1)
        );
        assert_eq!(chroma.samples, vec![1, 3, 0, 2]);
    }

    #[test]
    fn native_grid_alpha_plane_composes_ordered_cells() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("workspace root should exist")
            .join("test/images/external/avif/unsupported/sofa_grid1x5_420.avif");
        if !path.is_file() {
            eprintln!("external grid sample is unavailable; skipping alpha-grid unit oracle");
            return;
        }
        let info = crate::container::parse_avif(
            &std::fs::read(path).expect("external grid sample should be readable"),
        )
        .expect("external grid sample metadata should parse");
        let grid = info
            .primary_grid
            .expect("external sample should contain a grid");
        let (alpha, _) = decode_alpha_grid_plane(&grid).expect("grid alpha plane should compose");
        assert_eq!(alpha.layout.plane, 3);
        assert_eq!((alpha.layout.width, alpha.layout.height), (1024, 770));
        assert_eq!(alpha.samples.len(), 1024 * 770);
    }
}

fn grid_cell_info(info: &AvifInfo, cell: &GridCell) -> AvifInfo {
    let alpha_auxiliary_items = if info.alpha_grid.is_none() {
        info.primary_grid
            .as_ref()
            .and_then(|grid| grid_cell_alpha_item_id(&grid.payload, cell.item_id))
            .and_then(|alpha_id| {
                info.alpha_auxiliary_items
                    .iter()
                    .find(|item| item.item_id == alpha_id)
                    .cloned()
            })
            .into_iter()
            .collect()
    } else {
        Vec::new()
    };
    AvifInfo {
        major_brand: info.major_brand,
        compatible_brands: info.compatible_brands.clone(),
        primary_item_id: Some(cell.item_id),
        width: Some(cell.width),
        height: Some(cell.height),
        pixel_information: cell.pixel_information.clone(),
        color_information: cell
            .color_information
            .clone()
            .or_else(|| info.color_information.clone()),
        alpha_premultiplied: false,
        alpha_auxiliary_items,
        alpha_grid: None,
        primary_grid: None,
        clean_aperture: None,
        rotation: None,
        mirror: None,
        av1_config: cell.av1_config.clone(),
        primary_item_payload: cell.payload.clone(),
        sequence_sample_payloads: Vec::new(),
    }
}

fn decode_grid_cell(info: &AvifInfo, cell: &GridCell) -> Result<ImageBuffer, DecoderError> {
    let cell_info = grid_cell_info(info, cell);
    validate_public_container_preflight(&cell_info, true)?;
    let headers = parse_av1_headers(&cell_info)?;
    decode_still_image(&headers, Some(&cell_info))
}

fn decode_grid_cell_frame(info: &AvifInfo, cell: &GridCell) -> Result<DecodedFrame, DecoderError> {
    let cell_info = grid_cell_info(info, cell);
    validate_public_container_preflight(&cell_info, false)?;
    let headers = parse_av1_headers(&cell_info)?;
    decode_still_frame(&headers, Some(&cell_info))
}

pub(super) fn append_alpha_plane(
    frame: &mut DecodedFrame,
    alpha_frame: &DecodedFrame,
) -> Result<(), DecoderError> {
    if alpha_frame.width != frame.width || alpha_frame.height != frame.height {
        return Err(DecoderError::Bitstream(
            "AVIF alpha auxiliary dimensions do not match the primary image".to_string(),
        ));
    }
    if alpha_frame.buffers.planes.len() != 1 {
        return Err(DecoderError::Unsupported(
            "AVIF alpha auxiliary image must be monochrome".to_string(),
        ));
    }
    let alpha_plane = alpha_frame.buffers.planes.first().ok_or_else(|| {
        DecoderError::Bitstream("AVIF alpha auxiliary plane is missing".to_string())
    })?;
    append_alpha_plane_buffer(frame, alpha_plane.clone(), alpha_frame.bit_depth)
}

pub(super) fn append_alpha_plane_buffer(
    frame: &mut DecodedFrame,
    mut alpha_plane: PlaneBuffer,
    alpha_bit_depth: u8,
) -> Result<(), DecoderError> {
    if alpha_plane.layout.width == 0 || alpha_plane.layout.height == 0 {
        return Err(DecoderError::Bitstream(
            "AVIF alpha plane has zero dimensions".to_string(),
        ));
    }
    if frame
        .buffers
        .planes
        .iter()
        .any(|plane| plane.layout.plane == 3)
    {
        return Err(DecoderError::Bitstream(
            "AVIF primary image contains duplicate alpha planes".to_string(),
        ));
    }
    if alpha_bit_depth != frame.bit_depth {
        let source_max = (1u32 << alpha_bit_depth) - 1;
        let target_max = (1u32 << frame.bit_depth) - 1;
        for sample in &mut alpha_plane.samples {
            *sample = ((u32::from(*sample) * target_max + source_max / 2) / source_max) as u16;
        }
    }
    alpha_plane.layout.plane = 3;
    frame.buffers.planes.push(alpha_plane);
    Ok(())
}

pub(super) fn decode_alpha_auxiliary_frame(info: &AvifInfo) -> Result<DecodedFrame, DecoderError> {
    let auxiliary = info.alpha_auxiliary_items.first().ok_or_else(|| {
        DecoderError::Bitstream("AVIF alpha auxiliary item is missing".to_string())
    })?;
    let alpha_info = AvifInfo {
        major_brand: info.major_brand,
        compatible_brands: info.compatible_brands.clone(),
        primary_item_id: None,
        width: None,
        height: None,
        pixel_information: None,
        color_information: None,
        alpha_premultiplied: false,
        alpha_auxiliary_items: Vec::new(),
        alpha_grid: None,
        primary_grid: None,
        clean_aperture: None,
        rotation: None,
        mirror: None,
        av1_config: None,
        primary_item_payload: auxiliary.payload.clone(),
        sequence_sample_payloads: Vec::new(),
    };
    let headers = parse_av1_headers(&alpha_info)?;
    decode_still_frame(&headers, None)
}

pub(super) fn decode_alpha_grid_plane(
    grid: &crate::container::GridImage,
) -> Result<(PlaneBuffer, u8), DecoderError> {
    let rows = usize::from(grid.rows);
    let columns = usize::from(grid.columns);
    let cell_count = rows.checked_mul(columns).ok_or_else(|| {
        DecoderError::Bitstream("AVIF alpha grid cell count overflow".to_string())
    })?;
    if grid.cells.len() != cell_count {
        return Err(DecoderError::Bitstream(
            "AVIF alpha grid cell count does not match its dimensions".to_string(),
        ));
    }
    let width = usize::try_from(grid.output_width).map_err(|_| {
        DecoderError::InvalidParam("alpha grid output width is too large".to_string())
    })?;
    let height = usize::try_from(grid.output_height).map_err(|_| {
        DecoderError::InvalidParam("alpha grid output height is too large".to_string())
    })?;
    if width == 0 || height == 0 {
        return Err(DecoderError::Bitstream(
            "alpha grid output dimensions must be non-zero".to_string(),
        ));
    }
    let mut column_widths = vec![0usize; columns];
    let mut row_heights = vec![0usize; rows];
    let decoded_cells = decode_alpha_grid_cells(&grid.cells)?;
    for (index, cell) in grid.cells.iter().enumerate() {
        let cell_width = usize::try_from(cell.width).map_err(|_| {
            DecoderError::InvalidParam("alpha grid cell width is too large".to_string())
        })?;
        let cell_height = usize::try_from(cell.height).map_err(|_| {
            DecoderError::InvalidParam("alpha grid cell height is too large".to_string())
        })?;
        let row = index / columns;
        let column = index % columns;
        if column_widths[column] != 0 && column_widths[column] != cell_width {
            return Err(DecoderError::Bitstream(
                "alpha grid cells in one column have different widths".to_string(),
            ));
        }
        if row_heights[row] != 0 && row_heights[row] != cell_height {
            return Err(DecoderError::Bitstream(
                "alpha grid cells in one row have different heights".to_string(),
            ));
        }
        column_widths[column] = cell_width;
        row_heights[row] = cell_height;
        let decoded = decoded_cells.get(index).ok_or_else(|| {
            DecoderError::Bitstream("alpha grid cell decode result is missing".to_string())
        })?;
        if decoded.width != cell_width || decoded.height != cell_height {
            return Err(DecoderError::Bitstream(format!(
                "alpha grid cell {} decoded as {}x{}, metadata declares {}x{}",
                cell.item_id, decoded.width, decoded.height, cell_width, cell_height
            )));
        }
    }
    if column_widths.iter().sum::<usize>() < width || row_heights.iter().sum::<usize>() < height {
        return Err(DecoderError::Bitstream(
            "alpha grid cell dimensions do not cover the declared output".to_string(),
        ));
    }
    let first = decoded_cells
        .first()
        .ok_or_else(|| DecoderError::Bitstream("alpha grid has no cells".to_string()))?;
    let source = first
        .buffers
        .planes
        .first()
        .ok_or_else(|| DecoderError::Bitstream("alpha grid cell has no plane".to_string()))?;
    for cell in &decoded_cells {
        let plane =
            cell.buffers.planes.first().ok_or_else(|| {
                DecoderError::Bitstream("alpha grid cell has no plane".to_string())
            })?;
        if plane.layout.subsampling_x != source.layout.subsampling_x
            || plane.layout.subsampling_y != source.layout.subsampling_y
            || cell.bit_depth != first.bit_depth
        {
            return Err(DecoderError::Unsupported(
                "alpha grid cells use different plane configurations".to_string(),
            ));
        }
    }
    let subsampling_x = usize::from(source.layout.subsampling_x);
    let subsampling_y = usize::from(source.layout.subsampling_y);
    let scale_x = 1usize << subsampling_x;
    let scale_y = 1usize << subsampling_y;
    let plane_width = width.div_ceil(scale_x);
    let plane_height = height.div_ceil(scale_y);
    let sample_count = plane_width.checked_mul(plane_height).ok_or_else(|| {
        DecoderError::InvalidParam("alpha grid plane sample count overflows".to_string())
    })?;
    let mut output = PlaneBuffer {
        layout: PlaneLayout {
            plane: 3,
            width: plane_width,
            height: plane_height,
            subsampling_x: source.layout.subsampling_x,
            subsampling_y: source.layout.subsampling_y,
            sample_count,
        },
        samples: vec![0; sample_count],
    };
    let mut y_offset = 0usize;
    for row in 0..rows {
        let mut x_offset = 0usize;
        for column in 0..columns {
            let cell = &decoded_cells[row * columns + column];
            let source = cell.buffers.planes.first().ok_or_else(|| {
                DecoderError::Bitstream("alpha grid cell has no plane".to_string())
            })?;
            if !x_offset.is_multiple_of(scale_x) || !y_offset.is_multiple_of(scale_y) {
                return Err(DecoderError::Unsupported(
                    "alpha grid cell boundary is not aligned to chroma samples".to_string(),
                ));
            }
            let destination_x = x_offset / scale_x;
            let destination_y = y_offset / scale_y;
            if destination_x >= output.layout.width || destination_y >= output.layout.height {
                continue;
            }
            let copy_width = source.layout.width.min(output.layout.width - destination_x);
            let copy_height = source
                .layout
                .height
                .min(output.layout.height - destination_y);
            for source_y in 0..copy_height {
                let source_start = source_y * source.layout.width;
                let destination_start =
                    (destination_y + source_y) * output.layout.width + destination_x;
                output.samples[destination_start..destination_start + copy_width]
                    .copy_from_slice(&source.samples[source_start..source_start + copy_width]);
            }
            x_offset += column_widths[column];
        }
        y_offset += row_heights[row];
    }
    Ok((output, first.bit_depth))
}

fn decode_alpha_grid_cell_frame(cell: &GridCell) -> Result<DecodedFrame, DecoderError> {
    let info = AvifInfo {
        major_brand: *b"avif",
        compatible_brands: vec![*b"avif"],
        primary_item_id: Some(cell.item_id),
        width: Some(cell.width),
        height: Some(cell.height),
        pixel_information: cell.pixel_information.clone(),
        color_information: None,
        alpha_premultiplied: false,
        alpha_auxiliary_items: Vec::new(),
        alpha_grid: None,
        primary_grid: None,
        clean_aperture: None,
        rotation: None,
        mirror: None,
        av1_config: cell.av1_config.clone(),
        primary_item_payload: cell.payload.clone(),
        sequence_sample_payloads: Vec::new(),
    };
    validate_public_container_preflight(&info, false)?;
    let headers = parse_av1_headers(&info)?;
    decode_still_frame(&headers, None)
}

fn decode_alpha_grid_cells(cells: &[GridCell]) -> Result<Vec<DecodedFrame>, DecoderError> {
    decode_grid_items(cells, decode_alpha_grid_cell_frame)
}

fn apply_alpha_grid(
    image: &mut ImageBuffer,
    grid: &crate::container::GridImage,
) -> Result<(), DecoderError> {
    if grid.output_width != image.width as u32 || grid.output_height != image.height as u32 {
        return Err(DecoderError::Bitstream(
            "AVIF alpha grid dimensions do not match the primary grid".to_string(),
        ));
    }
    let rows = usize::from(grid.rows);
    let columns = usize::from(grid.columns);
    let cell_count = rows.checked_mul(columns).ok_or_else(|| {
        DecoderError::Bitstream("AVIF alpha grid cell count overflow".to_string())
    })?;
    if grid.cells.len() != cell_count {
        return Err(DecoderError::Bitstream(
            "AVIF alpha grid cell count does not match its dimensions".to_string(),
        ));
    }
    let mut column_widths = vec![0usize; columns];
    let mut row_heights = vec![0usize; rows];
    let mut decoded_cells = Vec::with_capacity(cell_count);
    for (index, cell) in grid.cells.iter().enumerate() {
        let width = usize::try_from(cell.width).map_err(|_| {
            DecoderError::InvalidParam("alpha grid cell width is too large".to_string())
        })?;
        let height = usize::try_from(cell.height).map_err(|_| {
            DecoderError::InvalidParam("alpha grid cell height is too large".to_string())
        })?;
        let row = index / columns;
        let column = index % columns;
        if column_widths[column] != 0 && column_widths[column] != width {
            return Err(DecoderError::Bitstream(
                "alpha grid cells in one column have different widths".to_string(),
            ));
        }
        if row_heights[row] != 0 && row_heights[row] != height {
            return Err(DecoderError::Bitstream(
                "alpha grid cells in one row have different heights".to_string(),
            ));
        }
        column_widths[column] = width;
        row_heights[row] = height;
        decoded_cells.push(decode_grid_cell_from_alpha(cell)?);
    }
    let alpha = compose_grid_images(grid, &decoded_cells, &column_widths, &row_heights)?;
    for (pixel, alpha_pixel) in image
        .rgba
        .chunks_exact_mut(4)
        .zip(alpha.rgba.chunks_exact(4))
    {
        pixel[3] = alpha_pixel[0];
    }
    Ok(())
}

fn decode_grid_cell_from_alpha(cell: &GridCell) -> Result<ImageBuffer, DecoderError> {
    let info = AvifInfo {
        major_brand: *b"avif",
        compatible_brands: vec![*b"avif"],
        primary_item_id: Some(cell.item_id),
        width: Some(cell.width),
        height: Some(cell.height),
        pixel_information: cell.pixel_information.clone(),
        color_information: cell.color_information.clone(),
        alpha_premultiplied: false,
        alpha_auxiliary_items: Vec::new(),
        alpha_grid: None,
        primary_grid: None,
        clean_aperture: None,
        rotation: None,
        mirror: None,
        av1_config: cell.av1_config.clone(),
        primary_item_payload: cell.payload.clone(),
        sequence_sample_payloads: Vec::new(),
    };
    validate_public_container_preflight(&info, true)?;
    let headers = parse_av1_headers(&info)?;
    decode_still_image(&headers, Some(&info))
}

fn apply_alpha_auxiliary(image: &mut ImageBuffer, info: &AvifInfo) -> Result<(), DecoderError> {
    let frame = decode_alpha_auxiliary_frame(info)?;
    let alpha_plane = frame.buffers.planes.first().ok_or_else(|| {
        DecoderError::Bitstream("AVIF alpha auxiliary plane is missing".to_string())
    })?;
    if frame.width != image.width || frame.height != image.height {
        return Err(DecoderError::Bitstream(
            "AVIF alpha auxiliary dimensions do not match the primary image".to_string(),
        ));
    }
    apply_alpha_plane_rows(
        &mut image.rgba,
        image.width,
        image.height,
        alpha_plane,
        frame.bit_depth,
    );
    Ok(())
}

#[cfg(not(target_family = "wasm"))]
const PARALLEL_ALPHA_MIN_PIXELS: usize = 256 * 1024;
#[cfg(not(target_family = "wasm"))]
const MAX_ALPHA_WORKERS: usize = 8;

fn apply_alpha_plane_rows(
    rgba: &mut [u8],
    width: usize,
    _height: usize,
    alpha_plane: &crate::av1::PlaneBuffer,
    bit_depth: u8,
) {
    #[cfg(not(target_family = "wasm"))]
    let workers = if width.saturating_mul(_height) < PARALLEL_ALPHA_MIN_PIXELS {
        1
    } else {
        std::thread::available_parallelism()
            .map_or(1, std::num::NonZeroUsize::get)
            .min(MAX_ALPHA_WORKERS)
            .min(_height.max(1))
    };
    #[cfg(target_family = "wasm")]
    let workers = 1;
    if workers <= 1 {
        apply_alpha_rows(
            rgba,
            0,
            width,
            alpha_plane.layout.width,
            alpha_plane.layout.height,
            alpha_plane.layout.subsampling_x,
            alpha_plane.layout.subsampling_y,
            &alpha_plane.samples,
            bit_depth,
        );
        return;
    }

    #[cfg(not(target_family = "wasm"))]
    {
        let rows_per_worker = _height.div_ceil(workers);
        let samples_per_chunk = rows_per_worker.saturating_mul(width).saturating_mul(4);
        let alpha_width = alpha_plane.layout.width;
        let alpha_height = alpha_plane.layout.height;
        let alpha_subsampling_x = alpha_plane.layout.subsampling_x;
        let alpha_subsampling_y = alpha_plane.layout.subsampling_y;
        let alpha_samples = &alpha_plane.samples;
        std::thread::scope(|scope| {
            for (chunk_index, chunk) in rgba.chunks_mut(samples_per_chunk).enumerate() {
                let first_row = chunk_index * rows_per_worker;
                scope.spawn(move || {
                    apply_alpha_rows(
                        chunk,
                        first_row,
                        width,
                        alpha_width,
                        alpha_height,
                        alpha_subsampling_x,
                        alpha_subsampling_y,
                        alpha_samples,
                        bit_depth,
                    );
                });
            }
        });
    }
}

pub(super) fn apply_alpha_rows(
    rgba: &mut [u8],
    first_row: usize,
    width: usize,
    alpha_width: usize,
    alpha_height: usize,
    alpha_subsampling_x: u8,
    alpha_subsampling_y: u8,
    alpha_samples: &[u16],
    bit_depth: u8,
) {
    let row_count = rgba.len() / width.max(1) / 4;
    for local_y in 0..row_count {
        let y = first_row + local_y;
        let alpha_y = (y >> usize::from(alpha_subsampling_y)).min(alpha_height.saturating_sub(1));
        let row = &mut rgba[local_y * width * 4..(local_y + 1) * width * 4];
        for (x, pixel) in row.chunks_exact_mut(4).enumerate() {
            let alpha_x =
                (x >> usize::from(alpha_subsampling_x)).min(alpha_width.saturating_sub(1));
            pixel[3] =
                scale_sample_to_u8(alpha_samples[alpha_y * alpha_width + alpha_x], bit_depth);
        }
    }
}

fn scale_sample_to_u8(sample: u16, bit_depth: u8) -> u8 {
    let maximum = (1u32 << bit_depth) - 1;
    ((u32::from(sample) * 255 + maximum / 2) / maximum) as u8
}
