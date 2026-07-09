use super::TileDecoder;
use crate::DecoderError;
use crate::av1::decode::PlaneBuffer;
use crate::av1::transform::TransformBlock;

impl<'a> TileDecoder<'a> {
    pub(super) fn reconstructed_extension_availability(
        &self,
        plane: &PlaneBuffer,
        transform: TransformBlock,
    ) -> Result<(bool, bool), DecoderError> {
        if usize::from(plane.layout.plane) != transform.plane {
            return Err(DecoderError::Bitstream(
                "AV1 reconstruction coverage plane does not match transform plane".to_string(),
            ));
        }
        let grid = self
            .reconstructed_mi_grid
            .get(transform.plane)
            .ok_or_else(|| {
                DecoderError::Bitstream("AV1 reconstruction coverage plane is invalid".to_string())
            })?;
        let width = transform.tx_size.width();
        let height = transform.tx_size.height();
        let top_right_available = if transform.y == 0 {
            true
        } else {
            let start = transform.x.saturating_add(width).min(plane.layout.width);
            let end = start.saturating_add(height).min(plane.layout.width);
            horizontal_span_is_reconstructed(
                grid,
                self.mi_cols,
                self.mi_rows,
                transform.y - 1,
                start,
                end,
            )
        };
        let bottom_left_available = if transform.x == 0 {
            true
        } else {
            let start = transform.y.saturating_add(height).min(plane.layout.height);
            let end = start.saturating_add(width).min(plane.layout.height);
            vertical_span_is_reconstructed(
                grid,
                self.mi_cols,
                self.mi_rows,
                transform.x - 1,
                start,
                end,
            )
        };
        Ok((top_right_available, bottom_left_available))
    }

    pub(super) fn mark_reconstructed_transform(
        &mut self,
        transform: TransformBlock,
    ) -> Result<(), DecoderError> {
        let grid = self
            .reconstructed_mi_grid
            .get_mut(transform.plane)
            .ok_or_else(|| {
                DecoderError::Bitstream("AV1 reconstruction coverage plane is invalid".to_string())
            })?;
        let start_col = transform.x >> 2;
        let start_row = transform.y >> 2;
        let end_col = transform
            .x
            .saturating_add(transform.tx_size.width())
            .min(self.mi_cols << 2)
            .saturating_add(3)
            >> 2;
        let end_row = transform
            .y
            .saturating_add(transform.tx_size.height())
            .min(self.mi_rows << 2)
            .saturating_add(3)
            >> 2;
        for mi_row in start_row..end_row.min(self.mi_rows) {
            for mi_col in start_col..end_col.min(self.mi_cols) {
                grid[mi_row * self.mi_cols + mi_col] = true;
            }
        }
        Ok(())
    }
}

fn horizontal_span_is_reconstructed(
    grid: &[bool],
    mi_cols: usize,
    mi_rows: usize,
    y: usize,
    start_x: usize,
    end_x: usize,
) -> bool {
    if start_x >= end_x {
        return true;
    }
    let mi_row = y >> 2;
    if mi_row >= mi_rows {
        return true;
    }
    let start_col = start_x >> 2;
    let end_col = (end_x - 1) >> 2;
    (start_col..=end_col).all(|mi_col| {
        mi_col < mi_cols
            && grid
                .get(mi_row * mi_cols + mi_col)
                .copied()
                .unwrap_or(false)
    })
}

fn vertical_span_is_reconstructed(
    grid: &[bool],
    mi_cols: usize,
    mi_rows: usize,
    x: usize,
    start_y: usize,
    end_y: usize,
) -> bool {
    if start_y >= end_y {
        return true;
    }
    let mi_col = x >> 2;
    if mi_col >= mi_cols {
        return true;
    }
    let start_row = start_y >> 2;
    let end_row = (end_y - 1) >> 2;
    (start_row..=end_row).all(|mi_row| {
        mi_row < mi_rows
            && grid
                .get(mi_row * mi_cols + mi_col)
                .copied()
                .unwrap_or(false)
    })
}
