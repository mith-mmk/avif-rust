use super::TileDecoder;
use crate::DecoderError;
use crate::av1::decode::PlaneBuffer;
use crate::av1::transform::TransformBlock;

impl<'a> TileDecoder<'a> {
    pub(super) fn reconstructed_extension_availability(
        &self,
        plane: &PlaneBuffer,
        transform: TransformBlock,
    ) -> Result<(usize, usize), DecoderError> {
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
        let subsampling_x = self.plane_subsampling_x[transform.plane];
        let subsampling_y = self.plane_subsampling_y[transform.plane];
        let top_right_available = if transform.y == 0 {
            0
        } else {
            let start = transform.x.saturating_add(width).min(plane.layout.width);
            horizontal_reconstructed_length(
                grid,
                self.mi_cols,
                self.mi_rows,
                (transform.y << subsampling_y).saturating_sub(1),
                subsampling_x,
                start,
                width,
                plane.layout.width,
            )
        };
        let bottom_left_available = if transform.x == 0 {
            0
        } else {
            let start = transform.y.saturating_add(height).min(plane.layout.height);
            vertical_reconstructed_length(
                grid,
                self.mi_cols,
                (transform.x << subsampling_x).saturating_sub(1),
                subsampling_y,
                start,
                height,
                plane.layout.height,
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
        let subsampling_x = self.plane_subsampling_x[transform.plane];
        let subsampling_y = self.plane_subsampling_y[transform.plane];
        let start_col = (transform.x << subsampling_x) >> 2;
        let start_row = (transform.y << subsampling_y) >> 2;
        let end_col = transform
            .x
            .saturating_add(transform.tx_size.width())
            .saturating_mul(1usize << subsampling_x)
            .min(self.mi_cols << 2)
            .saturating_add(3)
            >> 2;
        let end_row = transform
            .y
            .saturating_add(transform.tx_size.height())
            .saturating_mul(1usize << subsampling_y)
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

#[expect(
    clippy::too_many_arguments,
    reason = "coverage scan keeps coded-grid and plane-coordinate domains explicit"
)]
fn horizontal_reconstructed_length(
    grid: &[bool],
    mi_cols: usize,
    mi_rows: usize,
    y: usize,
    subsampling_x: usize,
    start_x: usize,
    maximum: usize,
    plane_width: usize,
) -> usize {
    let mi_row = y >> 2;
    if mi_row >= mi_rows {
        return 0;
    }
    (0..maximum)
        .take_while(|offset| {
            let x = start_x + offset;
            x < plane_width
                && grid
                    .get(mi_row * mi_cols + ((x << subsampling_x) >> 2))
                    .copied()
                    .unwrap_or(false)
        })
        .count()
}

fn vertical_reconstructed_length(
    grid: &[bool],
    mi_cols: usize,
    x: usize,
    subsampling_y: usize,
    start_y: usize,
    maximum: usize,
    plane_height: usize,
) -> usize {
    let mi_col = x >> 2;
    if mi_col >= mi_cols {
        return 0;
    }
    (0..maximum)
        .take_while(|offset| {
            let y = start_y + offset;
            y < plane_height
                && grid
                    .get(((y << subsampling_y) >> 2) * mi_cols + mi_col)
                    .copied()
                    .unwrap_or(false)
        })
        .count()
}
