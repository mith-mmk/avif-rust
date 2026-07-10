use super::TileDecoder;
use crate::av1::frame::FrameHeader;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CdefUnit {
    pub(crate) x: usize,
    pub(crate) y: usize,
    pub(crate) index: u32,
}

impl<'a> TileDecoder<'a> {
    pub(super) fn record_cdef_index(
        &mut self,
        frame: &FrameHeader,
        x: usize,
        y: usize,
        index: Option<u32>,
    ) {
        if !frame.cdef.enabled || frame.allow_intrabc {
            return;
        }

        let (unit_x, unit_y) = cdef_unit_origin(x, y);
        store_cdef_unit(&mut self.cdef_units, unit_x, unit_y, index);
    }
}

fn cdef_unit_origin(x: usize, y: usize) -> (usize, usize) {
    (x & !63, y & !63)
}

fn store_cdef_unit(units: &mut Vec<CdefUnit>, x: usize, y: usize, index: Option<u32>) {
    if let Some(unit) = units.iter_mut().find(|unit| unit.x == x && unit.y == y) {
        if let Some(index) = index {
            unit.index = index;
        }
        return;
    }
    units.push(CdefUnit {
        x,
        y,
        index: index.unwrap_or(0),
    });
}

#[cfg(test)]
mod tests {
    use super::{CdefUnit, cdef_unit_origin, store_cdef_unit};

    #[test]
    fn cdef_unit_is_addressed_on_a_64_pixel_grid() {
        let (x, y) = cdef_unit_origin(126, 191);
        let unit = CdefUnit { x, y, index: 3 };
        assert_eq!((unit.x, unit.y, unit.index), (64, 128, 3));
    }

    #[test]
    fn later_transmitted_index_replaces_skipped_unit_default() {
        let mut units = Vec::new();
        store_cdef_unit(&mut units, 0, 0, None);
        store_cdef_unit(&mut units, 0, 0, Some(2));
        assert_eq!(
            units,
            vec![CdefUnit {
                x: 0,
                y: 0,
                index: 2
            }]
        );
    }
}
