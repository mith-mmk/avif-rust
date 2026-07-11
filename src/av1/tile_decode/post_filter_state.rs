use super::TileDecoder;
use crate::av1::frame::FrameHeader;
use crate::av1::syntax::TxType;
use crate::av1::tile_decode::DecodedLumaBlock;
use crate::av1::transform::TransformBlock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CdefUnit {
    pub(crate) x: usize,
    pub(crate) y: usize,
    pub(crate) index: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TransformBoundary {
    pub(crate) block: TransformBlock,
    pub(crate) tx_type: TxType,
    pub(crate) non_zero_coefficients: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct PostFilterState {
    pub(crate) cdef_units: Vec<CdefUnit>,
    pub(crate) transform_boundaries: Vec<TransformBoundary>,
}

impl PostFilterState {
    pub(crate) fn merge(&mut self, other: Self) {
        for unit in other.cdef_units {
            if let Some(existing) = self
                .cdef_units
                .iter_mut()
                .find(|existing| existing.x == unit.x && existing.y == unit.y)
            {
                existing.index = unit.index;
            } else {
                self.cdef_units.push(unit);
            }
        }
        for boundary in other.transform_boundaries {
            if let Some(existing) = self
                .transform_boundaries
                .iter_mut()
                .find(|existing| existing.block == boundary.block)
            {
                *existing = boundary;
            } else {
                self.transform_boundaries.push(boundary);
            }
        }
    }

    pub(crate) fn record_luma_blocks(&mut self, blocks: &[DecodedLumaBlock]) {
        for block in blocks {
            for transform in &block.transforms {
                let boundary = TransformBoundary {
                    block: transform.transform,
                    tx_type: transform.tx_type,
                    non_zero_coefficients: transform
                        .coefficients
                        .iter()
                        .filter(|coefficient| **coefficient != 0)
                        .count(),
                };
                if !self
                    .transform_boundaries
                    .iter()
                    .any(|existing| existing.block == boundary.block)
                {
                    self.transform_boundaries.push(boundary);
                }
            }
        }
    }
}

impl<'a> TileDecoder<'a> {
    pub(super) fn take_post_filter_state(self) -> PostFilterState {
        PostFilterState {
            cdef_units: self.cdef_units,
            transform_boundaries: Vec::new(),
        }
    }

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
    use super::{CdefUnit, PostFilterState, cdef_unit_origin, store_cdef_unit};

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

    #[test]
    fn post_filter_state_merges_tile_units_by_origin() {
        let mut state = PostFilterState {
            cdef_units: vec![CdefUnit {
                x: 0,
                y: 0,
                index: 1,
            }],
            transform_boundaries: Vec::new(),
        };
        state.merge(PostFilterState {
            cdef_units: vec![
                CdefUnit {
                    x: 0,
                    y: 0,
                    index: 3,
                },
                CdefUnit {
                    x: 64,
                    y: 0,
                    index: 2,
                },
            ],
            transform_boundaries: Vec::new(),
        });

        assert_eq!(
            state.cdef_units,
            vec![
                CdefUnit {
                    x: 0,
                    y: 0,
                    index: 3,
                },
                CdefUnit {
                    x: 64,
                    y: 0,
                    index: 2,
                },
            ]
        );
    }
}
