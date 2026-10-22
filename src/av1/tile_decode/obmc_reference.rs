//! Fixed, stack-owned handles to the primary references of OBMC neighbors.
use super::*;
use std::sync::Arc;

pub(in crate::av1::tile_decode) struct Plan {
    references: [Option<Arc<FrameBuffers>>; 8],
}

impl Plan {
    pub(super) fn new(
        decoder: &TileDecoder<'_>,
        neighbors: &ObmcNeighbors,
    ) -> Result<Self, DecoderError> {
        let mut references = std::array::from_fn(|_| None);
        for neighbor in neighbors
            .above
            .iter()
            .chain(neighbors.left.iter())
            .flatten()
        {
            let slot = usize::from(neighbor.reference_frame);
            let target = references.get_mut(slot).ok_or_else(|| {
                DecoderError::Bitstream("AV1 OBMC reference slot is invalid".into())
            })?;
            if target.is_none() {
                *target = Some(decoder.reference_buffer(neighbor.reference_frame)?);
            }
        }
        Ok(Self { references })
    }
    pub(super) fn plane(&self, slot: u8, index: usize) -> Result<&PlaneBuffer, DecoderError> {
        self.references
            .get(usize::from(slot))
            .and_then(Option::as_ref)
            .and_then(|frame| frame.planes.get(index))
            .ok_or_else(|| {
                DecoderError::Unsupported("AV1 OBMC primary reference plane is unavailable".into())
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn obmc_cross_reference_neighbor_matches_aom_210_126_to_177() {
        // Actual coded order8 block(24,40): currentLAST(order0)=210,
        // leftALTREF(order16,MV32,-12)=126, first4-maskweight39 ->177.
        let plane = |value| PlaneBuffer {
            layout: crate::av1::decode::PlaneLayout {
                plane: 0,
                width: 16,
                height: 16,
                subsampling_x: 0,
                subsampling_y: 0,
                sample_count: 256,
            },
            samples: vec![value; 256],
        };
        let primary = plane(210);
        let other = Arc::new(FrameBuffers {
            width: 16,
            height: 16,
            planes: vec![plane(126)],
        });
        let mut references = std::array::from_fn(|_| None);
        references[1] = Some(other);
        let plan = Plan { references };
        let neighbor = super::super::super::diagnostic::ObmcNeighbor {
            reference_frame: 1,
            origin_x: 0,
            origin_y: 8,
            width: 8,
            height: 8,
            motion_vector: (0, 0),
            interpolation_filters: (InterpolationFilter::Regular, InterpolationFilter::Regular),
        };
        let neighbors = ObmcNeighbors {
            left: [Some(neighbor), None, None, None],
            ..Default::default()
        };
        let mut before = [210; 64];
        let mut output = [210; 64];
        let mut scratch = [0; 64];
        apply_obmc_edge_blend(
            &mut before,
            &primary,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            0,
            0,
            &neighbors,
            &mut scratch,
            8,
            None,
            0,
        )
        .unwrap();
        assert_eq!(
            before[0], 210,
            "previous same-reference fallback ignored the other frame"
        );
        apply_obmc_edge_blend(
            &mut output,
            &primary,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            0,
            0,
            &neighbors,
            &mut scratch,
            8,
            Some(&plan),
            0,
        )
        .unwrap();
        assert_eq!(&output[..8], &[177, 192, 203, 210, 210, 210, 210, 210]);
    }
}
