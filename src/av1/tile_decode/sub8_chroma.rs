//! Shared sub8 chroma uses each contributing luma block's ref, MV and filter.
use std::sync::Arc;

use super::super::TileDecoder;
use super::super::context_state::Sub8ChromaOwner;
use super::{floor_div_sixteen, motion_component_q4, predict_inter_sample};
use crate::DecoderError;
use crate::av1::decode::FrameBuffers;
use crate::av1::syntax::BlockSize;

pub(super) struct OwnerPrediction {
    owner: Sub8ChromaOwner,
    reference: Arc<FrameBuffers>,
}

pub(super) type PredictionPlan = [Option<OwnerPrediction>; 4];

pub(super) fn plan(
    decoder: &TileDecoder<'_>,
    x: usize,
    y: usize,
    block_size: BlockSize,
    subsampling_x: usize,
    subsampling_y: usize,
) -> Result<Option<PredictionPlan>, DecoderError> {
    let Some(owners) = decoder.sub8_chroma_owners(x, y, block_size, subsampling_x, subsampling_y)
    else {
        return Ok(None);
    };
    let mut predictions = std::array::from_fn(|_| None);
    for (target, owner) in predictions.iter_mut().zip(owners) {
        if let Some(owner) = owner {
            *target = Some(OwnerPrediction {
                reference: decoder.reference_buffer(owner.reference_frame)?,
                owner,
            });
        }
    }
    Ok(Some(predictions))
}

pub(super) fn predict(
    plan: &PredictionPlan,
    plane_index: usize,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    subsampling_x: usize,
    subsampling_y: usize,
    bit_depth: u8,
    output: &mut [u16],
) -> Result<(), DecoderError> {
    for row in 0..height {
        for col in 0..width {
            let px = x + col;
            let py = y + row;
            let prediction = plan
                .iter()
                .flatten()
                .find(|prediction| {
                    let owner = prediction.owner;
                    px >= owner.x
                        && px < owner.x + owner.width
                        && py >= owner.y
                        && py < owner.y + owner.height
                })
                .ok_or_else(|| {
                    DecoderError::Bitstream("AV1 shared chroma prediction has no luma owner".into())
                })?;
            let owner = prediction.owner;
            let plane = prediction
                .reference
                .planes
                .get(plane_index)
                .ok_or_else(|| {
                    DecoderError::Unsupported(
                        "AV1 shared chroma reference plane is unavailable".into(),
                    )
                })?;
            let fixed_x =
                (px as i64) * 16 + motion_component_q4(owner.motion_vector.1, subsampling_x);
            let fixed_y =
                (py as i64) * 16 + motion_component_q4(owner.motion_vector.0, subsampling_y);
            let source_x = floor_div_sixteen(fixed_x);
            let source_y = floor_div_sixteen(fixed_y);
            // Integer motion needs no filter work, including at replicated edges.
            if fixed_x == source_x * 16 && fixed_y == source_y * 16 {
                let sx = source_x.clamp(0, plane.layout.width.saturating_sub(1) as i64) as usize;
                let sy = source_y.clamp(0, plane.layout.height.saturating_sub(1) as i64) as usize;
                output[row * width + col] = plane.samples[sy * plane.layout.width + sx];
                continue;
            }
            output[row * width + col] = predict_inter_sample(
                plane,
                source_x,
                source_y,
                fixed_x - source_x * 16,
                fixed_y - source_y * 16,
                owner.interpolation_filters,
                bit_depth,
                (owner.width, owner.height),
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av1::decode::{PlaneBuffer, PlaneLayout};
    use crate::av1::frame::InterpolationFilter::Regular;

    #[test]
    fn sub8_chroma_owner_prediction_matches_aom_regions() {
        // AOM reconinter_template.inc builds four 2x2 regions for shared
        // 4:2:0 chroma. Integer luma MVs 0,16,32,48 shift chroma by 0,1,2,3.
        let plane = PlaneBuffer {
            layout: PlaneLayout {
                plane: 1,
                width: 32,
                height: 32,
                subsampling_x: 1,
                subsampling_y: 1,
                sample_count: 1024,
            },
            samples: (0..1024).map(|i| (i % 32) as u16).collect(),
        };
        let reference = Arc::new(FrameBuffers {
            width: 64,
            height: 64,
            planes: vec![plane.clone(), plane],
        });
        let plan: PredictionPlan = std::array::from_fn(|i| {
            Some(OwnerPrediction {
                owner: Sub8ChromaOwner {
                    x: 8 + (i % 2) * 2,
                    y: 8 + (i / 2) * 2,
                    width: 2,
                    height: 2,
                    reference_frame: 0,
                    motion_vector: (0, (i as i32) * 16),
                    interpolation_filters: (Regular, Regular),
                },
                reference: Arc::clone(&reference),
            })
        });
        let mut output = [0; 16];
        predict(&plan, 1, 8, 8, 4, 4, 1, 1, 8, &mut output).unwrap();
        assert_eq!(
            output,
            [8, 9, 11, 12, 8, 9, 11, 12, 10, 11, 13, 14, 10, 11, 13, 14]
        );
        // The previous whole-block path uses the lower-right owner's MV for
        // every quadrant; demonstrate that its output fails this oracle.
        let mut previous = [0; 16];
        super::super::predict_inter_block_into(
            &reference.planes[1],
            None,
            8,
            8,
            4,
            4,
            (0, 48),
            None,
            1,
            1,
            (Regular, Regular),
            None,
            None,
            8,
            super::super::MaskGeometry {
                x: 0,
                y: 0,
                width: 4,
                height: 4,
            },
            &mut previous,
        )
        .unwrap();
        assert_eq!(
            previous,
            [
                11, 12, 13, 14, 11, 12, 13, 14, 11, 12, 13, 14, 11, 12, 13, 14
            ]
        );
        assert_ne!(previous, output);
        // A transform subdivision must select the same owners and filters.
        let mut right = [0; 8];
        predict(&plan, 1, 10, 8, 2, 4, 1, 1, 8, &mut right).unwrap();
        assert_eq!(right, [11, 12, 11, 12, 13, 14, 13, 14]);
        let mut alternate = reference.as_ref().clone();
        alternate.planes[1].samples.fill(200);
        let mut plan = plan;
        plan[0].as_mut().unwrap().reference = Arc::new(alternate);
        predict(&plan, 1, 8, 8, 4, 4, 1, 1, 8, &mut output).unwrap();
        assert_eq!(&output[..8], &[200, 200, 11, 12, 200, 200, 11, 12]);
    }
}
