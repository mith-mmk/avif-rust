//! Normative inter-intra masks weight the intra predictor, then inter.
use super::*;
const WEIGHTS: [u8; 128] = [
    60, 58, 56, 54, 52, 50, 48, 47, 45, 44, 42, 41, 39, 38, 37, 35, 34, 33, 32, 31, 30, 29, 28, 27,
    26, 25, 24, 23, 22, 22, 21, 20, 19, 19, 18, 18, 17, 16, 16, 15, 15, 14, 14, 13, 13, 12, 12, 12,
    11, 11, 10, 10, 10, 9, 9, 9, 8, 8, 8, 8, 7, 7, 7, 7, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 4, 4, 4, 4,
    4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
];

fn mask(
    mode: InterIntraMode,
    wedge: Option<u8>,
    luma_width: usize,
    luma_height: usize,
    plane_width: usize,
    plane_height: usize,
    x: usize,
    y: usize,
    ssx: usize,
    ssy: usize,
) -> u8 {
    if let Some(index) = wedge {
        let mut sum = 0u32;
        for dy in 0..(1 << ssy) {
            for dx in 0..(1 << ssx) {
                sum += u32::from(
                    wedge_mask_value(
                        luma_width,
                        luma_height,
                        index,
                        false,
                        (x << ssx) + dx,
                        (y << ssy) + dy,
                    )
                    .unwrap_or(32),
                );
            }
        }
        let shift = ssx + ssy;
        return ((sum + ((1 << shift) / 2)) >> shift) as u8;
    }
    let scale = 128 / plane_width.max(plane_height);
    let position = match mode {
        InterIntraMode::Dc => return 32,
        InterIntraMode::Vertical => y,
        InterIntraMode::Horizontal => x,
        InterIntraMode::Smooth => x.min(y),
    };
    WEIGHTS[(position * scale).min(127)]
}

fn blend(inter: u16, intra: u16, mask: u8) -> u16 {
    ((u32::from(intra) * u32::from(mask) + u32::from(inter) * u32::from(64 - mask) + 32) >> 6)
        as u16
}

pub(super) fn apply(
    plane: &PlaneBuffer,
    output: &mut [u16],
    mode: InterIntraMode,
    wedge: Option<u8>,
    luma_width: usize,
    luma_height: usize,
    block_x: usize,
    block_y: usize,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    geometry: MaskGeometry,
    bit_depth: u8,
    ssx: usize,
    ssy: usize,
    enable_intra_edge_filter: bool,
    tile_left: usize,
    tile_top: usize,
    tile_right: usize,
    tile_bottom: usize,
    scratch: &mut [u16],
) -> Result<(), DecoderError> {
    let full_len = geometry.width * geometry.height;
    if scratch.len() < full_len {
        return Err(DecoderError::Bitstream(
            "AV1 inter-intra full-block scratch is too small".into(),
        ));
    }
    let intra = &mut scratch[..full_len];
    // Generate from the original prediction-block edges, never the edges of
    // a transform subdivision that may already contain reconstructed pixels.
    let prediction_mode = match mode {
        InterIntraMode::Dc => PredictionMode::Dc,
        InterIntraMode::Vertical => PredictionMode::Vertical,
        InterIntraMode::Horizontal => PredictionMode::Horizontal,
        InterIntraMode::Smooth => PredictionMode::Smooth,
    };
    if prediction_mode == PredictionMode::Dc {
        predict_dc_block_into_with_bounds(
            plane,
            block_x,
            block_y,
            geometry.width,
            geometry.height,
            bit_depth,
            tile_left,
            tile_top,
            tile_right,
            tile_bottom,
            intra,
        );
    } else {
        predict_block_into(
            plane,
            prediction_mode,
            block_x,
            block_y,
            geometry.width,
            geometry.height,
            None,
            None,
            bit_depth,
            enable_intra_edge_filter,
            false,
            0,
            0,
            tile_left,
            tile_top,
            tile_right,
            tile_bottom,
            intra,
        )?;
    }
    debug_assert_eq!((x - block_x, y - block_y), (geometry.x, geometry.y));
    for row in 0..height {
        for col in 0..width {
            let px = geometry.x + col;
            let py = geometry.y + row;
            let weight = mask(
                mode,
                wedge,
                luma_width,
                luma_height,
                geometry.width,
                geometry.height,
                px,
                py,
                ssx,
                ssy,
            );
            let index = row * width + col;
            output[index] = blend(output[index], intra[py * geometry.width + px], weight);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn interintra_fixture_wedge_weights_intra_predictor() {
        // AOM coded order16 block(8,16): inter=58, intra=31, wedge7[0]=64.
        // av1_combine_interintra produces31; reversing sources produces58.
        assert_eq!(
            mask(InterIntraMode::Horizontal, Some(7), 8, 8, 8, 8, 0, 0, 0, 0),
            64
        );
        assert_eq!(blend(58, 31, 64), 31);
    }
    #[test]
    fn interintra_masks_and_blends_match_aom_8_10_12_bit_oracles() {
        // Official av1_combine_interintra (8x8 luma / 4x4 4:2:0 chroma),
        // inter=58<<depth_shift, intra=31<<depth_shift, all four II modes
        // and wedge7. FNV-1a covers every output u16, not its byte encoding.
        let expected: [[[u64; 5]; 2]; 3] = [
            [
                [
                    17773294517074453605,
                    13743514577608875221,
                    251060901023615269,
                    9985555746663100123,
                    1731459518775907525,
                ],
                [
                    10851480456372878389,
                    16543672966490683529,
                    13288006801299430045,
                    14444681750731398780,
                    11238253970142237533,
                ],
            ],
            [
                [
                    11540505601218694949,
                    12116611073575186205,
                    13692712905426595941,
                    17006519946833284156,
                    8515583090582117477,
                ],
                [
                    11995621011786853925,
                    12809579193125248981,
                    16857918168159783197,
                    4810154649416701939,
                    3093877064971750949,
                ],
            ],
            [
                [
                    7968595847954535461,
                    4649877931545715869,
                    11051565424527198565,
                    1007208993969394112,
                    9563261701136650613,
                ],
                [
                    10471211574060387173,
                    15178651659331754637,
                    8912677777981622317,
                    8049080304009495751,
                    7272865833399135653,
                ],
            ],
        ];
        for (depth, expected_planes) in [8u8, 10, 12].into_iter().zip(expected) {
            for (chroma, expected_modes) in expected_planes.into_iter().enumerate() {
                let size = 8 >> chroma;
                for (mi, expected_hash) in expected_modes.into_iter().enumerate() {
                    let mode = match mi {
                        0 => InterIntraMode::Dc,
                        1 => InterIntraMode::Vertical,
                        3 => InterIntraMode::Smooth,
                        _ => InterIntraMode::Horizontal,
                    };
                    let wedge = (mi == 4).then_some(7);
                    let mut hash = 14695981039346656037u64;
                    for y in 0..size {
                        for x in 0..size {
                            let weight = mask(mode, wedge, 8, 8, size, size, x, y, chroma, chroma);
                            let value = blend(58 << (depth - 8), 31 << (depth - 8), weight);
                            hash = (hash ^ u64::from(value)).wrapping_mul(1099511628211);
                        }
                    }
                    assert_eq!(
                        hash, expected_hash,
                        "AOM II depth={depth} chroma={chroma} mode={mi}"
                    );
                }
            }
        }
    }

    #[test]
    fn interintra_transform_subdivisions_keep_original_block_edges() {
        use crate::av1::decode::PlaneLayout;
        let mut plane = PlaneBuffer {
            layout: PlaneLayout {
                plane: 0,
                width: 16,
                height: 16,
                subsampling_x: 0,
                subsampling_y: 0,
                sample_count: 256,
            },
            samples: (0..256)
                .map(|i| (23 + (i % 16) * 3 + (i / 16) * 5) as u16)
                .collect(),
        };
        let mut scratch = [0u16; 4096];
        for mode in [
            InterIntraMode::Dc,
            InterIntraMode::Vertical,
            InterIntraMode::Horizontal,
            InterIntraMode::Smooth,
        ] {
            let mut full = [100u16; 64];
            apply(
                &plane,
                &mut full,
                mode,
                None,
                8,
                8,
                4,
                4,
                4,
                4,
                8,
                8,
                MaskGeometry {
                    x: 0,
                    y: 0,
                    width: 8,
                    height: 8,
                },
                8,
                0,
                0,
                false,
                0,
                0,
                16,
                16,
                &mut scratch,
            )
            .unwrap();
            // Previously reconstructed transforms inside the block must never
            // become the intra reference edges of another transform.
            for y in 4..12 {
                for x in 4..12 {
                    plane.samples[y * 16 + x] = 240;
                }
            }
            for oy in [0, 4] {
                for ox in [0, 4] {
                    let mut fragment = [100u16; 16];
                    apply(
                        &plane,
                        &mut fragment,
                        mode,
                        None,
                        8,
                        8,
                        4,
                        4,
                        4 + ox,
                        4 + oy,
                        4,
                        4,
                        MaskGeometry {
                            x: ox,
                            y: oy,
                            width: 8,
                            height: 8,
                        },
                        8,
                        0,
                        0,
                        false,
                        0,
                        0,
                        16,
                        16,
                        &mut scratch,
                    )
                    .unwrap();
                    for row in 0..4 {
                        assert_eq!(
                            &fragment[row * 4..row * 4 + 4],
                            &full[(oy + row) * 8 + ox..(oy + row) * 8 + ox + 4]
                        );
                    }
                }
            }
        }
    }
}
