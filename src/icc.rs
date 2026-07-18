use crate::DecoderError;

const D50_TO_SRGB: [[f64; 3]; 3] = [
    [3.1338561, -1.6168667, -0.4906146],
    [-0.9787684, 1.9161415, 0.0334540],
    [0.0719453, -0.2289914, 1.4052427],
];

const D65_TO_SRGB: [[f64; 3]; 3] = [
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
];

#[derive(Debug, Clone)]
enum Curve {
    Identity,
    Gamma(f64),
    Table(Vec<u16>),
    Parametric { function: u16, parameters: [f64; 7] },
}

#[derive(Debug, Clone)]
struct MatrixShaperProfile {
    to_xyz: [[f64; 3]; 3],
    xyz_to_srgb: [[f64; 3]; 3],
    curves: [Curve; 3],
}

pub(crate) fn apply_to_rgba16(rgba: &mut [u16], profile: &[u8]) -> Result<(), DecoderError> {
    let profile = MatrixShaperProfile::parse(profile)?;
    for pixel in rgba.chunks_exact_mut(4) {
        let source = [
            f64::from(pixel[0]) / f64::from(u16::MAX),
            f64::from(pixel[1]) / f64::from(u16::MAX),
            f64::from(pixel[2]) / f64::from(u16::MAX),
        ];
        let linear = [
            profile.curves[0].decode(source[0]),
            profile.curves[1].decode(source[1]),
            profile.curves[2].decode(source[2]),
        ];
        let xyz = multiply(profile.to_xyz, linear);
        let rgb = multiply(profile.xyz_to_srgb, xyz);
        pixel[0] = encode_srgb(rgb[0]);
        pixel[1] = encode_srgb(rgb[1]);
        pixel[2] = encode_srgb(rgb[2]);
    }
    Ok(())
}

impl Curve {
    fn decode(&self, value: f64) -> f64 {
        match self {
            Self::Identity => value,
            Self::Gamma(gamma) => value.powf(*gamma),
            Self::Table(table) => {
                let position = value.clamp(0.0, 1.0) * (table.len() - 1) as f64;
                let index = position.floor() as usize;
                let next = (index + 1).min(table.len() - 1);
                let fraction = position - index as f64;
                let low = f64::from(table[index]);
                let high = f64::from(table[next]);
                (low + (high - low) * fraction) / f64::from(u16::MAX)
            }
            Self::Parametric {
                function,
                parameters,
            } => decode_parametric(*function, *parameters, value),
        }
    }
}

impl MatrixShaperProfile {
    fn parse(profile: &[u8]) -> Result<Self, DecoderError> {
        if profile.len() < 132 {
            return Err(unsupported("ICC profile header is truncated"));
        }
        let declared_size = read_u32(profile, 0)? as usize;
        if declared_size < 132 || declared_size > profile.len() {
            return Err(unsupported("ICC profile size is invalid"));
        }
        if &profile[12..16] != b"mntr" && &profile[12..16] != b"prtr" {
            return Err(unsupported("ICC profile device class is not supported"));
        }
        if &profile[16..20] != b"RGB " || &profile[20..24] != b"XYZ " {
            return Err(unsupported("ICC profile colour space is not RGB/XYZ"));
        }

        let tag_count = read_u32(profile, 128)? as usize;
        let table_end = 132usize
            .checked_add(
                tag_count
                    .checked_mul(12)
                    .ok_or_else(|| unsupported("ICC profile tag table is too large"))?,
            )
            .ok_or_else(|| unsupported("ICC profile tag table overflows"))?;
        if table_end > declared_size {
            return Err(unsupported("ICC profile tag table is truncated"));
        }
        let mut tags = [None; 7];
        for index in 0..tag_count {
            let entry = 132 + index * 12;
            let signature = &profile[entry..entry + 4];
            let offset = read_u32(profile, entry + 4)? as usize;
            let size = read_u32(profile, entry + 8)? as usize;
            let end = offset
                .checked_add(size)
                .ok_or_else(|| unsupported("ICC profile tag overflows"))?;
            if offset < 132 || end > declared_size || size < 8 {
                return Err(unsupported("ICC profile tag is outside the profile"));
            }
            let slot = match signature {
                b"wtpt" => Some(0),
                b"rXYZ" => Some(1),
                b"gXYZ" => Some(2),
                b"bXYZ" => Some(3),
                b"rTRC" => Some(4),
                b"gTRC" => Some(5),
                b"bTRC" => Some(6),
                _ => None,
            };
            if let Some(slot) = slot {
                tags[slot] = Some((offset, size));
            }
        }
        let white = read_xyz(profile, required_tag(tags[0], "wtpt")?)?;
        let xyz_to_srgb = if close_to_white(white, [0.9642, 1.0, 0.8249]) {
            D50_TO_SRGB
        } else if close_to_white(white, [0.9505, 1.0, 1.0890]) {
            D65_TO_SRGB
        } else {
            return Err(unsupported("ICC profile white point is not D50 or D65"));
        };
        let red = read_xyz(profile, required_tag(tags[1], "rXYZ")?)?;
        let green = read_xyz(profile, required_tag(tags[2], "gXYZ")?)?;
        let blue = read_xyz(profile, required_tag(tags[3], "bXYZ")?)?;
        Ok(Self {
            to_xyz: [
                [red[0], green[0], blue[0]],
                [red[1], green[1], blue[1]],
                [red[2], green[2], blue[2]],
            ],
            xyz_to_srgb,
            curves: [
                read_curve(profile, required_tag(tags[4], "rTRC")?)?,
                read_curve(profile, required_tag(tags[5], "gTRC")?)?,
                read_curve(profile, required_tag(tags[6], "bTRC")?)?,
            ],
        })
    }
}

fn required_tag(tag: Option<(usize, usize)>, name: &str) -> Result<(usize, usize), DecoderError> {
    tag.ok_or_else(|| unsupported(format!("ICC profile tag {name} is missing")))
}

fn read_xyz(profile: &[u8], tag: (usize, usize)) -> Result<[f64; 3], DecoderError> {
    let (offset, size) = tag;
    if size < 20 || &profile[offset..offset + 4] != b"XYZ " {
        return Err(unsupported("ICC XYZ tag is invalid"));
    }
    Ok([
        read_s15_fixed16(profile, offset + 8)?,
        read_s15_fixed16(profile, offset + 12)?,
        read_s15_fixed16(profile, offset + 16)?,
    ])
}

fn read_curve(profile: &[u8], tag: (usize, usize)) -> Result<Curve, DecoderError> {
    let (offset, size) = tag;
    if size < 12 {
        return Err(unsupported("ICC tone curve tag is truncated"));
    }
    let signature = profile
        .get(offset..offset + 4)
        .ok_or_else(|| unsupported("ICC tone curve tag is outside the profile"))?;
    if signature == b"para" {
        return read_parametric_curve(profile, offset, size);
    }
    if signature != b"curv" {
        return Err(unsupported(
            "ICC tone curve is not a supported curv/para tag",
        ));
    }
    let count = read_u32(profile, offset + 8)?;
    match count {
        0 => Ok(Curve::Identity),
        1 if size >= 14 => Ok(Curve::Gamma(
            f64::from(read_u16(profile, offset + 12)?) / 256.0,
        )),
        count => {
            const MAX_CURVE_ENTRIES: usize = 65_536;
            let count = usize::try_from(count)
                .map_err(|_| unsupported("ICC tone curve entry count overflows"))?;
            if count > MAX_CURVE_ENTRIES {
                return Err(unsupported("ICC tone curve lookup table is too large"));
            }
            let table_bytes = count
                .checked_mul(2)
                .and_then(|bytes| 12usize.checked_add(bytes))
                .ok_or_else(|| unsupported("ICC tone curve lookup table size overflows"))?;
            if size < table_bytes {
                return Err(unsupported("ICC tone curve lookup table is truncated"));
            }
            let mut table = Vec::with_capacity(count);
            for index in 0..count {
                table.push(read_u16(profile, offset + 12 + index * 2)?);
            }
            Ok(Curve::Table(table))
        }
    }
}

fn read_parametric_curve(
    profile: &[u8],
    offset: usize,
    size: usize,
) -> Result<Curve, DecoderError> {
    let function = read_u16(profile, offset + 8)?;
    let parameter_count = match function {
        0 => 1,
        1 => 3,
        2 => 4,
        3 => 5,
        4 => 7,
        _ => {
            return Err(unsupported(format!(
                "ICC parametric tone curve function {function} is not supported"
            )));
        }
    };
    let required_size = 12usize
        .checked_add(parameter_count * 4)
        .ok_or_else(|| unsupported("ICC parametric tone curve size overflows"))?;
    if size < required_size {
        return Err(unsupported("ICC parametric tone curve is truncated"));
    }
    let mut parameters = [0.0; 7];
    for (index, parameter) in parameters.iter_mut().take(parameter_count).enumerate() {
        *parameter = read_s15_fixed16(profile, offset + 12 + index * 4)?;
    }
    if matches!(function, 1 | 2) && parameters[1] == 0.0 {
        return Err(unsupported(
            "ICC parametric tone curve has a zero division coefficient",
        ));
    }
    Ok(Curve::Parametric {
        function,
        parameters,
    })
}

fn decode_parametric(function: u16, parameters: [f64; 7], value: f64) -> f64 {
    let x = value.clamp(0.0, 1.0);
    let gamma = parameters[0];
    let powered = |base: f64| base.max(0.0).powf(gamma);
    let output = match function {
        0 => powered(x),
        1 => {
            let threshold = -parameters[2] / parameters[1];
            if x >= threshold {
                powered(parameters[1] * x + parameters[2])
            } else {
                0.0
            }
        }
        2 => {
            let threshold = -parameters[2] / parameters[1];
            if x >= threshold {
                powered(parameters[1] * x + parameters[2]) + parameters[3]
            } else {
                parameters[3]
            }
        }
        3 => {
            if x >= parameters[4] {
                powered(parameters[1] * x + parameters[2])
            } else {
                parameters[3] * x
            }
        }
        4 => {
            if x >= parameters[4] {
                powered(parameters[1] * x + parameters[2]) + parameters[5]
            } else {
                parameters[3] * x + parameters[6]
            }
        }
        _ => x,
    };
    output.clamp(0.0, 1.0)
}

fn multiply(matrix: [[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    ]
}

fn encode_srgb(value: f64) -> u16 {
    let value = value.clamp(0.0, 1.0);
    let encoded = if value <= 0.0031308 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    };
    (encoded * f64::from(u16::MAX)).round() as u16
}

fn close_to_white(actual: [f64; 3], expected: [f64; 3]) -> bool {
    actual
        .into_iter()
        .zip(expected)
        .all(|(actual, expected)| (actual - expected).abs() <= 0.002)
}

fn read_u16(data: &[u8], offset: usize) -> Result<u16, DecoderError> {
    let bytes = data
        .get(offset..offset + 2)
        .ok_or_else(|| unsupported("ICC profile field is truncated"))?;
    Ok(u16::from_be_bytes([bytes[0], bytes[1]]))
}

fn read_u32(data: &[u8], offset: usize) -> Result<u32, DecoderError> {
    let bytes = data
        .get(offset..offset + 4)
        .ok_or_else(|| unsupported("ICC profile field is truncated"))?;
    Ok(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn read_s15_fixed16(data: &[u8], offset: usize) -> Result<f64, DecoderError> {
    Ok(f64::from(i32::from_be_bytes(read_u32(data, offset)?.to_be_bytes())) / 65536.0)
}

fn unsupported(message: impl Into<String>) -> DecoderError {
    DecoderError::Unsupported(format!("ICC matrix-shaper profile: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_encoding_round_trips_endpoints() {
        assert_eq!(encode_srgb(0.0), 0);
        assert_eq!(encode_srgb(1.0), u16::MAX);
    }

    #[test]
    fn white_point_tolerance_accepts_standard_profiles() {
        assert!(close_to_white([0.9642, 1.0, 0.8249], [0.9642, 1.0, 0.8249]));
        assert!(!close_to_white([0.9, 1.0, 0.9], [0.9642, 1.0, 0.8249]));
    }

    #[test]
    fn curve_lookup_table_interpolates_normalized_samples() {
        let curve = Curve::Table(vec![0, 16_384, 65_535]);

        assert_eq!(curve.decode(0.0), 0.0);
        assert!((curve.decode(0.25) - 0.125).abs() < 0.0001);
        assert_eq!(curve.decode(1.0), 1.0);
    }

    #[test]
    fn curve_lookup_table_rejects_truncated_payload() {
        let mut profile = vec![0; 16];
        profile[0..4].copy_from_slice(b"curv");
        profile[8..12].copy_from_slice(&3u32.to_be_bytes());
        profile[12..14].copy_from_slice(&0u16.to_be_bytes());
        let error = read_curve(&profile, (0, profile.len())).unwrap_err();

        assert!(
            matches!(error, DecoderError::Unsupported(message) if message.contains("truncated"))
        );
    }

    #[test]
    fn parametric_curve_evaluates_function_one() {
        let curve = Curve::Parametric {
            function: 1,
            parameters: [2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        };

        assert!((curve.decode(0.25) - 0.0625).abs() < 0.0001);
    }

    #[test]
    fn parametric_curve_function_two_uses_the_linear_breakpoint_branch() {
        let curve = Curve::Parametric {
            function: 2,
            parameters: [2.0, 1.0, 0.0, 0.25, 0.0, 0.0, 0.0],
        };

        assert_eq!(curve.decode(0.0), 0.25);
        assert!((curve.decode(0.5) - 0.5).abs() < 0.0001);
    }

    #[test]
    fn parametric_curve_function_types_have_finite_outputs() {
        let parameter_counts = [1, 3, 4, 5, 7];
        for (function, parameter_count) in parameter_counts.into_iter().enumerate() {
            let mut parameters = [0.0; 7];
            parameters[0] = 2.0;
            parameters[1] = 1.0;
            parameters[3] = 1.0;
            parameters[4] = 0.5;
            parameters[5] = 0.1;
            parameters[6] = 0.0;
            assert!(parameter_count <= parameters.len());
            let curve = Curve::Parametric {
                function: function as u16,
                parameters,
            };
            assert!(curve.decode(0.5).is_finite());
        }
    }

    #[test]
    fn parametric_curve_rejects_zero_division_coefficient() {
        let mut profile = vec![0; 24];
        profile[0..4].copy_from_slice(b"para");
        profile[8..10].copy_from_slice(&1u16.to_be_bytes());
        profile[12..16].copy_from_slice(&2i32.to_be_bytes());
        let error = read_curve(&profile, (0, profile.len())).unwrap_err();

        assert!(
            matches!(error, DecoderError::Unsupported(message) if message.contains("zero division"))
        );
    }
}
