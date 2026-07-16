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

#[derive(Debug, Clone, Copy)]
enum Curve {
    Identity,
    Gamma(f64),
}

#[derive(Debug, Clone, Copy)]
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
    fn decode(self, value: f64) -> f64 {
        match self {
            Self::Identity => value,
            Self::Gamma(gamma) => value.powf(gamma),
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
    if size < 12 || &profile[offset..offset + 4] != b"curv" {
        return Err(unsupported("ICC tone curve is not a supported curv tag"));
    }
    let count = read_u32(profile, offset + 8)?;
    match count {
        0 => Ok(Curve::Identity),
        1 if size >= 14 => Ok(Curve::Gamma(
            f64::from(read_u16(profile, offset + 12)?) / 256.0,
        )),
        _ => Err(unsupported(
            "ICC tone curves with lookup tables are not supported",
        )),
    }
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
}
