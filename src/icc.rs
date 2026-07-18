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

#[derive(Debug, Clone)]
struct LutProfile {
    matrix: [[f64; 3]; 3],
    xyz_to_srgb: [[f64; 3]; 3],
    input_tables: [Vec<u16>; 3],
    clut: Vec<u16>,
    output_tables: [Vec<u16>; 3],
    grid_points: usize,
}

pub(crate) fn apply_to_rgba16(rgba: &mut [u16], profile: &[u8]) -> Result<(), DecoderError> {
    if let Some(tag) = find_profile_tag(profile, b"A2B0")? {
        let signature = profile
            .get(tag.0..tag.0 + 4)
            .ok_or_else(|| unsupported("ICC A2B0 tag is outside the profile"))?;
        if signature == b"mft1" || signature == b"mft2" {
            let profile = LutProfile::parse(profile, tag)?;
            for pixel in rgba.chunks_exact_mut(4) {
                profile.apply(pixel);
            }
            return Ok(());
        }
        return Err(unsupported("ICC A2B0 tag is not an mft1/mft2 LUT"));
    }
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

impl LutProfile {
    fn parse(profile: &[u8], tag: (usize, usize)) -> Result<Self, DecoderError> {
        if profile.len() < 132 || &profile[16..20] != b"RGB " || &profile[20..24] != b"XYZ " {
            return Err(unsupported("ICC LUT profile must be RGB to XYZ"));
        }
        let (offset, size) = tag;
        if size < 48
            || offset
                .checked_add(size)
                .is_none_or(|end| end > profile.len())
        {
            return Err(unsupported("ICC LUT tag is truncated"));
        }
        let signature = &profile[offset..offset + 4];
        if signature != b"mft1" && signature != b"mft2" {
            return Err(unsupported("ICC LUT tag type is unsupported"));
        }
        let input_channels = profile[offset + 8];
        let output_channels = profile[offset + 9];
        let grid_points = usize::from(profile[offset + 10]);
        if input_channels != 3 || output_channels != 3 {
            return Err(unsupported(
                "ICC LUT profiles with non-RGB channel counts are not supported",
            ));
        }
        if !(2..=65).contains(&grid_points) {
            return Err(unsupported("ICC LUT grid point count is outside 2..=65"));
        }
        let mut matrix = [[0.0; 3]; 3];
        for (index, value) in matrix.iter_mut().flatten().enumerate() {
            *value = read_s15_fixed16(profile, offset + 12 + index * 4)?;
        }
        let (input_entries, output_entries, data_offset, bytes_per_value) = if signature == b"mft1"
        {
            (256usize, 256usize, offset + 48, 1usize)
        } else {
            let input_entries = usize::from(read_u16(profile, offset + 48)?);
            let output_entries = usize::from(read_u16(profile, offset + 50)?);
            if !(2..=4096).contains(&input_entries) || !(2..=4096).contains(&output_entries) {
                return Err(unsupported(
                    "ICC mft2 table entry count is outside 2..=4096",
                ));
            }
            (input_entries, output_entries, offset + 52, 2usize)
        };
        let grid_values = grid_points
            .checked_mul(grid_points)
            .and_then(|value| value.checked_mul(grid_points))
            .and_then(|value| value.checked_mul(output_channels as usize))
            .ok_or_else(|| unsupported("ICC LUT CLUT size overflows"))?;
        let input_values = input_channels as usize * input_entries;
        let output_values = output_channels as usize * output_entries;
        let value_count = input_values
            .checked_add(grid_values)
            .and_then(|value| value.checked_add(output_values))
            .ok_or_else(|| unsupported("ICC LUT table size overflows"))?;
        let required_bytes = value_count
            .checked_mul(bytes_per_value)
            .and_then(|value| data_offset.checked_add(value))
            .ok_or_else(|| unsupported("ICC LUT payload size overflows"))?;
        let tag_end = offset
            .checked_add(size)
            .ok_or_else(|| unsupported("ICC LUT tag size overflows"))?;
        if required_bytes > tag_end {
            return Err(unsupported("ICC LUT payload is truncated"));
        }
        let mut cursor = data_offset;
        let mut input_tables = std::array::from_fn(|_| Vec::with_capacity(input_entries));
        for table in &mut input_tables {
            for _ in 0..input_entries {
                table.push(read_lut_value(profile, &mut cursor, bytes_per_value)?);
            }
        }
        let mut clut = Vec::with_capacity(grid_values);
        for _ in 0..grid_values {
            clut.push(read_lut_value(profile, &mut cursor, bytes_per_value)?);
        }
        let mut output_tables = std::array::from_fn(|_| Vec::with_capacity(output_entries));
        for table in &mut output_tables {
            for _ in 0..output_entries {
                table.push(read_lut_value(profile, &mut cursor, bytes_per_value)?);
            }
        }
        let white = read_xyz(
            profile,
            find_profile_tag(profile, b"wtpt")?
                .ok_or_else(|| unsupported("ICC LUT profile media white point tag is missing"))?,
        )?;
        let xyz_to_srgb = xyz_to_srgb_matrix(white)?;
        Ok(Self {
            matrix,
            xyz_to_srgb,
            input_tables,
            clut,
            output_tables,
            grid_points,
        })
    }

    fn apply(&self, pixel: &mut [u16]) {
        let source = [
            f64::from(pixel[0]) / f64::from(u16::MAX),
            f64::from(pixel[1]) / f64::from(u16::MAX),
            f64::from(pixel[2]) / f64::from(u16::MAX),
        ];
        let matrix_input = multiply(self.matrix, source);
        let input = [
            lookup_table(&self.input_tables[0], matrix_input[0]),
            lookup_table(&self.input_tables[1], matrix_input[1]),
            lookup_table(&self.input_tables[2], matrix_input[2]),
        ];
        let pcs = self.lookup_clut(input);
        let xyz = [
            lookup_table(&self.output_tables[0], pcs[0]),
            lookup_table(&self.output_tables[1], pcs[1]),
            lookup_table(&self.output_tables[2], pcs[2]),
        ];
        let rgb = multiply(self.xyz_to_srgb, xyz);
        pixel[0] = encode_srgb(rgb[0]);
        pixel[1] = encode_srgb(rgb[1]);
        pixel[2] = encode_srgb(rgb[2]);
    }

    fn lookup_clut(&self, input: [f64; 3]) -> [f64; 3] {
        let scale = (self.grid_points - 1) as f64;
        let positions = input.map(|value| value.clamp(0.0, 1.0) * scale);
        let low = positions.map(|value| value.floor() as usize);
        let high = low.map(|value| (value + 1).min(self.grid_points - 1));
        let fraction = positions.map(|value| value - value.floor());
        let mut output = [0.0; 3];
        for blue_offset in 0..=1 {
            let blue = if blue_offset == 0 { low[2] } else { high[2] };
            let blue_weight = if blue_offset == 0 {
                1.0 - fraction[2]
            } else {
                fraction[2]
            };
            for green_offset in 0..=1 {
                let green = if green_offset == 0 { low[1] } else { high[1] };
                let green_weight = if green_offset == 0 {
                    1.0 - fraction[1]
                } else {
                    fraction[1]
                };
                for red_offset in 0..=1 {
                    let red = if red_offset == 0 { low[0] } else { high[0] };
                    let red_weight = if red_offset == 0 {
                        1.0 - fraction[0]
                    } else {
                        fraction[0]
                    };
                    let weight = blue_weight * green_weight * red_weight;
                    if weight == 0.0 {
                        continue;
                    }
                    let index = ((blue * self.grid_points + green) * self.grid_points + red) * 3;
                    for (channel, value) in output.iter_mut().enumerate() {
                        *value +=
                            weight * f64::from(self.clut[index + channel]) / f64::from(u16::MAX);
                    }
                }
            }
        }
        output
    }
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

fn find_profile_tag(
    profile: &[u8],
    wanted: &[u8; 4],
) -> Result<Option<(usize, usize)>, DecoderError> {
    if profile.len() < 132 {
        return Err(unsupported("ICC profile header is truncated"));
    }
    let declared_size = read_u32(profile, 0)? as usize;
    if declared_size < 132 || declared_size > profile.len() {
        return Err(unsupported("ICC profile size is invalid"));
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
    for index in 0..tag_count {
        let entry = 132 + index * 12;
        let signature = &profile[entry..entry + 4];
        let offset = read_u32(profile, entry + 4)? as usize;
        let size = read_u32(profile, entry + 8)? as usize;
        let end = offset
            .checked_add(size)
            .ok_or_else(|| unsupported("ICC profile tag overflows"))?;
        if offset < 132 || size < 8 || end > declared_size {
            return Err(unsupported("ICC profile tag is outside the profile"));
        }
        if signature == wanted {
            return Ok(Some((offset, size)));
        }
    }
    Ok(None)
}

fn read_lut_value(
    profile: &[u8],
    cursor: &mut usize,
    bytes_per_value: usize,
) -> Result<u16, DecoderError> {
    let value = match bytes_per_value {
        1 => {
            let value = *profile
                .get(*cursor)
                .ok_or_else(|| unsupported("ICC LUT value is truncated"))?;
            u16::from(value) * 257
        }
        2 => read_u16(profile, *cursor)?,
        _ => return Err(unsupported("ICC LUT precision is unsupported")),
    };
    *cursor = (*cursor)
        .checked_add(bytes_per_value)
        .ok_or_else(|| unsupported("ICC LUT cursor overflows"))?;
    Ok(value)
}

fn lookup_table(table: &[u16], value: f64) -> f64 {
    let position = value.clamp(0.0, 1.0) * (table.len() - 1) as f64;
    let index = position.floor() as usize;
    let next = (index + 1).min(table.len() - 1);
    let fraction = position - index as f64;
    let low = f64::from(table[index]);
    let high = f64::from(table[next]);
    (low + (high - low) * fraction) / f64::from(u16::MAX)
}

fn xyz_to_srgb_matrix(white: [f64; 3]) -> Result<[[f64; 3]; 3], DecoderError> {
    if close_to_white(white, [0.9642, 1.0, 0.8249]) {
        Ok(D50_TO_SRGB)
    } else if close_to_white(white, [0.9505, 1.0, 1.0890]) {
        Ok(D65_TO_SRGB)
    } else {
        Err(unsupported("ICC profile white point is not D50 or D65"))
    }
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

    const SRGB_D50_TO_XYZ: [[f64; 3]; 3] = [
        [0.4360747, 0.3850649, 0.1430804],
        [0.2225045, 0.7168786, 0.0606169],
        [0.0139322, 0.0971045, 0.7141733],
    ];

    fn synthetic_lut_profile(signature: &[u8; 4], grid_points: u8, entries: u16) -> Vec<u8> {
        let (input_entries, output_entries, bytes_per_value) = if signature == b"mft1" {
            (256usize, 256usize, 1usize)
        } else {
            (usize::from(entries), usize::from(entries), 2usize)
        };
        let grid = usize::from(grid_points);
        let clut_values = grid * grid * grid * 3;
        let payload_values = 3 * input_entries + clut_values + 3 * output_entries;
        let data_header = if signature == b"mft2" { 52 } else { 48 };
        let mut lut = vec![0; data_header + payload_values * bytes_per_value];
        lut[0..4].copy_from_slice(signature);
        lut[8] = 3;
        lut[9] = 3;
        lut[10] = grid_points;
        for index in 0..9 {
            let value = if index / 3 == index % 3 {
                1_i32 << 16
            } else {
                0
            };
            lut[12 + index * 4..16 + index * 4].copy_from_slice(&value.to_be_bytes());
        }
        if signature == b"mft2" {
            lut[48..50].copy_from_slice(&entries.to_be_bytes());
            lut[50..52].copy_from_slice(&entries.to_be_bytes());
        }
        let mut cursor = if signature == b"mft2" { 52 } else { 48 };
        for _ in 0..3 {
            for index in 0..input_entries {
                let value = if input_entries == 1 {
                    0
                } else {
                    ((index * 65_535) / (input_entries - 1)) as u16
                };
                write_lut_value(&mut lut, &mut cursor, value, bytes_per_value);
            }
        }
        for blue in 0..grid {
            for green in 0..grid {
                for red in 0..grid {
                    let source = [
                        red as f64 / (grid - 1) as f64,
                        green as f64 / (grid - 1) as f64,
                        blue as f64 / (grid - 1) as f64,
                    ];
                    for value in multiply(SRGB_D50_TO_XYZ, source) {
                        let value = (value.clamp(0.0, 1.0)
                            * if bytes_per_value == 1 {
                                255.0
                            } else {
                                65_535.0
                            })
                        .round() as u16;
                        let value = if bytes_per_value == 1 {
                            value * 257
                        } else {
                            value
                        };
                        write_lut_value(&mut lut, &mut cursor, value, bytes_per_value);
                    }
                }
            }
        }
        for _ in 0..3 {
            for index in 0..output_entries {
                let value = if output_entries == 1 {
                    0
                } else {
                    ((index * 65_535) / (output_entries - 1)) as u16
                };
                write_lut_value(&mut lut, &mut cursor, value, bytes_per_value);
            }
        }

        let lut_offset = 156usize;
        let white_offset = lut_offset + lut.len();
        let mut profile = vec![0; white_offset + 20];
        profile[12..16].copy_from_slice(b"mntr");
        profile[16..20].copy_from_slice(b"RGB ");
        profile[20..24].copy_from_slice(b"XYZ ");
        profile[128..132].copy_from_slice(&2_u32.to_be_bytes());
        profile[132..136].copy_from_slice(b"A2B0");
        profile[136..140].copy_from_slice(&(lut_offset as u32).to_be_bytes());
        profile[140..144].copy_from_slice(&(lut.len() as u32).to_be_bytes());
        profile[144..148].copy_from_slice(b"wtpt");
        profile[148..152].copy_from_slice(&(white_offset as u32).to_be_bytes());
        profile[152..156].copy_from_slice(&20_u32.to_be_bytes());
        profile[lut_offset..white_offset].copy_from_slice(&lut);
        profile[white_offset..white_offset + 4].copy_from_slice(b"XYZ ");
        for (index, value) in [0.9642_f64, 1.0, 0.8249].into_iter().enumerate() {
            let fixed = (value * 65_536.0).round() as i32;
            profile[white_offset + 8 + index * 4..white_offset + 12 + index * 4]
                .copy_from_slice(&fixed.to_be_bytes());
        }
        let profile_size = profile.len() as u32;
        profile[0..4].copy_from_slice(&profile_size.to_be_bytes());
        profile
    }

    fn write_lut_value(data: &mut [u8], cursor: &mut usize, value: u16, bytes_per_value: usize) {
        if bytes_per_value == 1 {
            data[*cursor] = (value / 257) as u8;
            *cursor += 1;
        } else {
            data[*cursor..*cursor + 2].copy_from_slice(&value.to_be_bytes());
            *cursor += 2;
        }
    }

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

    #[test]
    fn mft1_lut_profile_applies_clut_and_preserves_alpha() {
        let profile = synthetic_lut_profile(b"mft1", 2, 0);
        let mut rgba = [u16::MAX, 0, 0, 12_345];

        apply_to_rgba16(&mut rgba, &profile).unwrap();

        assert!(rgba[0] > 64_000);
        assert!(rgba[1] < 6_000);
        assert!(rgba[2] < 6_000);
        assert_eq!(rgba[3], 12_345);
    }

    #[test]
    fn mft2_lut_profile_interpolates_clut() {
        let profile = synthetic_lut_profile(b"mft2", 3, 17);
        let mut rgba = [0, u16::MAX, u16::MAX, u16::MAX];

        apply_to_rgba16(&mut rgba, &profile).unwrap();

        assert!(rgba[0] < 2_000);
        assert!(rgba[1] > 64_000);
        assert!(rgba[2] > 64_000);
    }

    #[test]
    fn lut_profile_rejects_unknown_a2b0_type() {
        let profile = synthetic_lut_profile(b"mAB ", 2, 0);
        let error = apply_to_rgba16(&mut [0, 0, 0, u16::MAX], &profile).unwrap_err();

        assert!(
            matches!(error, DecoderError::Unsupported(message) if message.contains("mft1/mft2"))
        );
    }

    #[test]
    fn lut_profile_rejects_truncated_payload() {
        let mut profile = synthetic_lut_profile(b"mft2", 2, 2);
        profile[140..144].copy_from_slice(&60_u32.to_be_bytes());
        let error = apply_to_rgba16(&mut [0, 0, 0, u16::MAX], &profile).unwrap_err();

        assert!(
            matches!(error, DecoderError::Unsupported(message) if message.contains("truncated"))
        );
    }

    #[test]
    fn lut_profile_rejects_invalid_grid_size() {
        let profile = synthetic_lut_profile(b"mft2", 1, 2);
        let error = apply_to_rgba16(&mut [0, 0, 0, u16::MAX], &profile).unwrap_err();

        assert!(
            matches!(error, DecoderError::Unsupported(message) if message.contains("grid point"))
        );
    }

    #[test]
    fn lut_profile_rejects_invalid_mft2_table_entries() {
        let profile = synthetic_lut_profile(b"mft2", 2, 1);
        let error = apply_to_rgba16(&mut [0, 0, 0, u16::MAX], &profile).unwrap_err();

        assert!(matches!(
            error,
            DecoderError::Unsupported(message) if message.contains("entry count")
        ));
    }
}
