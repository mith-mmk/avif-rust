use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TieRule {
    ToEven,
    AwayFromZero,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionPolicy {
    Exact,
    Nearest { ties: TieRule },
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedOutput {
    pub channels: PackedChannels,
    pub storage: SampleStorage,
    pub valid_bits: u8,
    pub row_stride_bytes: Option<usize>,
    pub precision: PrecisionPolicy,
}

fn divide(n: u64, d: u64, policy: PrecisionPolicy) -> Option<u64> {
    let q = n / d;
    let r = n % d;
    match policy {
        PrecisionPolicy::Exact => {
            if r == 0 {
                Some(q)
            } else {
                None
            }
        }
        PrecisionPolicy::Nearest { ties } => Some(
            q + u64::from(
                r > d - r || (r == d - r && (ties == TieRule::AwayFromZero || q % 2 == 1)),
            ),
        ),
    }
}

/// Full-range integer scaling only; not a native YUV/range/color transform.
pub fn scale_full_range(
    value: u16,
    from: u8,
    to: u8,
    policy: PrecisionPolicy,
) -> Result<u16, BufferError> {
    let a = sample_max(from)?;
    let b = sample_max(to)?;
    if u32::from(value) > a {
        return Err(BufferError::SampleRange {
            plane: 0,
            index: 0,
            value,
            bits: from,
        });
    }
    divide(u64::from(value) * u64::from(b), u64::from(a), policy)
        .map(|v| v as u16)
        .ok_or(BufferError::PrecisionLoss { value, from, to })
}

/// Explicit normalized precision/channel conversion, with zeroed output padding.
/// ICC and alpha association are retained without being applied or changed.
pub fn convert_packed(
    source: PackedImageViewEx<'_>,
    output: PackedOutput,
) -> Result<ImageBufferEx, BufferError> {
    ImageViewEx::Packed(source).validate()?;
    sample_max(output.valid_bits)?;
    if source.channels == PackedChannels::Rgba && output.channels == PackedChannels::Gray {
        return Err(BufferError::UnsupportedConversion(
            "RGBA to Gray requires a color policy",
        ));
    }
    if output.storage == SampleStorage::U8 && output.valid_bits != 8 {
        return Err(BufferError::Layout("U8 output requires8 bits"));
    }
    let width = source.plane.layout.width;
    let height = source.plane.layout.height;
    let row = width
        .checked_mul(output.channels.count())
        .and_then(|n| n.checked_mul(output.storage.bytes()))
        .ok_or(BufferError::Overflow)?;
    let stride = output.row_stride_bytes.unwrap_or(row);
    if stride < row || (output.storage == SampleStorage::U16 && !stride.is_multiple_of(2)) {
        return Err(BufferError::Layout("output stride"));
    }
    let length = stride.checked_mul(height).ok_or(BufferError::Overflow)?;
    let mut samples = if output.storage == SampleStorage::U16 {
        let mut words = Vec::new();
        words
            .try_reserve_exact(length / 2)
            .map_err(|_| BufferError::Allocation)?;
        words.resize(length / 2, 0);
        OwnedSamples::U16(words)
    } else {
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|_| BufferError::Allocation)?;
        bytes.resize(length, 0);
        if output.storage == SampleStorage::U8 {
            OwnedSamples::U8(bytes)
        } else {
            OwnedSamples::U16LeBytes(bytes)
        }
    };
    for y in 0..height {
        for x in 0..width {
            for c in 0..output.channels.count() {
                let value = if source.channels == PackedChannels::Gray && c == 3 {
                    sample_max(output.valid_bits)? as u16
                } else {
                    scale_full_range(
                        source.plane.sample(
                            x,
                            y,
                            if source.channels == PackedChannels::Gray {
                                0
                            } else {
                                c
                            },
                            source.channels.count(),
                        )?,
                        source.plane.layout.valid_bits,
                        output.valid_bits,
                        output.precision,
                    )?
                };
                let at = y * stride + (x * output.channels.count() + c) * output.storage.bytes();
                match &mut samples {
                    OwnedSamples::U8(bytes) => bytes[at] = value as u8,
                    OwnedSamples::U16(words) => words[at / 2] = value,
                    OwnedSamples::U16LeBytes(bytes) => {
                        bytes[at..at + 2].copy_from_slice(&value.to_le_bytes())
                    }
                }
            }
        }
    }
    ImageBufferEx::packed(
        samples,
        PlaneLayoutEx {
            width,
            height,
            valid_bits: output.valid_bits,
            offset_bytes: 0,
            row_stride_bytes: stride,
        },
        output.channels,
        if output.channels == PackedChannels::Gray {
            AlphaMode::Absent
        } else if source.channels == PackedChannels::Gray {
            AlphaMode::Straight
        } else {
            source.alpha
        },
        ColorMetadata::try_from_ref(source.color)?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tie_policies_are_explicit_even_when_bit_max_denominators_are_odd() {
        assert_eq!(
            divide(
                5,
                2,
                PrecisionPolicy::Nearest {
                    ties: TieRule::ToEven
                }
            ),
            Some(2)
        );
        assert_eq!(
            divide(
                5,
                2,
                PrecisionPolicy::Nearest {
                    ties: TieRule::AwayFromZero
                }
            ),
            Some(3)
        );
        assert_eq!(
            divide(
                3,
                2,
                PrecisionPolicy::Nearest {
                    ties: TieRule::ToEven
                }
            ),
            Some(2)
        );
    }
}
