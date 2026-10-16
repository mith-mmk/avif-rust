//! Checked pixel ownership/layout shared by the AVIF decoder and encoder.
//!
//! This crate has no codec, container, ICC engine or application dependency.
//! Native code values and full-range packed samples are distinct variants.
//! Context such as AVIF property order belongs to the caller, not this model.
#![forbid(unsafe_code)]

mod convert;
pub use convert::{PackedOutput, PrecisionPolicy, TieRule, convert_packed, scale_full_range};

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BufferError {
    Layout(&'static str),
    Overflow,
    Allocation,
    SampleRange {
        plane: usize,
        index: usize,
        value: u16,
        bits: u8,
    },
    PrecisionLoss {
        value: u16,
        from: u8,
        to: u8,
    },
    UnsupportedConversion(&'static str),
}
impl fmt::Display for BufferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Layout(message) => write!(f, "invalid image layout: {message}"),
            Self::Overflow => f.write_str("image layout arithmetic overflow"),
            Self::Allocation => f.write_str("image allocation failed"),
            Self::SampleRange {
                plane,
                index,
                value,
                bits,
            } => write!(
                f,
                "plane {plane} sample {index}: {value} exceeds {bits} valid bits"
            ),
            Self::PrecisionLoss { value, from, to } => {
                write!(f, "sample {value} is not exact from {from} to {to} bits")
            }
            Self::UnsupportedConversion(message) => {
                write!(f, "explicit conversion unsupported: {message}")
            }
        }
    }
}
impl std::error::Error for BufferError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleStorage {
    U8,
    U16,
    U16LeBytes,
}
impl SampleStorage {
    pub const fn bytes(self) -> usize {
        if matches!(self, Self::U8) { 1 } else { 2 }
    }
}

/// U16 is numeric storage; U16LeBytes is an explicitly little-endian byte view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplesRef<'a> {
    U8(&'a [u8]),
    U16(&'a [u16]),
    U16LeBytes(&'a [u8]),
}
impl SamplesRef<'_> {
    pub const fn storage(self) -> SampleStorage {
        match self {
            Self::U8(_) => SampleStorage::U8,
            Self::U16(_) => SampleStorage::U16,
            Self::U16LeBytes(_) => SampleStorage::U16LeBytes,
        }
    }
    pub fn length_bytes(self) -> Result<usize, BufferError> {
        match self {
            Self::U8(v) | Self::U16LeBytes(v) => Ok(v.len()),
            Self::U16(v) => v.len().checked_mul(2).ok_or(BufferError::Overflow),
        }
    }
    fn read(self, at: usize) -> Result<u16, BufferError> {
        let invalid = || BufferError::Layout("sample is outside storage or unaligned numeric U16");
        match self {
            Self::U8(v) => v.get(at).map(|v| u16::from(*v)).ok_or_else(invalid),
            Self::U16(v) if at.is_multiple_of(2) => v.get(at / 2).copied().ok_or_else(invalid),
            Self::U16(_) => Err(invalid()),
            Self::U16LeBytes(v) => Ok(u16::from_le_bytes([
                *v.get(at).ok_or_else(invalid)?,
                *v.get(at.checked_add(1).ok_or(BufferError::Overflow)?)
                    .ok_or_else(invalid)?,
            ])),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OwnedSamples {
    U8(Vec<u8>),
    U16(Vec<u16>),
    U16LeBytes(Vec<u8>),
}
impl OwnedSamples {
    pub fn view(&self) -> SamplesRef<'_> {
        match self {
            Self::U8(v) => SamplesRef::U8(v),
            Self::U16(v) => SamplesRef::U16(v),
            Self::U16LeBytes(v) => SamplesRef::U16LeBytes(v),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphaDomain {
    LinearLight,
    EncodedComponents,
    Unknown,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphaMode {
    Absent,
    Straight,
    Premultiplied(AlphaDomain),
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorRange {
    Full,
    Limited,
}
/// Raw CICP identifiers, including unknown/unspecified identifiers, are retained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cicp {
    pub primaries: u16,
    pub transfer: u16,
    pub matrix: u16,
}
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ColorMetadataRef<'a> {
    pub icc: Option<&'a [u8]>,
    pub cicp: Option<Cicp>,
}
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ColorMetadata {
    pub icc: Option<Vec<u8>>,
    pub cicp: Option<Cicp>,
}
impl ColorMetadata {
    pub fn view(&self) -> ColorMetadataRef<'_> {
        ColorMetadataRef {
            icc: self.icc.as_deref(),
            cicp: self.cicp,
        }
    }
    pub fn try_from_ref(value: ColorMetadataRef<'_>) -> Result<Self, BufferError> {
        let icc = value.icc.map(copy_values).transpose()?;
        Ok(Self {
            icc,
            cicp: value.cicp,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackedChannels {
    Gray,
    Rgba,
}
impl PackedChannels {
    pub const fn count(self) -> usize {
        if matches!(self, Self::Gray) { 1 } else { 4 }
    }
}

/// All offsets and strides use bytes, including numeric U16 storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaneLayoutEx {
    pub width: usize,
    pub height: usize,
    pub valid_bits: u8,
    pub offset_bytes: usize,
    pub row_stride_bytes: usize,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaneViewEx<'a> {
    pub samples: SamplesRef<'a>,
    pub layout: PlaneLayoutEx,
}
impl PlaneViewEx<'_> {
    pub fn sample(
        self,
        x: usize,
        y: usize,
        channel: usize,
        channels: usize,
    ) -> Result<u16, BufferError> {
        if x >= self.layout.width || y >= self.layout.height || channel >= channels {
            return Err(BufferError::Layout("sample coordinates"));
        }
        let column = x
            .checked_mul(channels)
            .and_then(|n| n.checked_add(channel))
            .and_then(|n| n.checked_mul(self.samples.storage().bytes()))
            .ok_or(BufferError::Overflow)?;
        let at = y
            .checked_mul(self.layout.row_stride_bytes)
            .and_then(|n| n.checked_add(self.layout.offset_bytes))
            .and_then(|n| n.checked_add(column))
            .ok_or(BufferError::Overflow)?;
        self.samples.read(at)
    }
    pub fn validate(self, channels: usize, plane: usize) -> Result<(), BufferError> {
        let p = self.layout;
        if p.width == 0 || p.height == 0 || channels == 0 {
            return Err(BufferError::Layout("zero dimensions/channels"));
        }
        let max = sample_max(p.valid_bits)?;
        if self.samples.storage() == SampleStorage::U8 && p.valid_bits != 8 {
            return Err(BufferError::Layout("U8 storage requires8 valid bits"));
        }
        if matches!(self.samples, SamplesRef::U16(_))
            && (!p.offset_bytes.is_multiple_of(2) || !p.row_stride_bytes.is_multiple_of(2))
        {
            return Err(BufferError::Layout(
                "numeric U16 offset/stride must be even",
            ));
        }
        let row = p
            .width
            .checked_mul(channels)
            .and_then(|n| n.checked_mul(self.samples.storage().bytes()))
            .ok_or(BufferError::Overflow)?;
        let end = (p.height - 1)
            .checked_mul(p.row_stride_bytes)
            .and_then(|n| n.checked_add(p.offset_bytes))
            .and_then(|n| n.checked_add(row))
            .ok_or(BufferError::Overflow)?;
        if p.row_stride_bytes < row || end > self.samples.length_bytes()? {
            return Err(BufferError::Layout("short stride/storage"));
        }
        for y in 0..p.height {
            for x in 0..p.width {
                for c in 0..channels {
                    let value = self.sample(x, y, c, channels)?;
                    if u32::from(value) > max {
                        return Err(BufferError::SampleRange {
                            plane,
                            index: (y * p.width + x) * channels + c,
                            value,
                            bits: p.valid_bits,
                        });
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedImageViewEx<'a> {
    pub plane: PlaneViewEx<'a>,
    pub channels: PackedChannels,
    pub alpha: AlphaMode,
    pub color: ColorMetadataRef<'a>,
}

/// Identity uses G/B/R slots; YUV uses Y/Cb/Cr. Monochrome has only slot0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeModel {
    Monochrome,
    IdentityGbr,
    Yuv444,
    Yuv422,
    Yuv420,
}
impl NativeModel {
    pub const fn chroma_subsampling(self) -> Option<(usize, usize)> {
        match self {
            Self::Monochrome => None,
            Self::IdentityGbr | Self::Yuv444 => Some((0, 0)),
            Self::Yuv422 => Some((1, 0)),
            Self::Yuv420 => Some((1, 1)),
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeDescriptor {
    pub width: usize,
    pub height: usize,
    pub model: NativeModel,
    pub alpha: AlphaMode,
    pub range: Option<ColorRange>,
    /// Codec-independent raw location identifier; None means absent, Some(0) unknown.
    pub chroma_location: Option<u8>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativePlanarViewEx<'a> {
    pub descriptor: NativeDescriptor,
    /// Color slots0..2 and optional alpha slot3. Absent chroma is None, not a dummy plane.
    pub planes: [Option<PlaneViewEx<'a>>; 4],
    pub color: ColorMetadataRef<'a>,
}

// Four immutable plane descriptors stay inline so a view never allocates.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageViewEx<'a> {
    Packed(PackedImageViewEx<'a>),
    NativePlanar(NativePlanarViewEx<'a>),
}
/// Owns an immutable descriptor snapshot and immutable sample borrows.
#[derive(Debug, Clone, Copy)]
pub struct ValidatedImageViewEx<'a>(ImageViewEx<'a>);
impl<'a> ValidatedImageViewEx<'a> {
    pub fn view(self) -> ImageViewEx<'a> {
        self.0
    }
    pub fn to_owned_compact(self) -> Result<ImageBufferEx, BufferError> {
        self.0.to_owned_compact()
    }
}
impl<'a> ImageViewEx<'a> {
    pub fn validate(self) -> Result<ValidatedImageViewEx<'a>, BufferError> {
        match self {
            Self::Packed(p) => {
                let absent = p.alpha == AlphaMode::Absent;
                if absent != (p.channels == PackedChannels::Gray) {
                    return Err(BufferError::Layout("packed channels/alpha mismatch"));
                }
                p.plane.validate(p.channels.count(), 0)?;
            }
            Self::NativePlanar(n) => {
                let d = n.descriptor;
                if d.width == 0 || d.height == 0 {
                    return Err(BufferError::Layout("zero native dimensions"));
                }
                let first =
                    n.planes[0].ok_or(BufferError::Layout("missing native primary plane"))?;
                for (index, p) in n.planes.iter().enumerate() {
                    let required = match index {
                        0 => true,
                        3 => d.alpha != AlphaMode::Absent,
                        _ => d.model != NativeModel::Monochrome,
                    };
                    if p.is_some() != required {
                        return Err(BufferError::Layout("native plane roles/alpha mismatch"));
                    }
                    if let Some(p) = p {
                        let (sx, sy) = if index == 1 || index == 2 {
                            d.model.chroma_subsampling().unwrap_or((0, 0))
                        } else {
                            (0, 0)
                        };
                        let w = d.width.div_ceil(1 << sx);
                        let h = d.height.div_ceil(1 << sy);
                        if p.layout.width != w
                            || p.layout.height != h
                            || (index < 3 && p.layout.valid_bits != first.layout.valid_bits)
                        {
                            return Err(BufferError::Layout(
                                "native plane dimensions/depth mismatch",
                            ));
                        }
                        p.validate(1, index)?;
                    }
                }
            }
        }
        Ok(ValidatedImageViewEx(self))
    }
    pub fn width(self) -> usize {
        match self {
            Self::Packed(p) => p.plane.layout.width,
            Self::NativePlanar(n) => n.descriptor.width,
        }
    }
    pub fn height(self) -> usize {
        match self {
            Self::Packed(p) => p.plane.layout.height,
            Self::NativePlanar(n) => n.descriptor.height,
        }
    }
    pub fn color(self) -> ColorMetadataRef<'a> {
        match self {
            Self::Packed(p) => p.color,
            Self::NativePlanar(n) => n.color,
        }
    }
    pub fn to_owned_compact(self) -> Result<ImageBufferEx, BufferError> {
        self.validate()?;
        match self {
            Self::Packed(p) => {
                let (samples, layout) = compact_plane(p.plane, p.channels.count())?;
                ImageBufferEx::packed(
                    samples,
                    layout,
                    p.channels,
                    p.alpha,
                    ColorMetadata::try_from_ref(p.color)?,
                )
            }
            Self::NativePlanar(n) => {
                let mut storage = Vec::new();
                storage
                    .try_reserve_exact(4)
                    .map_err(|_| BufferError::Allocation)?;
                let mut planes = [None; 4];
                for (index, p) in n.planes.iter().enumerate() {
                    if let Some(p) = p {
                        let (samples, layout) = compact_plane(*p, 1)?;
                        planes[index] = Some(PlaneBinding {
                            storage_index: storage.len(),
                            layout,
                        });
                        storage.push(samples);
                    }
                }
                ImageBufferEx::native(
                    n.descriptor,
                    storage,
                    planes,
                    ColorMetadata::try_from_ref(n.color)?,
                )
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaneBinding {
    pub storage_index: usize,
    pub layout: PlaneLayoutEx,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedImageBufferEx {
    samples: OwnedSamples,
    layout: PlaneLayoutEx,
    channels: PackedChannels,
    alpha: AlphaMode,
    color: ColorMetadata,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativePlanarBufferEx {
    descriptor: NativeDescriptor,
    storage: Vec<OwnedSamples>,
    planes: [Option<PlaneBinding>; 4],
    color: ColorMetadata,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageBufferEx {
    Packed(PackedImageBufferEx),
    NativePlanar(NativePlanarBufferEx),
}
impl ImageBufferEx {
    /// Adopt an existing allocation after validation, without copying its samples.
    pub fn packed(
        samples: OwnedSamples,
        layout: PlaneLayoutEx,
        channels: PackedChannels,
        alpha: AlphaMode,
        color: ColorMetadata,
    ) -> Result<Self, BufferError> {
        ImageViewEx::Packed(PackedImageViewEx {
            plane: PlaneViewEx {
                samples: samples.view(),
                layout,
            },
            channels,
            alpha,
            color: color.view(),
        })
        .validate()?;
        Ok(Self::Packed(PackedImageBufferEx {
            samples,
            layout,
            channels,
            alpha,
            color,
        }))
    }
    /// Supports separate allocations and planes bound into one allocation.
    pub fn native(
        descriptor: NativeDescriptor,
        storage: Vec<OwnedSamples>,
        planes: [Option<PlaneBinding>; 4],
        color: ColorMetadata,
    ) -> Result<Self, BufferError> {
        let value = Self::NativePlanar(NativePlanarBufferEx {
            descriptor,
            storage,
            planes,
            color,
        });
        value.view()?.validate()?;
        Ok(value)
    }
    pub fn view(&self) -> Result<ImageViewEx<'_>, BufferError> {
        match self {
            Self::Packed(p) => Ok(ImageViewEx::Packed(PackedImageViewEx {
                plane: PlaneViewEx {
                    samples: p.samples.view(),
                    layout: p.layout,
                },
                channels: p.channels,
                alpha: p.alpha,
                color: p.color.view(),
            })),
            Self::NativePlanar(n) => {
                let mut planes = [None; 4];
                for (index, binding) in n.planes.iter().enumerate() {
                    if let Some(b) = binding {
                        planes[index] = Some(PlaneViewEx {
                            samples: n
                                .storage
                                .get(b.storage_index)
                                .ok_or(BufferError::Layout("plane storage binding"))?
                                .view(),
                            layout: b.layout,
                        });
                    }
                }
                Ok(ImageViewEx::NativePlanar(NativePlanarViewEx {
                    descriptor: n.descriptor,
                    planes,
                    color: n.color.view(),
                }))
            }
        }
    }
}

pub fn sample_max(bits: u8) -> Result<u32, BufferError> {
    if !matches!(bits, 8 | 10 | 12 | 16) {
        return Err(BufferError::Layout("valid bits must be8/10/12/16"));
    }
    Ok((1u32 << bits) - 1)
}
fn copy_values<T: Copy>(source: &[T]) -> Result<Vec<T>, BufferError> {
    let mut out = Vec::new();
    out.try_reserve_exact(source.len())
        .map_err(|_| BufferError::Allocation)?;
    out.extend_from_slice(source);
    Ok(out)
}
fn compact_plane(
    p: PlaneViewEx<'_>,
    channels: usize,
) -> Result<(OwnedSamples, PlaneLayoutEx), BufferError> {
    let count = p
        .layout
        .width
        .checked_mul(p.layout.height)
        .and_then(|v| v.checked_mul(channels))
        .ok_or(BufferError::Overflow)?;
    let row = p
        .layout
        .width
        .checked_mul(channels)
        .and_then(|n| n.checked_mul(p.samples.storage().bytes()))
        .ok_or(BufferError::Overflow)?;
    let samples = if matches!(p.samples, SamplesRef::U16(_)) {
        let mut out = Vec::new();
        out.try_reserve_exact(count)
            .map_err(|_| BufferError::Allocation)?;
        for y in 0..p.layout.height {
            for x in 0..p.layout.width {
                for c in 0..channels {
                    out.push(p.sample(x, y, c, channels)?);
                }
            }
        }
        OwnedSamples::U16(out)
    } else {
        let mut out = Vec::new();
        out.try_reserve_exact(
            count
                .checked_mul(p.samples.storage().bytes())
                .ok_or(BufferError::Overflow)?,
        )
        .map_err(|_| BufferError::Allocation)?;
        for y in 0..p.layout.height {
            for x in 0..p.layout.width {
                for c in 0..channels {
                    let v = p.sample(x, y, c, channels)?;
                    if p.samples.storage() == SampleStorage::U8 {
                        out.push(v as u8);
                    } else {
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                }
            }
        }
        if p.samples.storage() == SampleStorage::U8 {
            OwnedSamples::U8(out)
        } else {
            OwnedSamples::U16LeBytes(out)
        }
    };
    Ok((
        samples,
        PlaneLayoutEx {
            offset_bytes: 0,
            row_stride_bytes: row,
            ..p.layout
        },
    ))
}
