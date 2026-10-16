//! Additive shared-buffer adapters. Pixel representation is separate from AVIF
//! context; old display-oriented helpers keep their ICC/unpremultiply behavior.
use crate::av1::{ChromaSamplePosition, ColorConfig, ColorRange as Av1Range, FrameBuffers};
use crate::buffer::*;
use crate::{
    DecodedFrame, DecoderError, ImageBuffer, NativeAvifInformation, NativeDecodeLimits,
    NativeDecodedFrame, Rgba16ImageBuffer,
};

/// Complete existing bounded context plus the decoded frame's original semantics.
/// Unknown property payloads absent from NativeAvifInformation are not invented.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AvifSourceContext {
    information: NativeAvifInformation,
    width: usize,
    height: usize,
    render_width: usize,
    render_height: usize,
    bit_depth: u8,
    color_config: ColorConfig,
    alpha_premultiplied: bool,
}
impl AvifSourceContext {
    pub fn information(&self) -> &NativeAvifInformation {
        &self.information
    }
    pub fn coded_dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }
    pub fn render_dimensions(&self) -> (usize, usize) {
        (self.render_width, self.render_height)
    }
    pub fn bit_depth(&self) -> u8 {
        self.bit_depth
    }
    pub fn color_config(&self) -> &ColorConfig {
        &self.color_config
    }
    pub fn alpha_premultiplied(&self) -> bool {
        self.alpha_premultiplied
    }
}

/// Native shared pixels and their AVIF source context, without display conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeImageEx {
    image: crate::buffer::ImageBufferEx,
    context: AvifSourceContext,
}
impl NativeImageEx {
    pub fn image(&self) -> &crate::buffer::ImageBufferEx {
        &self.image
    }
    pub fn view(&self) -> Result<ImageViewEx<'_>, BufferError> {
        self.image.view()
    }
    pub fn context(&self) -> &AvifSourceContext {
        &self.context
    }
    pub fn into_parts(self) -> (crate::buffer::ImageBufferEx, AvifSourceContext) {
        (self.image, self.context)
    }
}

fn descriptor(frame: &DecodedFrame) -> Result<NativeDescriptor, BufferError> {
    let c = frame.color_config;
    if c.bit_depth != frame.bit_depth {
        return Err(BufferError::Layout("frame and AV1 bit depth disagree"));
    }
    let matrix = c.color_description.map(|v| v.matrix_coefficients);
    let model = if c.monochrome {
        NativeModel::Monochrome
    } else if matrix == Some(0) {
        NativeModel::IdentityGbr
    } else {
        // These transforms have component roles outside this initial Y/Cb/Cr
        // model. Preserve/reject explicitly rather than relabel their planes.
        if matches!(matrix, Some(8 | 14)) {
            return Err(BufferError::UnsupportedConversion(
                "native component roles need a separate model",
            ));
        }
        match (c.subsampling_x, c.subsampling_y) {
            (false, false) => NativeModel::Yuv444,
            (true, false) => NativeModel::Yuv422,
            (true, true) => NativeModel::Yuv420,
            _ => return Err(BufferError::Layout("unsupported native subsampling")),
        }
    };
    let has_alpha = frame
        .buffers
        .planes
        .iter()
        .any(|p| p.layout.plane == 3 && !p.samples.is_empty());
    if frame.alpha_premultiplied && !has_alpha {
        return Err(BufferError::Layout(
            "premultiplied source has no alpha plane",
        ));
    }
    Ok(NativeDescriptor {
        width: frame.width,
        height: frame.height,
        model,
        alpha: if has_alpha {
            if frame.alpha_premultiplied {
                AlphaMode::Premultiplied(AlphaDomain::Unknown)
            } else {
                AlphaMode::Straight
            }
        } else {
            AlphaMode::Absent
        },
        range: Some(if c.color_range == Av1Range::Full {
            ColorRange::Full
        } else {
            ColorRange::Limited
        }),
        chroma_location: c.chroma_sample_position.map(|p| match p {
            ChromaSamplePosition::Unknown => 0,
            ChromaSamplePosition::Vertical => 1,
            ChromaSamplePosition::Colocated => 2,
            ChromaSamplePosition::Reserved => 3,
        }),
    })
}
fn color<'a>(frame: &DecodedFrame, icc: Option<&'a [u8]>) -> ColorMetadataRef<'a> {
    ColorMetadataRef {
        icc,
        cicp: frame.color_config.color_description.map(|c| Cicp {
            primaries: u16::from(c.color_primaries),
            transfer: u16::from(c.transfer_characteristics),
            matrix: u16::from(c.matrix_coefficients),
        }),
    }
}
fn layout(p: &crate::av1::PlaneBuffer, bits: u8) -> Result<PlaneLayoutEx, BufferError> {
    if p.layout.sample_count != p.samples.len()
        || p.layout.sample_count
            != p.layout
                .width
                .checked_mul(p.layout.height)
                .ok_or(BufferError::Overflow)?
    {
        return Err(BufferError::Layout("legacy native sample_count"));
    }
    Ok(PlaneLayoutEx {
        width: p.layout.width,
        height: p.layout.height,
        valid_bits: bits,
        offset_bytes: 0,
        row_stride_bytes: p.layout.width.checked_mul(2).ok_or(BufferError::Overflow)?,
    })
}

impl FrameBuffers {
    /// Borrow raw planes with explicit semantics; bit depth is not inferred from U16.
    pub fn view_ex<'a>(
        &'a self,
        descriptor: NativeDescriptor,
        bits: u8,
        color: ColorMetadataRef<'a>,
    ) -> Result<ImageViewEx<'a>, BufferError> {
        if self.width != descriptor.width || self.height != descriptor.height {
            return Err(BufferError::Layout("native frame dimensions/context"));
        }
        let mut planes = [None; 4];
        for p in &self.planes {
            let index = usize::from(p.layout.plane);
            if index >= 4 {
                return Err(BufferError::Layout("native plane role"));
            }
            if descriptor.model == NativeModel::Monochrome
                && matches!(index, 1 | 2)
                && p.samples.is_empty()
            {
                continue;
            }
            if planes[index].is_some() {
                return Err(BufferError::Layout("duplicate native plane role"));
            }
            let expected = if index == 1 || index == 2 {
                descriptor.model.chroma_subsampling().unwrap_or((0, 0))
            } else {
                (0, 0)
            };
            if (
                usize::from(p.layout.subsampling_x),
                usize::from(p.layout.subsampling_y),
            ) != expected
            {
                return Err(BufferError::Layout("native plane/context subsampling"));
            }
            planes[index] = Some(PlaneViewEx {
                samples: SamplesRef::U16(&p.samples),
                layout: layout(p, bits)?,
            });
        }
        let view = ImageViewEx::NativePlanar(NativePlanarViewEx {
            descriptor,
            planes,
            color,
        });
        view.validate()?;
        Ok(view)
    }
}
impl ImageBuffer {
    pub fn view_ex<'a>(
        &'a self,
        color: ColorMetadataRef<'a>,
    ) -> Result<ImageViewEx<'a>, BufferError> {
        packed(
            self.width,
            self.height,
            8,
            SamplesRef::U8(&self.rgba),
            color,
        )
    }
}
impl Rgba16ImageBuffer {
    /// Full-range16-bit storage stays16-bit; no native12-bit reinterpretation.
    pub fn view_ex<'a>(
        &'a self,
        color: ColorMetadataRef<'a>,
    ) -> Result<ImageViewEx<'a>, BufferError> {
        packed(
            self.width,
            self.height,
            16,
            SamplesRef::U16(&self.rgba),
            color,
        )
    }
}
fn packed<'a>(
    width: usize,
    height: usize,
    bits: u8,
    samples: SamplesRef<'a>,
    color: ColorMetadataRef<'a>,
) -> Result<ImageViewEx<'a>, BufferError> {
    let row = width
        .checked_mul(4)
        .and_then(|n| n.checked_mul(samples.storage().bytes()))
        .ok_or(BufferError::Overflow)?;
    let expected = row.checked_mul(height).ok_or(BufferError::Overflow)?;
    if samples.length_bytes() != Ok(expected) {
        return Err(BufferError::Layout("legacy RGBA exact length"));
    }
    let view = ImageViewEx::Packed(PackedImageViewEx {
        plane: PlaneViewEx {
            samples,
            layout: PlaneLayoutEx {
                width,
                height,
                valid_bits: bits,
                offset_bytes: 0,
                row_stride_bytes: row,
            },
        },
        channels: PackedChannels::Rgba,
        alpha: AlphaMode::Straight,
        color,
    });
    view.validate()?;
    Ok(view)
}

impl NativeDecodedFrame {
    /// Borrows native samples and source ICC without a display transform or copy.
    pub fn view_ex(&self) -> Result<ImageViewEx<'_>, BufferError> {
        let f = self.frame();
        f.buffers.view_ex(
            descriptor(f)?,
            f.bit_depth,
            color(
                f,
                self.information()
                    .color_information()
                    .icc_profile
                    .as_deref(),
            ),
        )
    }
    /// Move sample allocations into the shared model while retaining AVIF context.
    /// ICC bytes may be copied as metadata; plane sample allocations are moved.
    pub fn into_image_ex(self) -> Result<NativeImageEx, BufferError> {
        self.view_ex()?.validate()?;
        let d = descriptor(self.frame())?;
        let metadata = ColorMetadata::try_from_ref(color(
            self.frame(),
            self.information()
                .color_information()
                .icc_profile
                .as_deref(),
        ))?;
        let (frame, information) = self.into_parts();
        let context = AvifSourceContext {
            information,
            width: frame.width,
            height: frame.height,
            render_width: frame.render_width,
            render_height: frame.render_height,
            bit_depth: frame.bit_depth,
            color_config: frame.color_config,
            alpha_premultiplied: frame.alpha_premultiplied,
        };
        let mut storage = Vec::new();
        storage
            .try_reserve_exact(4)
            .map_err(|_| BufferError::Allocation)?;
        let mut planes = [None; 4];
        for p in frame.buffers.planes {
            let index = usize::from(p.layout.plane);
            if d.model == NativeModel::Monochrome && matches!(index, 1 | 2) && p.samples.is_empty()
            {
                continue;
            }
            let binding = PlaneBinding {
                storage_index: storage.len(),
                layout: layout(&p, frame.bit_depth)?,
            };
            planes[index] = Some(binding);
            storage.push(OwnedSamples::U16(p.samples));
        }
        Ok(NativeImageEx {
            image: crate::buffer::ImageBufferEx::native(d, storage, planes, metadata)?,
            context,
        })
    }
}

impl DecodedFrame {
    /// Pixel-only native view. For complete source context use NativeDecodedFrame.
    pub fn view_ex(&self) -> Result<ImageViewEx<'_>, BufferError> {
        self.buffers.view_ex(
            descriptor(self)?,
            self.bit_depth,
            color(
                self,
                self.color_information
                    .as_ref()
                    .and_then(|c| c.icc_profile()),
            ),
        )
    }
}

/// Decode once using the existing bounded strict native path; no legacy fallback.
pub fn decode_native_ex(
    data: &[u8],
    limits: &NativeDecodeLimits,
) -> Result<NativeImageEx, DecoderError> {
    crate::decode_frame_bytes_strict_with_limits(data, limits)?
        .into_image_ex()
        .map_err(|error| DecoderError::InvalidParam(error.to_string()))
}
