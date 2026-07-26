use std::collections::HashMap;

type Error = Box<dyn std::error::Error>;

/// Metadata map used by the compatibility wrapper.
pub type Metadata = HashMap<String, DataMap>;

/// Minimal metadata value type used by the compatibility wrapper.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataMap {
    UInt(u64),
    UIntAllay(Vec<u64>),
    Raw(Vec<u8>),
    Ascii(String),
    None,
}

/// Callback response command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseCommand {
    Abort,
    Continue,
}

/// Response returned by compatibility callbacks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CallbackResponse {
    pub response: ResponseCommand,
}

impl CallbackResponse {
    pub fn abort() -> Self {
        Self {
            response: ResponseCommand::Abort,
        }
    }

    pub fn cont() -> Self {
        Self {
            response: ResponseCommand::Continue,
        }
    }
}

/// Receives decoded image data from compatibility decode entry points.
pub trait DrawCallback: Sync + Send {
    fn init(
        &mut self,
        width: usize,
        height: usize,
        option: Option<InitOptions>,
    ) -> Result<Option<CallbackResponse>, Error>;
    fn draw(
        &mut self,
        start_x: usize,
        start_y: usize,
        width: usize,
        height: usize,
        data: &[u8],
        option: Option<DrawOptions>,
    ) -> Result<Option<CallbackResponse>, Error>;
    /// Starts the next animation frame.
    ///
    /// This method has a default implementation so existing compatibility
    /// clients remain source-compatible. Decoders that expose animation call
    /// it before each frame's draw operation.
    fn next(&mut self, _option: Option<NextOptions>) -> Result<Option<CallbackResponse>, Error> {
        Ok(Some(CallbackResponse::cont()))
    }
    fn terminate(
        &mut self,
        term: Option<TerminateOptions>,
    ) -> Result<Option<CallbackResponse>, Error>;
    fn verbose(
        &mut self,
        verbose: &str,
        option: Option<VerboseOptions>,
    ) -> Result<Option<CallbackResponse>, Error>;
    fn set_metadata(
        &mut self,
        key: &str,
        value: DataMap,
    ) -> Result<Option<CallbackResponse>, Error>;
}

/// Decoder initialization options.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InitOptions {
    pub loop_count: u32,
    pub animation: bool,
}

/// Draw options placeholder kept for shape compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DrawOptions {}

/// A rectangle occupied by one animation frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageRect {
    pub start_x: i32,
    pub start_y: i32,
    pub width: usize,
    pub height: usize,
}

/// Disposal behavior for the preceding animation frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NextDispose {
    None,
    Background,
    Previous,
}

/// Blending behavior for the animation frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NextBlend {
    Source,
    Override,
}

/// Per-frame animation timing and composition options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NextOptions {
    pub await_time: u64,
    pub image_rect: Option<ImageRect>,
    pub dispose: NextDispose,
    pub blend: NextBlend,
}

impl NextOptions {
    pub fn full_canvas(width: usize, height: usize, await_time: u64) -> Self {
        Self {
            await_time,
            image_rect: Some(ImageRect {
                start_x: 0,
                start_y: 0,
                width,
                height,
            }),
            dispose: NextDispose::None,
            blend: NextBlend::Override,
        }
    }
}

/// Termination options placeholder kept for shape compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TerminateOptions {}

/// Verbose options placeholder kept for shape compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VerboseOptions {}

/// Decoder call options.
pub struct DecodeOptions<'a> {
    pub debug_flag: usize,
    pub drawer: &'a mut dyn DrawCallback,
    pub options: Option<Metadata>,
}

impl<'a> DecodeOptions<'a> {
    pub fn new(drawer: &'a mut dyn DrawCallback) -> Self {
        Self {
            debug_flag: 0,
            drawer,
            options: None,
        }
    }
}
