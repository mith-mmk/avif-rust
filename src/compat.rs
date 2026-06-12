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
