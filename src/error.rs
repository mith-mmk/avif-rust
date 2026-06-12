use std::error::Error;
use std::fmt::{self, Display, Formatter};

/// Errors returned by AVIF parsing and decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecoderError {
    InvalidParam(String),
    NotEnoughData(String),
    Bitstream(String),
    Unsupported(String),
    Io(String),
}

impl Display for DecoderError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParam(message) => write!(f, "invalid parameter: {message}"),
            Self::NotEnoughData(message) => write!(f, "not enough data: {message}"),
            Self::Bitstream(message) => write!(f, "bitstream error: {message}"),
            Self::Unsupported(message) => write!(f, "unsupported feature: {message}"),
            Self::Io(message) => write!(f, "io error: {message}"),
        }
    }
}

impl Error for DecoderError {}
