pub(crate) fn mi_dimension(frame_dimension: u32) -> u32 {
    frame_dimension.div_ceil(8) * 2
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockSize {
    Block4x4,
    Block4x8,
    Block4x16,
    Block8x4,
    Block8x8,
    Block8x16,
    Block8x32,
    Block16x4,
    Block16x8,
    Block16x16,
    Block16x32,
    Block16x64,
    Block32x8,
    Block32x16,
    Block32x32,
    Block32x64,
    Block32x128,
    Block64x16,
    Block64x32,
    Block64x64,
    Block64x128,
    Block128x32,
    Block128x64,
    Block128x128,
}

impl BlockSize {
    pub fn square(width_mi_log2: u8) -> Self {
        match width_mi_log2 {
            0 => Self::Block4x4,
            1 => Self::Block8x8,
            2 => Self::Block16x16,
            3 => Self::Block32x32,
            4 => Self::Block64x64,
            _ => Self::Block128x128,
        }
    }

    pub fn from_dimensions(width: usize, height: usize) -> Option<Self> {
        match (width, height) {
            (4, 4) => Some(Self::Block4x4),
            (4, 8) => Some(Self::Block4x8),
            (4, 16) => Some(Self::Block4x16),
            (8, 4) => Some(Self::Block8x4),
            (8, 8) => Some(Self::Block8x8),
            (8, 16) => Some(Self::Block8x16),
            (8, 32) => Some(Self::Block8x32),
            (16, 4) => Some(Self::Block16x4),
            (16, 8) => Some(Self::Block16x8),
            (16, 16) => Some(Self::Block16x16),
            (16, 32) => Some(Self::Block16x32),
            (16, 64) => Some(Self::Block16x64),
            (32, 8) => Some(Self::Block32x8),
            (32, 16) => Some(Self::Block32x16),
            (32, 32) => Some(Self::Block32x32),
            (32, 64) => Some(Self::Block32x64),
            (32, 128) => Some(Self::Block32x128),
            (64, 16) => Some(Self::Block64x16),
            (64, 32) => Some(Self::Block64x32),
            (64, 64) => Some(Self::Block64x64),
            (64, 128) => Some(Self::Block64x128),
            (128, 32) => Some(Self::Block128x32),
            (128, 64) => Some(Self::Block128x64),
            (128, 128) => Some(Self::Block128x128),
            _ => None,
        }
    }

    pub fn width_mi_log2(self) -> u8 {
        match self {
            Self::Block4x4 | Self::Block4x8 | Self::Block4x16 => 0,
            Self::Block8x4 | Self::Block8x8 | Self::Block8x16 | Self::Block8x32 => 1,
            Self::Block16x4
            | Self::Block16x8
            | Self::Block16x16
            | Self::Block16x32
            | Self::Block16x64 => 2,
            Self::Block32x8
            | Self::Block32x16
            | Self::Block32x32
            | Self::Block32x64
            | Self::Block32x128 => 3,
            Self::Block64x16 | Self::Block64x32 | Self::Block64x64 | Self::Block64x128 => 4,
            Self::Block128x32 | Self::Block128x64 | Self::Block128x128 => 5,
        }
    }

    pub fn height_mi_log2(self) -> u8 {
        match self {
            Self::Block4x4 | Self::Block8x4 | Self::Block16x4 => 0,
            Self::Block4x8 | Self::Block8x8 | Self::Block16x8 | Self::Block32x8 => 1,
            Self::Block4x16
            | Self::Block8x16
            | Self::Block16x16
            | Self::Block32x16
            | Self::Block64x16 => 2,
            Self::Block8x32
            | Self::Block16x32
            | Self::Block32x32
            | Self::Block64x32
            | Self::Block128x32 => 3,
            Self::Block16x64 | Self::Block32x64 | Self::Block64x64 | Self::Block128x64 => 4,
            Self::Block32x128 | Self::Block64x128 | Self::Block128x128 => 5,
        }
    }

    pub fn width(self) -> usize {
        4usize << self.width_mi_log2()
    }

    pub fn height(self) -> usize {
        4usize << self.height_mi_log2()
    }

    pub fn motion_mode_cdf_index(self) -> usize {
        match self {
            Self::Block4x4 => 0,
            Self::Block4x8 => 1,
            Self::Block8x4 => 2,
            Self::Block8x8 => 3,
            Self::Block8x16 => 4,
            Self::Block16x8 => 5,
            Self::Block16x16 => 6,
            Self::Block16x32 => 7,
            Self::Block32x16 => 8,
            Self::Block32x32 => 9,
            Self::Block32x64 => 10,
            Self::Block64x32 => 11,
            Self::Block64x64 => 12,
            Self::Block64x128 => 13,
            Self::Block128x64 => 14,
            Self::Block128x128 => 15,
            Self::Block4x16 => 16,
            Self::Block16x4 => 17,
            Self::Block8x32 => 18,
            Self::Block32x8 => 19,
            Self::Block16x64 => 20,
            Self::Block64x16 | Self::Block32x128 | Self::Block128x32 => 21,
        }
    }

    pub fn size_group(self) -> usize {
        usize::from(self.width_mi_log2().min(self.height_mi_log2()).min(3))
    }

    pub fn filter_intra_cdf_index(self) -> usize {
        match self {
            Self::Block4x4 => 0,
            Self::Block4x8 => 1,
            Self::Block8x4 => 2,
            Self::Block8x8 => 3,
            Self::Block8x16 => 4,
            Self::Block16x8 => 5,
            Self::Block16x16 => 6,
            Self::Block16x32 => 7,
            Self::Block32x16 => 8,
            Self::Block32x32 => 9,
            Self::Block32x64 => 10,
            Self::Block64x32 => 11,
            Self::Block64x64 => 12,
            Self::Block64x128 => 13,
            Self::Block128x64 => 14,
            Self::Block128x128 => 15,
            Self::Block4x16 => 16,
            Self::Block16x4 => 17,
            Self::Block8x32 => 18,
            Self::Block32x8 => 19,
            Self::Block16x64 => 20,
            Self::Block64x16 => 21,
            Self::Block32x128 => 10,
            Self::Block128x32 => 11,
        }
    }

    pub fn largest_supported_tx_size(self) -> TxSize {
        let side = self.width().min(self.height()).min(64);
        match side {
            0..=4 => TxSize::Tx4x4,
            5..=8 => TxSize::Tx8x8,
            9..=16 => TxSize::Tx16x16,
            17..=32 => TxSize::Tx32x32,
            _ => TxSize::Tx64x64,
        }
    }

    pub fn largest_supported_rect_tx_size(self) -> TxSize {
        match self {
            Self::Block4x4 => TxSize::Tx4x4,
            Self::Block4x8 => TxSize::Tx4x8,
            Self::Block8x4 => TxSize::Tx8x4,
            Self::Block8x8 => TxSize::Tx8x8,
            Self::Block8x16 => TxSize::Tx8x16,
            Self::Block16x8 => TxSize::Tx16x8,
            Self::Block16x16 => TxSize::Tx16x16,
            Self::Block16x32 => TxSize::Tx16x32,
            Self::Block32x16 => TxSize::Tx32x16,
            Self::Block32x32 => TxSize::Tx32x32,
            Self::Block32x64 => TxSize::Tx32x64,
            Self::Block64x32 => TxSize::Tx64x32,
            Self::Block64x64
            | Self::Block32x128
            | Self::Block128x32
            | Self::Block64x128
            | Self::Block128x64
            | Self::Block128x128 => TxSize::Tx64x64,
            Self::Block4x16 => TxSize::Tx4x16,
            Self::Block16x4 => TxSize::Tx16x4,
            Self::Block8x32 => TxSize::Tx8x32,
            Self::Block32x8 => TxSize::Tx32x8,
            Self::Block16x64 => TxSize::Tx16x64,
            Self::Block64x16 => TxSize::Tx64x16,
        }
    }

    pub fn largest_supported_tx_dimensions(self) -> (usize, usize) {
        (self.width().min(64), self.height().min(64))
    }

    pub fn signals_tx_size(self) -> bool {
        self != Self::Block4x4
    }

    pub fn tx_size_category(self) -> usize {
        match self {
            Self::Block4x4 => 0,
            Self::Block4x8 | Self::Block8x4 | Self::Block8x8 => 0,
            Self::Block8x16
            | Self::Block16x8
            | Self::Block16x16
            | Self::Block4x16
            | Self::Block16x4 => 1,
            Self::Block16x32
            | Self::Block32x16
            | Self::Block32x32
            | Self::Block8x32
            | Self::Block32x8 => 2,
            _ => 3,
        }
    }

    pub fn max_tx_size_depth(self) -> usize {
        match self {
            Self::Block4x4 => 0,
            Self::Block4x8 | Self::Block8x4 | Self::Block8x8 => 1,
            Self::Block8x16
            | Self::Block16x8
            | Self::Block16x16
            | Self::Block16x32
            | Self::Block32x16
            | Self::Block32x32
            | Self::Block8x32
            | Self::Block32x8
            | Self::Block4x16
            | Self::Block16x4 => 2,
            _ => 2,
        }
    }

    pub fn tx_size_from_depth(self, depth: usize) -> TxSize {
        let mut tx_size = if self.width() == self.height() {
            self.largest_supported_tx_size()
        } else {
            self.largest_supported_rect_tx_size()
        };
        for _ in 0..depth.min(self.max_tx_size_depth()) {
            tx_size = tx_size.sub_size();
        }
        tx_size
    }

    pub fn split_subsize(self) -> Option<Self> {
        if self.width_mi_log2() == 0 || self.height_mi_log2() == 0 {
            return None;
        }
        Self::from_dimensions(self.width() / 2, self.height() / 2)
    }

    pub fn horizontal_subsize(self) -> Option<Self> {
        if self.height() <= 4 {
            return None;
        }
        Self::from_dimensions(self.width(), self.height() / 2)
    }

    pub fn vertical_subsize(self) -> Option<Self> {
        if self.width() <= 4 {
            return None;
        }
        Self::from_dimensions(self.width() / 2, self.height())
    }

    pub fn horizontal_4_subsize(self) -> Option<Self> {
        if self.height() <= 4 || !self.height().is_multiple_of(4) {
            return None;
        }
        Self::from_dimensions(self.width(), self.height() / 4)
    }

    pub fn vertical_4_subsize(self) -> Option<Self> {
        if self.width() <= 4 || !self.width().is_multiple_of(4) {
            return None;
        }
        Self::from_dimensions(self.width() / 4, self.height())
    }

    pub fn partition_contexts(self) -> (u8, u8) {
        match self {
            Self::Block4x4 => (31, 31),
            Self::Block4x8 => (31, 30),
            Self::Block8x4 => (30, 31),
            Self::Block8x8 => (30, 30),
            Self::Block8x16 => (30, 28),
            Self::Block16x8 => (28, 30),
            Self::Block16x16 => (28, 28),
            Self::Block16x32 => (28, 24),
            Self::Block32x16 => (24, 28),
            Self::Block32x32 => (24, 24),
            Self::Block32x64 => (24, 16),
            Self::Block64x32 => (16, 24),
            Self::Block64x64 => (16, 16),
            Self::Block64x128 => (16, 0),
            Self::Block128x64 => (0, 16),
            Self::Block128x128 => (0, 0),
            Self::Block4x16 => (31, 28),
            Self::Block16x4 => (28, 31),
            Self::Block8x32 => (30, 24),
            Self::Block32x8 => (24, 30),
            Self::Block16x64 => (28, 16),
            Self::Block64x16 => (16, 28),
            Self::Block32x128 => (24, 0),
            Self::Block128x32 => (0, 24),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Partition {
    None,
    Horizontal,
    Vertical,
    Split,
    HorizontalA,
    HorizontalB,
    VerticalA,
    VerticalB,
    Horizontal4,
    Vertical4,
}

impl Partition {
    pub fn from_symbol(block_size: BlockSize, symbol: usize) -> Option<Self> {
        match block_size.width_mi_log2() {
            1 => match symbol {
                0 => Some(Self::None),
                1 => Some(Self::Horizontal),
                2 => Some(Self::Vertical),
                3 => Some(Self::Split),
                _ => None,
            },
            5 => match symbol {
                0 => Some(Self::None),
                1 => Some(Self::Horizontal),
                2 => Some(Self::Vertical),
                3 => Some(Self::Split),
                4 => Some(Self::HorizontalA),
                5 => Some(Self::HorizontalB),
                6 => Some(Self::VerticalA),
                7 => Some(Self::VerticalB),
                _ => None,
            },
            _ => match symbol {
                0 => Some(Self::None),
                1 => Some(Self::Horizontal),
                2 => Some(Self::Vertical),
                3 => Some(Self::Split),
                4 => Some(Self::HorizontalA),
                5 => Some(Self::HorizontalB),
                6 => Some(Self::VerticalA),
                7 => Some(Self::VerticalB),
                8 => Some(Self::Horizontal4),
                9 => Some(Self::Vertical4),
                _ => None,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictionMode {
    Dc,
    Vertical,
    Horizontal,
    D45,
    D135,
    D113,
    D157,
    D203,
    D67,
    Smooth,
    SmoothVertical,
    SmoothHorizontal,
    Paeth,
}

impl PredictionMode {
    pub fn from_intra_symbol(symbol: usize) -> Option<Self> {
        match symbol {
            0 => Some(Self::Dc),
            1 => Some(Self::Vertical),
            2 => Some(Self::Horizontal),
            3 => Some(Self::D45),
            4 => Some(Self::D135),
            5 => Some(Self::D113),
            6 => Some(Self::D157),
            7 => Some(Self::D203),
            8 => Some(Self::D67),
            9 => Some(Self::Smooth),
            10 => Some(Self::SmoothVertical),
            11 => Some(Self::SmoothHorizontal),
            12 => Some(Self::Paeth),
            _ => None,
        }
    }

    pub fn is_directional(self) -> bool {
        matches!(
            self,
            Self::Vertical
                | Self::Horizontal
                | Self::D45
                | Self::D135
                | Self::D113
                | Self::D157
                | Self::D203
                | Self::D67
        )
    }

    pub fn is_smooth(self) -> bool {
        matches!(
            self,
            Self::Smooth | Self::SmoothVertical | Self::SmoothHorizontal
        )
    }

    pub fn directional_index(self) -> Option<usize> {
        match self {
            Self::Vertical => Some(0),
            Self::Horizontal => Some(1),
            Self::D45 => Some(2),
            Self::D135 => Some(3),
            Self::D113 => Some(4),
            Self::D157 => Some(5),
            Self::D203 => Some(6),
            Self::D67 => Some(7),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UvPredictionMode {
    Intra(PredictionMode),
    Cfl,
}

impl UvPredictionMode {
    pub fn from_symbol(symbol: usize) -> Option<Self> {
        if symbol == 13 {
            return Some(Self::Cfl);
        }
        PredictionMode::from_intra_symbol(symbol).map(Self::Intra)
    }

    pub fn is_directional(self) -> bool {
        matches!(self, Self::Intra(mode) if mode.is_directional())
    }

    pub fn directional_index(self) -> Option<usize> {
        match self {
            Self::Intra(mode) => mode.directional_index(),
            Self::Cfl => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TxSize {
    Tx4x4,
    Tx8x8,
    Tx16x16,
    Tx32x32,
    Tx64x64,
    Tx4x8,
    /// AV1's first rectangular transform (8 columns by 4 rows).
    Tx8x4,
    Tx8x16,
    Tx16x8,
    Tx16x32,
    Tx32x16,
    Tx32x64,
    Tx64x32,
    Tx4x16,
    Tx16x4,
    Tx8x32,
    Tx32x8,
    Tx16x64,
    Tx64x16,
}

impl TxSize {
    pub fn from_dimensions(width: usize, height: usize) -> Option<Self> {
        match (width, height) {
            (4, 4) => Some(Self::Tx4x4),
            (8, 8) => Some(Self::Tx8x8),
            (16, 16) => Some(Self::Tx16x16),
            (32, 32) => Some(Self::Tx32x32),
            (64, 64) => Some(Self::Tx64x64),
            (4, 8) => Some(Self::Tx4x8),
            (8, 4) => Some(Self::Tx8x4),
            (8, 16) => Some(Self::Tx8x16),
            (16, 8) => Some(Self::Tx16x8),
            (16, 32) => Some(Self::Tx16x32),
            (32, 16) => Some(Self::Tx32x16),
            (32, 64) => Some(Self::Tx32x64),
            (64, 32) => Some(Self::Tx64x32),
            (4, 16) => Some(Self::Tx4x16),
            (16, 4) => Some(Self::Tx16x4),
            (8, 32) => Some(Self::Tx8x32),
            (32, 8) => Some(Self::Tx32x8),
            (16, 64) => Some(Self::Tx16x64),
            (64, 16) => Some(Self::Tx64x16),
            _ => None,
        }
    }

    pub fn width(self) -> usize {
        1usize << self.width_log2()
    }

    pub fn height(self) -> usize {
        1usize << self.height_log2()
    }

    pub fn width_log2(self) -> u8 {
        match self {
            Self::Tx4x4 => 2,
            Self::Tx8x8 => 3,
            Self::Tx16x16 => 4,
            Self::Tx32x32 => 5,
            Self::Tx64x64 => 6,
            Self::Tx4x8 | Self::Tx4x16 => 2,
            Self::Tx8x4 => 3,
            Self::Tx8x16 | Self::Tx8x32 => 3,
            Self::Tx16x8 | Self::Tx16x32 | Self::Tx16x4 | Self::Tx16x64 => 4,
            Self::Tx32x16 | Self::Tx32x64 | Self::Tx32x8 => 5,
            Self::Tx64x32 | Self::Tx64x16 => 6,
        }
    }

    pub fn height_log2(self) -> u8 {
        match self {
            Self::Tx4x8 => 3,
            Self::Tx8x16 | Self::Tx4x16 => 4,
            Self::Tx16x32 | Self::Tx8x32 => 5,
            Self::Tx32x64 | Self::Tx16x64 => 6,
            Self::Tx16x8 | Self::Tx32x8 => 3,
            Self::Tx32x16 | Self::Tx64x16 => 4,
            Self::Tx64x32 => 5,
            Self::Tx16x4 => 2,
            Self::Tx8x4 => 2,
            _ => self.width_log2(),
        }
    }

    pub fn sample_count(self) -> usize {
        self.width() * self.height()
    }

    pub fn row_shift(self) -> u8 {
        match self {
            Self::Tx4x4 | Self::Tx4x8 | Self::Tx8x4 => 0,
            Self::Tx8x8 | Self::Tx8x16 | Self::Tx16x8 => 1,
            Self::Tx16x16
            | Self::Tx32x32
            | Self::Tx64x64
            | Self::Tx16x32
            | Self::Tx32x16
            | Self::Tx32x64
            | Self::Tx64x32
            | Self::Tx4x16
            | Self::Tx16x4 => 2,
            Self::Tx8x32 | Self::Tx32x8 | Self::Tx16x64 | Self::Tx64x16 => 2,
        }
    }

    pub fn dq_denom(self) -> i32 {
        match self.sample_count() {
            samples if samples > 1024 => 4,
            samples if samples > 256 => 2,
            _ => 1,
        }
    }

    pub fn coeff_cdf_index(self) -> usize {
        match self {
            Self::Tx4x4 => 0,
            Self::Tx8x8 | Self::Tx4x8 | Self::Tx8x4 | Self::Tx4x16 | Self::Tx16x4 => 1,
            Self::Tx16x16 | Self::Tx8x16 | Self::Tx16x8 | Self::Tx8x32 | Self::Tx32x8 => 2,
            Self::Tx32x32 | Self::Tx16x32 | Self::Tx32x16 | Self::Tx16x64 | Self::Tx64x16 => 3,
            Self::Tx64x64 | Self::Tx32x64 | Self::Tx64x32 => 4,
        }
    }

    pub fn sub_size(self) -> Self {
        match self {
            Self::Tx64x64 => Self::Tx32x32,
            Self::Tx32x32 => Self::Tx16x16,
            Self::Tx16x16 => Self::Tx8x8,
            Self::Tx8x8 => Self::Tx4x4,
            Self::Tx4x4 => Self::Tx4x4,
            Self::Tx4x8 | Self::Tx8x4 => Self::Tx4x4,
            Self::Tx8x16 | Self::Tx16x8 => Self::Tx8x8,
            Self::Tx16x32 | Self::Tx32x16 => Self::Tx16x16,
            Self::Tx32x64 | Self::Tx64x32 => Self::Tx32x32,
            Self::Tx4x16 => Self::Tx4x8,
            Self::Tx16x4 => Self::Tx8x4,
            Self::Tx8x32 => Self::Tx8x16,
            Self::Tx32x8 => Self::Tx16x8,
            Self::Tx16x64 => Self::Tx16x32,
            Self::Tx64x16 => Self::Tx32x16,
        }
    }

    pub fn is_rectangular(self) -> bool {
        self.width() != self.height()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TxType {
    DctDct,
    AdstDct,
    DctAdst,
    AdstAdst,
    Identity,
    VerticalDct,
    HorizontalDct,
    FlipAdstDct,
    DctFlipAdst,
    FlipAdstFlipAdst,
    AdstFlipAdst,
    FlipAdstAdst,
    VerticalAdst,
    HorizontalAdst,
    VerticalFlipAdst,
    HorizontalFlipAdst,
}

pub const TX_TYPES: usize = 16;

impl TxType {
    pub fn from_intra_ext_tx_set1_symbol(symbol: usize) -> Option<Self> {
        match symbol {
            0 => Some(Self::Identity),
            1 => Some(Self::DctDct),
            2 => Some(Self::VerticalDct),
            3 => Some(Self::HorizontalDct),
            4 => Some(Self::AdstAdst),
            5 => Some(Self::AdstDct),
            6 => Some(Self::DctAdst),
            _ => None,
        }
    }

    pub fn from_intra_ext_tx_set2_symbol(symbol: usize) -> Option<Self> {
        match symbol {
            0 => Some(Self::Identity),
            1 => Some(Self::DctDct),
            2 => Some(Self::AdstAdst),
            3 => Some(Self::AdstDct),
            4 => Some(Self::DctAdst),
            _ => None,
        }
    }

    pub fn from_inter_ext_tx_set3_symbol(symbol: usize) -> Option<Self> {
        match symbol {
            0 => Some(Self::Identity),
            1 => Some(Self::DctDct),
            _ => None,
        }
    }

    pub fn from_inter_ext_tx_set1_symbol(symbol: usize) -> Option<Self> {
        Some(match symbol {
            0 => Self::Identity,
            1 => Self::VerticalDct,
            2 => Self::HorizontalDct,
            3 => Self::VerticalAdst,
            4 => Self::HorizontalAdst,
            5 => Self::VerticalFlipAdst,
            6 => Self::HorizontalFlipAdst,
            7 => Self::DctDct,
            8 => Self::AdstDct,
            9 => Self::DctAdst,
            10 => Self::FlipAdstDct,
            11 => Self::DctFlipAdst,
            12 => Self::AdstAdst,
            13 => Self::FlipAdstFlipAdst,
            14 => Self::AdstFlipAdst,
            15 => Self::FlipAdstAdst,
            _ => return None,
        })
    }

    pub fn from_inter_ext_tx_set2_symbol(symbol: usize) -> Option<Self> {
        Some(match symbol {
            0 => Self::Identity,
            1 => Self::VerticalAdst,
            2 => Self::HorizontalAdst,
            3 => Self::DctDct,
            4 => Self::AdstDct,
            5 => Self::DctAdst,
            6 => Self::FlipAdstDct,
            7 => Self::DctFlipAdst,
            8 => Self::AdstAdst,
            9 => Self::FlipAdstFlipAdst,
            10 => Self::AdstFlipAdst,
            11 => Self::FlipAdstAdst,
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mi_dimension_rounds_frames_to_complete_eight_pixel_units() {
        assert_eq!(mi_dimension(1), 2);
        assert_eq!(mi_dimension(4), 2);
        assert_eq!(mi_dimension(8), 2);
        assert_eq!(mi_dimension(9), 4);
        assert_eq!(mi_dimension(900), 226);
    }

    #[test]
    fn square_block_split_subsize_halves_dimensions() {
        assert_eq!(
            BlockSize::Block128x128.split_subsize(),
            Some(BlockSize::Block64x64)
        );
        assert_eq!(
            BlockSize::Block64x64.split_subsize(),
            Some(BlockSize::Block32x32)
        );
        assert_eq!(BlockSize::Block4x4.split_subsize(), None);
        assert_eq!(
            BlockSize::Block64x32.split_subsize(),
            Some(BlockSize::Block32x16)
        );
    }

    #[test]
    fn block_size_from_dimensions_maps_supported_shapes() {
        assert_eq!(
            BlockSize::from_dimensions(128, 64),
            Some(BlockSize::Block128x64)
        );
        assert_eq!(
            BlockSize::from_dimensions(64, 128),
            Some(BlockSize::Block64x128)
        );
        assert_eq!(
            BlockSize::from_dimensions(128, 32),
            Some(BlockSize::Block128x32)
        );
        assert_eq!(BlockSize::from_dimensions(2, 4), None);
    }

    #[test]
    fn rectangular_blocks_retain_transform_context_dimensions() {
        assert_eq!(
            BlockSize::Block8x4.largest_supported_tx_dimensions(),
            (8, 4)
        );
        assert_eq!(
            BlockSize::Block4x8.largest_supported_tx_dimensions(),
            (4, 8)
        );
    }

    #[test]
    fn partition_context_lookup_matches_av1_table() {
        assert_eq!(BlockSize::Block4x4.partition_contexts(), (31, 31));
        assert_eq!(BlockSize::Block16x32.partition_contexts(), (28, 24));
        assert_eq!(BlockSize::Block64x128.partition_contexts(), (16, 0));
        assert_eq!(BlockSize::Block128x32.partition_contexts(), (0, 24));
    }

    #[test]
    fn block128_partition_symbols_exclude_four_way_partitions() {
        assert_eq!(
            Partition::from_symbol(BlockSize::Block128x128, 4),
            Some(Partition::HorizontalA)
        );
        assert_eq!(
            Partition::from_symbol(BlockSize::Block128x128, 5),
            Some(Partition::HorizontalB)
        );
        assert_eq!(
            Partition::from_symbol(BlockSize::Block128x128, 7),
            Some(Partition::VerticalB)
        );
        assert_eq!(Partition::from_symbol(BlockSize::Block128x128, 8), None);
    }

    #[test]
    fn horizontal_and_vertical_subsize_halves_one_axis() {
        assert_eq!(
            BlockSize::Block128x128.horizontal_subsize(),
            Some(BlockSize::Block128x64)
        );
        assert_eq!(
            BlockSize::Block128x128.vertical_subsize(),
            Some(BlockSize::Block64x128)
        );
        assert_eq!(
            BlockSize::Block8x8.horizontal_subsize(),
            Some(BlockSize::Block8x4)
        );
        assert_eq!(
            BlockSize::Block8x8.vertical_subsize(),
            Some(BlockSize::Block4x8)
        );
        assert_eq!(BlockSize::Block8x4.horizontal_subsize(), None);
        assert_eq!(BlockSize::Block4x8.vertical_subsize(), None);
    }

    #[test]
    fn horizontal_and_vertical_4_subsize_quarters_one_axis() {
        assert_eq!(
            BlockSize::Block128x128.horizontal_4_subsize(),
            Some(BlockSize::Block128x32)
        );
        assert_eq!(
            BlockSize::Block64x128.horizontal_4_subsize(),
            Some(BlockSize::Block64x32)
        );
        assert_eq!(
            BlockSize::Block128x64.vertical_4_subsize(),
            Some(BlockSize::Block32x64)
        );
        assert_eq!(BlockSize::Block16x8.horizontal_4_subsize(), None);
        assert_eq!(BlockSize::Block8x16.vertical_4_subsize(), None);
    }

    #[test]
    fn tx_size_depth_maps_to_square_tx_size() {
        assert_eq!(BlockSize::Block8x8.tx_size_category(), 0);
        assert_eq!(BlockSize::Block8x8.tx_size_from_depth(0), TxSize::Tx8x8);
        assert_eq!(BlockSize::Block8x8.tx_size_from_depth(1), TxSize::Tx4x4);
        assert_eq!(BlockSize::Block16x16.tx_size_category(), 1);
        assert_eq!(BlockSize::Block32x32.tx_size_category(), 2);
        assert_eq!(BlockSize::Block32x32.tx_size_from_depth(0), TxSize::Tx32x32);
        assert_eq!(BlockSize::Block32x32.tx_size_from_depth(2), TxSize::Tx8x8);
        assert_eq!(BlockSize::Block64x64.tx_size_category(), 3);
        assert_eq!(BlockSize::Block64x64.tx_size_from_depth(1), TxSize::Tx32x32);
    }

    #[test]
    fn intra_ext_tx_symbols_follow_av1_inverse_tables() {
        assert_eq!(
            TxType::from_intra_ext_tx_set1_symbol(0),
            Some(TxType::Identity)
        );
        assert_eq!(
            TxType::from_intra_ext_tx_set1_symbol(2),
            Some(TxType::VerticalDct)
        );
        assert_eq!(
            TxType::from_intra_ext_tx_set1_symbol(6),
            Some(TxType::DctAdst)
        );
        assert_eq!(
            TxType::from_intra_ext_tx_set2_symbol(2),
            Some(TxType::AdstAdst)
        );
        assert_eq!(TxType::from_intra_ext_tx_set2_symbol(5), None);
    }
}
