#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockSize {
    Block4x4,
    Block4x8,
    Block8x4,
    Block8x8,
    Block8x16,
    Block16x8,
    Block16x16,
    Block16x32,
    Block32x16,
    Block32x32,
    Block32x64,
    Block64x32,
    Block64x64,
    Block64x128,
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

    pub fn width_mi_log2(self) -> u8 {
        match self {
            Self::Block4x4 | Self::Block4x8 => 0,
            Self::Block8x4 | Self::Block8x8 | Self::Block8x16 => 1,
            Self::Block16x8 | Self::Block16x16 | Self::Block16x32 => 2,
            Self::Block32x16 | Self::Block32x32 | Self::Block32x64 => 3,
            Self::Block64x32 | Self::Block64x64 | Self::Block64x128 => 4,
            Self::Block128x64 | Self::Block128x128 => 5,
        }
    }

    pub fn height_mi_log2(self) -> u8 {
        match self {
            Self::Block4x4 | Self::Block8x4 => 0,
            Self::Block4x8 | Self::Block8x8 | Self::Block16x8 => 1,
            Self::Block8x16 | Self::Block16x16 | Self::Block32x16 => 2,
            Self::Block16x32 | Self::Block32x32 | Self::Block64x32 => 3,
            Self::Block32x64 | Self::Block64x64 | Self::Block128x64 => 4,
            Self::Block64x128 | Self::Block128x128 => 5,
        }
    }

    pub fn width(self) -> usize {
        4usize << self.width_mi_log2()
    }

    pub fn height(self) -> usize {
        4usize << self.height_mi_log2()
    }

    pub fn size_group(self) -> usize {
        match self.width().max(self.height()) {
            0..=8 => 0,
            9..=16 => 1,
            17..=32 => 2,
            _ => 3,
        }
    }

    pub fn largest_supported_tx_size(self) -> TxSize {
        let max_side = self.width().max(self.height()).min(32);
        match max_side {
            0..=4 => TxSize::Tx4x4,
            5..=8 => TxSize::Tx8x8,
            9..=16 => TxSize::Tx16x16,
            _ => TxSize::Tx32x32,
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
                4 => Some(Self::Horizontal4),
                5 => Some(Self::Vertical4),
                6 => Some(Self::HorizontalA),
                7 => Some(Self::VerticalA),
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
pub enum TxSize {
    Tx4x4,
    Tx8x8,
    Tx16x16,
    Tx32x32,
    Tx64x64,
}

impl TxSize {
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
        }
    }

    pub fn height_log2(self) -> u8 {
        self.width_log2()
    }

    pub fn sample_count(self) -> usize {
        self.width() * self.height()
    }

    pub fn row_shift(self) -> u8 {
        match self {
            Self::Tx4x4 => 0,
            Self::Tx8x8 => 1,
            Self::Tx16x16 | Self::Tx32x32 | Self::Tx64x64 => 2,
        }
    }

    pub fn dq_denom(self) -> i32 {
        match self {
            Self::Tx32x32 => 2,
            Self::Tx64x64 => 4,
            _ => 1,
        }
    }

    pub fn coeff_cdf_index(self) -> usize {
        match self {
            Self::Tx4x4 => 0,
            Self::Tx8x8 => 1,
            Self::Tx16x16 => 2,
            Self::Tx32x32 => 3,
            Self::Tx64x64 => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxType {
    DctDct,
    AdstDct,
    DctAdst,
    AdstAdst,
    Identity,
    VerticalDct,
    HorizontalDct,
}

impl TxType {
    pub fn from_intra_ext_tx_set1_symbol(symbol: usize) -> Option<Self> {
        match symbol {
            0 => Some(Self::DctDct),
            1 => Some(Self::AdstDct),
            2 => Some(Self::DctAdst),
            3 => Some(Self::AdstAdst),
            4 => Some(Self::Identity),
            5 => Some(Self::VerticalDct),
            6 => Some(Self::HorizontalDct),
            _ => None,
        }
    }

    pub fn from_intra_ext_tx_set2_symbol(symbol: usize) -> Option<Self> {
        match symbol {
            0 => Some(Self::DctDct),
            1 => Some(Self::Identity),
            2 => Some(Self::VerticalDct),
            3 => Some(Self::HorizontalDct),
            4 => Some(Self::AdstAdst),
            _ => None,
        }
    }
}
