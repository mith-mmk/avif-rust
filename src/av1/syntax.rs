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

    pub fn from_dimensions(width: usize, height: usize) -> Option<Self> {
        match (width, height) {
            (4, 4) => Some(Self::Block4x4),
            (4, 8) => Some(Self::Block4x8),
            (8, 4) => Some(Self::Block8x4),
            (8, 8) => Some(Self::Block8x8),
            (8, 16) => Some(Self::Block8x16),
            (16, 8) => Some(Self::Block16x8),
            (16, 16) => Some(Self::Block16x16),
            (16, 32) => Some(Self::Block16x32),
            (32, 16) => Some(Self::Block32x16),
            (32, 32) => Some(Self::Block32x32),
            (32, 64) => Some(Self::Block32x64),
            (64, 32) => Some(Self::Block64x32),
            (64, 64) => Some(Self::Block64x64),
            (64, 128) => Some(Self::Block64x128),
            (128, 64) => Some(Self::Block128x64),
            (128, 128) => Some(Self::Block128x128),
            _ => None,
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

    pub fn split_subsize(self) -> Option<Self> {
        if self.width() != self.height() || self.width_mi_log2() == 0 {
            return None;
        }
        Some(Self::square(self.width_mi_log2() - 1))
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
        if self.height() <= 4 || self.height() % 4 != 0 {
            return None;
        }
        Self::from_dimensions(self.width(), self.height() / 4)
    }

    pub fn vertical_4_subsize(self) -> Option<Self> {
        if self.width() <= 4 || self.width() % 4 != 0 {
            return None;
        }
        Self::from_dimensions(self.width() / 4, self.height())
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

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(BlockSize::Block64x32.split_subsize(), None);
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
        assert_eq!(BlockSize::from_dimensions(128, 32), None);
        assert_eq!(BlockSize::from_dimensions(2, 4), None);
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
            None,
            "128x32 is not represented by the supported MVP block-size enum"
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
}
