pub mod interpolation_search;

pub use interpolation_search::{
    BinaryInterpolation, BoundedLinearInterpolation, ExponentialProbing, InterpolationFunction,
    InterpolationSearchStrategy, LinearInterpolation, LogAmountBinarySearch, LogPriceInterpolation,
    LogarithmicBisection, SecantMethod, SqrtPriceInterpolation, TwoPhaseSearch,
};
