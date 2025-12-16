pub mod history_based;
pub mod history_based_archive;

// Main strategies (best performers)
pub use history_based::{
    BrentAndStrategy, BrentOrStrategy, BrentOriginalStrategy, BrentStrategy,
    ChandrupatlaStrategy, EVMBrentStrategy, EVMChandrupatlaStrategy, IqiStrategy,
};

// Archived strategies (kept for reference)
// Note: ChandrupatlaStrategy moved to history_based with correct criterion
pub use history_based_archive::{
    BlendedIqiSecantStrategy, BlendedIqiSecantV2Strategy, BrentV2Strategy,
    ChandrupatlaStrategy as ChandrupatlaStrategyArchived, ConvexSearchStrategy,
    CurvatureAdaptiveStrategy, ElasticityNewtonStrategy, HybridStrategy, IqiV2Strategy,
    ItpStrategy, NewtonCentralStrategy, NewtonLogStrategy, PiecewiseLinearStrategy,
    PrecisionLimitAwareStrategy, QuadraticRegressionStrategy, RiddersStrategy,
    StableSwapAwareStrategy, WeightedRegressionStrategy,
};