"""PULSE: Plastic Latent Dynamics Engine (可塑性隐态动力学引擎)."""

from pulse.baseline_transformer import TransformerBaseline, TransformerConfig
from pulse.config import PULSEConfig
from pulse.delta_resonance import (
    ChunkwiseGatedDeltaNet,
    ContrastiveEnergyCritic,
    ShortCausalConv,
    SlidingWindowEchoicAttention,
)
from pulse.experiment import (
    CCUBudgetGuard,
    ControlledExperimentRunner,
    ExperimentConfig,
    FairMuonAdamW,
)
from pulse.fomp_v3 import FOMPv3BackboneWrapper, FOMPv3Config, FOMPv3Module
from pulse.local_learning import LocalIOUpdater, LocalSynapticOptimizer
from pulse.manifold import (
    ConceptManifoldProjection,
    EMATargetEncoder,
    GatedConceptDecoder,
    TextConceptEncoder,
    VectorConceptEncoder,
)
from pulse.memory import (
    EpisodicTrace,
    FastWorkingStateSSM,
    SleepConsolidator,
    SurpriseGatedFastWeights,
)
from pulse.model import PULSEModel, ThinkOutput
from pulse.predictive_field import (
    BidirectionalPredictiveField,
    PredictiveLayer,
    RelaxationResult,
)
from pulse.pulse_lm import PULSELMConfig, PULSELanguageModel, ParallelSelectiveSSM
from pulse.pulse_v2 import PULSEv2Config, PULSEv2Model
from pulse.pulse_v3 import PULSEv3Config, PULSEv3Model
from pulse.pulse_v6 import PULSEv6Config, PULSEv6Model
from pulse.pulse_v7 import PULSEv7Config, PULSEv7Model
from pulse.pulse_v8 import PULSEv8Config, PULSEv8Model
from pulse.pulse_v7_plasticity import (
    DualTrackSemanticMemory,
    SemanticKeyProjector,
    TransitionConditionedVerifier,
    decore_contrastive_logits,
)
from pulse.scalable_transformer import ScalableTransformerConfig, ScalableTransformerLM
from pulse.tokenizer import PulseTokenizer
from pulse.v3_blocks import (
    CanonLayer,
    GatedDeltaNet2,
    GatedSlidingWindowAttention,
    ManifoldHyperConnection,
    RLSHippocampalMemory,
    TriSignalContrastiveCritic,
)

__version__ = "0.8.6"

__all__ = [
    "PULSEConfig",
    "PULSEModel",
    "ThinkOutput",
    "PULSELMConfig",
    "PULSELanguageModel",
    "ParallelSelectiveSSM",
    "PULSEv2Config",
    "PULSEv2Model",
    "PULSEv3Config",
    "PULSEv3Model",
    "PULSEv6Config",
    "PULSEv6Model",
    "PULSEv7Config",
    "PULSEv7Model",
    "PULSEv8Config",
    "PULSEv8Model",
    "FOMPv3Config",
    "FOMPv3Module",
    "FOMPv3BackboneWrapper",
    "CanonLayer",
    "ManifoldHyperConnection",
    "GatedDeltaNet2",
    "GatedSlidingWindowAttention",
    "RLSHippocampalMemory",
    "TriSignalContrastiveCritic",
    "ShortCausalConv",
    "ChunkwiseGatedDeltaNet",
    "SlidingWindowEchoicAttention",
    "ContrastiveEnergyCritic",
    "ScalableTransformerConfig",
    "ScalableTransformerLM",
    "CCUBudgetGuard",
    "ExperimentConfig",
    "ControlledExperimentRunner",
    "FairMuonAdamW",
    "PulseTokenizer",
    "ConceptManifoldProjection",
    "TextConceptEncoder",
    "VectorConceptEncoder",
    "EMATargetEncoder",
    "GatedConceptDecoder",
    "PredictiveLayer",
    "BidirectionalPredictiveField",
    "RelaxationResult",
    "FastWorkingStateSSM",
    "SurpriseGatedFastWeights",
    "SleepConsolidator",
    "EpisodicTrace",
    "LocalSynapticOptimizer",
    "LocalIOUpdater",
    "TransformerConfig",
    "TransformerBaseline",
]
