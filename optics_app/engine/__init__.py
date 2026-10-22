"""
Optics Engine Package: Optical modeling, sequential surface stack, ray tracing, and parameter optimization.
"""

from optics_app.engine.surface import Surface, SurfaceType
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.tracer import OpticalSystem
from optics_app.engine.ray import Ray
from optics_app.engine.optimizer import OpticalOptimizer, OptimizationResult
try:
    from .routes import OpticalRoute, OpticalRouteSet, RouteStep
except ImportError:
    from optics_app.engine.routes import OpticalRoute, OpticalRouteSet, RouteStep
try:
    from optics_app.engine.models import SpectrometerSystem
except ImportError:
    from v2.engine.models import SpectrometerSystem

try:
    from .non_sequential import (
        NonSequentialElement, NonSequentialSource, NonSequentialSystem, NonSequentialSurfaceType
    )
except ImportError:
    from optics_app.engine.non_sequential import (
        NonSequentialElement, NonSequentialSource, NonSequentialSystem, NonSequentialSurfaceType
    )

try:
    from .illumination import (
        IlluminationMaskProjectionTrain,
        IlluminationSource,
        SourceType,
        IlluminationMode,
        HomogenizerType,
        HomogenizerConfig,
        MaskType,
        MaskSubsystemConfig,
        CollectorType,
        ProjectionType,
        ProjectionLensConfig,
        EnergyThroughputMetrics,
    )
except ImportError:
    from optics_app.engine.illumination import (
        IlluminationMaskProjectionTrain,
        IlluminationSource,
        SourceType,
        IlluminationMode,
        HomogenizerType,
        HomogenizerConfig,
        MaskType,
        MaskSubsystemConfig,
        CollectorType,
        ProjectionType,
        ProjectionLensConfig,
        EnergyThroughputMetrics,
    )

__all__ = [
    "Surface",
    "SurfaceType",
    "SurfaceStack",
    "OpticalSystem",
    "Ray",
    "OpticalOptimizer",
    "OptimizationResult",
    "OpticalRoute",
    "OpticalRouteSet",
    "RouteStep",
    "SpectrometerSystem",
    "NonSequentialElement",
    "NonSequentialSource",
    "NonSequentialSystem",
    "NonSequentialSurfaceType",
    "IlluminationMaskProjectionTrain",
    "IlluminationSource",
    "SourceType",
    "IlluminationMode",
    "HomogenizerType",
    "HomogenizerConfig",
    "MaskType",
    "MaskSubsystemConfig",
    "CollectorType",
    "ProjectionType",
    "ProjectionLensConfig",
    "EnergyThroughputMetrics",
]
