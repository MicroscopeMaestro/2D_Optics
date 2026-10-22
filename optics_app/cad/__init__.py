"""
optics_app.cad: 3D CAD loading, Z-axis slicing, profile extraction, surface fitting, and 3D STEP exporter.
"""

from optics_app.cad.profile import (
    OpticalProfile,
    FitType,
    ProfileFitType,
)
from optics_app.cad.fitter import (
    SagFitter,
    ProfileFitter,
    ProfilePoints,
    fit_sag_profile,
)
from optics_app.cad.slicer import (
    ZSlicer,
    ZAxisSlicer,
    CADSlicer,
    CADAligner,
    SlicedLensGeometry,
    slice_z_axis,
)
from optics_app.cad.importer import (
    CADImporter,
    CADSurfaceImporter,
    CADLoadingTier,
    CADFileFormat,
    CADImportMetadata,
    CADImportResult,
)
from optics_app.cad.exporter import (
    STEPExporter,
    CADExporter,
    STEPExportEngine,
    STEPExportOptions,
    STEPExportResult,
    export_to_step,
)

__all__ = [
    "OpticalProfile",
    "FitType",
    "ProfileFitType",
    "SagFitter",
    "ProfileFitter",
    "ProfilePoints",
    "fit_sag_profile",
    "ZSlicer",
    "ZAxisSlicer",
    "CADSlicer",
    "CADAligner",
    "SlicedLensGeometry",
    "slice_z_axis",
    "CADImporter",
    "CADSurfaceImporter",
    "CADLoadingTier",
    "CADFileFormat",
    "CADImportMetadata",
    "CADImportResult",
    "STEPExporter",
    "CADExporter",
    "STEPExportEngine",
    "STEPExportOptions",
    "STEPExportResult",
    "export_to_step",
]
