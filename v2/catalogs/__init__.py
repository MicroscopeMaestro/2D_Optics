"""Built-in, source-traceable optical component catalogs."""

from .thorlabs import (
    THORLABS_CATALOG_SECTIONS,
    get_catalog_sections,
    get_lens_catalog,
    query_catalog,
)
from .microscopy import (
    FLUOROPHORE_STANDARDS,
    MICROSCOPE_OBJECTIVES,
    PMT_CATALOG,
    SCANNING_OPTICS,
    get_fluorophore_standard,
    get_microscope_objective,
    get_pmt,
)

__all__ = [
    "THORLABS_CATALOG_SECTIONS",
    "get_catalog_sections",
    "get_lens_catalog",
    "query_catalog",
    "FLUOROPHORE_STANDARDS",
    "MICROSCOPE_OBJECTIVES",
    "PMT_CATALOG",
    "SCANNING_OPTICS",
    "get_fluorophore_standard",
    "get_microscope_objective",
    "get_pmt",
]
