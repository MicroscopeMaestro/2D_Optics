"""
AI and Autonomous Agent Module for 2D Optics Engine powered by NVIDIA Nemotron & Google Gemma.
"""

from optics_app.ai.nemotron_client import (
    NemotronClient,
    GemmaClient,
    CloudAiClient,
    ModelTier,
    ModelFamily,
    TIER_MODEL_MAP,
    GEMMA_TIER_MODEL_MAP
)
from optics_app.ai.tools import (
    ToolRegistry,
    get_default_tool_registry,
    interpret_and_execute_command,
    rotate_optical_element,
    flip_optical_element,
    displace_optical_element,
    tilt_optical_element,
    replace_optical_element,
    replace_lens_from_thorlabs_query,
    optimize_focus_at_surface,
    offset_optical_element,
    run_ai_optimization,
    generate_optical_system,
    load_and_integrate_catalog_element,
    compute_matrix_propagation_and_q_factor
)
from optics_app.ai.agent import NemotronAgent
from optics_app.ai.generative_design import GenerativeOpticalDesigner
from optics_app.ai.catalog import OpticalCatalogManager

# OpticalAiAgent alias
OpticalAiAgent = NemotronAgent

__all__ = [
    "NemotronClient",
    "GemmaClient",
    "CloudAiClient",
    "ModelTier",
    "ModelFamily",
    "TIER_MODEL_MAP",
    "GEMMA_TIER_MODEL_MAP",
    "ToolRegistry",
    "get_default_tool_registry",
    "NemotronAgent",
    "OpticalAiAgent",
    "GenerativeOpticalDesigner",
    "OpticalCatalogManager",
    "interpret_and_execute_command",
    "rotate_optical_element",
    "flip_optical_element",
    "displace_optical_element",
    "tilt_optical_element",
    "replace_optical_element",
    "replace_lens_from_thorlabs_query",
    "optimize_focus_at_surface",
    "offset_optical_element",
    "run_ai_optimization",
    "generate_optical_system",
    "load_and_integrate_catalog_element",
    "compute_matrix_propagation_and_q_factor"
]
