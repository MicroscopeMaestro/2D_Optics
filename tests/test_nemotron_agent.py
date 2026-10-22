import os
import sys
import tempfile
import pytest

# Ensure repository root is on path for pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from optics_app.ai.nemotron_client import NemotronClient, ModelTier, TIER_MODEL_MAP
from optics_app.ai.tools import (
    ToolRegistry,
    get_default_tool_registry,
    read_file,
    write_file,
    edit_file,
    list_directory,
    search_codebase,
    run_command,
    lookup_thorlabs_catalog,
    trace_optical_design,
    verify_cad_export,
    set_active_optical_system,
)
from optics_app.ai.agent import NemotronAgent
from optics_app.engine.lde import SurfaceStack
from optics_app.engine.models import SpectrometerSystem
from optics_app.engine.surface import Surface, SurfaceType
from optics_app.gui.ai_dock import CommandWorker


def test_nemotron_client_initialization():
    """Test model tier resolution and config initialization."""
    client_high = NemotronClient(api_key="test-key-123", tier=ModelTier.HIGH)
    assert client_high.tier == ModelTier.HIGH
    assert client_high.model == TIER_MODEL_MAP[ModelTier.HIGH]
    assert client_high.is_configured() is True

    client_med = NemotronClient(api_key="test-key-123", tier=ModelTier.MEDIUM)
    assert client_med.tier == ModelTier.MEDIUM
    assert client_med.model == TIER_MODEL_MAP[ModelTier.MEDIUM]

    # Model switch
    client_med.set_tier("high")
    assert client_med.tier == ModelTier.HIGH
    assert client_med.model == TIER_MODEL_MAP[ModelTier.HIGH]


def test_file_tools_suite():
    """Test read, write, edit, list, and search tools in a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "sample.txt")
        
        # Write
        w_res = write_file(test_file, "line 1\nalpha_target\nline 3\n")
        assert "Successfully wrote" in w_res
        assert os.path.exists(test_file)

        # Read
        r_res = read_file(test_file, start_line=1, end_line=2)
        assert "alpha_target" in r_res
        assert "line 3" not in r_res

        # Edit
        e_res = edit_file(test_file, "alpha_target", "beta_replacement")
        assert "Successfully replaced" in e_res
        
        r_after = read_file(test_file)
        assert "beta_replacement" in r_after
        assert "alpha_target" not in r_after

        # List
        l_res = list_directory(tmpdir)
        assert "sample.txt" in l_res

        # Search
        s_res = search_codebase("beta_replacement", directory=tmpdir)
        assert "sample.txt" in s_res


def test_command_runner_tool():
    """Test shell command execution tool."""
    res = run_command("python -c \"print('nemotron_test_ok')\"")
    assert "Exit Code: 0" in res
    assert "nemotron_test_ok" in res


def test_thorlabs_catalog_lookup_tool():
    """Test querying Thorlabs database."""
    res = lookup_thorlabs_catalog("AC254-050-C")
    assert "N-LAK22" in res
    assert "50.0" in res

    res_asph = lookup_thorlabs_catalog("A230-C")
    assert "S-NPH1" in res_asph
    assert "0.390203" in res_asph


def test_optical_tracer_tool():
    """Test trace_optical_design tool on desktop design file if present."""
    candidate_paths = [
        "C:/Users/Juan/Desktop/second_mirror_spectrometer_NIR.json",
        "C:/Users/Juan/Desktop/mirror_spectrometer_NIR.json",
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            res = trace_optical_design(p)
            assert "=== Optical Trace Analysis" in res
            assert "Total Surfaces:" in res
            break


def test_mock_nemotron_agent_react_loop():
    """Test agent ReAct execution loop using a mock completion response."""
    class MockClient:
        def __init__(self):
            self.model = "mock-nemotron"
            self.tier = ModelTier.HIGH
            self.calls = 0

        def chat_completion(self, messages, tools=None, tool_choice=None):
            self.calls += 1
            if self.calls == 1:
                # Issue a tool call to lookup_thorlabs_catalog
                return {
                    "role": "assistant",
                    "content": "Let me look up the specs for AC254-050-C.",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup_thorlabs_catalog",
                            "arguments": '{"part_number": "AC254-050-C"}'
                        }
                    }]
                }
            else:
                # Final answer
                return {
                    "role": "assistant",
                    "content": "The lens AC254-050-C has a focal length of 50.0 mm with N-LAK22 and N-SF6HT glasses.",
                    "tool_calls": []
                }

    agent = NemotronAgent(client=MockClient())
    result = agent.run("What are the specs of AC254-050-C?", verbose=False)

    assert result["success"] is True
    assert "50.0 mm" in result["final_answer"]
    assert len(result["steps"]) == 1
    assert result["steps"][0]["tool"] == "lookup_thorlabs_catalog"


def test_ai_panel_agentic_worker_executes_tools_on_active_application_model():
    """Agentic UI mode runs the ReAct loop and applies its tool to the live stack."""
    stack = SurfaceStack()
    stack.add_surface(Surface(name="Object", surface_type=SurfaceType.OBJECT, thickness=20.0))
    stack.insert_catalog_lens(0, "AC254-050-C", air_gap_after=41.0)
    stack.add_surface(Surface(name="Detector", surface_type=SurfaceType.DETECTOR))
    system = SpectrometerSystem(surface_stack=stack)
    set_active_optical_system(system)

    client = NemotronClient(demo_mode=True)
    agent = NemotronAgent(client=client, tool_registry=get_default_tool_registry())
    worker = CommandWorker(
        "replace element 2 with AC254-030-C",
        system,
        client=client,
        agent=agent,
        agentic_mode=True,
    )
    result = worker.execute()

    assert result["agentic_mode"] is True
    assert result["success"] is True
    assert result["iterations"] == 2
    assert result["executed_results"][0]["tool"] == "replace_optical_element"
    assert [surface.group_name for surface in stack.surfaces[1:4]] == ["AC254-030-C"] * 3
