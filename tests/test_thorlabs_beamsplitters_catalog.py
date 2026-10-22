"""
Test Thorlabs Amplitude and Polarizing Beam Splitter Prescriptions and Catalog Queries.
"""

import pytest
from optics_app.catalogs.thorlabs import (
    get_catalog_sections,
    get_lens_catalog,
    get_catalog_lens,
    get_amplitude_beamsplitter_catalog,
    get_polarizing_beamsplitter_catalog,
    get_beamsplitter_catalog,
    query_catalog,
)
from v2.catalogs.thorlabs import (
    get_catalog_lens as get_catalog_lens_v2,
    get_amplitude_beamsplitter_catalog as get_amplitude_beamsplitter_catalog_v2,
    get_polarizing_beamsplitter_catalog as get_polarizing_beamsplitter_catalog_v2,
)


def test_amplitude_beamsplitters_catalog_count_and_keys():
    """Verify all 31 Thorlabs amplitude beamsplitters are present and correctly categorized."""
    amp_catalog = get_amplitude_beamsplitter_catalog()
    assert len(amp_catalog) >= 31, f"Expected >= 31 amplitude beamsplitters, got {len(amp_catalog)}"

    expected_parts = [
        "BS004", "BS007", "BS008",
        "BS013", "BS014", "BS015",
        "BS016", "BS017", "BS018",
        "BS025", "BS026", "BS027",
        "BS010", "BS011", "BS012", "BS019",
        "BSW04", "BSW05", "BSW06",
        "BSW10", "BSW11", "BSW12",
        "BSW26", "BSW27", "BSW28",
        "BSW29", "BSW30", "BSW31", "BSW32",
        "BP108", "BP145B2",
    ]

    for pn in expected_parts:
        assert pn in amp_catalog, f"Part {pn} missing from amplitude beamsplitters catalog"
        rec = amp_catalog[pn]
        assert rec["is_beam_splitter"] is True
        assert rec["is_polarizing"] is False
        assert rec["part_number"] == pn
        assert "surfaces" in rec
        assert len(rec["surfaces"]) >= 1
        assert "split_ratio_R" in rec
        assert "split_ratio_T" in rec
        assert abs(rec["split_ratio_R"] + rec["split_ratio_T"] - 1.0) < 1e-6


def test_polarizing_beamsplitters_catalog_count_and_keys():
    """Verify all 21 Thorlabs polarizing beamsplitters are present and correctly configured."""
    pbs_catalog = get_polarizing_beamsplitter_catalog()
    assert len(pbs_catalog) >= 21, f"Expected >= 21 polarizing beamsplitters, got {len(pbs_catalog)}"

    expected_pbs = [
        "PBS101", "PBS102", "PBS103", "PBS104",
        "PBS121", "PBS122", "PBS123", "PBS124",
        "PBS251", "PBS252", "PBS253", "PBS254", "PBS255",
        "PBS511", "PBS512", "PBS513", "PBS514",
        "WPBS25-VIS", "WPBS25-NIR", "WPBS50-VIS", "WPBS50-NIR",
    ]

    for pn in expected_pbs:
        assert pn in pbs_catalog, f"Part {pn} missing from polarizing beamsplitters catalog"
        rec = pbs_catalog[pn]
        assert rec["is_beam_splitter"] is True
        assert rec["is_polarizing"] is True
        assert rec["part_number"] == pn
        assert "extinction_ratio" in rec
        assert rec["extinction_ratio"] >= 800.0
        cfg = rec.get("beamsplitter_config", {})
        assert cfg.get("is_polarizing") is True
        assert cfg.get("Rs", 0.0) >= 0.90
        assert cfg.get("Tp", 0.0) >= 0.85


def test_combined_beamsplitter_catalog():
    """Verify get_beamsplitter_catalog combines amplitude and polarizing splitters."""
    combined = get_beamsplitter_catalog()
    amp = get_amplitude_beamsplitter_catalog()
    pbs = get_polarizing_beamsplitter_catalog()
    assert len(combined) == len(amp) + len(pbs)
    assert len(combined) >= 52


def test_query_catalog_beamsplitter_filters():
    """Verify query_catalog routing for beamsplitters."""
    all_bs = query_catalog(section="beamsplitter")
    assert len(all_bs) >= 52

    npbs_only = query_catalog(section="npbs")
    assert len(npbs_only) >= 31
    for r in npbs_only:
        assert r["is_polarizing"] is False

    pbs_only = query_catalog(section="pbs")
    assert len(pbs_only) >= 21
    for r in pbs_only:
        assert r["is_polarizing"] is True

    # Test diameter filtering
    inch1_bs = query_catalog(section="beamsplitter", diameter_mm=25.4)
    assert any(r["part_number"] == "BS016" for r in inch1_bs)
    assert any(r["part_number"] == "PBS251" for r in inch1_bs)


def test_parity_optics_app_and_v2_catalogs():
    """Verify strict 100% parity between optics_app and v2 catalogs."""
    amp1 = get_amplitude_beamsplitter_catalog()
    amp2 = get_amplitude_beamsplitter_catalog_v2()
    assert set(amp1.keys()) == set(amp2.keys())

    pbs1 = get_polarizing_beamsplitter_catalog()
    pbs2 = get_polarizing_beamsplitter_catalog_v2()
    assert set(pbs1.keys()) == set(pbs2.keys())

    for pn in ["BS016", "PBS251", "BS019", "WPBS25-VIS"]:
        rec1 = get_catalog_lens(pn)
        rec2 = get_catalog_lens_v2(pn)
        assert rec1 == rec2, f"Parity mismatch for {pn}"
