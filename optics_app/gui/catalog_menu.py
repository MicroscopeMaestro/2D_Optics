"""Qt menu helpers for the sectioned optical component catalog."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, Optional

from optics_app.catalogs.thorlabs import get_catalog_sections
from optics_app.gui.qt_compat import QAction


def diameter_menu_label(diameter_mm: float) -> str:
    """Return a recognizable metric/imperial size label for the catalog UI."""
    diameter_mm = float(diameter_mm)
    if abs(diameter_mm - 25.4) < 1e-6:
        return "1 inch (Ø25.4 mm)"
    if abs(diameter_mm - 12.7) < 1e-6:
        return "1/2 inch (Ø12.7 mm)"
    if abs(diameter_mm - 50.8) < 1e-6:
        return "2 inch (Ø50.8 mm)"
    if abs(diameter_mm - 75.0) < 1e-6 or abs(diameter_mm - 76.2) < 1e-6:
        return "3 inch (Ø75.0 mm)"
    if diameter_mm <= 10.0:
        return f"Ø{diameter_mm:g} mm (Miniature ≤10 mm)"
    return f"Ø{diameter_mm:g} mm"


def _format_catalog_action_label(part_number: str, record: Dict[str, Any]) -> str:
    """Format human-readable label for menu action."""
    if record.get("grooves_per_mm"):
        gr = record["grooves_per_mm"]
        blz = record.get("blaze_wavelength_um", 0.0) * 1000.0
        ang = record.get("blaze_angle_deg", 0.0)
        coat = record.get("coating_code", "Grating")
        if blz > 0:
            return f"{part_number} — {gr:g} l/mm, λb={blz:g} nm, θb={ang:g}° ({coat})"
        return f"{part_number} — {gr:g} l/mm ({coat})"
    elif record.get("is_polarizing") or record.get("section") == "polarizing_beamsplitters":
        ext = record.get("extinction_ratio", 1000)
        coat = record.get("coating_code", "PBS")
        dim = record.get("dimensions_mm", f"Ø{record.get('diameter_mm', 25.4):g}mm")
        return f"{part_number} — PBS {dim} (Ext > {ext:g}:1, {coat})"
    elif record.get("is_beam_splitter") or record.get("section") == "amplitude_beamsplitters":
        r_pct = int(round(record.get("split_ratio_R", 0.5) * 100))
        t_pct = int(round(record.get("split_ratio_T", 0.5) * 100))
        coat = record.get("coating_code", "VIS")
        dim = record.get("dimensions_mm", f"Ø{record.get('diameter_mm', 25.4):g}mm")
        return f"{part_number} — {r_pct}:{t_pct} Split ({dim}, {coat})"
    elif record.get("conic_k") == -1.0 or record.get("conic") == -1.0 or record.get("section") == "parabolic_mirrors":
        efl = record.get("nominal_focal_length_mm", record.get("focal_length_mm", 0.0))
        coat = record.get("coating_code", "Mirror")
        return f"{part_number} — Parabolic f={efl:g} mm ({coat})"
    elif record.get("section") == "spherical_mirrors":
        efl = record.get("nominal_focal_length_mm", record.get("focal_length_mm", 0.0))
        coat = record.get("coating_code", "Mirror")
        return f"{part_number} — Spherical f={efl:g} mm ({coat})"
    else:
        efl = record.get("nominal_focal_length_mm", record.get("focal_length_mm", 0.0))
        pwr = record.get("optical_power_diopters", 1000.0 / efl if efl else 0.0)
        return f"{part_number} — f={efl:g} mm ({pwr:.2f} D)"


def _catalog_sort_key(item: Tuple[str, Dict[str, Any]]) -> Tuple[float, float, str]:
    rec = item[1]
    if rec.get("grooves_per_mm"):
        return (rec.get("grooves_per_mm", 0.0), rec.get("blaze_wavelength_um", 0.0), item[0])
    if rec.get("is_beam_splitter") or "beamsplitter" in str(rec.get("section", "")):
        return (rec.get("diameter_mm", 25.4), rec.get("split_ratio_R", 0.5), item[0])
    return (rec.get("nominal_focal_length_mm", rec.get("focal_length_mm", 0.0)), 0.0, item[0])


def populate_lens_catalog_menu(
    parent_menu: Any,
    on_selected: Callable[[str], None],
    available_catalog: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """Populate a menu by catalog section, then diameter, with callable lens actions."""
    shown = set()
    for section in get_catalog_sections().values():
        section_menu = parent_menu.addMenu(section["label"])
        by_diameter = defaultdict(list)
        for part_number, record in section["parts"].items():
            by_diameter[record.get("diameter_mm", 25.4)].append((part_number, record))

        if len(by_diameter) <= 1:
            # Flat listing directly in section menu for single-diameter categories (e.g. Aspheric, Singlets)
            for part_number, record in sorted(
                section["parts"].items(),
                key=_catalog_sort_key,
            ):
                label = _format_catalog_action_label(part_number, record)
                action = QAction(label, parent_menu)
                action.triggered.connect(
                    lambda checked=False, key=part_number: on_selected(key)
                )
                section_menu.addAction(action)
                shown.add(part_number)
        else:
            # Multi-diameter grouping (e.g. Doublets with 1", 1/2", 2", miniature, Parabolic mirrors)
            for diameter_mm in sorted(by_diameter):
                diameter_menu = section_menu.addMenu(diameter_menu_label(diameter_mm))
                for part_number, record in sorted(
                    by_diameter[diameter_mm],
                    key=_catalog_sort_key,
                ):
                    label = _format_catalog_action_label(part_number, record)
                    action = QAction(label, parent_menu)
                    action.triggered.connect(
                        lambda checked=False, key=part_number: on_selected(key)
                    )
                    diameter_menu.addAction(action)
                    shown.add(part_number)

    if available_catalog:
        remaining = sorted(set(available_catalog) - shown)
        if remaining:
            other_menu = parent_menu.addMenu("Other / Custom Lenses")
            for part_number in remaining:
                action = QAction(part_number, parent_menu)
                action.triggered.connect(
                    lambda checked=False, key=part_number: on_selected(key)
                )
                other_menu.addAction(action)
