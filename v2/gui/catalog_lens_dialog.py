"""Interactive Thorlabs Catalog Lens Chooser & Replacement Tool Dialog.

Allows users to filter, inspect, and choose parameters when replacing or inserting
optical lens elements from the Thorlabs commercial catalog:
- Filter by coating/spectral band (A, B, C, Aspheric, Singlet)
- Filter by diameter (1", 1/2", 2", miniature)
- Real-time part search
- Detailed prescription preview (radii, thickness, glass, EFL, BFL, diopters)
- Parameter choices:
  * Normal or Reversed/Flipped orientation (critical for collimators/spectrometers)
  * Distance/air gap: preserve existing, set to catalog BFL, or custom thickness
  * Aperture: use catalog clear aperture or preserve existing
  * 3D Pose: preserve tilts and decenters
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from v2.gui.qt_compat import (
    QT_AVAILABLE, QtWidgets, QtCore,
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QComboBox, QLineEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QGroupBox, QRadioButton, QCheckBox, QDoubleSpinBox,
    QAbstractItemView, QMessageBox, Qt, QButtonGroup
)
from v2.catalogs.thorlabs import (
    get_catalog_sections, get_lens_catalog, get_catalog_lens, query_catalog
)
from v2.engine.element_colors import get_lens_elements


class CatalogLensDialog(QDialog if QT_AVAILABLE else object):
    """
    Interactive Dialog for choosing and applying Thorlabs catalog lenses.
    Supports both 'replace' mode (replacing an existing lens element) and
    'insert' mode (inserting after a specified surface).
    """

    def __init__(
        self,
        model: Any,
        parent: Optional[Any] = None,
        target: Optional[Any] = None,
        mode: str = "replace",
    ):
        if QT_AVAILABLE:
            super().__init__(parent)
            title = "Thorlabs Catalog Lens — Replace Selected Lens Tool" if mode == "replace" else "Thorlabs Catalog Lens — Insert Lens Tool"
            self.setWindowTitle(title)
            self.resize(880, 680)
        
        self.model = model
        self.mode = mode
        self.target = target
        self._selected_lens_record: Optional[Dict[str, Any]] = None
        self._all_lenses = get_lens_catalog()

        if QT_AVAILABLE:
            self._setup_ui()
            self._populate_targets()
            self._apply_filters()
            self._preselect_target(target)

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(8)

        # -----------------------------------------------------------------
        # 1. Target Selector Header
        # -----------------------------------------------------------------
        target_group = QGroupBox("Target Optical Element", self)
        target_layout = QHBoxLayout(target_group)

        lbl_text = "Target Lens to Replace:" if self.mode == "replace" else "Insert Lens After Surface:"
        target_layout.addWidget(QLabel(lbl_text))

        self.target_combo = QComboBox(target_group)
        self.target_combo.currentIndexChanged.connect(self._on_target_changed)
        target_layout.addWidget(self.target_combo, stretch=2)

        self.target_info_label = QLabel("", target_group)
        self.target_info_label.setStyleSheet("color: #555; font-style: italic;")
        target_layout.addWidget(self.target_info_label, stretch=2)

        main_layout.addWidget(target_group)

        # -----------------------------------------------------------------
        # 2. Middle Area: Catalog Filter & Selection (Left) + Details (Right)
        # -----------------------------------------------------------------
        mid_layout = QHBoxLayout()

        # Left Column: Catalog Browser
        cat_group = QGroupBox("Thorlabs Catalog Lens Browser", self)
        cat_layout = QVBoxLayout(cat_group)

        filter_row = QHBoxLayout()
        self.filter_section = QComboBox(cat_group)
        self.filter_section.addItem("All Categories / Coatings", None)
        sections = get_catalog_sections()
        for sec_key, sec_data in sections.items():
            self.filter_section.addItem(sec_data.get("label", sec_key), sec_key)
        self.filter_section.currentIndexChanged.connect(self._apply_filters)
        filter_row.addWidget(self.filter_section)

        self.filter_diameter = QComboBox(cat_group)
        self.filter_diameter.addItem("All Diameters", None)
        self.filter_diameter.addItem("1 inch (Ø25.4 mm)", 25.4)
        self.filter_diameter.addItem("1/2 inch (Ø12.7 mm)", 12.7)
        self.filter_diameter.addItem("2 inch (Ø50.8 mm)", 50.8)
        self.filter_diameter.addItem("3 inch (Ø75.0 mm)", 75.0)
        self.filter_diameter.addItem("Ø30.0 mm", 30.0)
        self.filter_diameter.addItem("1/4 inch (Ø6.35 mm)", 6.35)
        self.filter_diameter.addItem("Miniature (≤10 mm)", "mini")
        self.filter_diameter.currentIndexChanged.connect(self._apply_filters)
        filter_row.addWidget(self.filter_diameter)

        self.filter_coating = QComboBox(cat_group)
        self.filter_coating.addItem("All Coatings", None)
        self.filter_coating.addItem("C (1050-1700 nm)", "C")
        self.filter_coating.addItem("B (650-1050 nm)", "B")
        self.filter_coating.addItem("A (400-700 nm)", "A")
        self.filter_coating.addItem("Uncoated", "Uncoated")
        self.filter_coating.addItem("Mirror (Protected Silver -P01)", "P01")
        self.filter_coating.addItem("Mirror (Protected Gold -M01)", "M01")
        self.filter_coating.addItem("Mirror (Protected Aluminum -G01)", "G01")
        self.filter_coating.addItem("Grating / Reflective (AL)", "AL")
        self.filter_coating.addItem("Transmission Grating (TRANS)", "TRANS")
        self.filter_coating.addItem("Beamsplitter VIS (400-700 nm)", "VIS")
        self.filter_coating.addItem("Beamsplitter NIR (700-1100 nm)", "NIR")
        self.filter_coating.addItem("Beamsplitter IR (1100-1600 nm)", "IR")
        self.filter_coating.currentIndexChanged.connect(self._apply_filters)
        filter_row.addWidget(self.filter_coating)

        cat_layout.addLayout(filter_row)

        # Search Box and Count Status
        search_layout = QHBoxLayout()
        self.search_edit = QLineEdit(cat_group)
        self.search_edit.setPlaceholderText("🔍 Search part number, EFL, coating (e.g. AC254-050, 30mm, B, ACL, LA)...")
        self.search_edit.textChanged.connect(self._apply_filters)
        search_layout.addWidget(self.search_edit, stretch=3)

        self.lbl_catalog_count = QLabel("", cat_group)
        self.lbl_catalog_count.setStyleSheet("color: #455A64; font-weight: bold; padding-left: 4px;")
        search_layout.addWidget(self.lbl_catalog_count, stretch=1)
        cat_layout.addLayout(search_layout)

        # Table of catalog items
        self.lens_table = QTableWidget(cat_group)
        self.lens_table.setColumnCount(7)
        self.lens_table.setHorizontalHeaderLabels([
            "Part Number", "Type / Family", "Coating", "Ø (mm)", "EFL (mm)", "BFL (mm)", "Power (D)"
        ])
        self.lens_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        if hasattr(self.lens_table.horizontalHeader(), "setStretchLastSection"):
            self.lens_table.horizontalHeader().setStretchLastSection(True)
        self.lens_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.lens_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.lens_table.itemSelectionChanged.connect(self._on_table_row_selected)
        cat_layout.addWidget(self.lens_table)

        mid_layout.addWidget(cat_group, stretch=6)

        # Right Column: Lens Prescription Preview Card
        details_group = QGroupBox("Selected Lens Prescription", self)
        details_layout = QVBoxLayout(details_group)

        self.lbl_lens_title = QLabel("<b>Select a lens to view specifications</b>", details_group)
        self.lbl_lens_title.setTextFormat(Qt.RichText)
        details_layout.addWidget(self.lbl_lens_title)

        self.lbl_lens_specs = QLabel("", details_group)
        self.lbl_lens_specs.setWordWrap(True)
        details_layout.addWidget(self.lbl_lens_specs)

        # Surface Breakdown Table
        self.surf_table = QTableWidget(details_group)
        self.surf_table.setColumnCount(4)
        self.surf_table.setHorizontalHeaderLabels(["Surface", "Radius (mm)", "Thickness", "Glass"])
        self.surf_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.surf_table.setFixedHeight(120)
        details_layout.addWidget(self.surf_table)

        details_layout.addStretch()
        mid_layout.addWidget(details_group, stretch=4)

        main_layout.addLayout(mid_layout)

        # -----------------------------------------------------------------
        # 3. Parameter Controls: Orientation, Air Gap, Aperture, Pose
        # -----------------------------------------------------------------
        params_group = QGroupBox("Choose Placement Parameters", self)
        params_layout = QGridLayout(params_group)

        # Orientation
        params_layout.addWidget(QLabel("<b>Lens Orientation:</b>"), 0, 0)
        self.radio_normal = QRadioButton("Normal (R₁ → R₂ → R₃) [Forward Beam]", params_group)
        self.radio_normal.setChecked(True)
        self.radio_reverse = QRadioButton("Flipped / Reversed (−R₃ → −R₂ → −R₁) [Collimator Mode]", params_group)
        self.radio_reverse.setToolTip("Flips the lens element 180°. Useful when collimating light from a point source or slit.")
        params_layout.addWidget(self.radio_normal, 0, 1)
        params_layout.addWidget(self.radio_reverse, 0, 2)

        self.group_orientation = QButtonGroup(self)
        self.group_orientation.addButton(self.radio_normal)
        self.group_orientation.addButton(self.radio_reverse)

        # Trailing Air Gap / Distance
        params_layout.addWidget(QLabel("<b>Distance After Lens:</b>"), 1, 0)
        self.radio_gap_preserve = QRadioButton("Preserve Existing Air Gap", params_group)
        self.radio_gap_preserve.setChecked(True)
        self.radio_gap_autofocus = QRadioButton("🎯 Auto-correct focal distance (focus to image/detector)", params_group)
        self.radio_gap_autofocus.setToolTip("Automatically calculates and sets the trailing distance so light focuses sharply on downstream detector/image surfaces.")
        self.radio_gap_bfl = QRadioButton("Set to Catalog Back Focal Length (BFL)", params_group)
        self.radio_gap_custom = QRadioButton("Custom Distance:", params_group)

        self.spin_custom_gap = QDoubleSpinBox(params_group)
        self.spin_custom_gap.setRange(0.0, 2000.0)
        self.spin_custom_gap.setDecimals(3)
        self.spin_custom_gap.setSuffix(" mm")
        self.spin_custom_gap.setValue(10.0)
        self.spin_custom_gap.setEnabled(False)
        self.radio_gap_custom.toggled.connect(self.spin_custom_gap.setEnabled)

        self.group_gap = QButtonGroup(self)
        self.group_gap.addButton(self.radio_gap_preserve)
        self.group_gap.addButton(self.radio_gap_autofocus)
        self.group_gap.addButton(self.radio_gap_bfl)
        self.group_gap.addButton(self.radio_gap_custom)

        gap_layout = QVBoxLayout()
        gap_row1 = QHBoxLayout()
        gap_row1.addWidget(self.radio_gap_preserve)
        gap_row1.addWidget(self.radio_gap_autofocus)
        gap_row1.addStretch()
        gap_row2 = QHBoxLayout()
        gap_row2.addWidget(self.radio_gap_bfl)
        gap_row2.addWidget(self.radio_gap_custom)
        gap_row2.addWidget(self.spin_custom_gap)
        gap_row2.addStretch()
        gap_layout.addLayout(gap_row1)
        gap_layout.addLayout(gap_row2)
        params_layout.addLayout(gap_layout, 1, 1, 1, 2)

        # Aperture & Pose Checkboxes
        opt_layout = QHBoxLayout()
        self.check_set_aperture = QCheckBox("Set clear aperture to catalog lens semi-diameter", params_group)
        self.check_set_aperture.setChecked(True)
        opt_layout.addWidget(self.check_set_aperture)

        self.check_preserve_pose = QCheckBox("Preserve existing element 3D tilts & decenters", params_group)
        self.check_preserve_pose.setChecked(True)
        opt_layout.addWidget(self.check_preserve_pose)
        opt_layout.addStretch()
        params_layout.addLayout(opt_layout, 2, 1, 1, 2)

        main_layout.addWidget(params_group)

        # -----------------------------------------------------------------
        # 4. Action Buttons
        # -----------------------------------------------------------------
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.btn_cancel = QPushButton("Cancel", self)
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)

        action_text = "🔄 Replace Selected Lens" if self.mode == "replace" else "➕ Insert Lens"
        self.btn_apply = QPushButton(action_text, self)
        self.btn_apply.setStyleSheet("font-weight: bold; padding: 6px 14px;")
        self.btn_apply.setEnabled(False)
        self.btn_apply.clicked.connect(self._on_apply)
        btn_layout.addWidget(self.btn_apply)

        main_layout.addLayout(btn_layout)

    def _populate_targets(self):
        """Populate the target dropdown with existing lenses or surfaces."""
        if not self.model:
            return
        stack = getattr(self.model, "stack", None) or (self.model.get_surface_stack() if hasattr(self.model, "get_surface_stack") else None)
        if not stack:
            return

        self.target_combo.clear()
        if self.mode == "replace":
            unlocked = getattr(self.model, "_unlocked_lens_groups", set())
            lens_elems = get_lens_elements(stack.surfaces, unlocked_groups=unlocked)
            if lens_elems:
                for elem in lens_elems:
                    label = f"{elem['group_name']} (Surfaces {elem['surface_indices'][0]}-{elem['surface_indices'][-1]})"
                    self.target_combo.addItem(label, elem)
            else:
                # Fallback: allow replacing individual surfaces if no multi-surface element exists
                for i, surf in enumerate(stack.surfaces):
                    sname = getattr(surf, "name", f"Surface {i}")
                    elem = {"group_name": sname, "surface_indices": [i]}
                    self.target_combo.addItem(f"Surface {i}: {sname}", elem)
        else:
            for i, surf in enumerate(stack.surfaces):
                sname = getattr(surf, "name", f"Surface {i}")
                self.target_combo.addItem(f"Surface {i}: {sname}", i)

    def _preselect_target(self, target: Any):
        """Pre-select the target in the dropdown if given."""
        if target is None:
            return
        target_str = str(target).strip().upper()
        for idx in range(self.target_combo.count()):
            data = self.target_combo.itemData(idx)
            if self.mode == "replace" and isinstance(data, dict):
                indices = data.get("surface_indices", [])
                g_name = str(data.get("group_name", "")).strip().upper()
                if (isinstance(target, int) and target in indices) or \
                   (target_str.isdigit() and int(target_str) in indices) or \
                   (target_str == g_name):
                    self.target_combo.setCurrentIndex(idx)
                    return
            elif self.mode != "replace":
                if data == target or str(data) == target_str:
                    self.target_combo.setCurrentIndex(idx)
                    return

    def _on_target_changed(self):
        """Update information label when target lens selection changes."""
        data = self.target_combo.currentData()
        if not data or not self.model:
            self.target_info_label.setText("")
            return

        stack = getattr(self.model, "stack", None) or (self.model.get_surface_stack() if hasattr(self.model, "get_surface_stack") else None)
        if not stack:
            return

        if self.mode == "replace" and isinstance(data, dict):
            indices = data.get("surface_indices", [])
            if indices and indices[-1] < len(stack.surfaces):
                rear_thic = stack.surfaces[indices[-1]].thickness
                total_thic = sum(stack.surfaces[k].thickness for k in indices[:-1]) if len(indices) > 1 else stack.surfaces[indices[0]].thickness
                glasses = [stack.surfaces[k].glass for k in indices if getattr(stack.surfaces[k], 'is_glass', False)]
                g_str = "/".join(glasses) if glasses else "AIR"
                self.target_info_label.setText(f"Glass: {g_str} (t={total_thic:.2f} mm) | Air Gap: {rear_thic:.2f} mm")
                self.radio_gap_preserve.setText(f"Preserve Existing Air Gap ({rear_thic:.2f} mm)")
        elif self.mode != "replace" and isinstance(data, int):
            if 0 <= data < len(stack.surfaces):
                surf = stack.surfaces[data]
                self.target_info_label.setText(f"After {surf.name} (Thickness={surf.thickness:.2f} mm)")

        if getattr(self, '_selected_lens_record', None):
            self._update_preview(self._selected_lens_record)

    def _apply_filters(self):
        """Filter catalog table rows based on section, diameter, coating, and search text."""
        sec_key = self.filter_section.currentData() if hasattr(self, 'filter_section') else None
        diam_val = self.filter_diameter.currentData() if hasattr(self, 'filter_diameter') else None
        coat_val = self.filter_coating.currentData() if hasattr(self, 'filter_coating') else None
        search_txt = self.search_edit.text().strip().lower() if hasattr(self, 'search_edit') else ""

        # Query matching records
        diam_mm = float(diam_val) if isinstance(diam_val, (int, float)) else None
        matches = query_catalog(section=sec_key, diameter_mm=diam_mm, coating_code=coat_val, search_text=search_txt)

        if diam_val == "mini":
            matches = [m for m in matches if m.get("diameter_mm", 0.0) <= 10.0]

        # Also search legacy/extended catalog lenses if present in _all_lenses
        matched_keys = {m["part_number"] for m in matches}
        for k, v in self._all_lenses.items():
            if k in matched_keys:
                continue
            if sec_key and v.get("section") != sec_key:
                continue
            if diam_mm is not None and abs(v.get("diameter_mm", 0.0) - diam_mm) > 1e-4:
                continue
            if diam_val == "mini" and v.get("diameter_mm", 0.0) > 10.0:
                continue
            if coat_val and coat_val != "ALL":
                rec_coat = str(v.get("coating_code", "")).upper()
                if coat_val == "P01" and "P01" not in rec_coat:
                    continue
                elif coat_val != "P01" and coat_val not in rec_coat:
                    continue
            if search_txt:
                haystack = f"{k} {v.get('family', '')} {v.get('type', '')} {v.get('focal_length_mm', '')} {v.get('coating_code', '')}".lower()
                if not all(term in haystack for term in search_txt.split()):
                    continue
            matches.append(v)

        # Sort
        matches.sort(key=lambda item: (item.get("diameter_mm", 0.0), item.get("focal_length_mm", 0.0), item["part_number"]))

        total_catalog_count = len(self._all_lenses)
        if hasattr(self, 'lbl_catalog_count'):
            self.lbl_catalog_count.setText(f"Showing {len(matches)} of {total_catalog_count} items")

        self.lens_table.setRowCount(len(matches))
        for row, rec in enumerate(matches):
            pn_item = QTableWidgetItem(rec["part_number"])
            pn_item.setData(Qt.UserRole, rec)
            self.lens_table.setItem(row, 0, pn_item)

            type_str = rec.get("family", rec.get("type", "Lens"))
            self.lens_table.setItem(row, 1, QTableWidgetItem(type_str))

            coat_str = str(rec.get("coating_code", ""))
            self.lens_table.setItem(row, 2, QTableWidgetItem(coat_str))

            diam_str = f"{rec.get('diameter_mm', 25.4):g}"
            self.lens_table.setItem(row, 3, QTableWidgetItem(diam_str))

            if rec.get("grooves_per_mm"):
                efl_str = f"{rec['grooves_per_mm']:g} l/mm"
                blz = rec.get("blaze_wavelength_um", 0.0) * 1000.0
                bfl_str = f"λb {blz:g} nm" if blz > 0 else "—"
                ang = rec.get("blaze_angle_deg", 0.0)
                pwr_str = f"{ang:g}°" if ang > 0 else "0°"
            elif rec.get("conic_k") == -1.0 or rec.get("conic") == -1.0 or rec.get("section") == "parabolic_mirrors":
                efl_val = rec.get('focal_length_mm', rec.get('nominal_focal_length_mm', 0.0))
                efl_str = f"f={efl_val:g} mm"
                bfl_str = f"{rec.get('back_focal_length_mm', efl_val):.1f}"
                pwr_str = f"k=-1.0"
            else:
                efl_str = f"{rec.get('focal_length_mm', rec.get('nominal_focal_length_mm', 0.0)):g}"
                bfl_str = f"{rec.get('back_focal_length_mm', 0.0):.1f}"
                pwr_str = f"{rec.get('optical_power_diopters', 0.0):.2f}"

            self.lens_table.setItem(row, 4, QTableWidgetItem(efl_str))
            self.lens_table.setItem(row, 5, QTableWidgetItem(bfl_str))
            self.lens_table.setItem(row, 6, QTableWidgetItem(pwr_str))

        if matches:
            self.lens_table.selectRow(0)
            self._on_table_row_selected()
        else:
            self._update_preview(None)

    def _on_table_row_selected(self):
        row = -1
        if hasattr(self.lens_table, 'selectionModel') and self.lens_table.selectionModel():
            selected_rows = self.lens_table.selectionModel().selectedRows()
            if selected_rows:
                row = selected_rows[0].row()
        if row < 0 and hasattr(self.lens_table, 'currentRow'):
            row = self.lens_table.currentRow()
        if row < 0:
            self._update_preview(None)
            return
        item = self.lens_table.item(row, 0)
        rec = item.data(Qt.UserRole) if item and hasattr(item, 'data') else None
        self._update_preview(rec)

    def _update_preview(self, record: Optional[Dict[str, Any]]):
        self._selected_lens_record = record
        if not record:
            self.lbl_lens_title.setText("<b>No lens selected</b>")
            self.lbl_lens_specs.setText("")
            self.surf_table.setRowCount(0)
            self.btn_apply.setEnabled(False)
            return

        pn = record.get("part_number", "")
        family = record.get("family", record.get("type", "Lens"))
        coating = record.get("coating_code", "")
        coat_rng = record.get("coating_range_nm", [])
        coat_str = f"Coating: {coating} ({coat_rng[0]:g}–{coat_rng[1]:g} nm)" if coat_rng else f"Coating: {coating}"
        efl = record.get("focal_length_mm", record.get("nominal_focal_length_mm", 0.0))
        bfl = record.get("back_focal_length_mm", 0.0)
        diam = record.get("diameter_mm", 25.4)
        ct = record.get("center_thickness_mm", 0.0)
        pwr = record.get("optical_power_diopters", 1000.0 / efl if efl else 0.0)

        is_grating = bool(record.get("grooves_per_mm")) or record.get("section") == "diffraction_gratings"
        is_parabolic = (record.get("conic_k") == -1.0 or record.get("conic") == -1.0 or record.get("section") == "parabolic_mirrors")
        is_beamsplitter = bool(record.get("is_beam_splitter")) or ("beamsplitter" in str(record.get("section", "")))

        self.lbl_lens_title.setText(f"<h3 style='margin:0;'>{pn}</h3><span style='color:#0288D1;'>{family}</span>")
        if is_beamsplitter:
            dims = record.get("dimensions_mm", f"Ø{diam:g} mm")
            if record.get("is_polarizing"):
                ext = record.get("extinction_ratio", 1000.0)
                tp = record.get("split_ratio_T", 0.90) * 100.0
                rs = record.get("split_ratio_R", 0.995) * 100.0
                specs_html = (
                    f"<b>Polarizing Beam Splitter:</b> {record.get('type', family)}<br>"
                    f"<b>Extinction Ratio:</b> > {ext:g}:1 | <b>Tp (P-pol):</b> {tp:g}% | <b>Rs (S-pol):</b> {rs:g}%<br>"
                    f"<b>Clear Aperture / Size:</b> {dims} | <b>{coat_str}</b>"
                )
            else:
                r_pct = int(round(record.get("split_ratio_R", 0.5) * 100))
                t_pct = int(round(record.get("split_ratio_T", 0.5) * 100))
                specs_html = (
                    f"<b>Amplitude Beam Splitter:</b> {record.get('type', family)}<br>"
                    f"<b>Split Ratio:</b> {r_pct}:{t_pct} (R={r_pct}%, T={t_pct}%)<br>"
                    f"<b>Clear Aperture / Size:</b> {dims} | <b>Total Thickness:</b> {ct:.2f} mm | <b>{coat_str}</b>"
                )
        elif is_grating:
            gr = record.get("grooves_per_mm", 0.0)
            blaze_nm = record.get("blaze_wavelength_um", 0.0) * 1000.0
            blaze_ang = record.get("blaze_angle_deg", 0.0)
            dims = record.get("dimensions_mm", f"Ø{diam:g} mm")
            gtype = "Transmission" if record.get("is_transmission") else "Reflective"
            specs_html = (
                f"<b>Diffraction Grating:</b> {gr:g} lines/mm ({gtype})<br>"
                f"<b>Blaze Wavelength:</b> {blaze_nm:g} nm | <b>Blaze Angle:</b> {blaze_ang:g}°<br>"
                f"<b>Aperture / Size:</b> {dims} | <b>{coat_str}</b>"
            )
        elif is_parabolic:
            r_mirror = record.get("radius_mm", -2.0 * efl)
            specs_html = (
                f"<b>Figure:</b> Concave Parabolic Mirror (Conic k = -1.0)<br>"
                f"<b>Focal Length:</b> {efl:.2f} mm | <b>Radius of Curvature:</b> {r_mirror:.2f} mm<br>"
                f"<b>Diameter:</b> Ø{diam:g} mm | <b>Substrate Thickness:</b> {ct:.2f} mm | <b>{coat_str}</b>"
            )
        else:
            specs_html = (
                f"<b>Diameter:</b> Ø{diam:g} mm (Semi-Dia: {diam/2:.2f} mm)<br>"
                f"<b>EFL:</b> {efl:.2f} mm | <b>BFL:</b> {bfl:.2f} mm | <b>Power:</b> {pwr:.2f} D<br>"
                f"<b>Total Center Thickness:</b> {ct:.2f} mm | <b>{coat_str}</b>"
            )
        self.lbl_lens_specs.setText(specs_html)

        # Update BFL radio text
        self.radio_gap_bfl.setText(f"Set to Catalog Back Focal Length ({bfl:.2f} mm)")

        # Update Autofocus radio text
        if self.mode == "replace" and self.model:
            stack = getattr(self.model, "stack", None) or (self.model.get_surface_stack() if hasattr(self.model, "get_surface_stack") else None)
            target_data = self.target_combo.currentData() if hasattr(self, 'target_combo') else None
            target_key = target_data.get("surface_indices", [0])[0] if isinstance(target_data, dict) else target_data
            if stack and target_key is not None and hasattr(stack, "calculate_autofocus_gap"):
                try:
                    af_val = stack.calculate_autofocus_gap(target_key, pn)
                    self.radio_gap_autofocus.setText(f"🎯 Auto-correct focal distance ({af_val:.2f} mm)")
                except Exception:
                    self.radio_gap_autofocus.setText(f"🎯 Auto-correct focal distance ({bfl:.2f} mm)")
            else:
                self.radio_gap_autofocus.setText(f"🎯 Auto-correct focal distance ({bfl:.2f} mm)")
        else:
            self.radio_gap_autofocus.setText(f"🎯 Auto-correct focal distance ({bfl:.2f} mm)")

        # Populate Surface breakdown
        surfs = record.get("surfaces", [])
        self.surf_table.setRowCount(len(surfs))
        for s_idx, s in enumerate(surfs):
            s_name = f"Surface {s_idx+1}"
            r_val = float(s.get('radius', 0.0))
            r_str = f"{r_val:.2f}" if r_val != 0 else "Flat (∞)"
            t_str = f"{float(s.get('thickness', 0.0)):.2f}"
            g_str = str(s.get("glass", "AIR"))
            stype = str(s.get("surface_type", ""))

            if stype == "Grating" or s.get("grating_lines_per_mm", 0) > 0 or record.get("grooves_per_mm"):
                gr_val = s.get("grating_lines_per_mm", record.get("grooves_per_mm", 0))
                r_str = f"Grating ({gr_val:g} l/mm)"
            elif s.get("is_beam_splitter") or (is_beamsplitter and s.get("tilt_x")):
                tilt = s.get("tilt_x", 45.0)
                r_str = f"Beamsplitter ({tilt:g}°)"
            elif abs(float(s.get("conic", 0.0)) - (-1.0)) < 1e-6:
                r_str += f" [Parabolic, k=-1.0]"
            elif "conic" in s or "asphere_coeffs" in s:
                r_str += f" (Asphere)"

            self.surf_table.setItem(s_idx, 0, QTableWidgetItem(s_name))
            self.surf_table.setItem(s_idx, 1, QTableWidgetItem(r_str))
            self.surf_table.setItem(s_idx, 2, QTableWidgetItem(t_str))
            self.surf_table.setItem(s_idx, 3, QTableWidgetItem(g_str))

        self.btn_apply.setEnabled(True)

    def _on_apply(self):
        """Execute the lens replacement or insertion with user-chosen parameters."""
        if not self._selected_lens_record or not self.model:
            return

        part_number = self._selected_lens_record["part_number"]
        reverse = self.radio_reverse.isChecked()

        # Air gap choice
        air_gap_after = None
        use_catalog_bfl = False
        auto_adjust_focus = False
        if self.radio_gap_autofocus.isChecked():
            auto_adjust_focus = True
        elif self.radio_gap_bfl.isChecked():
            use_catalog_bfl = True
        elif self.radio_gap_custom.isChecked():
            air_gap_after = self.spin_custom_gap.value()

        preserve_aperture = not self.check_set_aperture.isChecked()

        try:
            if self.mode == "replace":
                data = self.target_combo.currentData()
                if not data:
                    QMessageBox.warning(self, "No Target", "Please select a target lens element to replace.")
                    return
                target_key = data.get("surface_indices", [0])[0] if isinstance(data, dict) else data

                success = self.model.replace_lens_from_catalog(
                    target=target_key,
                    part_number=part_number,
                    reverse=reverse,
                    air_gap_after=air_gap_after,
                    use_catalog_bfl=use_catalog_bfl,
                    auto_adjust_focus=auto_adjust_focus,
                    preserve_aperture=preserve_aperture,
                )
                if not success:
                    QMessageBox.critical(self, "Replace Lens Error", f"Failed to replace lens with '{part_number}'.")
                    return
            else:
                target_row = int(self.target_combo.currentData() or 0)
                eff_gap = self._selected_lens_record.get("back_focal_length_mm", 10.0) if (use_catalog_bfl or auto_adjust_focus) else (air_gap_after or 10.0)
                stack = getattr(self.model, "stack", None) or self.model.get_surface_stack()
                stack.insert_catalog_lens(
                    after_surface_index=target_row,
                    part_number=part_number,
                    air_gap_after=eff_gap,
                    reverse=reverse,
                )
                self.model.set_surface_stack(stack)

            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Lens Tool Error", str(e))
