#!/usr/bin/env python3
"""Qt front end to `plot.py`: load a Gocad TSurf, pick a vector3 property and
tune the streamlines interactively.

    python scripts/plot_gui.py [file.ts]

The loading, the streamline generation and the polyline packing are `plot.py`'s,
so the two scripts always agree on what they draw.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from qtpy import QtCore, QtGui, QtWidgets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot import (generate_streamlines, lines_to_polydata, load_gocad_ts,
                  select_vectors)

SEEDINGS = ["all", "density", "threshold", "probability"]

# what each strategy does with --seeding-param, shown next to the spin box
SEEDING_PARAM_LABEL = {
    "all": "unused",
    "density": "density factor",
    "threshold": "threshold",
    "probability": "exponent",
}


# ---------------------------------------------------------------------------
# Local axes
# ---------------------------------------------------------------------------

def vertex_normals(vertices, triangles):
    """Area-weighted vertex normals of a triangulated surface."""
    v0, v1, v2 = (vertices[triangles[:, 0]], vertices[triangles[:, 1]],
                  vertices[triangles[:, 2]])
    # the cross product is left unnormalized on purpose: its length is twice the
    # triangle area, which is the weight we want in the average
    face = np.cross(v1 - v0, v2 - v0)

    normals = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(normals, triangles[:, k], face)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.where(length > 1e-12, length, 1.0)


def compute_local_axes(vertices, triangles):
    """The (normal, strike, dip) frame at each vertex.

    `strike` is the horizontal direction lying in the surface, `dip` the
    in-surface direction perpendicular to it, pointing downslope. Both are unit
    vectors, so either can be streamlined directly.
    """
    normal = vertex_normals(vertices, triangles)

    # a vector lies in the surface iff it is perpendicular to the normal;
    # (-ny, nx, 0) is the horizontal one, i.e. the strike
    strike = np.column_stack([-normal[:, 1], normal[:, 0], np.zeros(len(normal))])
    length = np.linalg.norm(strike, axis=1, keepdims=True)
    # on a horizontal facet the strike is undefined — any direction will do
    flat = length[:, 0] < 1e-9
    strike[flat] = (1.0, 0.0, 0.0)
    length[flat] = 1.0
    strike /= length

    dip = np.cross(normal, strike)
    dip[dip[:, 2] > 0] *= -1  # point down, not up

    return {"normal": normal, "strike": strike, "dip": dip}


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

class ColorButton(QtWidgets.QPushButton):
    """A button showing a colour swatch, opening a colour dialog when clicked."""

    color_changed = QtCore.Signal()

    def __init__(self, color, parent=None):
        super().__init__(parent)
        self.setFixedHeight(24)
        self._color = QtGui.QColor(color)
        self._refresh()
        self.clicked.connect(self._pick)

    def _refresh(self):
        self.setStyleSheet(
            f"background-color: {self._color.name()}; border: 1px solid grey;")
        self.setText(self._color.name())

    def _pick(self):
        color = QtWidgets.QColorDialog.getColor(self._color, self)
        if color.isValid():
            self._color = color
            self._refresh()
            self.color_changed.emit()

    def rgb(self):
        """The colour as the (r, g, b) floats pyvista expects."""
        return (self._color.redF(), self._color.greenF(), self._color.blueF())


def spin(minimum, maximum, value, step=0.1, decimals=3):
    w = QtWidgets.QDoubleSpinBox()
    w.setRange(minimum, maximum)
    w.setSingleStep(step)
    w.setDecimals(decimals)
    w.setValue(value)
    return w


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, filename=None):
        super().__init__()
        self.setWindowTitle("Surface streamlines")
        self.resize(1280, 820)

        self.parts = []       # every part of the loaded file
        self.selected = []    # the parts currently displayed
        self.lines = {}       # part name -> list of (n, 3) polylines
        self.surface_actors = []
        self.line_actors = []
        self._loading = False  # guards the signals while the panel is repopulated

        self.plotter = QtInteractor(self)
        self.setCentralWidget(self.plotter.interactor)
        self.plotter.set_background("white")
        self.plotter.add_axes()

        dock = QtWidgets.QDockWidget("Controls", self)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable |
                         QtWidgets.QDockWidget.DockWidgetFloatable)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._build_panel())
        scroll.setMinimumWidth(330)
        dock.setWidget(scroll)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)

        self.status = self.statusBar()
        self.status.showMessage("Open a Gocad TSurf file to start")

        if filename:
            self.load(filename)

    # -- panel ------------------------------------------------------------

    def _build_panel(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        # --- surface
        box = QtWidgets.QGroupBox("Surface")
        form = QtWidgets.QFormLayout(box)
        open_button = QtWidgets.QPushButton("Open .ts file…")
        open_button.clicked.connect(self.on_open)
        form.addRow(open_button)

        self.file_label = QtWidgets.QLabel("(none)")
        self.file_label.setWordWrap(True)
        form.addRow("File", self.file_label)

        self.part_combo = QtWidgets.QComboBox()
        self.part_combo.currentIndexChanged.connect(self.on_part_changed)
        form.addRow("Part", self.part_combo)

        self.axes_button = QtWidgets.QPushButton("Compute local axes (strike / dip)")
        self.axes_button.setToolTip(
            "Derive the normal, strike and dip unit vectors from the geometry "
            "and add them to the list of streamable properties")
        self.axes_button.clicked.connect(self.on_compute_axes)
        self.axes_button.setEnabled(False)
        form.addRow(self.axes_button)
        layout.addWidget(box)

        # --- streamlines
        box = QtWidgets.QGroupBox("Streamlines")
        form = QtWidgets.QFormLayout(box)

        self.vector_combo = QtWidgets.QComboBox()
        self.vector_combo.setToolTip("The vector3 property to trace")
        form.addRow("Vector field", self.vector_combo)

        self.density = spin(0.01, 100.0, 1.0, step=0.25, decimals=2)
        self.density.setToolTip(">1 packs the streamlines closer, <1 spreads them")
        form.addRow("Density", self.density)

        self.separation = spin(0.0, 1e9, 0.0, step=1.0, decimals=4)
        self.separation.setToolTip("Separation distance in model units; "
                                   "0 derives it from the density")
        form.addRow("Separation", self.separation)

        self.step = spin(0.0, 1e9, 0.0, step=1.0, decimals=4)
        self.step.setToolTip("Integration step; 0 derives it from the bounding box")
        form.addRow("Integration step", self.step)

        self.max_iterations = QtWidgets.QSpinBox()
        self.max_iterations.setRange(1, 1000000)
        self.max_iterations.setValue(1000)
        form.addRow("Max iterations", self.max_iterations)

        self.seeding = QtWidgets.QComboBox()
        self.seeding.addItems(SEEDINGS)
        self.seeding.currentTextChanged.connect(self.on_seeding_changed)
        form.addRow("Seeding", self.seeding)

        self.seeding_param = spin(-1e9, 1e9, 1.0, step=0.1, decimals=3)
        self.seeding_param_label = QtWidgets.QLabel("unused")
        form.addRow(self.seeding_param_label, self.seeding_param)

        self.seed_scalar = QtWidgets.QComboBox()
        self.seed_scalar.setToolTip("The scalar field the seeding strategy reads")
        form.addRow("Seeding scalar", self.seed_scalar)

        self.random_seed = QtWidgets.QSpinBox()
        self.random_seed.setRange(0, 2**31 - 1)
        self.random_seed.setToolTip("0 draws a new seed on every run")
        form.addRow("Random seed", self.random_seed)

        self.project = QtWidgets.QCheckBox("Project vectors onto the surface")
        self.project.setChecked(True)
        self.project.setToolTip(
            "The field read from the file has no reason to be tangent to the "
            "surface, and the generator integrates inside the triangle planes")
        form.addRow(self.project)

        self.compute_button = QtWidgets.QPushButton("Compute streamlines")
        self.compute_button.setEnabled(False)
        self.compute_button.clicked.connect(self.on_compute)
        form.addRow(self.compute_button)

        self.clear_button = QtWidgets.QPushButton("Clear streamlines")
        self.clear_button.clicked.connect(self.on_clear)
        form.addRow(self.clear_button)
        layout.addWidget(box)

        # --- display: applied live, no recomputation
        box = QtWidgets.QGroupBox("Display")
        form = QtWidgets.QFormLayout(box)

        self.surface_color = ColorButton("#d3d3d3")
        self.surface_color.color_changed.connect(self.render_surfaces)
        form.addRow("Surface colour", self.surface_color)

        self.opacity = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(100)
        self.opacity.valueChanged.connect(self.render_surfaces)
        form.addRow("Opacity", self.opacity)

        self.smooth = QtWidgets.QCheckBox("Smooth shading")
        self.smooth.setChecked(True)
        self.smooth.toggled.connect(self.render_surfaces)
        form.addRow(self.smooth)

        self.show_edges = QtWidgets.QCheckBox("Show mesh edges")
        self.show_edges.toggled.connect(self.render_surfaces)
        form.addRow(self.show_edges)

        self.line_color = ColorButton("#000000")
        self.line_color.color_changed.connect(self.render_lines)
        form.addRow("Streamline colour", self.line_color)

        self.line_width = spin(0.1, 50.0, 2.0, step=0.5, decimals=1)
        self.line_width.valueChanged.connect(self.render_lines)
        form.addRow("Line width", self.line_width)

        self.tube = spin(0.0, 1e9, 0.0, step=1.0, decimals=4)
        self.tube.setToolTip("Draw the streamlines as tubes of this radius; "
                             "0 keeps them as lines")
        self.tube.valueChanged.connect(self.render_lines)
        form.addRow("Tube radius", self.tube)

        self.background = ColorButton("#ffffff")
        self.background.color_changed.connect(
            lambda: self.plotter.set_background(self.background.rgb()))
        form.addRow("Background", self.background)

        reset = QtWidgets.QPushButton("Reset camera")
        reset.clicked.connect(self.plotter.reset_camera)
        form.addRow(reset)
        layout.addWidget(box)

        layout.addStretch(1)
        return panel

    # -- loading ----------------------------------------------------------

    def on_open(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open a Gocad TSurf", "", "Gocad TSurf (*.ts);;All files (*)")
        if filename:
            self.load(filename)

    def load(self, filename):
        try:
            parts = load_gocad_ts(filename)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Cannot read the file", str(e))
            return

        self.parts = parts
        self.lines = {}
        self.file_label.setText(os.path.basename(filename))
        self.axes_button.setEnabled(True)

        self._loading = True
        self.part_combo.clear()
        if len(parts) > 1:
            self.part_combo.addItem("All parts", None)
        for i, part in enumerate(parts):
            self.part_combo.addItem(
                f"{part['name']} ({len(part['vertices'])} v, "
                f"{len(part['triangles'])} t)", i)
        self._loading = False

        self.on_part_changed()
        self.plotter.reset_camera()

    def on_part_changed(self):
        if self._loading or not self.parts:
            return
        index = self.part_combo.currentData()
        self.selected = self.parts if index is None else [self.parts[index]]
        self.lines = {}
        self.refresh_properties()
        self.render_surfaces()
        self.render_lines()

    def refresh_properties(self):
        """Repopulate the two property combos from the selected parts."""
        vectors, scalars = [], []
        for part in self.selected:
            for name, field in part["properties"].items():
                target = vectors if field.shape[1] == 3 else scalars
                if field.shape[1] in (1, 3) and name not in target:
                    target.append(name)

        self._loading = True
        previous = self.vector_combo.currentText()
        self.vector_combo.clear()
        self.vector_combo.addItems(vectors)
        if previous in vectors:
            self.vector_combo.setCurrentText(previous)

        previous = self.seed_scalar.currentText()
        self.seed_scalar.clear()
        self.seed_scalar.addItem("magnitude of the vector field")
        self.seed_scalar.addItems(scalars)
        if previous:
            self.seed_scalar.setCurrentText(previous)
        self._loading = False

        self.compute_button.setEnabled(bool(vectors))
        if not vectors:
            self.status.showMessage(
                "No vector3 property here — use “Compute local axes” to derive "
                "strike and dip from the geometry")

    def on_seeding_changed(self, name):
        self.seeding_param_label.setText(SEEDING_PARAM_LABEL[name])
        self.seeding_param.setEnabled(name != "all")
        self.seed_scalar.setEnabled(name != "all")

    def on_compute_axes(self):
        for part in self.selected:
            axes = compute_local_axes(part["vertices"], part["triangles"])
            part["properties"].update(axes)
        self.refresh_properties()
        # dip is usually the interesting one on a fault, so preselect it
        if self.vector_combo.findText("dip") >= 0:
            self.vector_combo.setCurrentText("dip")
        self.status.showMessage(
            f"Added normal, strike and dip to {len(self.selected)} part(s)")

    # -- streamlines ------------------------------------------------------

    def params(self):
        """The knobs of the panel, shaped like `plot.py`'s argparse namespace."""
        return SimpleNamespace(
            density=self.density.value(),
            separation=self.separation.value() or None,
            step=self.step.value() or None,
            max_iterations=self.max_iterations.value(),
            seeding=self.seeding.currentText(),
            seeding_param=self.seeding_param.value(),
            random_seed=self.random_seed.value(),
            no_project=not self.project.isChecked(),
        )

    def on_compute(self):
        name = self.vector_combo.currentText()
        if not name:
            return
        args = self.params()
        self.lines = {}
        skipped, total = [], 0

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            for part in self.selected:
                try:
                    vectors = np.nan_to_num(select_vectors(part, name))
                except KeyError as e:
                    skipped.append(f"{part['name']} ({e})")
                    continue

                scalars = None
                if args.seeding != "all":
                    # every strategy but AllTrianglesSeeding reads the scalar field
                    if self.seed_scalar.currentIndex() <= 0:
                        scalars = np.linalg.norm(vectors, axis=1)
                    else:
                        field = part["properties"].get(
                            self.seed_scalar.currentText())
                        if field is None:
                            skipped.append(f"{part['name']} (no seeding scalar)")
                            continue
                        scalars = np.nan_to_num(field[:, 0])

                lines = generate_streamlines(part["vertices"], part["triangles"],
                                             vectors, scalars, args)
                self.lines[part["name"]] = lines
                total += len(lines)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Generation failed", str(e))
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        self.render_lines()
        message = f"{total} streamlines from “{name}”"
        if skipped:
            message += " — skipped " + ", ".join(skipped)
        self.status.showMessage(message)

    def on_clear(self):
        self.lines = {}
        self.render_lines()
        self.status.showMessage("Cleared")

    # -- rendering --------------------------------------------------------

    def render_surfaces(self):
        for actor in self.surface_actors:
            self.plotter.remove_actor(actor, render=False)
        self.surface_actors = []

        for part in self.selected:
            triangles = part["triangles"]
            faces = np.hstack([np.full((len(triangles), 1), 3), triangles]).ravel()
            surface = pv.PolyData(part["vertices"], faces)
            self.surface_actors.append(self.plotter.add_mesh(
                surface,
                color=self.surface_color.rgb(),
                opacity=self.opacity.value() / 100.0,
                show_edges=self.show_edges.isChecked(),
                edge_color="grey",
                smooth_shading=self.smooth.isChecked(),
                reset_camera=False,
            ))
        self.plotter.render()

    def render_lines(self):
        for actor in self.line_actors:
            self.plotter.remove_actor(actor, render=False)
        self.line_actors = []

        radius = self.tube.value()
        for lines in self.lines.values():
            if not lines:
                continue
            polylines = lines_to_polydata(lines)
            if radius:
                polylines = polylines.tube(radius=radius)
            self.line_actors.append(self.plotter.add_mesh(
                polylines,
                color=self.line_color.rgb(),
                line_width=self.line_width.value(),
                reset_camera=False,
            ))
        self.plotter.render()

    def closeEvent(self, event):
        # the render window owns an OpenGL context Qt does not know about
        self.plotter.close()
        super().closeEvent(event)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    app = QtWidgets.QApplication(argv)
    window = MainWindow(argv[1] if len(argv) > 1 else None)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
