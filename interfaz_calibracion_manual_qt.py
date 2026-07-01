import csv
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    print("[INFO] pyvistaqt no instalado — modo 3D no disponible. "
          "Instala con:  pip install pyvistaqt vtk")


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "csv_entrada"
CALIBRATION_DIR = BASE_DIR / "calibracion"
OUTPUT_DIR = BASE_DIR / "salidas"
DXF_OUTPUT_DIR = OUTPUT_DIR / "dxf"
CSV_OUTPUT_DIR = OUTPUT_DIR / "csv"
IMAGE_OUTPUT_DIR = OUTPUT_DIR / "imagenes"
TXT_OUTPUT_DIR = OUTPUT_DIR / "txt"

for directory in (INPUT_DIR, CALIBRATION_DIR, DXF_OUTPUT_DIR, CSV_OUTPUT_DIR, IMAGE_OUTPUT_DIR, TXT_OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

DEFAULT_CSV_NAME = "test5.csv"
BEAM_DIAMETER_MM = 0.055
MIN_DXF_SEGMENT_MM = BEAM_DIAMETER_MM
CROSS_SIZE_MM = 0.500

DXF_LEVEL_CONFIG = {
    2: {"pin": "PIN_1", "layer": "PIN_1_NIVEL_2", "color": "3"},
    3: {"pin": "PIN_2", "layer": "PIN_2_NIVEL_3", "color": "6"},
    4: {"pin": "PIN_3", "layer": "PIN_3_NIVEL_4", "color": "1"},
}

LEVEL_STYLE_COLORS = [
    (0, 200, 0),
    (220, 0, 220),
    (220, 0, 0),
    (0, 145, 255),
    (255, 150, 0),
    (120, 80, 255),
    (80, 80, 80),
]

DXF_COLOR_CODES = ["3", "6", "1", "4", "30", "5", "8"]

HEIGHT_LUT_POSITIONS = np.array([0.00, 0.35, 0.50, 0.68, 0.82, 1.00], dtype=float)
HEIGHT_LUT_COLORS = np.array([
    [7, 87, 200],
    [126, 200, 255],
    [22, 163, 74],
    [255, 216, 77],
    [255, 138, 31],
    [215, 25, 32],
], dtype=np.uint8)


@dataclass
class ControlPoint:
    point_id: str
    profile_x_mm: float
    profile_y_mm: float
    samlight_x_mm: float
    samlight_y_mm: float
    use_for_affine: bool = True


@dataclass
class WorkPoint:
    point_id: str
    profile_x_mm: float
    profile_y_mm: float
    samlight_x_mm: float
    samlight_y_mm: float
    use_for_affine: bool = False


def clean_cell(value):
    value = str(value).strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1]
    return value


def resolve_existing_file(arg, default_dir):
    path = Path(arg)
    candidates = [path, BASE_DIR / path, default_dir / path.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"No se encontro '{arg}'. Prueba en: {default_dir}")


def read_keyence_height_csv(path):
    print(f"Leyendo archivo: {path}")
    with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.reader(f))

    header = {}
    height_start = None
    for i, row in enumerate(rows):
        cells = [clean_cell(c) for c in row]
        if cells and cells[0] == "Height":
            height_start = i + 1
            break
        if len(cells) >= 2 and cells[0]:
            header[cells[0]] = cells[1]

    if height_start is None:
        raise ValueError('No se encontro la seccion "Height" en el CSV.')

    nx = int(float(header["Horizontal"]))
    declared_ny = int(float(header["Vertical"]))
    pixel_size = float(header.get("XY Calibration", "1000")) / 1000.0

    data_rows = rows[height_start:]
    while data_rows and not any(clean_cell(c) for c in data_rows[-1]):
        data_rows.pop()
    ny = min(declared_ny, len(data_rows))

    z = np.full((ny, nx), np.nan, dtype=np.float32)
    for y_idx in range(ny):
        row = data_rows[y_idx]
        for x_idx in range(min(nx, len(row))):
            value = clean_cell(row[x_idx])
            if value == "":
                continue
            try:
                z[y_idx, x_idx] = float(value)
            except ValueError:
                pass
    return z, pixel_size, header


def robust_percentile(values, percentile, fallback):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return fallback
    return float(np.nanpercentile(finite, percentile))


def dxf_header(layer_configs):
    lines = [
        "0", "SECTION",
        "2", "HEADER",
        "9", "$INSUNITS",
        "70", "4",
        "0", "ENDSEC",
        "0", "SECTION",
        "2", "TABLES",
        "0", "TABLE",
        "2", "LAYER",
        "70", str(len(layer_configs)),
    ]
    for cfg in layer_configs:
        lines.extend([
            "0", "LAYER",
            "2", cfg["layer"],
            "70", "0",
            "62", str(cfg["color"]),
            "6", "CONTINUOUS",
        ])
    lines.extend(["0", "ENDTAB", "0", "ENDSEC", "0", "SECTION", "2", "ENTITIES"])
    return lines


def add_dxf_line(lines, layer, color, x0, y0, x1, y1):
    lines.extend([
        "0", "LINE",
        "8", layer,
        "62", str(color),
        "10", f"{x0:.6f}",
        "20", f"{y0:.6f}",
        "30", "0",
        "11", f"{x1:.6f}",
        "21", f"{y1:.6f}",
        "31", "0",
    ])


class CalibrationViewBox(pg.ViewBox):
    clicked = QtCore.Signal(float, float)
    dragStarted = QtCore.Signal(float, float)
    dragMoved = QtCore.Signal(float, float)
    dragFinished = QtCore.Signal(float, float)

    def __init__(self):
        super().__init__()
        self.nearest_callback = None
        self.dragging_point = False

    def mouseClickEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            pos = self.mapSceneToView(event.scenePos())
            self.clicked.emit(float(pos.x()), float(pos.y()))
            event.accept()
            return
        super().mouseClickEvent(event)

    def mouseDragEvent(self, event, axis=None):
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            super().mouseDragEvent(event, axis=axis)
            return

        pos = self.mapSceneToView(event.scenePos())
        x_val = float(pos.x())
        y_val = float(pos.y())

        if event.isStart():
            can_drag = False
            if self.nearest_callback is not None:
                can_drag = self.nearest_callback(x_val, y_val)
            self.dragging_point = bool(can_drag)
            if self.dragging_point:
                self.dragStarted.emit(x_val, y_val)
                event.accept()
                return
            super().mouseDragEvent(event, axis=axis)
            return

        if self.dragging_point:
            self.dragMoved.emit(x_val, y_val)
            if event.isFinish():
                self.dragFinished.emit(x_val, y_val)
                self.dragging_point = False
            event.accept()
            return

        super().mouseDragEvent(event, axis=axis)


class WorkPointDialog(QtWidgets.QDialog):
    def __init__(self, parent, default_id, profile_x, profile_y, samlight_x, samlight_y):
        super().__init__(parent)
        self.setWindowTitle("Confirmar punto de trabajo")
        self.setMinimumWidth(340)
        layout = QtWidgets.QFormLayout(self)

        self._edit_id = QtWidgets.QLineEdit(default_id)
        layout.addRow("ID", self._edit_id)
        layout.addRow("Perfil X (mm)", QtWidgets.QLabel(f"{profile_x:.6f}"))
        layout.addRow("Perfil Y (mm)", QtWidgets.QLabel(f"{profile_y:.6f}"))

        def spin(val):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(-10000, 10000)
            s.setDecimals(6)
            s.setSingleStep(0.001)
            s.setValue(val)
            return s

        self._spin_sx = spin(samlight_x)
        self._spin_sy = spin(samlight_y)
        layout.addRow("SAMLight X (mm)", self._spin_sx)
        layout.addRow("SAMLight Y (mm)", self._spin_sy)

        note = QtWidgets.QLabel("Las coords SAMLight se calculan de la calibración afín.\nPuedes editarlas si lo necesitas.")
        note.setStyleSheet("color: #888; font-size: 10px;")
        note.setWordWrap(True)
        layout.addRow(note)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def point_id(self):   return self._edit_id.text().strip() or "WP"
    def samlight_x(self): return float(self._spin_sx.value())
    def samlight_y(self): return float(self._spin_sy.value())


class ManualCalibrationQt(QtWidgets.QMainWindow):
    def __init__(self, csv_file, calibration_file=None):
        super().__init__()
        self.csv_file = csv_file
        self.calibration_file = calibration_file
        self.z, self.pixel_size, self.header = read_keyence_height_csv(csv_file)
        self.ny, self.nx = self.z.shape
        self.x_min = 0.0
        self.x_max = self.nx * self.pixel_size
        self.y_min = -self.ny * self.pixel_size
        self.y_max = 0.0
        self.z_finite = self.z[np.isfinite(self.z)]
        self.z_min = float(np.nanmin(self.z_finite)) if self.z_finite.size else -1.0
        self.z_max = float(np.nanmax(self.z_finite)) if self.z_finite.size else 1.0
        self.color_min = min(robust_percentile(self.z, 2, self.z_min), -0.001)
        self.color_max = max(robust_percentile(self.z, 98, self.z_max), 0.001)

        positive = self.z_finite[self.z_finite > 0]
        positive_max = float(np.nanmax(positive)) if positive.size else max(self.z_max, 0.001)
        self.level_n2 = positive_max * 0.60
        self.level_n3 = positive_max * 0.75
        self.level_n4 = positive_max * 0.90
        self.level_thresholds = {
            2: self.level_n2,
            3: self.level_n3,
            4: self.level_n4,
        }

        self.points = []
        self.selected_index = None
        self.adding_point = False
        self.dragging_index = None
        self.affine_x = None
        self.affine_y = None
        self.inverse_affine = None
        self.profile_data = None
        self.dxf_profile_data = None
        self.comsol_profile_data = None
        self.comsol_pick_mode = None
        self.comsol_start = None
        self.comsol_end = None
        self.dragging_comsol_endpoint = None
        self.block_updates = False
        self.level_controls = {}
        self._view_mode = '2d'
        self._3d_actor = None
        self.work_points = []
        self.profile_list = []
        self.work_point_items = []
        self.adding_work_point = False

        self.point_scatter = None
        self.point_labels = []
        self.cross_items = []
        self.profile_line_item = None
        self.comsol_items = []
        self.level_items = []

        if calibration_file:
            self.load_calibration_points(calibration_file)

        self.build_ui()
        self.update_image()
        self.refresh_calibration()
        self.refresh_points()
        self.refresh_comsol_items()
        self.update_profile_plot()

    def build_ui(self):
        self.setWindowTitle(f"Calibracion manual rapida - {self.csv_file.name}")
        self.resize(1500, 900)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        self.viewbox = CalibrationViewBox()
        self.viewbox.nearest_callback = self.prepare_drag_at
        self.plot = pg.PlotWidget(viewBox=self.viewbox)
        self.plot.setBackground("w")
        self.plot.setAspectLocked(True)
        self.plot.showGrid(x=True, y=True, alpha=0.20)
        self.plot.setLabel("bottom", "x perfil", units="mm")
        self.plot.setLabel("left", "y perfil", units="mm")
        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.plot.addItem(self.image_item)
        self.image_item.setRect(QtCore.QRectF(self.x_min, self.y_min, self.x_max - self.x_min, self.y_max - self.y_min))
        self.plot.setXRange(self.x_min, self.x_max, padding=0.01)
        self.plot.setYRange(self.y_min, self.y_max, padding=0.01)
        self.map_stack = QtWidgets.QStackedWidget()
        self.map_stack.addWidget(self.plot)
        if HAS_PYVISTA:
            self._plotter = QtInteractor(self.map_stack)
            self.map_stack.addWidget(self._plotter)
        main_layout.addWidget(self.map_stack, 4)

        self.viewbox.clicked.connect(self.on_map_clicked)
        self.viewbox.dragMoved.connect(self.on_drag_moved)
        self.viewbox.dragFinished.connect(self.on_drag_finished)

        side = QtWidgets.QWidget()
        side.setMinimumWidth(500)
        side_layout = QtWidgets.QVBoxLayout(side)
        main_layout.addWidget(side, 0)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.make_height_tab(), "Heightmap")
        self.tabs.addTab(self.make_calibration_tab(), "Calibracion / DXF")
        self.tabs.addTab(self.make_perfiles_tab(), "Perfiles")
        self.tabs.addTab(self.make_comsol_tab(), "COMSOL")
        side_layout.addWidget(self.tabs, 4)
        side_layout.addWidget(self.make_status_group(), 1)

    def make_scroll_tab(self, widget):
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setWidget(widget)
        return scroll

    def make_height_tab(self):
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.addWidget(self.make_height_group())
        layout.addStretch(1)
        return content

    def make_calibration_tab(self):
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(self.make_calibration_help_group())
        layout.addWidget(self.make_points_group(), 2)
        layout.addWidget(self.make_work_points_group(), 1)
        layout.addStretch(1)
        return content

    def make_perfiles_tab(self):
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(self.make_dxf_profile_group(), 3)
        layout.addWidget(self.make_levels_group())
        layout.addWidget(self.make_profile_list_group())
        export_bar = QtWidgets.QHBoxLayout()
        self.btn_export_bar_dxf = QtWidgets.QPushButton("⬇ Exportar DXF perfil")
        self.btn_export_bar_csv = QtWidgets.QPushButton("⬇ Exportar CSV perfil")
        self.btn_export_bar_dxf.setStyleSheet("font-weight:bold; background:#1a7abf; color:white; padding:6px;")
        self.btn_export_bar_csv.setStyleSheet("font-weight:bold; background:#2e7d32; color:white; padding:6px;")
        self.btn_export_bar_dxf.clicked.connect(self.export_profile_dxf)
        self.btn_export_bar_csv.clicked.connect(self.export_profile_csv)
        export_bar.addWidget(self.btn_export_bar_dxf)
        export_bar.addWidget(self.btn_export_bar_csv)
        layout.addLayout(export_bar)
        return content

    def make_comsol_tab(self):
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.addWidget(self.make_comsol_controls_group())
        layout.addWidget(self.make_comsol_profile_group(), 1)
        return content

    def make_height_group(self):
        group = QtWidgets.QGroupBox("Heightmap")
        layout = QtWidgets.QFormLayout(group)
        zm, zM = self.z_min * 1000, self.z_max * 1000
        self.spin_color_min = self.make_spin(self.color_min * 1000, zm, zM, decimals=2, step=0.5)
        self.spin_color_max = self.make_spin(self.color_max * 1000, zm, zM, decimals=2, step=0.5)
        self.slider_color_min = self.make_slider(self.color_min)
        self.slider_color_max = self.make_slider(self.color_max)
        self.spin_color_min.valueChanged.connect(self.on_color_spin_changed)
        self.spin_color_max.valueChanged.connect(self.on_color_spin_changed)
        self.slider_color_min.valueChanged.connect(self.on_color_slider_changed)
        self.slider_color_max.valueChanged.connect(self.on_color_slider_changed)
        row_min = QtWidgets.QHBoxLayout()
        row_min.addWidget(self.spin_color_min)
        row_min.addWidget(self.slider_color_min, 1)
        row_max = QtWidgets.QHBoxLayout()
        row_max.addWidget(self.spin_color_max)
        row_max.addWidget(self.slider_color_max, 1)
        layout.addRow("Min (µm)", row_min)
        layout.addRow("Max (µm)", row_max)

        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addRow(sep)

        self._combo_view_mode = QtWidgets.QComboBox()
        self._combo_view_mode.addItem("2D Heightmap")
        if HAS_PYVISTA:
            self._combo_view_mode.addItem("3D Surface (Keyence)")
        else:
            self._combo_view_mode.addItem("3D — instala: pip install pyvistaqt vtk")
            self._combo_view_mode.model().item(1).setEnabled(False)
        layout.addRow("Vista", self._combo_view_mode)

        row_mag = QtWidgets.QHBoxLayout()
        self._spin_height_mag = self.make_spin(1.0, 0.1, 500.0, decimals=1, step=0.5)
        row_mag.addWidget(self._spin_height_mag)
        for lbl, val in [("×1", 1.0), ("×10", 10.0), ("×50", 50.0), ("×100", 100.0)]:
            btn = QtWidgets.QPushButton(lbl)
            btn.setFixedWidth(38)
            btn.clicked.connect(lambda _, v=val: self._spin_height_mag.setValue(v))
            row_mag.addWidget(btn)
        layout.addRow("Altura ×", row_mag)

        self._combo_view_mode.currentTextChanged.connect(self._switch_view_mode)
        self._spin_height_mag.valueChanged.connect(self._on_height_mag_changed)

        self._lbl_3d_controls = QtWidgets.QLabel(
            "🖱 Arrastrar = rotar  |  Shift+Arrastrar = pan  |  Scroll = zoom"
        )
        self._lbl_3d_controls.setStyleSheet("color: #888; font-size: 10px;")
        self._lbl_3d_controls.setWordWrap(True)
        self._lbl_3d_controls.setVisible(False)
        layout.addRow(self._lbl_3d_controls)

        return group

    def make_points_group(self):
        group = QtWidgets.QGroupBox("Puntos de calibracion")
        layout = QtWidgets.QVBoxLayout(group)

        form = QtWidgets.QFormLayout()
        self.edit_point_id = QtWidgets.QLineEdit(self.next_point_id())
        self.spin_laser_x = self.make_spin(0.0, -10000, 10000, decimals=6, step=0.001)
        self.spin_laser_y = self.make_spin(0.0, -10000, 10000, decimals=6, step=0.001)
        form.addRow("ID", self.edit_point_id)
        form.addRow("X SAMLight (mm)", self.spin_laser_x)
        form.addRow("Y SAMLight (mm)", self.spin_laser_y)
        layout.addLayout(form)

        buttons = QtWidgets.QHBoxLayout()
        self.btn_add_point = QtWidgets.QPushButton("Nuevo punto")
        self.btn_update_point = QtWidgets.QPushButton("Actualizar")
        self.btn_delete_point = QtWidgets.QPushButton("Borrar")
        buttons.addWidget(self.btn_add_point)
        buttons.addWidget(self.btn_update_point)
        buttons.addWidget(self.btn_delete_point)
        layout.addLayout(buttons)

        self.btn_add_point.clicked.connect(self.start_add_point)
        self.btn_update_point.clicked.connect(self.update_selected_point)
        self.btn_delete_point.clicked.connect(self.delete_selected_point)

        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["ID", "X laser", "Y laser", "X perfil", "Y perfil", "err um"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.on_table_selection_changed)
        layout.addWidget(self.table, 1)

        buttons2 = QtWidgets.QHBoxLayout()
        self.btn_save_cal = QtWidgets.QPushButton("Guardar calib.")
        self.btn_cross_dxf = QtWidgets.QPushButton("DXF cruces")
        buttons2.addWidget(self.btn_save_cal)
        buttons2.addWidget(self.btn_cross_dxf)
        layout.addLayout(buttons2)
        self.btn_save_cal.clicked.connect(self.save_calibration_points)
        self.btn_cross_dxf.clicked.connect(self.export_crosses_dxf)
        return group

    def make_calibration_help_group(self):
        group = QtWidgets.QGroupBox("Como usar esta pestaña")
        layout = QtWidgets.QVBoxLayout(group)
        text = QtWidgets.QLabel(
            "1. Escribe la coordenada real del laser/Samlight.\n"
            "2. Pulsa Nuevo punto y haz click en el pico correspondiente del heightmap.\n"
            "3. Repite con al menos 3 puntos, mejor 5 repartidos.\n"
            "4. Guarda calibracion o genera DXF de cruces para validar.\n"
            "5. Para DXF de perfil, elige dos puntos A/B, pulsa Ver perfil y ajusta niveles."
        )
        text.setWordWrap(True)
        layout.addWidget(text)
        return group

    def make_dxf_profile_group(self):
        group = QtWidgets.QGroupBox("Perfil para DXF Samlight")
        layout = QtWidgets.QVBoxLayout(group)

        top = QtWidgets.QHBoxLayout()
        self.combo_p1 = QtWidgets.QComboBox()
        self.combo_p2 = QtWidgets.QComboBox()
        self.btn_profile = QtWidgets.QPushButton("Ver perfil")
        self.btn_add_to_list = QtWidgets.QPushButton("➕ Añadir a lista")
        top.addWidget(QtWidgets.QLabel("A"))
        top.addWidget(self.combo_p1)
        top.addWidget(QtWidgets.QLabel("B"))
        top.addWidget(self.combo_p2)
        top.addWidget(self.btn_profile)
        top.addWidget(self.btn_add_to_list)
        layout.addLayout(top)
        self.btn_profile.clicked.connect(self.update_profile_from_controls)
        self.btn_add_to_list.clicked.connect(self.add_profile_to_list)

        self.dxf_profile_plot = pg.PlotWidget()
        self.dxf_profile_plot.setBackground("w")
        self.dxf_profile_plot.showGrid(x=True, y=True, alpha=0.25)
        self.dxf_profile_plot.setLabel("bottom", "distancia", units="mm")
        self.dxf_profile_plot.setLabel("left", "altura", units="µm")
        self.dxf_profile_plot.setMinimumHeight(230)
        layout.addWidget(self.dxf_profile_plot, 1)
        self.profile_plot = self.dxf_profile_plot

        export_buttons = QtWidgets.QHBoxLayout()
        self.btn_export_profile_dxf = QtWidgets.QPushButton("Exportar DXF")
        self.btn_export_profile_csv = QtWidgets.QPushButton("Exportar CSV")
        export_buttons.addWidget(self.btn_export_profile_dxf)
        export_buttons.addWidget(self.btn_export_profile_csv)
        layout.addLayout(export_buttons)
        self.btn_export_profile_dxf.clicked.connect(self.export_profile_dxf)
        self.btn_export_profile_csv.clicked.connect(self.export_profile_csv)
        return group

    def make_work_points_group(self):
        group = QtWidgets.QGroupBox("Puntos de trabajo — click en mapa → coords SAMLight")
        layout = QtWidgets.QVBoxLayout(group)

        note = QtWidgets.QLabel(
            "Con calibración activa (≥3 pts): haz click → obtén coords SAMLight automáticamente.\n"
            "Aparecen como cruces en el mapa y en los selectores A/B de perfil."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(note)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_add_wp = QtWidgets.QPushButton("➕ Nuevo punto")
        self.btn_del_wp = QtWidgets.QPushButton("Borrar sel.")
        self.btn_clear_wp = QtWidgets.QPushButton("Borrar todos")
        btn_row.addWidget(self.btn_add_wp)
        btn_row.addWidget(self.btn_del_wp)
        btn_row.addWidget(self.btn_clear_wp)
        layout.addLayout(btn_row)

        self.work_table = QtWidgets.QTableWidget(0, 3)
        self.work_table.setHorizontalHeaderLabels(["ID", "X SAMLight (mm)", "Y SAMLight (mm)"])
        self.work_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.work_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.work_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.work_table.setMaximumHeight(110)
        layout.addWidget(self.work_table)

        self.btn_add_wp.clicked.connect(self.start_add_work_point)
        self.btn_del_wp.clicked.connect(self.delete_selected_work_point)
        self.btn_clear_wp.clicked.connect(self.clear_all_work_points)
        return group

    def make_levels_group(self):
        group = QtWidgets.QGroupBox("Niveles de altura para DXF")
        layout = QtWidgets.QVBoxLayout(group)
        self.levels_layout = QtWidgets.QVBoxLayout()
        layout.addLayout(self.levels_layout)

        buttons = QtWidgets.QHBoxLayout()
        self.btn_add_level = QtWidgets.QPushButton("Añadir nivel")
        self.btn_remove_level = QtWidgets.QPushButton("Quitar ultimo")
        buttons.addWidget(self.btn_add_level)
        buttons.addWidget(self.btn_remove_level)
        layout.addLayout(buttons)

        self.btn_add_level.clicked.connect(self.add_level)
        self.btn_remove_level.clicked.connect(self.remove_highest_level)
        self.rebuild_level_controls()
        return group

    def make_comsol_controls_group(self):
        group = QtWidgets.QGroupBox("Perfil 2D para COMSOL")
        layout = QtWidgets.QVBoxLayout(group)

        form = QtWidgets.QFormLayout()
        self.spin_comsol_length = self.make_spin(1.0, 0.001, max(self.x_max - self.x_min, self.y_max - self.y_min), decimals=6, step=0.001)
        self.spin_comsol_samples = QtWidgets.QSpinBox()
        self.spin_comsol_samples.setRange(2, 20000)
        default_samples = max(2, min(5000, int(np.ceil(1.0 / max(self.pixel_size, 1e-9))) + 1))
        self.spin_comsol_samples.setValue(default_samples)
        form.addRow("Longitud mm", self.spin_comsol_length)
        form.addRow("Puntos perfil", self.spin_comsol_samples)
        layout.addLayout(form)

        pick_buttons = QtWidgets.QHBoxLayout()
        self.btn_comsol_start = QtWidgets.QPushButton("Marcar inicio")
        self.btn_comsol_end = QtWidgets.QPushButton("Marcar fin")
        self.btn_comsol_profile = QtWidgets.QPushButton("Ver COMSOL")
        pick_buttons.addWidget(self.btn_comsol_start)
        pick_buttons.addWidget(self.btn_comsol_end)
        pick_buttons.addWidget(self.btn_comsol_profile)
        layout.addLayout(pick_buttons)

        export_buttons = QtWidgets.QHBoxLayout()
        self.btn_export_comsol_csv = QtWidgets.QPushButton("COMSOL CSV")
        self.btn_export_comsol_txt = QtWidgets.QPushButton("COMSOL TXT")
        export_buttons.addWidget(self.btn_export_comsol_csv)
        export_buttons.addWidget(self.btn_export_comsol_txt)
        layout.addLayout(export_buttons)

        self.spin_comsol_length.valueChanged.connect(self.on_comsol_length_changed)
        self.spin_comsol_samples.valueChanged.connect(self.on_comsol_samples_changed)
        self.btn_comsol_start.clicked.connect(self.start_pick_comsol_start)
        self.btn_comsol_end.clicked.connect(self.start_pick_comsol_end)
        self.btn_comsol_profile.clicked.connect(self.compute_comsol_profile)
        self.btn_export_comsol_csv.clicked.connect(self.export_comsol_csv)
        self.btn_export_comsol_txt.clicked.connect(self.export_comsol_txt)
        return group

    def make_comsol_profile_group(self):
        group = QtWidgets.QGroupBox("Perfil seleccionado")
        layout = QtWidgets.QVBoxLayout(group)
        self.comsol_profile_plot = pg.PlotWidget()
        self.comsol_profile_plot.setBackground("w")
        self.comsol_profile_plot.showGrid(x=True, y=True, alpha=0.25)
        self.comsol_profile_plot.setLabel("bottom", "distancia", units="mm")
        self.comsol_profile_plot.setLabel("left", "altura", units="µm")
        layout.addWidget(self.comsol_profile_plot, 1)
        return group

    def make_status_group(self):
        group = QtWidgets.QGroupBox("Estado")
        layout = QtWidgets.QVBoxLayout(group)
        self.status = QtWidgets.QPlainTextEdit()
        self.status.setReadOnly(True)
        self.status.setMaximumBlockCount(500)
        layout.addWidget(self.status)
        return group

    def make_spin(self, value, minimum, maximum, decimals=4, step=0.001):
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(float(minimum), float(maximum))
        spin.setSingleStep(float(step))
        spin.setValue(float(value))
        spin.setKeyboardTracking(False)
        return spin

    def make_slider(self, value):
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(0, 10000)
        slider.setValue(self.value_to_slider(value))
        return slider

    def value_to_slider(self, value):
        if self.z_max <= self.z_min:
            return 0
        ratio = (float(value) - self.z_min) / (self.z_max - self.z_min)
        return int(np.clip(round(ratio * 10000), 0, 10000))

    def slider_to_value(self, slider_value):
        if self.z_max <= self.z_min:
            return self.z_min
        return self.z_min + (float(slider_value) / 10000.0) * (self.z_max - self.z_min)

    def set_spin_quiet(self, spin, value):
        blocker = QtCore.QSignalBlocker(spin)
        spin.setValue(float(value))
        del blocker

    def set_slider_quiet(self, slider, value):
        blocker = QtCore.QSignalBlocker(slider)
        slider.setValue(self.value_to_slider(value))
        del blocker

    def on_color_spin_changed(self):
        if self.block_updates:
            return
        self.color_min = float(self.spin_color_min.value()) / 1000.0
        self.color_max = float(self.spin_color_max.value()) / 1000.0
        if self.color_min >= self.color_max:
            self.color_max = self.color_min + 0.001
            self.set_spin_quiet(self.spin_color_max, self.color_max * 1000.0)
        self.set_slider_quiet(self.slider_color_min, self.color_min)
        self.set_slider_quiet(self.slider_color_max, self.color_max)
        self.update_image()

    def on_color_slider_changed(self):
        if self.block_updates:
            return
        self.color_min = self.slider_to_value(self.slider_color_min.value())
        self.color_max = self.slider_to_value(self.slider_color_max.value())
        if self.color_min >= self.color_max:
            return
        self.set_spin_quiet(self.spin_color_min, self.color_min * 1000.0)
        self.set_spin_quiet(self.spin_color_max, self.color_max * 1000.0)
        self.update_image()

    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                self.clear_layout(child_layout)

    def sorted_level_ids(self):
        return sorted(self.level_thresholds)

    def level_color(self, level):
        return LEVEL_STYLE_COLORS[(int(level) - 2) % len(LEVEL_STYLE_COLORS)]

    def dxf_config_for_level(self, level):
        idx = (int(level) - 2) % len(DXF_COLOR_CODES)
        pin = f"PIN_{int(level) - 1}"
        return {
            "pin": pin,
            "layer": f"{pin}_NIVEL_{int(level)}",
            "color": DXF_COLOR_CODES[idx],
        }

    def rebuild_level_controls(self):
        if not hasattr(self, "levels_layout"):
            return
        self.clear_layout(self.levels_layout)
        self.level_controls = {}
        for level in self.sorted_level_ids():
            row_widget = QtWidgets.QWidget()
            row = QtWidgets.QGridLayout(row_widget)
            row.setContentsMargins(0, 0, 0, 0)

            color = self.level_color(level)
            label = QtWidgets.QLabel(f"N{level}")
            label.setStyleSheet(f"font-weight: 700; color: rgb({color[0]}, {color[1]}, {color[2]});")
            spin = self.make_spin(self.level_thresholds[level] * 1000, self.z_min * 1000, self.z_max * 1000, decimals=2, step=0.5)
            slider = self.make_slider(self.level_thresholds[level])
            remove = QtWidgets.QPushButton("Quitar")
            remove.setEnabled(len(self.level_thresholds) > 1)

            row.addWidget(label, 0, 0)
            row.addWidget(spin, 0, 1)
            row.addWidget(remove, 0, 2)
            row.addWidget(slider, 1, 0, 1, 3)

            spin.valueChanged.connect(lambda value, lvl=level: self.set_level_threshold(lvl, float(value) / 1000.0, "spin"))
            slider.valueChanged.connect(lambda value, lvl=level: self.set_level_threshold(lvl, self.slider_to_value(value), "slider"))
            remove.clicked.connect(lambda _checked=False, lvl=level: self.remove_level(lvl))

            self.level_controls[level] = {"spin": spin, "slider": slider, "row": row_widget}
            self.levels_layout.addWidget(row_widget)

    def set_level_threshold(self, level, value, source):
        if self.block_updates:
            return
        self.level_thresholds[int(level)] = float(value)
        self.enforce_level_order()
        self.sync_level_controls(source_level=int(level), source=source)
        self.update_dxf_profile_plot()

    def add_level(self):
        next_level = max(self.level_thresholds) + 1 if self.level_thresholds else 2
        current_max = max(self.level_thresholds.values()) if self.level_thresholds else max(self.z_min, 0.0)
        step = max((self.z_max - self.z_min) * 0.03, 0.001)
        self.level_thresholds[next_level] = float(np.clip(current_max + step, self.z_min, self.z_max))
        self.enforce_level_order()
        self.rebuild_level_controls()
        self.update_dxf_profile_plot()
        self.log(f"Nivel N{next_level} añadido.")

    def remove_highest_level(self):
        if not self.level_thresholds:
            return
        self.remove_level(max(self.level_thresholds))

    def remove_level(self, level):
        if len(self.level_thresholds) <= 1:
            self.log("Debe quedar al menos un nivel.")
            return
        removed = int(level)
        self.level_thresholds.pop(removed, None)
        self.rebuild_level_controls()
        self.update_dxf_profile_plot()
        self.log(f"Nivel N{removed} quitado.")

    def enforce_level_order(self):
        previous = None
        for level in self.sorted_level_ids():
            value = float(self.level_thresholds[level])
            if previous is not None and value <= previous:
                value = previous + 0.001
            value = float(np.clip(value, self.z_min, self.z_max))
            self.level_thresholds[level] = value
            previous = value
        self.level_n2 = self.level_thresholds.get(2, self.z_min)
        self.level_n3 = self.level_thresholds.get(3, self.level_n2 + 0.001)
        self.level_n4 = self.level_thresholds.get(4, self.level_n3 + 0.001)

    def sync_level_controls(self, source_level=None, source=None):
        self.block_updates = True
        try:
            for level, controls in self.level_controls.items():
                value = self.level_thresholds[level]
                self.set_spin_quiet(controls["spin"], value * 1000.0)
                self.set_slider_quiet(controls["slider"], value)
        finally:
            self.block_updates = False

    def update_image(self):
        img = self.colorize_heightmap(self.z, self.color_min, self.color_max)
        self.image_item.setImage(np.flipud(img), autoLevels=False)
        self.image_item.setRect(QtCore.QRectF(self.x_min, self.y_min, self.x_max - self.x_min, self.y_max - self.y_min))
        if self._view_mode == '3d':
            self._update_3d_clim()

    def colorize_heightmap(self, values, vmin, vmax):
        vmin = min(float(vmin), -0.001)
        vmax = max(float(vmax), 0.001)
        finite = np.isfinite(values)
        t = np.zeros(values.shape, dtype=np.float32)

        neg = finite & (values <= 0)
        pos = finite & (values > 0)
        if vmin < 0:
            t[neg] = 0.5 * np.clip((values[neg] - vmin) / (0.0 - vmin), 0.0, 1.0)
        if vmax > 0:
            t[pos] = 0.5 + 0.5 * np.clip(values[pos] / vmax, 0.0, 1.0)

        idx = np.clip(np.round(t * 255), 0, 255).astype(np.uint8)
        lut = self.height_lookup_table()
        rgb = lut[idx]
        rgba = np.empty((*values.shape, 4), dtype=np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = 255
        rgba[~finite] = np.array([255, 255, 255, 255], dtype=np.uint8)
        return rgba

    def height_lookup_table(self):
        x = np.linspace(0.0, 1.0, 256)
        lut = np.empty((256, 3), dtype=np.uint8)
        for channel in range(3):
            lut[:, channel] = np.interp(x, HEIGHT_LUT_POSITIONS, HEIGHT_LUT_COLORS[:, channel]).astype(np.uint8)
        return lut

    # ── 3D Keyence-style view ─────────────────────────────────────────────────

    def _make_keyence_cmap(self):
        try:
            from matplotlib.colors import LinearSegmentedColormap
            positions = HEIGHT_LUT_POSITIONS.tolist()
            colors_norm = [[r / 255.0, g / 255.0, b / 255.0] for r, g, b in HEIGHT_LUT_COLORS]
            return LinearSegmentedColormap.from_list(
                'keyence', list(zip(positions, colors_norm)), N=256)
        except ImportError:
            return 'rainbow'

    def _switch_view_mode(self, mode_text):
        if '3D' in mode_text and HAS_PYVISTA:
            self._view_mode = '3d'
            self.map_stack.setCurrentIndex(1)
            self._lbl_3d_controls.setVisible(True)
            self._update_3d_view()
        else:
            self._view_mode = '2d'
            self.map_stack.setCurrentIndex(0)
            self._lbl_3d_controls.setVisible(False)

    def _on_height_mag_changed(self):
        if self._view_mode == '3d':
            self._update_3d_view()

    def _update_3d_view(self):
        if not HAS_PYVISTA or self._view_mode != '3d':
            return

        nr, nc = self.z.shape
        x_arr = np.linspace(self.x_min, self.x_max, nc)
        y_arr = np.linspace(self.y_max, self.y_min, nr)  # y_max->y_min: fila 0 en arriba (y=0), igual que 2D
        xx, yy = np.meshgrid(x_arr, y_arr)

        mag = float(self._spin_height_mag.value())
        z_finite = np.isfinite(self.z)
        fallback = float(np.nanmedian(self.z[z_finite])) if z_finite.any() else 0.0
        zz_vis = np.where(z_finite, self.z * mag, fallback * mag)

        points = np.column_stack([xx.ravel(), yy.ravel(), zz_vis.ravel()])
        grid = pv.StructuredGrid()
        grid.points = points
        grid.dimensions = [nc, nr, 1]
        grid.point_data["height_mm"] = np.where(z_finite, self.z, np.nan).ravel()

        try:
            cmap = self._make_keyence_cmap()
        except Exception:
            cmap = 'rainbow'

        self._plotter.clear()
        self._plotter.set_background('black')
        self._plotter.enable_trackball_style()

        self._3d_actor = self._plotter.add_mesh(
            grid,
            scalars="height_mm",
            cmap=cmap,
            clim=[self.color_min, self.color_max],
            nan_color='dimgray',
            nan_opacity=0.5,
            show_scalar_bar=True,
            scalar_bar_args={
                'title': 'Height (mm)',
                'vertical': True,
                'position_x': 0.02,
                'position_y': 0.05,
                'width': 0.07,
                'height': 0.90,
                'label_font_size': 11,
                'title_font_size': 11,
                'color': 'white',
                'fmt': '%.4f',
                'n_labels': 5,
            },
            lighting=True,
        )

        try:
            self._plotter.show_bounds(
                grid=True,
                location='outer',
                ticks='both',
                xlabel='X (mm)',
                ylabel='Y (mm)',
                zlabel=f'Z×{mag:.0f} (mm)',
                color='gray',
                font_size=8,
            )
        except Exception:
            pass

        try:
            self._plotter.add_axes(color='white')
        except Exception:
            pass

        self._plotter.view_isometric()
        self._plotter.render()

    def _update_3d_clim(self):
        if not HAS_PYVISTA or self._view_mode != '3d':
            return
        if self._3d_actor is None:
            self._update_3d_view()
            return
        try:
            self._plotter.update_scalar_bar_range([self.color_min, self.color_max])
            self._plotter.render()
        except Exception:
            self._update_3d_view()

    # ─────────────────────────────────────────────────────────────────────────

    # ── Puntos de trabajo ────────────────────────────────────────────────────

    def next_work_point_id(self):
        return f"WP{len(self.work_points) + 1}"

    def start_add_work_point(self):
        if self.affine_x is None:
            self.log("Necesitas ≥3 puntos de calibración para usar puntos de trabajo.")
            return
        self.adding_work_point = True
        self.adding_point = False
        self.comsol_pick_mode = None
        self.log("Haz click en el heightmap → se calculan coords SAMLight automáticamente.")

    def add_work_point_at(self, x_mm, y_mm):
        sx, sy = self.profile_to_samlight(x_mm, y_mm)
        dlg = WorkPointDialog(self, self.next_work_point_id(), x_mm, y_mm, sx, sy)
        self.adding_work_point = False
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        wp = WorkPoint(
            point_id=dlg.point_id(),
            profile_x_mm=float(np.clip(x_mm, self.x_min, self.x_max)),
            profile_y_mm=float(np.clip(y_mm, self.y_min, self.y_max)),
            samlight_x_mm=dlg.samlight_x(),
            samlight_y_mm=dlg.samlight_y(),
        )
        self.work_points.append(wp)
        self.refresh_work_points()
        self.refresh_profile_combos()
        self.log(f"Punto trabajo {wp.point_id}: SAMLight ({wp.samlight_x_mm:.4f}, {wp.samlight_y_mm:.4f}) mm")
        self._autosave_calibration()

    def refresh_work_points(self):
        self.clear_work_point_items()
        half = CROSS_SIZE_MM / 2.0
        pen = pg.mkPen((255, 80, 200), width=1.6)
        for wp in self.work_points:
            h = pg.PlotDataItem([wp.profile_x_mm - half, wp.profile_x_mm + half],
                                [wp.profile_y_mm, wp.profile_y_mm], pen=pen)
            v = pg.PlotDataItem([wp.profile_x_mm, wp.profile_x_mm],
                                [wp.profile_y_mm - half, wp.profile_y_mm + half], pen=pen)
            lbl = pg.TextItem(wp.point_id, color=(255, 80, 200), anchor=(0, 1))
            lbl.setPos(wp.profile_x_mm, wp.profile_y_mm)
            self.plot.addItem(h)
            self.plot.addItem(v)
            self.plot.addItem(lbl)
            self.work_point_items.extend([h, v, lbl])
        self.work_table.setRowCount(len(self.work_points))
        for row, wp in enumerate(self.work_points):
            self.work_table.setItem(row, 0, QtWidgets.QTableWidgetItem(wp.point_id))
            self.work_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{wp.samlight_x_mm:.6f}"))
            self.work_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{wp.samlight_y_mm:.6f}"))

    def clear_work_point_items(self):
        for item in self.work_point_items:
            self.plot.removeItem(item)
        self.work_point_items = []

    def delete_selected_work_point(self):
        rows = self.work_table.selectionModel().selectedRows()
        if not rows:
            self.log("No hay punto de trabajo seleccionado.")
            return
        idx = rows[0].row()
        if 0 <= idx < len(self.work_points):
            removed = self.work_points.pop(idx)
            self.refresh_work_points()
            self.refresh_profile_combos()
            self.log(f"Punto trabajo {removed.point_id} borrado.")
            self._autosave_calibration()

    def clear_all_work_points(self):
        self.work_points.clear()
        self.refresh_work_points()
        self.refresh_profile_combos()
        self.log("Todos los puntos de trabajo borrados.")

    def _point_by_id_combined(self, point_id):
        for p in self.points:
            if p.point_id == point_id:
                return p
        for wp in self.work_points:
            if wp.point_id == point_id:
                return wp
        return None

    # ─────────────────────────────────────────────────────────────────────────

    def _autosave_calibration(self):
        if len(self.points) < 2:
            return
        output = CALIBRATION_DIR / f"calibracion_manual_{self.csv_file.stem}_autosave.csv"
        residuals = self.residuals()
        rows = [[
            "id", "x_profile_mm", "y_profile_mm", "x_samlight_mm", "y_samlight_mm",
            "use_for_affine", "residual_x_mm", "residual_y_mm", "error_mm",
        ]]
        for point in self.points:
            rx, ry, err = residuals.get(point.point_id, ("", "", ""))
            rows.append([
                point.point_id,
                f"{point.profile_x_mm:.6f}",
                f"{point.profile_y_mm:.6f}",
                f"{point.samlight_x_mm:.6f}",
                f"{point.samlight_y_mm:.6f}",
                "yes" if point.use_for_affine else "no",
                f"{rx:.6f}" if rx != "" else "",
                f"{ry:.6f}" if ry != "" else "",
                f"{err:.6f}" if err != "" else "",
            ])
        with output.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        self.log(f"[autosave] {output.name}")

    def log(self, text):
        self.status.appendPlainText(str(text))

    def next_point_id(self):
        return f"P{len(self.points) + 1}"

    def start_add_point(self):
        self.adding_point = True
        self.comsol_pick_mode = None
        self.log("Haz click en el heightmap para colocar el punto.")

    def on_map_clicked(self, x_mm, y_mm):
        if self.comsol_pick_mode == "start":
            self.set_comsol_start(x_mm, y_mm)
            self.comsol_pick_mode = None
            return
        if self.comsol_pick_mode == "end":
            self.set_comsol_end_from_direction(x_mm, y_mm)
            self.comsol_pick_mode = None
            return
        if self.adding_work_point:
            self.add_work_point_at(x_mm, y_mm)
            return
        if self.adding_point:
            self.add_point_at(x_mm, y_mm)
            return
        idx = self.nearest_point_index(x_mm, y_mm)
        if idx is not None:
            self.select_point(idx)

    def prepare_drag_at(self, x_mm, y_mm):
        endpoint = self.nearest_comsol_endpoint(x_mm, y_mm)
        if endpoint is not None:
            self.dragging_comsol_endpoint = endpoint
            self.dragging_index = None
            return True
        idx = self.nearest_point_index(x_mm, y_mm)
        if idx is None:
            return False
        self.dragging_index = idx
        self.dragging_comsol_endpoint = None
        self.select_point(idx)
        return True

    def on_drag_moved(self, x_mm, y_mm):
        if self.dragging_comsol_endpoint is not None:
            self.move_comsol_endpoint(self.dragging_comsol_endpoint, x_mm, y_mm)
            return
        if self.dragging_index is None:
            return
        point = self.points[self.dragging_index]
        point.profile_x_mm = float(np.clip(x_mm, self.x_min, self.x_max))
        point.profile_y_mm = float(np.clip(y_mm, self.y_min, self.y_max))
        self.refresh_calibration()
        self.refresh_points()
        self.update_profile_plot()

    def on_drag_finished(self, _x_mm, _y_mm):
        if self.dragging_comsol_endpoint is not None:
            self.log("Perfil COMSOL movido.")
            self.dragging_comsol_endpoint = None
            return
        if self.dragging_index is not None:
            self.log(f"Punto {self.points[self.dragging_index].point_id} movido.")
            self.dragging_index = None
            self._autosave_calibration()
            return
        self.dragging_index = None

    def nearest_point_index(self, x_mm, y_mm):
        if not self.points:
            return None
        x_range, y_range = self.viewbox.viewRange()
        threshold = max((x_range[1] - x_range[0]) / max(self.viewbox.width(), 1), (y_range[1] - y_range[0]) / max(self.viewbox.height(), 1)) * 16
        best_idx = None
        best_dist = None
        for idx, point in enumerate(self.points):
            dist = float(np.hypot(point.profile_x_mm - x_mm, point.profile_y_mm - y_mm))
            if best_dist is None or dist < best_dist:
                best_idx = idx
                best_dist = dist
        if best_dist is not None and best_dist <= threshold:
            return best_idx
        return None

    def start_pick_comsol_start(self):
        self.adding_point = False
        self.comsol_pick_mode = "start"
        self.log("COMSOL: haz click en el inicio del perfil. Ese punto sera x=0.")

    def start_pick_comsol_end(self):
        if self.comsol_start is None:
            self.log("COMSOL: primero marca el inicio.")
            return
        self.adding_point = False
        self.comsol_pick_mode = "end"
        self.log("COMSOL: haz click hacia el final/direccion. La longitud se fuerza al valor indicado.")

    def clip_profile_point(self, x_mm, y_mm):
        return (
            float(np.clip(x_mm, self.x_min, self.x_max)),
            float(np.clip(y_mm, self.y_min, self.y_max)),
        )

    def current_comsol_length(self):
        return float(self.spin_comsol_length.value())

    def set_comsol_start(self, x_mm, y_mm):
        start = self.clip_profile_point(x_mm, y_mm)
        direction = self.comsol_direction()
        if direction is None:
            direction = np.array([1.0, 0.0], dtype=float)
        self.comsol_start = start
        self.comsol_end = self.comsol_endpoint_from_direction(start, direction)
        self.refresh_comsol_items()
        self.compute_comsol_profile()
        self.log(f"COMSOL inicio: ({start[0]:.6f}, {start[1]:.6f}) mm.")

    def set_comsol_end_from_direction(self, x_mm, y_mm):
        if self.comsol_start is None:
            self.set_comsol_start(x_mm, y_mm)
            return
        raw_end = np.array(self.clip_profile_point(x_mm, y_mm), dtype=float)
        start = np.array(self.comsol_start, dtype=float)
        direction = raw_end - start
        norm = float(np.hypot(direction[0], direction[1]))
        if norm <= 1e-12:
            self.log("COMSOL: el fin coincide con el inicio; elige otro punto.")
            return
        self.comsol_end = self.comsol_endpoint_from_direction(self.comsol_start, direction / norm)
        self.refresh_comsol_items()
        self.compute_comsol_profile()
        self.log(f"COMSOL fin forzado a {self.current_comsol_length():.6f} mm.")

    def comsol_direction(self):
        if self.comsol_start is None or self.comsol_end is None:
            return None
        start = np.array(self.comsol_start, dtype=float)
        end = np.array(self.comsol_end, dtype=float)
        direction = end - start
        norm = float(np.hypot(direction[0], direction[1]))
        if norm <= 1e-12:
            return None
        return direction / norm

    def comsol_endpoint_from_direction(self, start, direction):
        start_vec = np.array(start, dtype=float)
        direction_vec = np.array(direction, dtype=float)
        norm = float(np.hypot(direction_vec[0], direction_vec[1]))
        if norm <= 1e-12:
            direction_vec = np.array([1.0, 0.0], dtype=float)
        else:
            direction_vec = direction_vec / norm
        end_vec = start_vec + direction_vec * self.current_comsol_length()
        return float(end_vec[0]), float(end_vec[1])

    def on_comsol_length_changed(self):
        if self.comsol_start is None:
            return
        direction = self.comsol_direction()
        if direction is None:
            direction = np.array([1.0, 0.0], dtype=float)
        self.comsol_end = self.comsol_endpoint_from_direction(self.comsol_start, direction)
        self.refresh_comsol_items()
        self.compute_comsol_profile(quiet=True)

    def on_comsol_samples_changed(self):
        if self.comsol_profile_data is not None:
            self.compute_comsol_profile(quiet=True)

    def nearest_comsol_endpoint(self, x_mm, y_mm):
        candidates = []
        if self.comsol_start is not None:
            candidates.append(("start", self.comsol_start))
        if self.comsol_end is not None:
            candidates.append(("end", self.comsol_end))
        if not candidates:
            return None
        x_range, y_range = self.viewbox.viewRange()
        threshold = max((x_range[1] - x_range[0]) / max(self.viewbox.width(), 1), (y_range[1] - y_range[0]) / max(self.viewbox.height(), 1)) * 18
        best_name = None
        best_dist = None
        for name, point in candidates:
            dist = float(np.hypot(point[0] - x_mm, point[1] - y_mm))
            if best_dist is None or dist < best_dist:
                best_name = name
                best_dist = dist
        if best_dist is not None and best_dist <= threshold:
            return best_name
        return None

    def move_comsol_endpoint(self, endpoint, x_mm, y_mm):
        if endpoint == "start":
            old_start = np.array(self.comsol_start if self.comsol_start is not None else (x_mm, y_mm), dtype=float)
            old_end = np.array(self.comsol_end if self.comsol_end is not None else (x_mm + self.current_comsol_length(), y_mm), dtype=float)
            direction = old_end - old_start
            norm = float(np.hypot(direction[0], direction[1]))
            if norm <= 1e-12:
                direction = np.array([1.0, 0.0], dtype=float)
            else:
                direction = direction / norm
            self.comsol_start = self.clip_profile_point(x_mm, y_mm)
            self.comsol_end = self.comsol_endpoint_from_direction(self.comsol_start, direction)
        elif endpoint == "end":
            if self.comsol_start is None:
                return
            raw_end = np.array(self.clip_profile_point(x_mm, y_mm), dtype=float)
            start = np.array(self.comsol_start, dtype=float)
            direction = raw_end - start
            norm = float(np.hypot(direction[0], direction[1]))
            if norm <= 1e-12:
                return
            self.comsol_end = self.comsol_endpoint_from_direction(self.comsol_start, direction / norm)
        self.refresh_comsol_items()
        self.compute_comsol_profile(quiet=True)

    def refresh_comsol_items(self):
        for item in self.comsol_items:
            self.plot.removeItem(item)
        self.comsol_items = []
        if self.comsol_start is None:
            return

        start = self.comsol_start
        end = self.comsol_end
        if end is not None:
            line = pg.PlotDataItem([start[0], end[0]], [start[1], end[1]], pen=pg.mkPen((255, 120, 0), width=2.0))
            self.plot.addItem(line)
            self.comsol_items.append(line)

        points = [{"pos": start, "data": "start", "brush": pg.mkBrush(255, 255, 255, 220), "pen": pg.mkPen((255, 120, 0), width=2), "size": 14}]
        if end is not None:
            points.append({"pos": end, "data": "end", "brush": pg.mkBrush(255, 120, 0, 220), "pen": pg.mkPen("k", width=1), "size": 14})
        scatter = pg.ScatterPlotItem(points, pxMode=True)
        self.plot.addItem(scatter)
        self.comsol_items.append(scatter)

        label_a = pg.TextItem("COMSOL x=0", color=(180, 70, 0), anchor=(0, 1))
        label_a.setPos(start[0], start[1])
        self.plot.addItem(label_a)
        self.comsol_items.append(label_a)
        if end is not None:
            label_b = pg.TextItem(f"x={self.current_comsol_length():.3f}", color=(180, 70, 0), anchor=(0, 0))
            label_b.setPos(end[0], end[1])
            self.plot.addItem(label_b)
            self.comsol_items.append(label_b)

    def add_point_at(self, x_mm, y_mm):
        point_id = clean_cell(self.edit_point_id.text()) or self.next_point_id()
        point = ControlPoint(
            point_id=point_id,
            profile_x_mm=float(np.clip(x_mm, self.x_min, self.x_max)),
            profile_y_mm=float(np.clip(y_mm, self.y_min, self.y_max)),
            samlight_x_mm=float(self.spin_laser_x.value()),
            samlight_y_mm=float(self.spin_laser_y.value()),
        )
        self.points.append(point)
        self.adding_point = False
        self.selected_index = len(self.points) - 1
        self.edit_point_id.setText(self.next_point_id())
        self.refresh_calibration()
        self.refresh_points()
        self.log(f"Punto {point.point_id} colocado.")
        self._autosave_calibration()

    def update_selected_point(self):
        if self.selected_index is None or not (0 <= self.selected_index < len(self.points)):
            self.log("No hay punto seleccionado.")
            return
        point = self.points[self.selected_index]
        point.point_id = clean_cell(self.edit_point_id.text()) or point.point_id
        point.samlight_x_mm = float(self.spin_laser_x.value())
        point.samlight_y_mm = float(self.spin_laser_y.value())
        self.refresh_calibration()
        self.refresh_points()
        self.log(f"Punto {point.point_id} actualizado.")
        self._autosave_calibration()

    def delete_selected_point(self):
        if self.selected_index is None or not (0 <= self.selected_index < len(self.points)):
            self.log("No hay punto seleccionado.")
            return
        removed = self.points.pop(self.selected_index)
        self.selected_index = None
        self.refresh_calibration()
        self.refresh_points()
        self.log(f"Punto {removed.point_id} borrado.")
        self._autosave_calibration()

    def select_point(self, idx):
        if idx is None or not (0 <= idx < len(self.points)):
            return
        self.selected_index = idx
        point = self.points[idx]
        self.edit_point_id.setText(point.point_id)
        self.set_spin_quiet(self.spin_laser_x, point.samlight_x_mm)
        self.set_spin_quiet(self.spin_laser_y, point.samlight_y_mm)
        self.table.selectRow(idx)
        self.refresh_points()

    def on_table_selection_changed(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        if 0 <= row < len(self.points):
            self.selected_index = row
            point = self.points[row]
            self.edit_point_id.setText(point.point_id)
            self.set_spin_quiet(self.spin_laser_x, point.samlight_x_mm)
            self.set_spin_quiet(self.spin_laser_y, point.samlight_y_mm)
            self.refresh_points()

    def refresh_points(self):
        self.clear_point_items()
        if self.points:
            points_data = [
                {
                    "pos": (p.profile_x_mm, p.profile_y_mm),
                    "data": i,
                    "brush": pg.mkBrush(255, 255, 0, 160) if i == self.selected_index else pg.mkBrush(0, 255, 255, 120),
                    "pen": pg.mkPen("k", width=1) if i == self.selected_index else pg.mkPen("c", width=1),
                    "size": 13 if i == self.selected_index else 10,
                }
                for i, p in enumerate(self.points)
            ]
            self.point_scatter = pg.ScatterPlotItem(points_data, pxMode=True)
            self.plot.addItem(self.point_scatter)

        half = CROSS_SIZE_MM / 2.0
        for idx, point in enumerate(self.points):
            label = pg.TextItem(point.point_id, color=(0, 120, 160), anchor=(0, 1))
            label.setPos(point.profile_x_mm, point.profile_y_mm)
            self.plot.addItem(label)
            self.point_labels.append(label)
            pen = pg.mkPen("y", width=1.4) if idx == self.selected_index else pg.mkPen("c", width=1.0)
            for (x0, y0), (x1, y1) in self.display_cross_segments(point, half):
                item = pg.PlotDataItem([x0, x1], [y0, y1], pen=pen)
                self.plot.addItem(item)
                self.cross_items.append(item)

        self.refresh_table()
        self.refresh_profile_combos()
        self.update_status()

    def clear_point_items(self):
        if self.point_scatter is not None:
            self.plot.removeItem(self.point_scatter)
            self.point_scatter = None
        for item in self.point_labels:
            self.plot.removeItem(item)
        self.point_labels = []
        for item in self.cross_items:
            self.plot.removeItem(item)
        self.cross_items = []

    def refresh_table(self):
        selected = self.selected_index
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.points))
        residuals = self.residuals()
        for row, point in enumerate(self.points):
            err_um = ""
            if point.point_id in residuals:
                err_um = f"{residuals[point.point_id][2] * 1000:.1f}"
            values = [
                point.point_id,
                f"{point.samlight_x_mm:.6f}",
                f"{point.samlight_y_mm:.6f}",
                f"{point.profile_x_mm:.6f}",
                f"{point.profile_y_mm:.6f}",
                err_um,
            ]
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                self.table.setItem(row, col, item)
        self.table.blockSignals(False)
        if selected is not None and 0 <= selected < len(self.points):
            self.table.selectRow(selected)

    def refresh_profile_combos(self):
        old_a = self.combo_p1.currentText()
        old_b = self.combo_p2.currentText()
        ids = [p.point_id for p in self.points] + [wp.point_id for wp in self.work_points]
        self.combo_p1.blockSignals(True)
        self.combo_p2.blockSignals(True)
        self.combo_p1.clear()
        self.combo_p2.clear()
        self.combo_p1.addItems(ids)
        self.combo_p2.addItems(ids)
        if old_a in ids:
            self.combo_p1.setCurrentText(old_a)
        if old_b in ids:
            self.combo_p2.setCurrentText(old_b)
        elif len(ids) >= 2:
            self.combo_p2.setCurrentIndex(1)
        self.combo_p1.blockSignals(False)
        self.combo_p2.blockSignals(False)

    def point_by_id(self, point_id):
        for point in self.points:
            if point.point_id == point_id:
                return point
        return None

    def refresh_calibration(self):
        active = [p for p in self.points if p.use_for_affine]
        self.affine_x = None
        self.affine_y = None
        self.inverse_affine = None
        if len(active) < 3:
            return

        source = np.array([[p.profile_x_mm, p.profile_y_mm, 1.0] for p in active], dtype=float)
        target_x = np.array([p.samlight_x_mm for p in active], dtype=float)
        target_y = np.array([p.samlight_y_mm for p in active], dtype=float)
        self.affine_x, *_ = np.linalg.lstsq(source, target_x, rcond=None)
        self.affine_y, *_ = np.linalg.lstsq(source, target_y, rcond=None)
        matrix = np.array([
            [self.affine_x[0], self.affine_x[1], self.affine_x[2]],
            [self.affine_y[0], self.affine_y[1], self.affine_y[2]],
            [0.0, 0.0, 1.0],
        ])
        try:
            self.inverse_affine = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            self.inverse_affine = None

    def profile_to_samlight(self, x_mm, y_mm):
        if self.affine_x is None or self.affine_y is None:
            raise RuntimeError("Faltan 3 puntos para transformacion afin.")
        sx = self.affine_x[0] * x_mm + self.affine_x[1] * y_mm + self.affine_x[2]
        sy = self.affine_y[0] * x_mm + self.affine_y[1] * y_mm + self.affine_y[2]
        return sx, sy

    def samlight_to_profile(self, sx, sy):
        if self.inverse_affine is None:
            return None
        result = self.inverse_affine @ np.array([sx, sy, 1.0], dtype=float)
        return float(result[0]), float(result[1])

    def residuals(self):
        if self.affine_x is None:
            return {}
        values = {}
        for point in self.points:
            sx, sy = self.profile_to_samlight(point.profile_x_mm, point.profile_y_mm)
            dx = sx - point.samlight_x_mm
            dy = sy - point.samlight_y_mm
            values[point.point_id] = (dx, dy, float(np.hypot(dx, dy)))
        return values

    def display_cross_segments(self, point, half):
        if self.inverse_affine is not None:
            endpoints = [
                ((point.samlight_x_mm - half, point.samlight_y_mm), (point.samlight_x_mm + half, point.samlight_y_mm)),
                ((point.samlight_x_mm, point.samlight_y_mm - half), (point.samlight_x_mm, point.samlight_y_mm + half)),
            ]
            converted = []
            for start, end in endpoints:
                p0 = self.samlight_to_profile(*start)
                p1 = self.samlight_to_profile(*end)
                if p0 is not None and p1 is not None:
                    converted.append((p0, p1))
            if converted:
                return converted
        return [
            ((point.profile_x_mm - half, point.profile_y_mm), (point.profile_x_mm + half, point.profile_y_mm)),
            ((point.profile_x_mm, point.profile_y_mm - half), (point.profile_x_mm, point.profile_y_mm + half)),
        ]

    def update_status(self):
        lines = [
            f"CSV: {self.csv_file.name}",
            f"Pixel: {self.pixel_size * 1000:.3f} um",
            f"Puntos: {len(self.points)}",
        ]
        if self.affine_x is None:
            lines.append("Calibracion: faltan 3 puntos")
        else:
            residuals = self.residuals()
            errors = [v[2] for v in residuals.values()]
            lines.append("Calibracion: afin activa")
            lines.append(f"Error medio: {np.mean(errors) * 1000:.1f} um")
            lines.append(f"Error max: {np.max(errors) * 1000:.1f} um")
        self.status.setPlainText("\n".join(lines))

    def load_calibration_points(self, path):
        print(f"Cargando puntos: {path}")
        with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                point_id = clean_cell(row.get("id", "")) or clean_cell(row.get("point_id", ""))
                if not point_id:
                    continue
                use_value = clean_cell(row.get("use_for_affine", "yes")).lower()
                self.points.append(ControlPoint(
                    point_id=point_id,
                    profile_x_mm=float(clean_cell(row["x_profile_mm"])),
                    profile_y_mm=float(clean_cell(row["y_profile_mm"])),
                    samlight_x_mm=float(clean_cell(row["x_samlight_mm"])),
                    samlight_y_mm=float(clean_cell(row["y_samlight_mm"])),
                    use_for_affine=use_value not in ("no", "false", "0"),
                ))

    def save_calibration_points(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = CALIBRATION_DIR / f"calibracion_manual_{self.csv_file.stem}_{timestamp}.csv"
        affine_output = CALIBRATION_DIR / f"calibracion_affine_{self.csv_file.stem}_{timestamp}.csv"
        residuals = self.residuals()
        rows = [[
            "id", "x_profile_mm", "y_profile_mm", "x_samlight_mm", "y_samlight_mm",
            "use_for_affine", "residual_x_mm", "residual_y_mm", "error_mm",
        ]]
        for point in self.points:
            rx, ry, err = residuals.get(point.point_id, ("", "", ""))
            rows.append([
                point.point_id,
                f"{point.profile_x_mm:.6f}",
                f"{point.profile_y_mm:.6f}",
                f"{point.samlight_x_mm:.6f}",
                f"{point.samlight_y_mm:.6f}",
                "yes" if point.use_for_affine else "no",
                f"{rx:.6f}" if rx != "" else "",
                f"{ry:.6f}" if ry != "" else "",
                f"{err:.6f}" if err != "" else "",
            ])
        with output.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)

        affine_rows = [[
            "id", "x_scanner_rel_mm", "y_scanner_rel_mm", "x_samlight_mm", "y_samlight_mm",
            "use_for_affine", "origin_x_px", "origin_y_px", "profile_x_sign", "profile_y_sign",
            "origin_samlight_x_mm", "origin_samlight_y_mm", "transform_mode",
        ]]
        for idx, point in enumerate(self.points):
            affine_rows.append([
                point.point_id,
                f"{point.profile_x_mm:.6f}",
                f"{point.profile_y_mm:.6f}",
                f"{point.samlight_x_mm:.6f}",
                f"{point.samlight_y_mm:.6f}",
                "yes" if point.use_for_affine else "no",
                "0" if idx == 0 else "",
                "0" if idx == 0 else "",
                "1" if idx == 0 else "",
                "-1" if idx == 0 else "",
                "",
                "",
                "affine" if idx == 0 else "",
            ])
        with affine_output.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(affine_rows)

        self.log(f"Calibracion guardada:\n{output}\n{affine_output}")

    def export_crosses_dxf(self):
        if not self.points:
            self.log("No hay puntos para exportar cruces.")
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = DXF_OUTPUT_DIR / f"Cruces_Calibracion_{self.csv_file.stem}_{timestamp}.dxf"
        cfg = {"layer": "CALIB_CRUCES_0p5mm", "color": "4"}
        lines = dxf_header([cfg])
        half = CROSS_SIZE_MM / 2.0
        for point in self.points:
            add_dxf_line(lines, cfg["layer"], cfg["color"], point.samlight_x_mm - half, point.samlight_y_mm, point.samlight_x_mm + half, point.samlight_y_mm)
            add_dxf_line(lines, cfg["layer"], cfg["color"], point.samlight_x_mm, point.samlight_y_mm - half, point.samlight_x_mm, point.samlight_y_mm + half)
        lines.extend(["0", "ENDSEC", "0", "EOF"])
        output.write_text("\n".join(lines) + "\n", encoding="ascii")
        self.log(f"DXF cruces guardado:\n{output}")

    def update_profile_from_controls(self):
        p1 = self._point_by_id_combined(self.combo_p1.currentText())
        p2 = self._point_by_id_combined(self.combo_p2.currentText())
        if p1 is None or p2 is None or p1.point_id == p2.point_id:
            self.log("Perfil: selecciona dos puntos distintos.")
            return
        self.compute_profile(p1, p2)
        self.update_profile_plot()

    def compute_profile(self, p1, p2):
        dx = p2.profile_x_mm - p1.profile_x_mm
        dy = p2.profile_y_mm - p1.profile_y_mm
        length = float(np.hypot(dx, dy))
        if length <= 0:
            self.dxf_profile_data = None
            self.profile_data = None
            return
        step = min(self.pixel_size, BEAM_DIAMETER_MM / 2.0)
        count = max(2, int(np.ceil(length / step)) + 1)
        t = np.linspace(0.0, 1.0, count)
        xs = p1.profile_x_mm + t * dx
        ys = p1.profile_y_mm + t * dy
        distance = t * length
        height = self.sample_height(xs, ys)
        if self.affine_x is not None:
            sx, sy = self.profile_to_samlight(xs, ys)
        else:
            sx = p1.samlight_x_mm + t * (p2.samlight_x_mm - p1.samlight_x_mm)
            sy = p1.samlight_y_mm + t * (p2.samlight_y_mm - p1.samlight_y_mm)
        self.dxf_profile_data = {
            "p1": p1.point_id,
            "p2": p2.point_id,
            "profile_x_mm": xs,
            "profile_y_mm": ys,
            "samlight_x_mm": sx,
            "samlight_y_mm": sy,
            "distance_mm": distance,
            "height_mm": height,
            "length_mm": length,
        }
        self.profile_data = self.dxf_profile_data
        self.log(f"Perfil {p1.point_id}->{p2.point_id}: {length:.3f} mm.")

    def compute_comsol_profile(self, quiet=False):
        if self.comsol_start is None or self.comsol_end is None:
            if not quiet:
                self.log("COMSOL: marca inicio y fin/direccion antes de ver el perfil.")
            return
        start = np.array(self.comsol_start, dtype=float)
        end = np.array(self.comsol_end, dtype=float)
        direction = end - start
        direction_length = float(np.hypot(direction[0], direction[1]))
        if direction_length <= 1e-12:
            if not quiet:
                self.log("COMSOL: perfil sin longitud.")
            return
        direction = direction / direction_length
        length = self.current_comsol_length()
        count = int(self.spin_comsol_samples.value())
        distance = np.linspace(0.0, length, count)
        xs = start[0] + direction[0] * distance
        ys = start[1] + direction[1] * distance
        height = self.sample_height(xs, ys)
        if self.affine_x is not None:
            sx, sy = self.profile_to_samlight(xs, ys)
        else:
            sx = np.full(xs.shape, np.nan, dtype=float)
            sy = np.full(ys.shape, np.nan, dtype=float)
        self.comsol_profile_data = {
            "p1": "COMSOL_x0",
            "p2": f"COMSOL_x{length:.3f}",
            "profile_x_mm": xs,
            "profile_y_mm": ys,
            "samlight_x_mm": sx,
            "samlight_y_mm": sy,
            "distance_mm": distance,
            "height_mm": height,
            "length_mm": length,
            "is_comsol": True,
        }
        self.profile_data = self.comsol_profile_data
        self.update_comsol_profile_plot()
        if not quiet:
            invalid = int(np.count_nonzero(~np.isfinite(height)))
            suffix = f" | {invalid} puntos sin altura" if invalid else ""
            self.log(f"COMSOL perfil calculado: x=0..{length:.6f} mm, {count} puntos{suffix}.")

    def sample_height(self, xs, ys):
        cols = xs / self.pixel_size
        rows = -ys / self.pixel_size
        result = np.full(xs.shape, np.nan, dtype=float)
        c0 = np.floor(cols).astype(int)
        r0 = np.floor(rows).astype(int)
        dc = cols - c0
        dr = rows - r0
        valid = (c0 >= 0) & (r0 >= 0) & (c0 + 1 < self.nx) & (r0 + 1 < self.ny)
        for i in np.where(valid)[0]:
            z00 = self.z[r0[i], c0[i]]
            z10 = self.z[r0[i], c0[i] + 1]
            z01 = self.z[r0[i] + 1, c0[i]]
            z11 = self.z[r0[i] + 1, c0[i] + 1]
            if not np.isfinite(np.array([z00, z10, z01, z11])).all():
                continue
            top = z00 * (1.0 - dc[i]) + z10 * dc[i]
            bottom = z01 * (1.0 - dc[i]) + z11 * dc[i]
            result[i] = top * (1.0 - dr[i]) + bottom * dr[i]
        return result

    def classify_profile_levels(self, profile_data=None):
        data = self.dxf_profile_data if profile_data is None else profile_data
        if data is None:
            return None
        h = data["height_mm"]
        levels = np.zeros(h.shape, dtype=int)
        valid = np.isfinite(h)
        ordered = [(level, self.level_thresholds[level]) for level in self.sorted_level_ids()]
        for idx, (level, threshold) in enumerate(ordered):
            if idx + 1 < len(ordered):
                next_threshold = ordered[idx + 1][1]
                mask = valid & (h >= threshold) & (h < next_threshold)
            else:
                mask = valid & (h >= threshold)
            levels[mask] = int(level)
        return levels

    def update_profile_plot(self):
        self.update_dxf_profile_plot()
        self.update_comsol_profile_plot()

    def update_dxf_profile_plot(self):
        self.update_single_profile_plot(
            self.dxf_profile_plot,
            self.dxf_profile_data,
            show_levels=True,
            empty_title="Perfil DXF no seleccionado",
        )

    def update_comsol_profile_plot(self):
        self.update_single_profile_plot(
            self.comsol_profile_plot,
            self.comsol_profile_data,
            show_levels=False,
            empty_title="Perfil COMSOL no seleccionado",
        )

    def update_single_profile_plot(self, plot_widget, profile_data, show_levels, empty_title):
        if plot_widget is None:
            return
        plot_widget.clear()
        if profile_data is None:
            plot_widget.setTitle(empty_title)
            return
        distance = profile_data["distance_mm"]
        height_um = profile_data["height_mm"] * 1000.0   # mm → µm para display
        plot_widget.plot(distance, height_um, pen=pg.mkPen("k", width=1.2))
        if show_levels:
            levels = self.classify_profile_levels(profile_data)
            for level in self.sorted_level_ids():
                mask = levels == int(level)
                if np.any(mask):
                    plot_widget.plot(
                        distance[mask],
                        height_um[mask],
                        pen=None,
                        symbol="o",
                        symbolSize=5,
                        symbolBrush=self.level_color(level),
                    )
                plot_widget.addLine(
                    y=self.level_thresholds[level] * 1000.0,   # mm → µm
                    pen=pg.mkPen(self.level_color(level), style=QtCore.Qt.PenStyle.DashLine),
                )
        plot_widget.setTitle(f"Perfil {profile_data['p1']} -> {profile_data['p2']} | {profile_data['length_mm']:.3f} mm")

    def profile_runs(self, levels):
        if levels is None or len(levels) == 0:
            return
        start = 0
        current = int(levels[0])
        for idx in range(1, len(levels)):
            value = int(levels[idx])
            if value == current:
                continue
            yield start, idx - 1, current
            start = idx
            current = value
        yield start, len(levels) - 1, current

    def make_profile_list_group(self):
        group = QtWidgets.QGroupBox("Lista de perfiles para exportar")
        layout = QtWidgets.QVBoxLayout(group)

        self.profile_list_table = QtWidgets.QTableWidget(0, 2)
        self.profile_list_table.setHorizontalHeaderLabels(["Perfil", ""])
        self.profile_list_table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.profile_list_table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.profile_list_table.setColumnWidth(1, 32)
        self.profile_list_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.profile_list_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.profile_list_table.setMaximumHeight(120)
        layout.addWidget(self.profile_list_table)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_clear_profile_list = QtWidgets.QPushButton("Limpiar lista")
        self.btn_export_all_dxf = QtWidgets.QPushButton("⬇ Exportar TODOS los perfiles (DXF)")
        self.btn_export_all_dxf.setStyleSheet(
            "font-weight:bold; background:#7b1fa2; color:white; padding:6px;")
        btn_row.addWidget(self.btn_clear_profile_list)
        btn_row.addWidget(self.btn_export_all_dxf, 1)
        layout.addLayout(btn_row)

        self.btn_clear_profile_list.clicked.connect(self.clear_profile_list)
        self.btn_export_all_dxf.clicked.connect(self.export_all_profiles_dxf)
        return group

    def add_profile_to_list(self):
        p1 = self._point_by_id_combined(self.combo_p1.currentText())
        p2 = self._point_by_id_combined(self.combo_p2.currentText())
        if p1 is None or p2 is None or p1.point_id == p2.point_id:
            self.log("Perfil: selecciona dos puntos distintos.")
            return
        pair = (p1.point_id, p2.point_id)
        if pair in self.profile_list:
            self.log(f"Perfil {pair[0]}→{pair[1]} ya está en la lista.")
            return
        self.profile_list.append(pair)
        self.refresh_profile_list_table()
        self.log(f"Perfil {pair[0]}→{pair[1]} añadido a la lista ({len(self.profile_list)} total).")

    def refresh_profile_list_table(self):
        self.profile_list_table.setRowCount(len(self.profile_list))
        for row, (p1_id, p2_id) in enumerate(self.profile_list):
            self.profile_list_table.setItem(
                row, 0, QtWidgets.QTableWidgetItem(f"{p1_id} → {p2_id}"))
            btn = QtWidgets.QPushButton("✕")
            btn.setFixedWidth(28)
            btn.clicked.connect(lambda _, r=row: self._remove_profile_at(r))
            self.profile_list_table.setCellWidget(row, 1, btn)

    def _remove_profile_at(self, idx):
        if 0 <= idx < len(self.profile_list):
            removed = self.profile_list.pop(idx)
            self.refresh_profile_list_table()
            self.log(f"Perfil {removed[0]}→{removed[1]} eliminado de la lista.")

    def clear_profile_list(self):
        self.profile_list.clear()
        self.refresh_profile_list_table()
        self.log("Lista de perfiles vaciada.")

    def export_all_profiles_dxf(self):
        if not self.profile_list:
            self.log("La lista de perfiles está vacía. Usa ➕ para añadir perfiles.")
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = DXF_OUTPUT_DIR / f"Perfiles_Samlight_{self.csv_file.stem}_{timestamp}.dxf"
        layer_configs = [self.dxf_config_for_level(level) for level in self.sorted_level_ids()]
        wp_cfg = {"layer": "TRABAJO_CRUCES_0p5mm", "color": "6"}
        all_configs = layer_configs + ([wp_cfg] if self.work_points else [])
        lines = dxf_header(all_configs)
        total_exported = {int(level): 0 for level in self.sorted_level_ids()}
        total_skipped = 0
        ok = 0
        for p1_id, p2_id in self.profile_list:
            p1 = self._point_by_id_combined(p1_id)
            p2 = self._point_by_id_combined(p2_id)
            if p1 is None or p2 is None:
                self.log(f"[AVISO] {p1_id}/{p2_id} no encontrados, omitido.")
                continue
            self.compute_profile(p1, p2)
            if self.dxf_profile_data is None:
                continue
            levels = self.classify_profile_levels(self.dxf_profile_data)
            sx = self.dxf_profile_data["samlight_x_mm"]
            sy = self.dxf_profile_data["samlight_y_mm"]
            distance = self.dxf_profile_data["distance_mm"]
            for start_idx, end_idx, level in self.profile_runs(levels):
                if level not in total_exported:
                    continue
                if end_idx <= start_idx or (distance[end_idx] - distance[start_idx]) < MIN_DXF_SEGMENT_MM:
                    total_skipped += 1
                    continue
                cfg = self.dxf_config_for_level(level)
                add_dxf_line(lines, cfg["layer"], cfg["color"],
                             sx[start_idx], sy[start_idx], sx[end_idx], sy[end_idx])
                total_exported[level] += 1
            ok += 1
        half = CROSS_SIZE_MM / 2.0
        for wp in self.work_points:
            add_dxf_line(lines, wp_cfg["layer"], wp_cfg["color"],
                         wp.samlight_x_mm - half, wp.samlight_y_mm,
                         wp.samlight_x_mm + half, wp.samlight_y_mm)
            add_dxf_line(lines, wp_cfg["layer"], wp_cfg["color"],
                         wp.samlight_x_mm, wp.samlight_y_mm - half,
                         wp.samlight_x_mm, wp.samlight_y_mm + half)
        lines.extend(["0", "ENDSEC", "0", "EOF"])
        output.write_text("\n".join(lines) + "\n", encoding="ascii")
        exported_text = " ".join(f"N{level}={count}" for level, count in total_exported.items())
        wp_text = f" | cruces trabajo={len(self.work_points)}" if self.work_points else ""
        self.log(
            f"DXF multi-perfil guardado ({ok}/{len(self.profile_list)} perfiles):\n"
            f"{output}\n{exported_text} | cortos={total_skipped}{wp_text}")

    def export_profile_dxf(self):
        if self.dxf_profile_data is None:
            self.log("No hay perfil seleccionado.")
            return
        levels = self.classify_profile_levels(self.dxf_profile_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        p1 = self.dxf_profile_data["p1"]
        p2 = self.dxf_profile_data["p2"]
        output = DXF_OUTPUT_DIR / f"Perfil_Samlight_{self.csv_file.stem}_{p1}_{p2}_{timestamp}.dxf"
        layer_configs = [self.dxf_config_for_level(level) for level in self.sorted_level_ids()]
        wp_cfg = {"layer": "TRABAJO_CRUCES_0p5mm", "color": "6"}
        all_configs = layer_configs + ([wp_cfg] if self.work_points else [])
        lines = dxf_header(all_configs)
        sx = self.dxf_profile_data["samlight_x_mm"]
        sy = self.dxf_profile_data["samlight_y_mm"]
        distance = self.dxf_profile_data["distance_mm"]
        exported = {int(level): 0 for level in self.sorted_level_ids()}
        skipped = 0
        for start_idx, end_idx, level in self.profile_runs(levels):
            if level not in exported:
                continue
            if end_idx <= start_idx or (distance[end_idx] - distance[start_idx]) < MIN_DXF_SEGMENT_MM:
                skipped += 1
                continue
            cfg = self.dxf_config_for_level(level)
            add_dxf_line(lines, cfg["layer"], cfg["color"], sx[start_idx], sy[start_idx], sx[end_idx], sy[end_idx])
            exported[level] += 1
        half = CROSS_SIZE_MM / 2.0
        for wp in self.work_points:
            add_dxf_line(lines, wp_cfg["layer"], wp_cfg["color"],
                         wp.samlight_x_mm - half, wp.samlight_y_mm,
                         wp.samlight_x_mm + half, wp.samlight_y_mm)
            add_dxf_line(lines, wp_cfg["layer"], wp_cfg["color"],
                         wp.samlight_x_mm, wp.samlight_y_mm - half,
                         wp.samlight_x_mm, wp.samlight_y_mm + half)

        lines.extend(["0", "ENDSEC", "0", "EOF"])
        output.write_text("\n".join(lines) + "\n", encoding="ascii")
        exported_text = " ".join(f"N{level}={count}" for level, count in exported.items())
        wp_text = f" | cruces trabajo={len(self.work_points)}" if self.work_points else ""
        self.log(f"DXF perfil guardado:\n{output}\n{exported_text} | cortos ignorados={skipped}{wp_text}")

    def export_profile_csv(self):
        if self.dxf_profile_data is None:
            self.log("No hay perfil seleccionado.")
            return
        levels = self.classify_profile_levels(self.dxf_profile_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        p1 = self.dxf_profile_data["p1"]
        p2 = self.dxf_profile_data["p2"]
        output = CSV_OUTPUT_DIR / f"Perfil_{self.csv_file.stem}_{p1}_{p2}_{timestamp}.csv"
        rows = [["distance_mm", "height_mm", "level", "x_profile_mm", "y_profile_mm", "x_samlight_mm", "y_samlight_mm"]]
        for idx in range(len(self.dxf_profile_data["distance_mm"])):
            rows.append([
                f"{self.dxf_profile_data['distance_mm'][idx]:.6f}",
                f"{self.dxf_profile_data['height_mm'][idx]:.6f}",
                int(levels[idx]),
                f"{self.dxf_profile_data['profile_x_mm'][idx]:.6f}",
                f"{self.dxf_profile_data['profile_y_mm'][idx]:.6f}",
                f"{self.dxf_profile_data['samlight_x_mm'][idx]:.6f}",
                f"{self.dxf_profile_data['samlight_y_mm'][idx]:.6f}",
            ])
        with output.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        self.log(f"CSV perfil guardado:\n{output}")

    def comsol_xy_for_export(self):
        if self.comsol_profile_data is None:
            self.compute_comsol_profile()
        if self.comsol_profile_data is None:
            return None, None, 0

        x = np.asarray(self.comsol_profile_data["distance_mm"], dtype=float)
        y = np.asarray(self.comsol_profile_data["height_mm"], dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(finite) < 2:
            self.log("COMSOL: no hay suficientes alturas validas en el segmento.")
            return None, None, 0

        filled = y.copy()
        invalid_count = int(np.count_nonzero(~np.isfinite(filled)))
        if invalid_count:
            filled[~np.isfinite(filled)] = np.interp(x[~np.isfinite(filled)], x[finite], y[finite])
        return x, filled, invalid_count

    def export_comsol_csv(self):
        x, y, filled = self.comsol_xy_for_export()
        if x is None:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = CSV_OUTPUT_DIR / f"Perfil_COMSOL_{self.csv_file.stem}_{timestamp}.csv"
        rows = [["x_mm", "y_mm"]]
        for x_val, y_val in zip(x, y):
            rows.append([f"{x_val:.9f}", f"{y_val:.9f}"])
        with output.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        note = f"\nHuecos interpolados: {filled}" if filled else ""
        self.log(f"COMSOL CSV guardado:\n{output}{note}")

    def export_comsol_txt(self):
        x, y, filled = self.comsol_xy_for_export()
        if x is None:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = TXT_OUTPUT_DIR / f"Perfil_COMSOL_{self.csv_file.stem}_{timestamp}.txt"
        lines = [f"{x_val:.9f}\t{y_val:.9f}" for x_val, y_val in zip(x, y)]
        output.write_text("\n".join(lines) + "\n", encoding="ascii")
        note = f"\nHuecos interpolados: {filled}" if filled else ""
        self.log(f"COMSOL TXT guardado:\n{output}{note}")


def main():
    csv_file = resolve_existing_file(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV_NAME, INPUT_DIR)
    calibration_file = resolve_existing_file(sys.argv[2], CALIBRATION_DIR) if len(sys.argv) > 2 else None
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)
    window = ManualCalibrationQt(csv_file, calibration_file)
    window.show()
    print("Interfaz rapida abierta.")
    print("Click: seleccionar/anadir punto. Drag sobre un punto: moverlo.")
    print("Rueda: zoom. Arrastre fuera de puntos: pan.")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
