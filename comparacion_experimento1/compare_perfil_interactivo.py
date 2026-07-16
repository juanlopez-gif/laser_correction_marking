"""
compare_perfil_interactivo.py  —  PySide6 + pyqtgraph
======================================================
Comparacion interactiva de perfiles PRE/POST sobre el mapa de diferencias.

Uso:
    python compare_perfil_interactivo.py <pre.csv> <post.csv>
    python compare_perfil_interactivo.py <pre.csv> <post.csv> --out resultados/

Flujo:
  1. Alinea PRE y POST (mismo pipeline que compare_heights.py)
  2. Pestaña "Puntos": define puntos por clic en mapa o entrada manual SAMLight
  3. Pestaña "Perfiles": selecciona P1/P2 por combo o clic directo en mapa
  4. [X] para borrar un perfil individual, "Borrar todos" para limpiar
  5. "Guardar todos PNG" exporta todas las tarjetas a la carpeta de salida

Coordenadas:
  Los ejes del mapa usan coordenadas de perfil (x_profile_mm / y_profile_mm).
  Si hay calibracion cargada, la barra de estado muestra tambien SAMLight mm.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

# ── Fix Qt platform plugin path (Windows) ────────────────────────────────────
def _fix_qt_plugin_path():
    import importlib.util
    spec = importlib.util.find_spec("PySide6")
    if not spec or not spec.origin:
        return
    plugin_dir = Path(spec.origin).parent / "plugins" / "platforms"
    if plugin_dir.exists():
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(plugin_dir))

_fix_qt_plugin_path()
# ─────────────────────────────────────────────────────────────────────────────

from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg

_pg_ver = tuple(int(x) for x in pg.__version__.split('.')[:2])
if _pg_ver < (0, 13):
    sys.exit(
        f"\n[ERROR] pyqtgraph {pg.__version__} en uso es demasiado antiguo "
        f"(necesita >= 0.13 para PySide6).\n"
        f"  Ruta detectada: {pg.__file__}\n\n"
        f"  Ejecuta con Python311 explicitamente:\n"
        f"  C:\\Users\\mss\\AppData\\Local\\Programs\\Python\\Python311\\python.exe "
        f"{sys.argv[0]} <pre.csv> <post.csv>\n"
    )

from pyqtgraph.exporters import ImageExporter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_heights import (
    load_vr6000_csv, ComparisonConfig,
    coarse_search, fine_search,
    search_inplane_rotation, _rotate_image, _overlap_region,
    compute_z_offset, compute_rotation, apply_rotation_correction,
    _smooth_profile, _safe_name,
)

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

COLORES = [
    '#ef5350', '#42a5f5', '#66bb6a', '#ab47bc',
    '#ffa726', '#26c6da', '#ff7043', '#8d6e63',
]
SMOOTH_PX = 5


# =============================================================================
# PUNTO DEFINIDO
# =============================================================================

@dataclass
class PuntoDefinido:
    punto_id:  str
    x_profile: float
    y_profile: float
    x_samlight: float
    y_samlight: float


# =============================================================================
# ALINEACION
# =============================================================================

def alinear(pre_csv, post_csv):
    cfg = ComparisonConfig()
    print("Cargando CSVs...", flush=True)
    H_pre,  px = load_vr6000_csv(pre_csv)
    H_post, _  = load_vr6000_csv(post_csv)
    cfg.pixel_size_mm = px
    print(f"  PRE  {H_pre.shape}  ({H_pre.shape[1]*px:.2f} x {H_pre.shape[0]*px:.2f} mm)")
    print(f"  POST {H_post.shape}  ({H_post.shape[1]*px:.2f} x {H_post.shape[0]*px:.2f} mm)")

    print("Alineando (coarse + fine POC)...", flush=True)
    coarse = coarse_search(H_pre, H_post, cfg)
    if coarse is None or coarse[3] < cfg.confidence_threshold:
        y_pos, x_pos, snr, conf = 0, 0, 0.0, 0.0
    else:
        y_pos, x_pos, snr, conf = coarse
        y_pos, x_pos, snr, conf = fine_search(H_pre, H_post, y_pos, x_pos, cfg)
    print(f"  Y={y_pos:+d}px  X={x_pos:+d}px  conf={conf:.3f}", flush=True)

    print("Buscando rotacion en plano...", flush=True)
    angle = search_inplane_rotation(H_pre, H_post, y_pos, x_pos, cfg)
    if abs(angle) > cfg.inplane_angle_step_deg * 0.5:
        print(f"  Rotacion: {angle:+.3f} deg", flush=True)
        H_post = _rotate_image(H_post, angle)
    else:
        print("  Rotacion negligible", flush=True)

    slices = _overlap_region(H_pre.shape, H_post.shape, y_pos, x_pos)
    if slices is None:
        raise RuntimeError("Sin solapamiento entre PRE y POST")
    pre_sr, pre_sc, post_sr, post_sc = slices
    H_pre_reg  = H_pre[pre_sr, pre_sc]
    H_post_reg = H_post[post_sr, post_sc]

    delta_z = compute_z_offset(H_pre_reg, H_post_reg)
    theta_x, theta_y, r2 = compute_rotation(H_pre_reg, H_post_reg, delta_z, px)
    theta_mag = np.sqrt(theta_x**2 + theta_y**2)
    H_post_f = H_post_reg.copy()
    if theta_mag > cfg.rotation_mag_threshold_deg and r2 > cfg.rotation_r2_threshold:
        H_post_f = apply_rotation_correction(H_post_reg, theta_x, theta_y, px)
        delta_z  = compute_z_offset(H_pre_reg, H_post_f)

    delta_raw = (H_post_f - delta_z - H_pre_reg) * 1000.0
    rms = float(np.nanstd(delta_raw))
    print(f"  dZ={delta_z*1000:+.3f} um  RMS delta={rms:.2f} um\nListo.\n", flush=True)

    return H_pre_reg, H_post_f, delta_z, delta_raw, px, pre_sr.start, pre_sc.start


# =============================================================================
# MUESTREO BILINEAL
# =============================================================================

def _samp(H, rows_f, cols_f):
    from scipy.ndimage import map_coordinates
    nr, nc   = H.shape
    nan_mask = np.isnan(H)
    filled   = np.where(nan_mask, 0.0, H)
    rr_c     = np.clip(rows_f, 0, nr - 1)
    cc_c     = np.clip(cols_f, 0, nc - 1)
    vals     = map_coordinates(filled, [rr_c, cc_c], order=1, mode='constant', cval=np.nan)
    nf       = map_coordinates(nan_mask.astype(np.float32), [rr_c, cc_c],
                               order=1, mode='constant', cval=1.0)
    oob = (rows_f < 0) | (rows_f > nr-1) | (cols_f < 0) | (cols_f > nc-1)
    # 0.5: NaN solo si la mayoría del peso bilineal cae en píxeles sin dato
    vals[(nf > 0.5) | oob] = np.nan
    return vals


# =============================================================================
# CALIBRACION: transformacion profile <-> SAMLight
# =============================================================================

class Calibracion:
    """
    Transformacion bidireccional profile <-> SAMLight.

    Se construye de dos formas:
      1. CSV de calibracion guardado por el interface
         (calibracion/calibracion_manual_{stem}_*.csv)
         -> transformacion afin completa (6 param, >= 3 puntos no colineales)

      2. CSV de perfil exportado por el interface
         (salidas/csv/Perfil_{stem}_*.csv)
         -> transformacion de semejanza (escala + rotacion + traslacion)
           derivada de los dos extremos del perfil
    """

    def __init__(self, a: complex = None, b: complex = None,
                 affine_x=None, affine_y=None, inverse_affine=None,
                 source=''):
        self._a  = a
        self._ai = 1.0 / a if a is not None else None
        self._b  = b
        self.affine_x = affine_x
        self.affine_y = affine_y
        self.inverse_affine = inverse_affine
        self.source = source

    def perfil_a_samlight(self, x_p, y_p):
        if self.affine_x is not None and self.affine_y is not None:
            xs = self.affine_x[0] * x_p + self.affine_x[1] * y_p + self.affine_x[2]
            ys = self.affine_y[0] * x_p + self.affine_y[1] * y_p + self.affine_y[2]
            return float(xs), float(ys)
        z = self._a * complex(x_p, y_p) + self._b
        return z.real, z.imag

    def samlight_a_perfil(self, x_s, y_s):
        if self.inverse_affine is not None:
            result = self.inverse_affine @ np.array([x_s, y_s, 1.0], dtype=float)
            return float(result[0]), float(result[1])
        z = self._ai * (complex(x_s, y_s) - self._b)
        return z.real, z.imag

    @classmethod
    def desde_calibracion_manual(cls, path):
        import csv as _csv
        rows = []
        with Path(path).open(newline='', encoding='utf-8', errors='replace') as f:
            for r in _csv.DictReader(f):
                rows.append({k.strip(): v.strip() for k, v in r.items()})
        pts = []
        for r in rows:
            try:
                use = r.get('use_for_affine', 'yes').lower() not in ('no','false','0')
                if not use:
                    continue
                xp = float(r['x_profile_mm']); yp = float(r['y_profile_mm'])
                xs = float(r['x_samlight_mm']); ys = float(r['y_samlight_mm'])
                pts.append((xp, yp, xs, ys))
            except (ValueError, KeyError):
                continue
        if len(pts) < 2:
            raise ValueError(f"Se necesitan >= 2 puntos, encontrados {len(pts)}")
        pts = np.array(pts, dtype=float)
        if len(pts) >= 3:
            source = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts))])
            target_x = pts[:, 2]
            target_y = pts[:, 3]
            affine_x, *_ = np.linalg.lstsq(source, target_x, rcond=None)
            affine_y, *_ = np.linalg.lstsq(source, target_y, rcond=None)
            matrix = np.array([
                [affine_x[0], affine_x[1], affine_x[2]],
                [affine_y[0], affine_y[1], affine_y[2]],
                [0.0, 0.0, 1.0],
            ])
            try:
                inverse_affine = np.linalg.inv(matrix)
            except np.linalg.LinAlgError as e:
                raise ValueError("La calibracion afin no es invertible") from e
            pred_x = source @ affine_x
            pred_y = source @ affine_y
            err = np.hypot(pred_x - target_x, pred_y - target_y)
            print(f"  Calibracion afin OK [{Path(path).name}]  "
                  f"puntos={len(pts)}  err_med={np.mean(err)*1000:.2f} um  "
                  f"err_max={np.max(err)*1000:.2f} um")
            return cls(affine_x=affine_x, affine_y=affine_y,
                       inverse_affine=inverse_affine, source=Path(path).name)
        i, j = 0, len(pts) - 1
        return cls._from_two_points(pts[i, :2], pts[i, 2:4], pts[j, :2], pts[j, 2:4],
                                    source=Path(path).name)

    @classmethod
    def desde_csv_perfil(cls, path):
        import csv as _csv
        rows = []
        with Path(path).open(newline='', encoding='utf-8', errors='replace') as f:
            for r in _csv.DictReader(f):
                rows.append(r)
        if len(rows) < 2:
            raise ValueError("CSV de perfil demasiado corto")
        def _pt(r):
            return (float(r['x_profile_mm']), float(r['y_profile_mm'])), \
                   (float(r['x_samlight_mm']), float(r['y_samlight_mm']))
        p1_prof, p1_sam = _pt(rows[0])
        p2_prof, p2_sam = _pt(rows[-1])
        return cls._from_two_points(p1_prof, p1_sam, p2_prof, p2_sam,
                                    source=Path(path).name)

    @classmethod
    def _from_two_points(cls, p1_prof, p1_sam, p2_prof, p2_sam, source=''):
        z1p = complex(*p1_prof); z2p = complex(*p2_prof)
        z1s = complex(*p1_sam);  z2s = complex(*p2_sam)
        if abs(z2p - z1p) < 1e-9:
            raise ValueError("Los dos puntos de calibracion son identicos")
        a = (z2s - z1s) / (z2p - z1p)
        b = z1s - a * z1p
        print(f"  Calibracion OK [{source}]  "
              f"escala={abs(a):.4f}  rotacion={np.degrees(np.angle(a)):.3f}  "
              f"trasl=({b.real:.3f}, {b.imag:.3f}) mm")
        return cls(a=a, b=b, source=source)


def buscar_calibracion_auto(pre_stem, base_dir):
    base = Path(base_dir)

    patron1 = sorted((base / 'calibracion').glob(f'calibracion_manual_{pre_stem}_*.csv'),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    if patron1:
        try:
            return Calibracion.desde_calibracion_manual(patron1[0])
        except Exception as e:
            print(f"  [AVISO] {patron1[0].name}: {e}", file=sys.stderr)

    patron2 = sorted((base / 'salidas' / 'csv').glob(f'Perfil_{pre_stem}_*.csv'),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    for p in patron2:
        try:
            import csv as _csv
            with p.open(newline='', encoding='utf-8') as f:
                hdr = next(_csv.reader(f), [])
            if 'x_samlight_mm' in hdr and 'y_samlight_mm' in hdr:
                return Calibracion.desde_csv_perfil(p)
        except Exception as e:
            print(f"  [AVISO] {p.name}: {e}", file=sys.stderr)

    return None


# =============================================================================
# VIEWBOX CLICKABLE
# =============================================================================

class PerfilViewBox(pg.ViewBox):
    clicked = QtCore.Signal(float, float)
    pointDragMoved = QtCore.Signal(float, float)
    pointDragFinished = QtCore.Signal(float, float)

    def __init__(self):
        super().__init__()
        self.nearest_callback = None
        self.dragging_point = False

    def _to_view(self, scene_pos):
        """Convierte posicion de escena a coordenadas de vista (mm)."""
        pos = self.mapSceneToView(scene_pos)
        return float(pos.x()), float(pos.y())

    def mouseClickEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            sp = event.scenePos()
            c = self._to_view(sp)
            if c is not None:
                self.clicked.emit(c[0], c[1])
            event.accept()
            return
        super().mouseClickEvent(event)

    def mouseDragEvent(self, event, axis=None):
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            super().mouseDragEvent(event, axis=axis)
            return

        c = self._to_view(event.scenePos())

        if event.isStart():
            can_drag = False
            if c is not None and self.nearest_callback is not None:
                can_drag = self.nearest_callback(c[0], c[1])
            self.dragging_point = bool(can_drag)
            if self.dragging_point:
                self.pointDragMoved.emit(c[0], c[1])
                event.accept()
                return

            c0 = self._to_view(event.buttonDownScenePos())
            if c0 is not None:
                self.clicked.emit(c0[0], c0[1])
            event.accept()
            return

        if self.dragging_point:
            if c is not None:
                self.pointDragMoved.emit(c[0], c[1])
                if event.isFinish():
                    self.pointDragFinished.emit(c[0], c[1])
            if event.isFinish():
                self.dragging_point = False
            event.accept()
            return

        event.accept()


# =============================================================================
# WIDGET DE PLOT DE UN PERFIL
# =============================================================================

def _cmap_vr6000():
    stops = np.array([0.0, 0.20, 0.38, 0.56, 0.73, 1.0])
    colors = np.array([
        [0,   0,   139, 255], [0,   0,   255, 255],
        [0,   255, 255, 255], [255, 255,   0, 255],
        [255, 165,   0, 255], [255,   0,   0, 255],
    ], dtype=np.uint8)
    return pg.ColorMap(stops, colors)


class PerfilPlotWidget(pg.GraphicsLayoutWidget):
    def __init__(self, d, pre_um, post_um, delta_um, color, label, parent=None):
        super().__init__(parent=parent)
        self.setBackground('#0d1117')
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.setFixedHeight(260)

        pen_pre   = pg.mkPen('#4fc3f7', width=1.5)
        pen_post  = pg.mkPen(color,     width=1.5,
                             style=QtCore.Qt.PenStyle.DashLine)
        pen_delta = pg.mkPen('#66bb6a', width=1.3)

        p1 = self.addPlot(row=0, col=0, title=label)
        p1.plot(d, pre_um,  pen=pen_pre,  name='PRE')
        p1.plot(d, post_um, pen=pen_post, name='POST')
        p1.addLegend(offset=(-8, 8), labelTextSize='8pt',
                     brush=pg.mkBrush(0, 0, 0, 160))
        p1.setLabel('left', 'Altura (µm)', color='#9e9e9e', size='8pt')
        p1.showGrid(x=True, y=True, alpha=0.18)
        p1.getAxis('bottom').setStyle(showValues=False)
        p1.getAxis('bottom').hide()

        p2 = self.addPlot(row=1, col=0)
        p2.setXLink(p1)

        dv = delta_um[~np.isnan(delta_um)]
        y_max = float(np.nanmax(np.abs(dv))) * 1.25 if len(dv) else 10.0

        p2.plot(d, np.where(delta_um < 0, delta_um, 0.0),
                pen=None, fillLevel=0, brush=pg.mkBrush('#4575b4aa'))
        p2.plot(d, np.where(delta_um > 0, delta_um, 0.0),
                pen=None, fillLevel=0, brush=pg.mkBrush('#d73027aa'))
        p2.plot(d, delta_um, pen=pen_delta)
        p2.plot([d[0], d[-1]], [0.0, 0.0],
                pen=pg.mkPen('#ffffff', width=0.7, alpha=0.5))

        p2.setYRange(-y_max, y_max, padding=0)
        p2.setLabel('left',   'Delta (µm)', color='#9e9e9e', size='8pt')
        p2.setLabel('bottom', 'Dist (mm)',  color='#9e9e9e', size='8pt')
        p2.showGrid(x=True, y=True, alpha=0.18)

        if len(dv):
            stats_txt = (f"mean {dv.mean():+.1f}  "
                         f"std {dv.std():.1f}  "
                         f"min {dv.min():+.1f}  "
                         f"max {dv.max():+.1f}  µm")
            t = pg.TextItem(stats_txt, color='#9e9e9e', anchor=(1, 0))
            t.setFont(QtGui.QFont('Courier', 7))
            p2.addItem(t)
            t.setPos(d[-1], y_max * 0.9)

        self.ci.layout.setRowStretchFactor(0, 2)
        self.ci.layout.setRowStretchFactor(1, 1)
        self.ci.layout.setSpacing(2)


# =============================================================================
# TARJETA DE PERFIL
# =============================================================================

class TarjetaPerfil(QtWidgets.QFrame):
    eliminado = QtCore.Signal(object)

    def __init__(self, idx, p1, p2, color,
                 d, pre_um, post_um, delta_um,
                 map_items, label_override=None, parent=None):
        super().__init__(parent=parent)
        self.idx       = idx
        self.p1        = p1
        self.p2        = p2
        self.color     = color
        self.map_items = map_items
        self._d        = d
        self._pre_um   = pre_um
        self._post_um  = post_um
        self._delta_um = delta_um

        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            f"TarjetaPerfil {{ border: 2px solid {color}; border-radius: 5px;"
            f"  margin: 2px; background: #111827; }}"
        )

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(3)

        hdr = QtWidgets.QWidget()
        hl  = QtWidgets.QHBoxLayout(hdr)
        hl.setContentsMargins(0, 0, 0, 0)

        n_lbl = QtWidgets.QLabel(f"<b>Perfil {idx + 1}</b>")
        n_lbl.setStyleSheet(f"color: {color}; font-size: 10pt; border: none;")

        L = np.hypot(p2[0]-p1[0], p2[1]-p1[1])
        if label_override:
            coord_txt = label_override
        else:
            coord_txt = (f"P1({p1[0]:.3f}, {p1[1]:.3f})  →  "
                         f"P2({p2[0]:.3f}, {p2[1]:.3f})  "
                         f"L = {L:.3f} mm")
        info = QtWidgets.QLabel(coord_txt)
        info.setStyleSheet("color: #9e9e9e; font-size: 8pt; border: none;")

        btn_del = QtWidgets.QPushButton("✕")
        btn_del.setFixedSize(22, 22)
        btn_del.setStyleSheet(
            "QPushButton { color: #ef9a9a; border: 1px solid #555; "
            "  border-radius: 3px; font-weight: bold; }"
            "QPushButton:hover { background: #3a1a1a; }"
        )
        btn_del.clicked.connect(lambda: self.eliminado.emit(self))

        hl.addWidget(n_lbl)
        hl.addWidget(info, 1)
        hl.addWidget(btn_del)
        lay.addWidget(hdr)

        label = (f"Perfil {idx+1}  —  {pre_um.size} pts  |  "
                 f"media delta = {np.nanmean(delta_um):+.1f} µm")
        self.plot_w = PerfilPlotWidget(d, pre_um, post_um, delta_um, color, label)
        lay.addWidget(self.plot_w)

    def exportar(self, out_dir, pre_name, post_name, ts):
        nombre = (f"perfil_{self.idx+1}_"
                  f"{_safe_name(pre_name)}_vs_{_safe_name(post_name)}_{ts}.png")
        out = Path(out_dir) / nombre
        try:
            exp = ImageExporter(self.plot_w.scene())
            exp.parameters()['width'] = 1400
            exp.export(str(out))
            print(f"  Guardado: {nombre}")
        except Exception as e:
            print(f"  Error al exportar perfil {self.idx+1}: {e}")


# =============================================================================
# DIALOGO: NUEVO PUNTO
# =============================================================================

class DialogNuevoPunto(QtWidgets.QDialog):
    """
    Confirma o edita un punto antes de añadirlo a la lista.

    manual=False (clic en mapa): profile coords fijos (labels), SAMLight editable.
    manual=True  (entrada manual): SAMLight editable, profile se computa en vivo.
    """

    def __init__(self, parent, default_id, x_profile, y_profile,
                 x_samlight, y_samlight, cal=None, manual=False):
        super().__init__(parent)
        self._cal      = cal
        self._manual   = manual
        self._x_prof   = x_profile
        self._y_prof   = y_profile

        titulo = ("Nuevo punto  —  entrada manual" if manual
                  else "Nuevo punto  —  confirmacion")
        self.setWindowTitle(titulo)
        self.setMinimumWidth(400)
        self.setModal(True)
        self.setStyleSheet(
            "QDialog{background:#0d1117;color:#e0e0e0;}"
            "QLabel{border:none;} QLineEdit{background:#111827;color:#e0e0e0;"
            "border:1px solid #374151;border-radius:3px;padding:3px;}"
        )

        lay = QtWidgets.QFormLayout(self)
        lay.setSpacing(8)
        lay.setContentsMargins(16, 16, 16, 16)

        self._edit_id = QtWidgets.QLineEdit(default_id)
        lay.addRow("ID", self._edit_id)

        def spin(val):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(-99999, 99999); s.setDecimals(4); s.setSingleStep(0.1)
            s.setValue(val)
            s.setStyleSheet(
                "QDoubleSpinBox{background:#111827;color:#e0e0e0;"
                "border:1px solid #374151;border-radius:3px;padding:3px;}"
            )
            return s

        if manual and cal is not None:
            # SAMLight editable → perfil se calcula en vivo
            self._spin_sx = spin(x_samlight)
            self._spin_sy = spin(y_samlight)
            self._lbl_xp  = QtWidgets.QLabel(f"{x_profile:.4f}")
            self._lbl_yp  = QtWidgets.QLabel(f"{y_profile:.4f}")
            self._lbl_xp.setStyleSheet("color:#9e9e9e;")
            self._lbl_yp.setStyleSheet("color:#9e9e9e;")
            lay.addRow("SAMLight X (mm)", self._spin_sx)
            lay.addRow("SAMLight Y (mm)", self._spin_sy)
            lay.addRow("→ Perfil X (mm)", self._lbl_xp)
            lay.addRow("→ Perfil Y (mm)", self._lbl_yp)
            self._spin_sx.valueChanged.connect(self._update_profile_labels)
            self._spin_sy.valueChanged.connect(self._update_profile_labels)
            self._update_profile_labels()
        elif manual and cal is None:
            # Sin calibracion en modo manual: editar perfil directamente
            self._spin_sx = spin(x_profile)
            self._spin_sy = spin(y_profile)
            self._lbl_xp  = None
            self._lbl_yp  = None
            lay.addRow("Perfil X (mm)", self._spin_sx)
            lay.addRow("Perfil Y (mm)", self._spin_sy)
            info = QtWidgets.QLabel("(sin calibracion — se usan coords de perfil)")
            info.setStyleSheet("color:#9e9e9e;font-size:8pt;")
            lay.addRow("", info)
        else:
            # Clic en mapa: perfil fijo, SAMLight editable si hay cal
            xp_lbl = QtWidgets.QLabel(f"{x_profile:.4f}")
            yp_lbl = QtWidgets.QLabel(f"{y_profile:.4f}")
            xp_lbl.setStyleSheet("color:#9e9e9e;")
            yp_lbl.setStyleSheet("color:#9e9e9e;")
            lay.addRow("Perfil X (mm)", xp_lbl)
            lay.addRow("Perfil Y (mm)", yp_lbl)
            if cal is not None:
                self._spin_sx = spin(x_samlight)
                self._spin_sy = spin(y_samlight)
                lay.addRow("SAMLight X (mm)", self._spin_sx)
                lay.addRow("SAMLight Y (mm)", self._spin_sy)
            else:
                self._spin_sx = None
                self._spin_sy = None
                info = QtWidgets.QLabel(f"SAMLight: {x_samlight:.4f} , {y_samlight:.4f}  (sin calibracion)")
                info.setStyleSheet("color:#9e9e9e;font-size:8pt;")
                lay.addRow("", info)
            self._lbl_xp = None
            self._lbl_yp = None

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        btns.setStyleSheet(
            "QPushButton{background:#1f2937;color:#e0e0e0;"
            "border:1px solid #374151;border-radius:4px;padding:4px 14px;}"
            "QPushButton:hover{background:#374151;}"
        )
        lay.addRow(btns)

    def _update_profile_labels(self):
        if self._cal and self._lbl_xp and self._lbl_yp:
            xp, yp = self._cal.samlight_a_perfil(
                self._spin_sx.value(), self._spin_sy.value())
            self._x_prof = xp
            self._y_prof = yp
            self._lbl_xp.setText(f"{xp:.4f}")
            self._lbl_yp.setText(f"{yp:.4f}")

    def punto_id(self):
        return self._edit_id.text().strip() or "PT"

    def perfil_xy(self):
        """Devuelve siempre coordenadas de perfil."""
        if self._manual and self._cal:
            self._update_profile_labels()
            return self._x_prof, self._y_prof
        if self._manual and self._cal is None and self._spin_sx:
            return float(self._spin_sx.value()), float(self._spin_sy.value())
        return self._x_prof, self._y_prof

    def samlight_xy(self):
        """Devuelve coordenadas SAMLight."""
        if self._spin_sx and self._spin_sy:
            return float(self._spin_sx.value()), float(self._spin_sy.value())
        if self._cal:
            return self._cal.perfil_a_samlight(self._x_prof, self._y_prof)
        return self._x_prof, self._y_prof


# =============================================================================
# VENTANA PRINCIPAL
# =============================================================================

_TAB_PUNTOS   = 0
_TAB_PERFILES = 1

_SS_BTN = ("QPushButton{background:#1f2937;color:#e0e0e0;"
           "border:1px solid #374151;border-radius:4px;padding:3px 10px;}"
           "QPushButton:hover{background:#374151;}"
           "QPushButton:checked{background:#1d4ed8;border-color:#3b82f6;}")

_SS_COMBO = ("QComboBox{background:#111827;color:#e0e0e0;"
             "border:1px solid #374151;border-radius:3px;padding:2px 6px;}"
             "QComboBox QAbstractItemView{background:#111827;color:#e0e0e0;"
             "selection-background-color:#1d4ed8;}")

_SS_TABLE = ("QTableWidget{background:#111827;color:#e0e0e0;"
             "gridline-color:#374151;border:1px solid #374151;}"
             "QHeaderView::section{background:#1f2937;color:#9e9e9e;"
             "border:none;padding:3px;}"
             "QTableWidget::item:selected{background:#1d4ed8;}")


class VentanaPrincipal(QtWidgets.QMainWindow):
    def __init__(self, Hp, Hq, dz, delta_raw, px, r0, c0,
                 pre_name, post_name, out_dir, cal=None):
        super().__init__()
        self.Hp        = Hp
        self.Hq        = Hq
        self.dz        = dz
        self.delta_raw = delta_raw
        self.px        = px
        self.r0        = r0
        self.c0        = c0
        self.pre_name  = pre_name
        self.post_name = post_name
        self.out_dir   = out_dir
        self.cal       = cal

        nr, nc = Hp.shape
        self.x_off = c0 * px
        self.y_bot = -(r0 + nr) * px
        self.W     = nc * px
        self.H     = nr * px

        # Puntos definidos
        self._puntos_def        = []       # list[PuntoDefinido]
        self._punto_map_items   = {}       # id -> [scatter, label]
        self._modo_agregar_punto = False
        self._drag_punto_idx    = None

        # Perfiles
        self._perfiles   = []
        self._p1_pend    = None
        self._color_idx  = 0

        self.setWindowTitle(
            f"Perfiles interactivos  —  {pre_name}  vs  {post_name}")
        self.resize(1650, 960)
        self._aplicar_tema()
        self._build_ui()
        self._actualizar_mapa('delta')

    # ── Tema ──────────────────────────────────────────────────────────────────
    def _aplicar_tema(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #0d1117; color: #e0e0e0; }
            QLabel   { border: none; }
            QPushButton {
                background: #1f2937; color: #e0e0e0;
                border: 1px solid #374151; border-radius: 4px;
                padding: 3px 10px;
            }
            QPushButton:hover   { background: #374151; }
            QPushButton:checked { background: #1d4ed8; border-color: #3b82f6; }
            QTabWidget::pane    { border: 1px solid #374151; }
            QTabBar::tab {
                background: #1f2937; color: #9e9e9e;
                padding: 6px 16px; border-radius: 0px;
            }
            QTabBar::tab:selected { background: #111827; color: #e0e0e0;
                border-bottom: 2px solid #4fc3f7; }
            QScrollArea { border: none; background: #0d1117; }
            QScrollBar:vertical {
                background: #1f2937; width: 10px; border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #374151; border-radius: 5px; min-height: 30px;
            }
        """)

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 4)
        root.setSpacing(4)

        root.addWidget(self._build_toolbar())

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_mapa())
        splitter.addWidget(self._build_right_tabs())
        splitter.setSizes([900, 750])
        root.addWidget(splitter, 1)

        self.statusBar().setStyleSheet(
            "QStatusBar { background: #111827; color: #9e9e9e; font-size: 8pt; }")
        self.statusBar().showMessage(
            "Pestaña Puntos: define puntos por clic o manualmente. "
            "Pestaña Perfiles: selecciona P1/P2 o haz clic en el mapa.")

    # ── Toolbar ───────────────────────────────────────────────────────────────
    def _build_toolbar(self):
        tb = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(tb)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        lay.addWidget(QtWidgets.QLabel(
            f"<b style='color:#4fc3f7'>PRE</b> {self.pre_name}   "
            f"<b style='color:#ef9a9a'>POST</b> {self.post_name}"))

        lay.addSpacing(16)
        lay.addWidget(QtWidgets.QLabel("Mapa:"))

        self.btn_pre   = QtWidgets.QPushButton("PRE")
        self.btn_post  = QtWidgets.QPushButton("POST")
        self.btn_delta = QtWidgets.QPushButton("DELTA")
        for b, mode in [(self.btn_pre, 'pre'), (self.btn_post, 'post'),
                        (self.btn_delta, 'delta')]:
            b.setCheckable(True)
            b.setFixedHeight(26)
            b.clicked.connect(lambda _, m=mode: self._actualizar_mapa(m))
        self.btn_delta.setChecked(True)

        bg = QtWidgets.QButtonGroup(self)
        bg.setExclusive(True)
        for b in [self.btn_pre, self.btn_post, self.btn_delta]:
            bg.addButton(b)
            lay.addWidget(b)

        lay.addStretch()

        self.lbl_estado = QtWidgets.QLabel("")
        self.lbl_estado.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight |
                                     QtCore.Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(self.lbl_estado)
        return tb

    # ── Mapa ──────────────────────────────────────────────────────────────────
    def _build_mapa(self):
        self.vb = PerfilViewBox()

        self.plot_mapa = pg.PlotWidget(viewBox=self.vb)
        self.plot_mapa.setAspectLocked(True)   # igual que en interfaz_calibracion_manual_qt
        self.plot_mapa.setBackground('#0d1117')
        self.plot_mapa.showGrid(x=True, y=True, alpha=0.12)
        self.plot_mapa.setLabel('bottom', 'X (mm)', color='#9e9e9e', size='9pt')
        self.plot_mapa.setLabel('left',   'Y (mm)', color='#9e9e9e', size='9pt')

        self.img_item = pg.ImageItem(axisOrder='row-major')
        self.plot_mapa.addItem(self.img_item)
        self._aplicar_geometria_mapa()

        self.cbar = pg.ColorBarItem(
            colorMap=pg.colormap.get('RdBu_r', source='matplotlib'),
            label='Delta altura (µm)', interactive=True, orientation='v',
        )
        self.cbar.setImageItem(self.img_item,
                               insert_in=self.plot_mapa.getPlotItem())
        self._aplicar_geometria_mapa()

        # Marcador P1 pendiente
        self._p1_scatter = pg.ScatterPlotItem(
            size=13, pen=pg.mkPen('w', width=1.5), brush=pg.mkBrush('#00E676'))
        self._p1_scatter.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.plot_mapa.addItem(self._p1_scatter)

        # Label cursor
        self._cur_lbl = pg.TextItem('', color='#ffffff', anchor=(0, 0))
        self._cur_lbl.setFont(QtGui.QFont('Courier', 8))
        self._cur_lbl.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.plot_mapa.addItem(self._cur_lbl)

        self._proxy = pg.SignalProxy(
            self.plot_mapa.scene().sigMouseMoved,
            rateLimit=60, slot=self._on_mouse_move)
        self.vb.nearest_callback = self._preparar_arrastre_punto
        self.vb.clicked.connect(self._on_click_mapa)
        self.vb.pointDragMoved.connect(self._mover_punto_arrastrado)
        self.vb.pointDragFinished.connect(self._finalizar_arrastre_punto)

        return self.plot_mapa

    def _aplicar_geometria_mapa(self):
        rect = QtCore.QRectF(self.x_off, self.y_bot, self.W, self.H)
        self.img_item.setRect(rect)
        self.plot_mapa.setXRange(self.x_off, self.x_off + self.W, padding=0.02)
        self.plot_mapa.setYRange(self.y_bot, self.y_bot + self.H, padding=0.02)

    # ── Panel derecho: tabs ───────────────────────────────────────────────────
    def _build_right_tabs(self):
        self._right_tabs = QtWidgets.QTabWidget()
        self._right_tabs.addTab(self._build_tab_puntos(),   "Puntos")
        self._right_tabs.addTab(self._build_tab_perfiles(), "Perfiles")
        self._right_tabs.currentChanged.connect(self._on_tab_changed)
        return self._right_tabs

    # ── Tab Puntos ────────────────────────────────────────────────────────────
    def _build_tab_puntos(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        # Botones
        btn_row = QtWidgets.QHBoxLayout()

        self.btn_add_clic = QtWidgets.QPushButton("+ Clic en mapa")
        self.btn_add_clic.setCheckable(True)
        self.btn_add_clic.setStyleSheet(_SS_BTN)
        self.btn_add_clic.setToolTip(
            "Activa el modo 'añadir punto'.\n"
            "El siguiente clic en el mapa abrirá un diálogo para confirmar el punto.")
        self.btn_add_clic.toggled.connect(self._toggle_modo_agregar)

        btn_manual = QtWidgets.QPushButton("+ Manual (SAMLight)")
        btn_manual.setStyleSheet(_SS_BTN)
        btn_manual.setToolTip("Introduce directamente las coordenadas SAMLight del punto.")
        btn_manual.clicked.connect(self._agregar_punto_manual)

        btn_del = QtWidgets.QPushButton("✕ Borrar seleccionado")
        btn_del.setStyleSheet(_SS_BTN + "QPushButton{color:#ff7043;}")
        btn_del.clicked.connect(self._borrar_punto_seleccionado)

        btn_row.addWidget(self.btn_add_clic)
        btn_row.addWidget(btn_manual)
        btn_row.addWidget(btn_del)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        # Info de modo
        self.lbl_modo_punto = QtWidgets.QLabel(
            "Click en el mapa para definir puntos de análisis con nombre.")
        self.lbl_modo_punto.setStyleSheet("color:#9e9e9e;font-size:8pt;")
        self.lbl_modo_punto.setWordWrap(True)
        lay.addWidget(self.lbl_modo_punto)

        # Tabla
        self.tabla_puntos = QtWidgets.QTableWidget(0, 5)
        self.tabla_puntos.setHorizontalHeaderLabels(
            ["ID", "X perfil", "Y perfil", "X SAMLight", "Y SAMLight"])
        self.tabla_puntos.setStyleSheet(_SS_TABLE)
        self.tabla_puntos.horizontalHeader().setStretchLastSection(True)
        self.tabla_puntos.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.tabla_puntos.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tabla_puntos.setAlternatingRowColors(True)
        self.tabla_puntos.verticalHeader().setVisible(False)
        lay.addWidget(self.tabla_puntos, 1)

        # Hint calibracion
        if self.cal is None:
            hint = QtWidgets.QLabel(
                "Sin calibracion SAMLight — los puntos se guardan en coords de perfil.")
            hint.setStyleSheet(
                "color:#ffa726;font-size:8pt;"
                "background:#1f2937;border-radius:4px;padding:6px;")
            hint.setWordWrap(True)
            lay.addWidget(hint)

        return w

    # ── Tab Perfiles ──────────────────────────────────────────────────────────
    def _build_tab_perfiles(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        # Selector P1 / P2
        sel_grp = QtWidgets.QGroupBox("Perfil entre dos puntos definidos")
        sel_grp.setStyleSheet(
            "QGroupBox{border:1px solid #374151;border-radius:4px;"
            "margin-top:6px;color:#9e9e9e;}"
            "QGroupBox::title{subcontrol-origin:margin;left:8px;}")
        sel_lay = QtWidgets.QGridLayout(sel_grp)
        sel_lay.setSpacing(6)

        sel_lay.addWidget(QtWidgets.QLabel("Punto A:"), 0, 0)
        self.combo_p1 = QtWidgets.QComboBox()
        self.combo_p1.setStyleSheet(_SS_COMBO)
        self.combo_p1.setMinimumWidth(100)
        sel_lay.addWidget(self.combo_p1, 0, 1)

        sel_lay.addWidget(QtWidgets.QLabel("Punto B:"), 0, 2)
        self.combo_p2 = QtWidgets.QComboBox()
        self.combo_p2.setStyleSheet(_SS_COMBO)
        self.combo_p2.setMinimumWidth(100)
        sel_lay.addWidget(self.combo_p2, 0, 3)

        btn_ver = QtWidgets.QPushButton("Ver perfil")
        btn_ver.setStyleSheet(_SS_BTN + "QPushButton{color:#4fc3f7;font-weight:bold;}")
        btn_ver.clicked.connect(self._ver_perfil_desde_combos)
        sel_lay.addWidget(btn_ver, 0, 4)

        btn_ver_poly = QtWidgets.QPushButton("Ver con intermedios")
        btn_ver_poly.setStyleSheet(_SS_BTN + "QPushButton{color:#66bb6a;font-weight:bold;}")
        btn_ver_poly.clicked.connect(self._ver_perfil_con_intermedios)
        sel_lay.addWidget(btn_ver_poly, 0, 5)

        lay.addWidget(sel_grp)

        seq_grp = QtWidgets.QGroupBox("Secuencia de puntos")
        seq_grp.setStyleSheet(
            "QGroupBox{border:1px solid #374151;border-radius:4px;"
            "margin-top:6px;color:#9e9e9e;}"
            "QGroupBox::title{subcontrol-origin:margin;left:8px;}")
        seq_lay = QtWidgets.QVBoxLayout(seq_grp)
        seq_lay.setSpacing(6)

        self.lista_puntos_perfil = QtWidgets.QListWidget()
        self.lista_puntos_perfil.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.lista_puntos_perfil.setStyleSheet(_SS_TABLE)
        self.lista_puntos_perfil.setMaximumHeight(92)
        seq_lay.addWidget(self.lista_puntos_perfil)

        btn_seq_row = QtWidgets.QHBoxLayout()
        btn_seq = QtWidgets.QPushButton("Ver secuencia seleccionada")
        btn_seq.setStyleSheet(_SS_BTN + "QPushButton{color:#66bb6a;font-weight:bold;}")
        btn_seq.clicked.connect(self._ver_perfil_secuencia_seleccionada)
        btn_seq_row.addWidget(btn_seq)
        btn_seq_row.addStretch()
        seq_lay.addLayout(btn_seq_row)

        lay.addWidget(seq_grp)

        # Info clic directo
        lbl_clic = QtWidgets.QLabel(
            "O bien: clic P1 → clic P2 directamente en el mapa para perfil rapido.")
        lbl_clic.setStyleSheet("color:#9e9e9e;font-size:8pt;")
        lay.addWidget(lbl_clic)

        self.lbl_p1_status = QtWidgets.QLabel(
            "Click en <b style='color:#00E676'>P1</b> sobre el mapa")
        self.lbl_p1_status.setStyleSheet("font-size:8pt;")
        lay.addWidget(self.lbl_p1_status)

        # Botones de gestión de perfiles
        btn_row2 = QtWidgets.QHBoxLayout()
        btn_clear = QtWidgets.QPushButton("Borrar todos")
        btn_clear.setStyleSheet(_SS_BTN + "QPushButton{color:#ff7043;}")
        btn_clear.clicked.connect(self._borrar_todos)
        btn_save = QtWidgets.QPushButton("Guardar todos PNG")
        btn_save.setStyleSheet(_SS_BTN + "QPushButton{color:#66bb6a;}")
        btn_save.clicked.connect(self._guardar_todos)
        btn_row2.addWidget(btn_clear)
        btn_row2.addWidget(btn_save)
        btn_row2.addStretch()
        lay.addLayout(btn_row2)

        # Tarjetas de perfil
        hdr = QtWidgets.QLabel(
            "<b>PERFILES</b>")
        hdr.setStyleSheet("color: #9e9e9e; font-size: 9pt; padding: 2px;")
        lay.addWidget(hdr)

        self._scroll_area = QtWidgets.QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._perfiles_widget = QtWidgets.QWidget()
        self._perfiles_lay = QtWidgets.QVBoxLayout(self._perfiles_widget)
        self._perfiles_lay.setContentsMargins(2, 2, 2, 2)
        self._perfiles_lay.setSpacing(6)
        self._perfiles_lay.addStretch()
        self._scroll_area.setWidget(self._perfiles_widget)
        lay.addWidget(self._scroll_area, 1)

        return w

    # ── Mapa: actualizar imagen ───────────────────────────────────────────────
    def _actualizar_mapa(self, modo):
        if modo == 'pre':
            lv = (np.nanpercentile(self.Hp, 2), np.nanpercentile(self.Hp, 98))
            self.img_item.setImage(np.flipud(self.Hp), autoLevels=False)
            self._aplicar_geometria_mapa()
            self.img_item.setLevels(lv)
            self.img_item.setColorMap(_cmap_vr6000())
            self.cbar.setLevels(lv)
        elif modo == 'post':
            H  = self.Hq - self.dz
            lv = (np.nanpercentile(self.Hp, 2), np.nanpercentile(self.Hp, 98))
            self.img_item.setImage(np.flipud(H), autoLevels=False)
            self._aplicar_geometria_mapa()
            self.img_item.setLevels(lv)
            self.img_item.setColorMap(_cmap_vr6000())
            self.cbar.setLevels(lv)
        else:
            d     = self.delta_raw
            d_max = max(abs(np.nanpercentile(d, 1)),
                        abs(np.nanpercentile(d, 99)), 3.0)
            self.img_item.setImage(np.flipud(d), autoLevels=False)
            self._aplicar_geometria_mapa()
            self.img_item.setLevels((-d_max, d_max))
            self.img_item.setColorMap(
                pg.colormap.get('RdBu_r', source='matplotlib'))
            self.cbar.setLevels((-d_max, d_max))

    # ── Mouse move ────────────────────────────────────────────────────────────
    def _on_mouse_move(self, evt):
        pos = evt[0]
        if not self.plot_mapa.sceneBoundingRect().contains(pos):
            return
        pt  = self.vb.mapSceneToView(pos)
        x, y = pt.x(), pt.y()
        if not (self.x_off <= x <= self.x_off + self.W and
                self.y_bot <= y <= self.y_bot + self.H):
            return
        self._cur_lbl.setPos(x, self.y_bot + self.H * 0.01)
        if self.cal is not None:
            xs, ys = self.cal.perfil_a_samlight(x, y)
            self._cur_lbl.setText(f"  SAMLight ({xs:.3f}, {ys:.3f}) mm")
            self.statusBar().showMessage(
                f"SAMLight:  x={xs:.4f}  y={ys:.4f} mm   |   "
                f"Perfil:  x={x:.4f}  y={y:.4f} mm")
        else:
            self._cur_lbl.setText(f"  ({x:.4f}, {y:.4f}) mm")
            self.statusBar().showMessage(
                f"Perfil:  x={x:.4f}  y={y:.4f} mm")

    # ── Click en mapa ─────────────────────────────────────────────────────────
    def _on_click_mapa(self, x, y):
        if not (self.x_off <= x <= self.x_off + self.W and
                self.y_bot <= y <= self.y_bot + self.H):
            self.statusBar().showMessage(
                f"Click fuera del heightmap: perfil x={x:.4f}, y={y:.4f} mm")
            return

        tab = self._right_tabs.currentIndex()

        if tab == _TAB_PUNTOS and self._modo_agregar_punto:
            self._confirmar_nuevo_punto(x, y)
            return

        if tab == _TAB_PUNTOS:
            idx = self._indice_punto_cercano(x, y)
            if idx is not None:
                self.tabla_puntos.selectRow(idx)
                self.statusBar().showMessage(
                    f"{self._puntos_def[idx].punto_id} seleccionado. Arrastralo para moverlo.")
            return

        if tab == _TAB_PERFILES:
            if self._p1_pend is None:
                self._p1_pend = (x, y)
                self._p1_scatter.setData([x], [y])
                self.lbl_p1_status.setText(
                    f"<b style='color:#00E676'>P1</b> = ({x:.4f}, {y:.4f})  "
                    "→ Ahora click en <b style='color:#ff7043'>P2</b>")
            else:
                p1, p2 = self._p1_pend, (x, y)
                self._p1_pend = None
                self._p1_scatter.clear()
                self.lbl_p1_status.setText(
                    "Click en <b style='color:#00E676'>P1</b> para otro perfil")
                self._agregar_perfil(p1, p2)

    def _on_tab_changed(self, idx):
        # Cancelar P1 pendiente si se cambia de tab
        if idx != _TAB_PERFILES and self._p1_pend is not None:
            self._p1_pend = None
            self._p1_scatter.clear()
        # Desactivar modo añadir si se sale de tab Puntos
        if idx != _TAB_PUNTOS and self._modo_agregar_punto:
            self.btn_add_clic.setChecked(False)

    # ── Gestión de puntos ─────────────────────────────────────────────────────
    def _toggle_modo_agregar(self, checked):
        self._modo_agregar_punto = checked
        if checked:
            self.lbl_modo_punto.setText(
                "MODO ACTIVO — haz clic en el mapa para colocar un punto.")
            self.lbl_modo_punto.setStyleSheet("color:#ffd54f;font-size:8pt;font-weight:bold;")
            self.statusBar().showMessage(
                "Clic en el mapa para colocar un punto de analisis")
        else:
            self.lbl_modo_punto.setText(
                "Click en el mapa para definir puntos de analisis con nombre. "
                "Arrastra un punto para moverlo.")
            self.lbl_modo_punto.setStyleSheet("color:#9e9e9e;font-size:8pt;")

    def _clip_mapa(self, x, y):
        return (
            float(np.clip(x, self.x_off, self.x_off + self.W)),
            float(np.clip(y, self.y_bot, self.y_bot + self.H)),
        )

    def _umbral_arrastre_mm(self):
        x_range, y_range = self.vb.viewRange()
        px_w = max(float(self.vb.width()), 1.0)
        px_h = max(float(self.vb.height()), 1.0)
        return max((x_range[1] - x_range[0]) / px_w,
                   (y_range[1] - y_range[0]) / px_h) * 16.0

    def _indice_punto_cercano(self, x, y):
        if not self._puntos_def:
            return None
        threshold = self._umbral_arrastre_mm()
        best_idx = None
        best_dist = None
        for idx, pt in enumerate(self._puntos_def):
            dist = float(np.hypot(pt.x_profile - x, pt.y_profile - y))
            if best_dist is None or dist < best_dist:
                best_idx = idx
                best_dist = dist
        if best_dist is not None and best_dist <= threshold:
            return best_idx
        return None

    def _preparar_arrastre_punto(self, x, y):
        if self._right_tabs.currentIndex() != _TAB_PUNTOS:
            return False
        if self._modo_agregar_punto:
            return False
        idx = self._indice_punto_cercano(x, y)
        if idx is None:
            return False
        self._drag_punto_idx = idx
        self.tabla_puntos.selectRow(idx)
        self.statusBar().showMessage(
            f"Moviendo {self._puntos_def[idx].punto_id}...")
        return True

    def _actualizar_posicion_punto(self, idx, x, y):
        if idx is None or not (0 <= idx < len(self._puntos_def)):
            return
        x, y = self._clip_mapa(x, y)
        pt = self._puntos_def[idx]
        pt.x_profile = x
        pt.y_profile = y
        if self.cal:
            pt.x_samlight, pt.y_samlight = self.cal.perfil_a_samlight(x, y)
        else:
            pt.x_samlight, pt.y_samlight = x, y
        self._refresh_puntos()
        self.tabla_puntos.selectRow(idx)

    def _mover_punto_arrastrado(self, x, y):
        self._actualizar_posicion_punto(self._drag_punto_idx, x, y)

    def _finalizar_arrastre_punto(self, x, y):
        idx = self._drag_punto_idx
        self._actualizar_posicion_punto(idx, x, y)
        if idx is not None and 0 <= idx < len(self._puntos_def):
            pt = self._puntos_def[idx]
            self.statusBar().showMessage(
                f"{pt.punto_id} movido: Perfil ({pt.x_profile:.4f}, {pt.y_profile:.4f}) mm | "
                f"SAMLight ({pt.x_samlight:.4f}, {pt.y_samlight:.4f}) mm")
        self._drag_punto_idx = None

    def _confirmar_nuevo_punto(self, x_profile, y_profile):
        default_id = f"PT{len(self._puntos_def)+1}"
        if self.cal:
            xs, ys = self.cal.perfil_a_samlight(x_profile, y_profile)
        else:
            xs, ys = x_profile, y_profile

        dlg = DialogNuevoPunto(self, default_id, x_profile, y_profile,
                               xs, ys, cal=self.cal, manual=False)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        xp, yp = dlg.perfil_xy()
        xsl, ysl = dlg.samlight_xy()
        pt = PuntoDefinido(dlg.punto_id(), xp, yp, xsl, ysl)
        self._puntos_def.append(pt)
        self._refresh_puntos()

        # Desactivar modo tras añadir (one-shot)
        self.btn_add_clic.setChecked(False)

    def _agregar_punto_manual(self):
        """Abre diálogo para introducir SAMLight directamente (sin clic en mapa)."""
        default_id = f"PT{len(self._puntos_def)+1}"
        # Centro del mapa como punto de partida
        x_mid = self.x_off + self.W / 2
        y_mid = self.y_bot + self.H / 2
        if self.cal:
            xs, ys = self.cal.perfil_a_samlight(x_mid, y_mid)
        else:
            xs, ys = x_mid, y_mid

        dlg = DialogNuevoPunto(self, default_id, x_mid, y_mid,
                               xs, ys, cal=self.cal, manual=True)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        xp, yp   = dlg.perfil_xy()
        xsl, ysl  = dlg.samlight_xy()
        pt = PuntoDefinido(dlg.punto_id(), xp, yp, xsl, ysl)
        self._puntos_def.append(pt)
        self._refresh_puntos()

    def _refresh_puntos(self):
        # Tabla
        self.tabla_puntos.setRowCount(len(self._puntos_def))
        for i, pt in enumerate(self._puntos_def):
            items = [
                pt.punto_id,
                f"{pt.x_profile:.4f}",
                f"{pt.y_profile:.4f}",
                f"{pt.x_samlight:.4f}",
                f"{pt.y_samlight:.4f}",
            ]
            for j, val in enumerate(items):
                item = QtWidgets.QTableWidgetItem(val)
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self.tabla_puntos.setItem(i, j, item)
        self.tabla_puntos.resizeColumnsToContents()

        # Marcadores en el mapa
        self._redraw_point_markers()

        # Combos en tab Perfiles
        self._refresh_combos()

    def _redraw_point_markers(self):
        for items in self._punto_map_items.values():
            for item in items:
                try:
                    self.plot_mapa.removeItem(item)
                except Exception:
                    pass
        self._punto_map_items = {}

        for pt in self._puntos_def:
            sc = pg.ScatterPlotItem(
                [pt.x_profile], [pt.y_profile],
                size=12, symbol='o',
                pen=pg.mkPen('#ffd54f', width=1.5),
                brush=pg.mkBrush('#ffd54f80'))
            sc.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
            lbl = pg.TextItem(pt.punto_id, color='#ffd54f', anchor=(0.5, 1.6))
            lbl.setFont(QtGui.QFont('Courier', 8, QtGui.QFont.Weight.Bold))
            lbl.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
            lbl.setPos(pt.x_profile, pt.y_profile)
            self.plot_mapa.addItem(sc)
            self.plot_mapa.addItem(lbl)
            self._punto_map_items[pt.punto_id] = [sc, lbl]

    def _refresh_combos(self):
        ids = [pt.punto_id for pt in self._puntos_def]
        p1_curr = self.combo_p1.currentText()
        p2_curr = self.combo_p2.currentText()
        for combo in (self.combo_p1, self.combo_p2):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(ids)
            combo.blockSignals(False)
        idx1 = self.combo_p1.findText(p1_curr)
        idx2 = self.combo_p2.findText(p2_curr)
        if idx1 >= 0: self.combo_p1.setCurrentIndex(idx1)
        if idx2 >= 0: self.combo_p2.setCurrentIndex(idx2)
        # Si hay 2+ puntos, seleccionar los últimos dos por defecto
        if len(ids) >= 2 and idx1 < 0:
            self.combo_p1.setCurrentIndex(len(ids) - 2)
        if len(ids) >= 1 and idx2 < 0:
            self.combo_p2.setCurrentIndex(len(ids) - 1)
        if hasattr(self, "lista_puntos_perfil"):
            selected = {item.text().split()[0]
                        for item in self.lista_puntos_perfil.selectedItems()}
            self.lista_puntos_perfil.clear()
            for pt in self._puntos_def:
                item = QtWidgets.QListWidgetItem(
                    f"{pt.punto_id}   perfil=({pt.x_profile:.3f}, {pt.y_profile:.3f})")
                item.setData(QtCore.Qt.ItemDataRole.UserRole, pt.punto_id)
                self.lista_puntos_perfil.addItem(item)
                if pt.punto_id in selected:
                    item.setSelected(True)

    def _borrar_punto_seleccionado(self):
        rows = self.tabla_puntos.selectionModel().selectedRows()
        if not rows:
            return
        idx = rows[0].row()
        if idx >= len(self._puntos_def):
            return
        pt = self._puntos_def[idx]
        for item in self._punto_map_items.get(pt.punto_id, []):
            try:
                self.plot_mapa.removeItem(item)
            except Exception:
                pass
        self._punto_map_items.pop(pt.punto_id, None)
        self._puntos_def.pop(idx)
        self._refresh_puntos()

    # ── Perfiles ──────────────────────────────────────────────────────────────
    def _ver_perfil_desde_combos(self):
        p1_id = self.combo_p1.currentText()
        p2_id = self.combo_p2.currentText()
        if not p1_id or not p2_id:
            self.statusBar().showMessage(
                "Define al menos dos puntos en la pestaña Puntos primero.")
            return
        pt1 = next((p for p in self._puntos_def if p.punto_id == p1_id), None)
        pt2 = next((p for p in self._puntos_def if p.punto_id == p2_id), None)
        if pt1 is None or pt2 is None:
            return
        L = np.hypot(pt2.x_profile - pt1.x_profile,
                     pt2.y_profile - pt1.y_profile)
        if self.cal:
            label_override = (
                f"{p1_id}({pt1.x_samlight:.3f}, {pt1.y_samlight:.3f})  →  "
                f"{p2_id}({pt2.x_samlight:.3f}, {pt2.y_samlight:.3f})  "
                f"L={L:.3f} mm  [SAMLight mm]"
            )
        else:
            label_override = (
                f"{p1_id}({pt1.x_profile:.3f}, {pt1.y_profile:.3f})  →  "
                f"{p2_id}({pt2.x_profile:.3f}, {pt2.y_profile:.3f})  "
                f"L={L:.3f} mm"
            )
        self._agregar_perfil(
            (pt1.x_profile, pt1.y_profile),
            (pt2.x_profile, pt2.y_profile),
            label_override=label_override)
        # Ir a tab perfiles para ver la tarjeta
        self._right_tabs.setCurrentIndex(_TAB_PERFILES)

    def _punto_def_por_id(self, punto_id):
        return next((p for p in self._puntos_def if p.punto_id == punto_id), None)

    def _label_puntos_perfil(self, puntos, ids=None):
        if ids is None:
            ids = [f"P{i+1}" for i in range(len(puntos))]
        L = sum(float(np.hypot(puntos[i+1][0] - puntos[i][0],
                               puntos[i+1][1] - puntos[i][1]))
                for i in range(len(puntos) - 1))
        return f"{' -> '.join(ids)}  L={L:.3f} mm"

    def _ver_perfil_secuencia_seleccionada(self):
        if not hasattr(self, "lista_puntos_perfil"):
            return
        selected_rows = sorted(
            self.lista_puntos_perfil.row(item)
            for item in self.lista_puntos_perfil.selectedItems())
        if len(selected_rows) < 2:
            self.statusBar().showMessage(
                "Selecciona al menos dos puntos en la lista de secuencia.")
            return

        pts_def = []
        for row in selected_rows:
            item = self.lista_puntos_perfil.item(row)
            if item is None:
                continue
            pt = self._punto_def_por_id(item.data(QtCore.Qt.ItemDataRole.UserRole))
            if pt is not None:
                pts_def.append(pt)
        if len(pts_def) < 2:
            return

        puntos = [(pt.x_profile, pt.y_profile) for pt in pts_def]
        ids = [pt.punto_id for pt in pts_def]
        self._agregar_perfil_polilinea(
            puntos,
            label_override=self._label_puntos_perfil(puntos, ids=ids),
            waypoint_labels=ids)
        self.statusBar().showMessage(f"Perfil secuencia: {' -> '.join(ids)}")
        self._right_tabs.setCurrentIndex(_TAB_PERFILES)

    def _ver_perfil_con_intermedios(self):
        p1_id = self.combo_p1.currentText()
        p2_id = self.combo_p2.currentText()
        pt1 = self._punto_def_por_id(p1_id)
        pt2 = self._punto_def_por_id(p2_id)
        if pt1 is None or pt2 is None:
            self.statusBar().showMessage(
                "Define al menos dos puntos en la pestaña Puntos primero.")
            return

        a = np.array([pt1.x_profile, pt1.y_profile], dtype=float)
        b = np.array([pt2.x_profile, pt2.y_profile], dtype=float)
        ab = b - a
        L = float(np.hypot(ab[0], ab[1]))
        if L < 1e-6:
            return

        unit = ab / L
        threshold = max(self._umbral_arrastre_mm() * 1.5, self.px * 8.0)
        intermedios = []
        for pt in self._puntos_def:
            if pt.punto_id in (p1_id, p2_id):
                continue
            p = np.array([pt.x_profile, pt.y_profile], dtype=float)
            along = float(np.dot(p - a, unit))
            if along <= self.px or along >= L - self.px:
                continue
            rel = p - a
            perp = float(abs(unit[0] * rel[1] - unit[1] * rel[0]))
            if perp <= threshold:
                intermedios.append((along, pt))

        intermedios.sort(key=lambda item: item[0])
        pts_def = [pt1] + [pt for _, pt in intermedios] + [pt2]
        puntos = [(pt.x_profile, pt.y_profile) for pt in pts_def]
        ids = [pt.punto_id for pt in pts_def]
        if len(pts_def) == 2:
            self.statusBar().showMessage(
                "No hay puntos intermedios cerca de la línea A-B; usando recta directa.")
        else:
            self.statusBar().showMessage(
                f"Perfil con intermedios: {' -> '.join(ids)}")
        self._agregar_perfil_polilinea(
            puntos,
            label_override=self._label_puntos_perfil(puntos, ids=ids),
            waypoint_labels=ids)
        self._right_tabs.setCurrentIndex(_TAB_PERFILES)

    def _sig_color(self):
        c = COLORES[self._color_idx % len(COLORES)]
        self._color_idx += 1
        return c

    def _agregar_perfil(self, p1, p2, label_override=None):
        return self._agregar_perfil_polilinea(
            [p1, p2], label_override=label_override)

        x1, y1 = p1
        x2, y2 = p2
        L = float(np.hypot(x2-x1, y2-y1))
        if L < 1e-6:
            return

        n = max(500, int(L / self.px * 2))
        t = np.linspace(0, 1, n)
        x_arr = x1 + t*(x2-x1)
        y_arr = y1 + t*(y2-y1)
        dists = t * L

        rows_f = -y_arr / self.px - self.r0
        cols_f =  x_arr / self.px - self.c0

        h_pre  = _samp(self.Hp,            rows_f, cols_f)
        h_post = _samp(self.Hq - self.dz, rows_f, cols_f)
        valid  = ~np.isnan(h_pre) & ~np.isnan(h_post)

        nr, nc = self.Hp.shape
        print(f"[Perfil] p1=({x1:.4f},{y1:.4f})  p2=({x2:.4f},{y2:.4f})  L={L:.4f} mm")
        print(f"  imagen: {nr}x{nc} px  r0={self.r0}  c0={self.c0}  px={self.px}")
        print(f"  rows_f = [{rows_f.min():.1f} .. {rows_f.max():.1f}]")
        print(f"  cols_f = [{cols_f.min():.1f} .. {cols_f.max():.1f}]")
        print(f"  puntos validos: {valid.sum()} / {len(valid)}")

        if valid.sum() < 5:
            self.lbl_p1_status.setText(
                f"<font color='#ff5252'>Solo {valid.sum()} puntos validos de {len(valid)} — "
                f"rows=[{rows_f.min():.0f}..{rows_f.max():.0f}] (max={nr-1})  "
                f"cols=[{cols_f.min():.0f}..{cols_f.max():.0f}] (max={nc-1})</font>")
            return

        ref      = np.nanmean(h_pre[valid])
        pre_um   = (h_pre  - ref) * 1000.0
        post_um  = (h_post - ref) * 1000.0
        delta_um = post_um - pre_um

        pre_s   = _smooth_profile(pre_um,   SMOOTH_PX)[valid]
        post_s  = _smooth_profile(post_um,  SMOOTH_PX)[valid]
        delta_s = _smooth_profile(delta_um, SMOOTH_PX)[valid]
        d       = dists[valid]

        color = self._sig_color()
        idx   = len(self._perfiles)

        pen  = pg.mkPen(color, width=2.5)
        line = self.plot_mapa.plot([x1, x2], [y1, y2], pen=pen)
        line.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)

        pts = pg.ScatterPlotItem([x1, x2], [y1, y2], size=11,
                                 pen=pg.mkPen('w', width=1.5),
                                 brush=[pg.mkBrush(color)]*2)
        pts.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.plot_mapa.addItem(pts)

        lbl1 = pg.TextItem(f'P{idx*2+1}', color=color, anchor=(0.5, 1.3))
        lbl2 = pg.TextItem(f'P{idx*2+2}', color=color, anchor=(0.5, 1.3))
        lbl1.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        lbl2.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        lbl1.setPos(x1, y1); lbl2.setPos(x2, y2)
        self.plot_mapa.addItem(lbl1)
        self.plot_mapa.addItem(lbl2)
        map_items = [line, pts, lbl1, lbl2]

        tarjeta = TarjetaPerfil(idx, p1, p2, color,
                                d, pre_s, post_s, delta_s,
                                map_items=map_items,
                                label_override=label_override)
        tarjeta.eliminado.connect(self._eliminar_perfil)
        self._perfiles.append(tarjeta)

        n_items = self._perfiles_lay.count()
        self._perfiles_lay.insertWidget(n_items - 1, tarjeta)

        QtCore.QTimer.singleShot(
            100, lambda: self._scroll_area.verticalScrollBar().setValue(99999))

    def _agregar_perfil_polilinea(self, puntos, label_override=None, waypoint_labels=None):
        puntos = [(float(x), float(y)) for x, y in puntos]
        if len(puntos) < 2:
            return

        xs_all, ys_all, d_all = [], [], []
        dist0 = 0.0
        for i, (p1, p2) in enumerate(zip(puntos[:-1], puntos[1:])):
            x1, y1 = p1
            x2, y2 = p2
            L = float(np.hypot(x2 - x1, y2 - y1))
            if L < 1e-6:
                continue
            n = max(2, int(L / self.px * 2))
            t = np.linspace(0.0, 1.0, n)
            if i > 0:
                t = t[1:]
            xs_all.append(x1 + t * (x2 - x1))
            ys_all.append(y1 + t * (y2 - y1))
            d_all.append(dist0 + t * L)
            dist0 += L

        if not xs_all:
            return

        x_arr = np.concatenate(xs_all)
        y_arr = np.concatenate(ys_all)
        dists = np.concatenate(d_all)

        rows_f = -y_arr / self.px - self.r0
        cols_f =  x_arr / self.px - self.c0

        h_pre  = _samp(self.Hp,           rows_f, cols_f)
        h_post = _samp(self.Hq - self.dz, rows_f, cols_f)
        valid  = ~np.isnan(h_pre) & ~np.isnan(h_post)

        nr, nc = self.Hp.shape
        if valid.sum() < 5:
            self.lbl_p1_status.setText(
                f"<font color='#ff5252'>Solo {valid.sum()} puntos validos de {len(valid)} - "
                f"rows=[{rows_f.min():.0f}..{rows_f.max():.0f}] (max={nr-1})  "
                f"cols=[{cols_f.min():.0f}..{cols_f.max():.0f}] (max={nc-1})</font>")
            return

        ref      = np.nanmean(h_pre[valid])
        pre_um   = (h_pre  - ref) * 1000.0
        post_um  = (h_post - ref) * 1000.0
        delta_um = post_um - pre_um

        pre_s   = _smooth_profile(pre_um,   SMOOTH_PX)[valid]
        post_s  = _smooth_profile(post_um,  SMOOTH_PX)[valid]
        delta_s = _smooth_profile(delta_um, SMOOTH_PX)[valid]
        d       = dists[valid]

        color = self._sig_color()
        idx   = len(self._perfiles)

        pen  = pg.mkPen(color, width=2.5)
        line = self.plot_mapa.plot(
            [p[0] for p in puntos], [p[1] for p in puntos], pen=pen)
        line.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)

        pts = pg.ScatterPlotItem(
            [p[0] for p in puntos], [p[1] for p in puntos], size=11,
            pen=pg.mkPen('w', width=1.5),
            brush=[pg.mkBrush(color)] * len(puntos))
        pts.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.plot_mapa.addItem(pts)

        labels = waypoint_labels or [f'P{idx*2+i+1}' for i in range(len(puntos))]
        lbl_items = []
        for label, (x, y) in zip(labels, puntos):
            lbl = pg.TextItem(label, color=color, anchor=(0.5, 1.3))
            lbl.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
            lbl.setPos(x, y)
            self.plot_mapa.addItem(lbl)
            lbl_items.append(lbl)
        map_items = [line, pts] + lbl_items

        tarjeta = TarjetaPerfil(idx, puntos[0], puntos[-1], color,
                                d, pre_s, post_s, delta_s,
                                map_items=map_items,
                                label_override=label_override)
        tarjeta.eliminado.connect(self._eliminar_perfil)
        self._perfiles.append(tarjeta)

        n_items = self._perfiles_lay.count()
        self._perfiles_lay.insertWidget(n_items - 1, tarjeta)

        QtCore.QTimer.singleShot(
            100, lambda: self._scroll_area.verticalScrollBar().setValue(99999))

    def _eliminar_perfil(self, tarjeta):
        for item in tarjeta.map_items:
            try:
                self.plot_mapa.removeItem(item)
            except Exception:
                pass
        self._perfiles_lay.removeWidget(tarjeta)
        tarjeta.setParent(None)
        tarjeta.deleteLater()
        if tarjeta in self._perfiles:
            self._perfiles.remove(tarjeta)

    def _borrar_todos(self):
        for t in list(self._perfiles):
            self._eliminar_perfil(t)

    def _guardar_todos(self):
        if not self._perfiles:
            self.statusBar().showMessage("No hay perfiles para guardar.")
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        for t in self._perfiles:
            t.exportar(self.out_dir, self.pre_name, self.post_name, ts)
        self.statusBar().showMessage(
            f"Guardados {len(self._perfiles)} perfiles en {self.out_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Comparacion interactiva de perfiles PRE/POST (Qt/pyqtgraph)")
    ap.add_argument("pre",  help="CSV PRE (VR-6000)")
    ap.add_argument("post", help="CSV POST (VR-6000)")
    ap.add_argument("--out", default="resultados",
                    help="Carpeta de salida (default: resultados/)")
    args = ap.parse_args()

    pre_name  = Path(args.pre).stem
    post_name = Path(args.post).stem
    base_dir  = Path(__file__).resolve().parent.parent
    out_dir   = str(Path(__file__).resolve().parent / args.out)

    print(f"\n{'='*60}")
    print(f"  PRE : {pre_name}")
    print(f"  POST: {post_name}")
    print(f"{'='*60}")

    print("Buscando calibracion SAMLight...", flush=True)
    cal = buscar_calibracion_auto(pre_name, base_dir)
    if cal is None:
        print("  No encontrada — se usaran coordenadas de perfil", flush=True)

    Hp, Hq, dz, delta_raw, px, r0, c0 = alinear(args.pre, args.post)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = VentanaPrincipal(Hp, Hq, dz, delta_raw, px, r0, c0,
                           pre_name, post_name, out_dir, cal=cal)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
