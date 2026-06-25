import csv
import sys
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Polygon
from matplotlib.path import Path as MplPath
from matplotlib.widgets import Button, Slider, TextBox


# --- CONFIGURACION ---
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "csv_entrada"
CALIBRATION_DIR = BASE_DIR / "calibracion"
OUTPUT_DIR = BASE_DIR / "salidas"
DXF_OUTPUT_DIR = OUTPUT_DIR / "dxf"
CSV_OUTPUT_DIR = OUTPUT_DIR / "csv"
IMAGE_OUTPUT_DIR = OUTPUT_DIR / "imagenes"

DEFAULT_CSV_NAME = "test2.csv"
DEFAULT_CALIBRATION_NAME = "calibracion_test2.csv"

for directory in (INPUT_DIR, CALIBRATION_DIR, DXF_OUTPUT_DIR, CSV_OUTPUT_DIR, IMAGE_OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def resolve_existing_file(arg, default_dir):
    path = Path(arg)
    candidates = [
        path,
        BASE_DIR / path,
        default_dir / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"No se encontro '{arg}'. Prueba a ponerlo en: {default_dir}"
    )


CSV_FILE = resolve_existing_file(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV_NAME, INPUT_DIR)
CALIBRATION_FILE = resolve_existing_file(
    sys.argv[2] if len(sys.argv) > 2 else DEFAULT_CALIBRATION_NAME,
    CALIBRATION_DIR,
)

# Niveles iniciales. Si el scan tiene maximo menor/mayor, se recalculan abajo.
N4_INIT = 0.070  # Negro: z >= N4
N3_INIT = 0.065  # Rosa:  N3 <= z < N4
N2_INIT = 0.060  # Verde: N2 <= z < N3

STEP = 0.001

DXF_LEVEL_CONFIG = {
    # El nivel bajo exportado se puede asignar al pin 1 en Samlight, etc.
    2: {"pin": "PIN_1", "layer": "PIN_1_NIVEL_2", "color": "3"},  # Verde
    3: {"pin": "PIN_2", "layer": "PIN_2_NIVEL_3", "color": "6"},  # Magenta
    4: {"pin": "PIN_3", "layer": "PIN_3_NIVEL_4", "color": "1"},  # Rojo
}
DXF_HATCH_SPACING_MM = 0.025
DXF_INCLUDE_CONTOURS = False

# Area de actuacion seleccionada en coordenadas scanner (mm).
# Si no hay 3+ puntos, se procesa todo el scan.
area_points = []
area_patch = None
area_mask_cache = None
selecting_area = False


def clean_cell(value):
    value = str(value).strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1]
    return value


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

    z = np.full((ny, nx), np.nan, dtype=float)
    for y in range(ny):
        row = data_rows[y]
        for x in range(min(nx, len(row))):
            value = clean_cell(row[x])
            if value == "":
                continue
            try:
                z[y, x] = float(value)
            except ValueError:
                pass

    return z, pixel_size, header


def read_calibration_csv(path):
    points = []
    with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            use_value = clean_cell(row.get("use_for_affine", "yes")).lower()
            if use_value not in ("yes", "y", "si", "s", "true", "1"):
                continue
            points.append((
                float(clean_cell(row["x_scanner_rel_mm"])),
                float(clean_cell(row["y_scanner_rel_mm"])),
                float(clean_cell(row["x_samlight_mm"])),
                float(clean_cell(row["y_samlight_mm"])),
            ))

    if len(points) < 3:
        raise ValueError(
            f"La calibracion necesita al menos 3 puntos activos. Archivo: {path}"
        )
    return points


def affine_from_points(points):
    scanner = np.array([[p[0], p[1], 1.0] for p in points], dtype=float)
    sam_x = np.array([p[2] for p in points], dtype=float)
    sam_y = np.array([p[3] for p in points], dtype=float)
    if len(points) == 3:
        ax = np.linalg.solve(scanner, sam_x)
        ay = np.linalg.solve(scanner, sam_y)
    else:
        ax, *_ = np.linalg.lstsq(scanner, sam_x, rcond=None)
        ay, *_ = np.linalg.lstsq(scanner, sam_y, rcond=None)
    return ax, ay


def scanner_to_samlight(x_mm, y_mm):
    sx = affine_x[0] * x_mm + affine_x[1] * y_mm + affine_x[2]
    sy = affine_y[0] * x_mm + affine_y[1] * y_mm + affine_y[2]
    return sx, sy


def dxf_config_for_level(level):
    return DXF_LEVEL_CONFIG.get(
        level,
        {"pin": f"PIN_{level}", "layer": f"PIN_{level}_NIVEL_{level}", "color": "7"},
    )


def add_dxf_line(lines, layer, color, x0, y0, x1, y1):
    lines.extend([
        "0", "LINE",
        "8", layer,
        "62", color,
        "10", f"{x0:.6f}",
        "20", f"{y0:.6f}",
        "30", "0",
        "11", f"{x1:.6f}",
        "21", f"{y1:.6f}",
        "31", "0",
    ])


def add_dxf_polyline(lines, layer, color, points):
    lines.extend(["0", "POLYLINE", "8", layer, "62", color, "66", "1", "70", "1"])
    for x_coord, y_coord in points:
        lines.extend([
            "0", "VERTEX",
            "8", layer,
            "62", color,
            "10", f"{x_coord:.6f}",
            "20", f"{y_coord:.6f}",
            "30", "0",
        ])
    lines.extend(["0", "SEQEND"])


def iter_contiguous_runs(sorted_xs):
    if len(sorted_xs) == 0:
        return

    start = int(sorted_xs[0])
    previous = start
    for value in sorted_xs[1:]:
        value = int(value)
        if value <= previous + 1:
            previous = value
            continue
        yield start, previous
        start = value
        previous = value
    yield start, previous


def iter_node_hatch_segments(node):
    step_px = max(1, int(round(DXF_HATCH_SPACING_MM / pixel_size)))
    rows = {}
    for x_px, y_px in zip(node["xs"], node["ys"]):
        rows.setdefault(int(y_px), []).append(int(x_px))

    selected_rows = sorted(rows)[::step_px]
    if not selected_rows and rows:
        selected_rows = [sorted(rows)[len(rows) // 2]]

    for y_px in selected_rows:
        xs_row = np.asarray(rows[y_px], dtype=int)
        xs_row.sort()
        for start_px, end_px in iter_contiguous_runs(xs_row):
            x0_mm = start_px * pixel_size
            x1_mm = (end_px + 1) * pixel_size
            y_mm = -y_px * pixel_size
            yield (*scanner_to_samlight(x0_mm, y_mm), *scanner_to_samlight(x1_mm, y_mm))


def make_height_colormap():
    # Negativo azul -> cero verde -> positivo amarillo/naranja/rojo
    colors = [
        (0.00, "#0757c8"),
        (0.35, "#7ec8ff"),
        (0.50, "#16a34a"),
        (0.68, "#ffd84d"),
        (0.82, "#ff8a1f"),
        (1.00, "#d71920"),
    ]
    return LinearSegmentedColormap.from_list("zero_green", colors, N=256)


def create_level_map(n4, n3, n2):
    level_map = np.ones((*z.shape, 3), dtype=float)
    actuation_mask = get_actuation_mask()
    mask2 = (z >= n2) & (z < n3)
    mask3 = (z >= n3) & (z < n4)
    mask4 = z >= n4
    mask2 &= actuation_mask
    mask3 &= actuation_mask
    mask4 &= actuation_mask

    level_map[mask2] = [0.0, 1.0, 0.0]       # Verde fosforito
    level_map[mask3] = [1.0, 0.078, 0.576]   # Rosa
    level_map[mask4] = [0.0, 0.0, 0.0]       # Negro
    return level_map, mask4, mask3, mask2


def get_actuation_mask():
    global area_mask_cache
    if len(area_points) < 3:
        return np.ones(z.shape, dtype=bool)
    if area_mask_cache is None:
        polygon = MplPath(np.asarray(area_points, dtype=float))
        coords = np.column_stack([X_GRID.ravel(), Y_GRID.ravel()])
        area_mask_cache = polygon.contains_points(coords).reshape(z.shape)
    return area_mask_cache


def find_nodes(mask):
    if not np.any(mask):
        return []
    mask_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def summarize_nodes(level, mask, contours):
    summaries = []
    for node_id, contour in enumerate(contours, start=1):
        node_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(node_mask, [contour], -1, 255, thickness=-1)
        node_mask = node_mask.astype(bool)

        ys, xs = np.where(node_mask)
        if len(xs) == 0:
            continue

        heights = z[node_mask]
        summaries.append({
            "level": level,
            "node": node_id,
            "mean_height": float(np.nanmean(heights)),
            "area_mm2": float(np.sum(node_mask) * pixel_size**2),
            "xs": xs,
            "ys": ys,
            "contour": contour,
        })
    return summaries


def current_levels():
    n4 = slider_n4.val
    n3 = slider_n3.val
    n2 = slider_n2.val

    if n3 >= n4:
        n3 = n4 - STEP
        slider_n3.set_val(n3)
    if n2 >= n3:
        n2 = n3 - STEP
        slider_n2.set_val(n2)

    return n4, n3, n2


def compute_current_nodes():
    n4, n3, n2 = current_levels()
    _, mask4, mask3, mask2 = create_level_map(n4, n3, n2)

    contours4 = find_nodes(mask4)
    contours3 = find_nodes(mask3)
    contours2 = find_nodes(mask2)

    nodes = []
    nodes.extend(summarize_nodes(4, mask4, contours4))
    nodes.extend(summarize_nodes(3, mask3, contours3))
    nodes.extend(summarize_nodes(2, mask2, contours2))

    stats = {
        4: {"mask": mask4, "contours": contours4},
        3: {"mask": mask3, "contours": contours3},
        2: {"mask": mask2, "contours": contours2},
    }
    return nodes, stats


def stats_text(stats, n4, n3, n2):
    lines = []
    if len(area_points) >= 3:
        lines.append(f"AREA ACTUACION\nPuntos: {len(area_points)}\n")
    else:
        lines.append("AREA ACTUACION\nTodo el scan\n")
    for level, label, threshold in [
        (4, "NIVEL 4 (Negro)", f">= {n4:.3f} mm"),
        (3, "NIVEL 3 (Rosa)", f"[{n3:.3f}, {n4:.3f}) mm"),
        (2, "NIVEL 2 (Verde)", f"[{n2:.3f}, {n3:.3f}) mm"),
    ]:
        mask = stats[level]["mask"]
        pixels = int(np.sum(mask))
        area = pixels * pixel_size**2
        nodes = len(stats[level]["contours"])
        lines.append(
            f"{label}\n"
            f"Umbral: {threshold}\n"
            f"Pixeles: {pixels}\n"
            f"Area: {area:.4f} mm2\n"
            f"Nodos: {nodes}\n"
        )
    lines.append(f"RESTO (Blanco)\nUmbral: < {n2:.3f} mm")
    return "\n".join(lines)


def update_nodes(_=None):
    n4, n3, n2 = current_levels()
    level_map, mask4, mask3, mask2 = create_level_map(n4, n3, n2)
    im_nodes.set_data(level_map)

    contours4 = find_nodes(mask4)
    contours3 = find_nodes(mask3)
    contours2 = find_nodes(mask2)
    stats = {
        4: {"mask": mask4, "contours": contours4},
        3: {"mask": mask3, "contours": contours3},
        2: {"mask": mask2, "contours": contours2},
    }

    ax_nodes.set_title(
        f"Filtrado por niveles | N4>={n4:.3f} | N3:[{n3:.3f},{n4:.3f}) | N2:[{n2:.3f},{n3:.3f})",
        fontsize=11,
        fontweight="bold",
    )
    ax_stats.clear()
    ax_stats.axis("off")
    ax_stats.text(
        0.05,
        0.95,
        stats_text(stats, n4, n3, n2),
        transform=ax_stats.transAxes,
        fontsize=10,
        va="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85),
    )
    fig_nodes.canvas.draw_idle()


def apply_height_range(vmin, vmax):
    if vmin >= 0:
        vmin = -STEP
    if vmax <= 0:
        vmax = STEP
    if vmin >= vmax:
        return

    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    im_height.set_norm(norm)
    ax_height.set_title(f"Height range | min={vmin:.4f} mm | max={vmax:.4f} mm")
    fig_height.canvas.draw_idle()


def update_height(_=None):
    try:
        vmin = float(tb_vmin.text)
        vmax = float(tb_vmax.text)
    except ValueError:
        return
    apply_height_range(vmin, vmax)


def update_height_from_sliders(_=None):
    vmin = slider_hmin.val
    vmax = slider_hmax.val
    tb_vmin.set_val(f"{vmin:.4f}")
    tb_vmax.set_val(f"{vmax:.4f}")
    apply_height_range(vmin, vmax)


def nudge_active_level(delta):
    sliders = {4: slider_n4, 3: slider_n3, 2: slider_n2}
    slider = sliders[active_level[0]]
    slider.set_val(round(slider.val + delta, 6))
    update_nodes()


def on_key(event):
    if event.key in ("4", "3", "2"):
        active_level[0] = int(event.key)
        print(f"Nivel activo para flechas: N{active_level[0]}")
    elif event.key == "up":
        nudge_active_level(STEP)
    elif event.key == "down":
        nudge_active_level(-STEP)
    elif event.key == "right":
        nudge_active_level(STEP * 10)
    elif event.key == "left":
        nudge_active_level(-STEP * 10)


def start_area_selection(_=None):
    global selecting_area, area_mask_cache
    area_points.clear()
    area_mask_cache = None
    selecting_area = True
    redraw_area_patch()
    update_nodes()
    print("Seleccion area: haz 4 clicks en la ventana NODOS (A, B, C, D).")


def clear_area_selection(_=None):
    global selecting_area, area_mask_cache
    area_points.clear()
    area_mask_cache = None
    selecting_area = False
    redraw_area_patch()
    update_nodes()
    print("Area de actuacion borrada. Se procesara todo el scan.")


def on_nodes_click(event):
    global selecting_area, area_mask_cache
    if not selecting_area or event.inaxes is not ax_nodes:
        return
    if event.xdata is None or event.ydata is None:
        return
    area_points.append((float(event.xdata), float(event.ydata)))
    area_mask_cache = None
    print(f"Punto area {len(area_points)}: x={event.xdata:.3f} mm, y={event.ydata:.3f} mm")
    redraw_area_patch()
    if len(area_points) >= 4:
        selecting_area = False
        print("Area de actuacion definida. Los nodos/CSV/DXF usaran solo esa zona.")
        update_nodes()


def redraw_area_patch():
    global area_patch
    if area_patch is not None:
        area_patch.remove()
        area_patch = None
    if len(area_points) >= 2:
        area_patch = Polygon(
            area_points,
            closed=len(area_points) >= 3,
            fill=False,
            edgecolor="yellow",
            linewidth=2,
            linestyle="-",
        )
        ax_nodes.add_patch(area_patch)
    fig_nodes.canvas.draw_idle()


def export_nodes_csv(_=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = CSV_OUTPUT_DIR / f"Niveles_Nodos_{CSV_FILE.stem}_{timestamp}.csv"
    image_output = IMAGE_OUTPUT_DIR / f"Figura_Nodos_{CSV_FILE.stem}_{timestamp}.png"

    nodes, _ = compute_current_nodes()
    rows = [[
        "Nivel",
        "Pin_sugerido",
        "Capa_DXF",
        "Nodo",
        "Altura_media_mm",
        "Area_mm2",
        "X_mm",
        "Y_mm",
        "Z_mm",
        "X_samlight_mm",
        "Y_samlight_mm",
    ]]

    for node in nodes:
        dxf_cfg = dxf_config_for_level(node["level"])
        rows.append([
            node["level"],
            dxf_cfg["pin"],
            dxf_cfg["layer"],
            node["node"],
            f"{node['mean_height']:.6f}",
            f"{node['area_mm2']:.6f}",
            "",
            "",
            "",
            "",
            "",
        ])
        for x_px, y_px in zip(node["xs"], node["ys"]):
            x_mm = x_px * pixel_size
            y_mm = -y_px * pixel_size
            z_mm = z[y_px, x_px]
            sx, sy = scanner_to_samlight(x_mm, y_mm)
            rows.append([
                node["level"],
                dxf_cfg["pin"],
                dxf_cfg["layer"],
                node["node"],
                "",
                "",
                f"{x_mm:.6f}",
                f"{y_mm:.6f}",
                f"{z_mm:.6f}",
                f"{sx:.6f}",
                f"{sy:.6f}",
            ])

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    fig_nodes.savefig(image_output, dpi=200)
    print(f"CSV generado: {output}")
    print(f"Imagen generada: {image_output}")


def export_nodes_dxf(_=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = DXF_OUTPUT_DIR / f"Nodos_Samlight_{CSV_FILE.stem}_{timestamp}.dxf"
    nodes, _ = compute_current_nodes()

    used_levels = sorted({node["level"] for node in nodes})
    lines = [
        "0", "SECTION",
        "2", "TABLES",
        "0", "TABLE",
        "2", "LAYER",
        "70", str(len(used_levels)),
    ]
    for level in used_levels:
        cfg = dxf_config_for_level(level)
        lines.extend([
            "0", "LAYER",
            "2", cfg["layer"],
            "70", "0",
            "62", cfg["color"],
            "6", "CONTINUOUS",
        ])
    lines.extend([
        "0", "ENDTAB",
        "0", "ENDSEC",
        "0", "SECTION",
        "2", "ENTITIES",
    ])

    hatch_count = 0
    contour_count = 0
    for node in nodes:
        cfg = dxf_config_for_level(node["level"])
        layer = cfg["layer"]
        color = cfg["color"]

        for sx0, sy0, sx1, sy1 in iter_node_hatch_segments(node):
            add_dxf_line(lines, layer, color, sx0, sy0, sx1, sy1)
            hatch_count += 1

        if DXF_INCLUDE_CONTOURS:
            contour = node["contour"]
            if len(contour) < 3:
                continue
            approx = cv2.approxPolyDP(contour, epsilon=1.5, closed=True)
            contour_points = []
            for point in approx[:, 0, :]:
                x_px, y_px = int(point[0]), int(point[1])
                x_mm = x_px * pixel_size
                y_mm = -y_px * pixel_size
                contour_points.append(scanner_to_samlight(x_mm, y_mm))
            add_dxf_polyline(lines, layer, color, contour_points)
            contour_count += 1

    lines.extend(["0", "ENDSEC", "0", "EOF"])
    output.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"DXF generado: {output}")
    print(f"Lineas internas de relleno: {hatch_count}")
    if DXF_INCLUDE_CONTOURS:
        print(f"Contornos exportados: {contour_count}")
    print("Capas DXF:")
    for level in used_levels:
        cfg = dxf_config_for_level(level)
        print(f"  {cfg['layer']} -> {cfg['pin']}")


# --- LECTURA ---
z, pixel_size, header = read_keyence_height_csv(CSV_FILE)
ny, nx = z.shape
x = np.arange(nx) * pixel_size
y = -np.arange(ny) * pixel_size
X_GRID, Y_GRID = np.meshgrid(x, y)

CALIBRATION_POINTS = read_calibration_csv(CALIBRATION_FILE)
affine_x, affine_y = affine_from_points(CALIBRATION_POINTS)

z_min = float(np.nanmin(z))
z_max = float(np.nanmax(z))
valid = np.sum(~np.isnan(z))

print("\n--- INFORMACION DEL SCAN ---")
print(f"CSV entrada: {CSV_FILE}")
print(f"Calibracion: {CALIBRATION_FILE}")
print(f"Dimensiones: {nx} x {ny} pixeles")
print(f"Area fisica: {nx * pixel_size:.2f} x {ny * pixel_size:.2f} mm")
print(f"Altura minima: {z_min:.3f} mm")
print(f"Altura maxima: {z_max:.3f} mm")
print(f"Valores validos: {valid} / {z.size}")
print("\n--- TRANSFORMACION SCANNER -> SAMLIGHT ---")
print(f"X = {affine_x[0]:.6f}*x + {affine_x[1]:.6f}*y + {affine_x[2]:.6f}")
print(f"Y = {affine_y[0]:.6f}*x + {affine_y[1]:.6f}*y + {affine_y[2]:.6f}")

# Ajuste inicial de nodos para test2
positive_max = max(z_max, 0.001)
N4_INIT = positive_max * 0.90
N3_INIT = positive_max * 0.80
N2_INIT = positive_max * 0.70

height_cmap = make_height_colormap()
height_norm = TwoSlopeNorm(vmin=z_min, vcenter=0.0, vmax=z_max)


# --- VENTANA 1: HEIGHT RANGE ---
fig_height, ax_height = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(bottom=0.26)

im_height = ax_height.imshow(
    z,
    extent=[x.min(), x.max(), y.min(), y.max()],
    cmap=height_cmap,
    norm=height_norm,
    aspect="auto",
    origin="upper",
)
ax_height.set_xlabel("X [mm]")
ax_height.set_ylabel("Y [mm]")
ax_height.set_title(f"Height range | min={z_min:.4f} mm | max={z_max:.4f} mm")
ax_height.set_xlim(0, x.max())
ax_height.set_ylim(y.min(), y.max())
fig_height.colorbar(im_height, ax=ax_height, label="Altura [mm]")

ax_hmin = plt.axes([0.15, 0.14, 0.62, 0.025])
ax_hmax = plt.axes([0.15, 0.10, 0.62, 0.025])
slider_hmin = Slider(
    ax_hmin,
    "Min color",
    z_min,
    min(-STEP, z_max - STEP),
    valinit=z_min,
    valstep=STEP,
    color="#0757c8",
)
slider_hmax = Slider(
    ax_hmax,
    "Max color",
    max(STEP, z_min + STEP),
    z_max,
    valinit=z_max,
    valstep=STEP,
    color="#d71920",
)

ax_tb_min = plt.axes([0.15, 0.04, 0.22, 0.04])
ax_tb_max = plt.axes([0.55, 0.04, 0.22, 0.04])
tb_vmin = TextBox(ax_tb_min, "Min height", initial=f"{z_min:.4f}")
tb_vmax = TextBox(ax_tb_max, "Max height", initial=f"{z_max:.4f}")
slider_hmin.on_changed(update_height_from_sliders)
slider_hmax.on_changed(update_height_from_sliders)
tb_vmin.on_submit(update_height)
tb_vmax.on_submit(update_height)


# --- VENTANA 2: NODOS ---
fig_nodes = plt.figure(figsize=(14, 10))
ax_nodes = plt.subplot2grid((6, 3), (0, 0), rowspan=5, colspan=2)
ax_stats = plt.subplot2grid((6, 3), (0, 2), rowspan=5)
ax_stats.axis("off")

level_map, _, _, _ = create_level_map(N4_INIT, N3_INIT, N2_INIT)
im_nodes = ax_nodes.imshow(
    level_map,
    extent=[x.min(), x.max(), y.min(), y.max()],
    aspect="auto",
    origin="upper",
)
ax_nodes.set_xlabel("X [mm]")
ax_nodes.set_ylabel("Y [mm]")
ax_nodes.set_xlim(0, x.max())
ax_nodes.set_ylim(y.min(), y.max())
ax_nodes.grid(True, alpha=0.25)

ax_s4 = plt.axes([0.08, 0.13, 0.55, 0.025])
ax_s3 = plt.axes([0.08, 0.09, 0.55, 0.025])
ax_s2 = plt.axes([0.08, 0.05, 0.55, 0.025])

slider_n4 = Slider(ax_s4, "N4 Negro", 0.0, max(positive_max * 1.1, 0.001), valinit=N4_INIT, valstep=STEP, color="black")
slider_n3 = Slider(ax_s3, "N3 Rosa", 0.0, max(positive_max * 1.1, 0.001), valinit=N3_INIT, valstep=STEP, color="deeppink")
slider_n2 = Slider(ax_s2, "N2 Verde", 0.0, max(positive_max * 1.1, 0.001), valinit=N2_INIT, valstep=STEP, color="lime")

slider_n4.on_changed(update_nodes)
slider_n3.on_changed(update_nodes)
slider_n2.on_changed(update_nodes)

ax_csv = plt.axes([0.70, 0.08, 0.12, 0.04])
ax_dxf = plt.axes([0.84, 0.08, 0.12, 0.04])
ax_area = plt.axes([0.70, 0.02, 0.12, 0.04])
ax_clear = plt.axes([0.84, 0.02, 0.12, 0.04])
btn_csv = Button(ax_csv, "Exportar CSV", color="lightblue", hovercolor="skyblue")
btn_dxf = Button(ax_dxf, "Exportar DXF", color="lightgreen", hovercolor="lime")
btn_area = Button(ax_area, "Marcar area", color="khaki", hovercolor="gold")
btn_clear = Button(ax_clear, "Borrar area", color="mistyrose", hovercolor="lightcoral")
btn_csv.on_clicked(export_nodes_csv)
btn_dxf.on_clicked(export_nodes_dxf)
btn_area.on_clicked(start_area_selection)
btn_clear.on_clicked(clear_area_selection)

active_level = [4]
fig_nodes.canvas.mpl_connect("key_press_event", on_key)
fig_nodes.canvas.mpl_connect("button_press_event", on_nodes_click)

update_nodes()

print("\nControles ventana NODOS:")
print("  Tecla 4/3/2: selecciona nivel activo")
print("  Flecha arriba/abajo: cambia umbral +/- 0.001 mm")
print("  Flecha derecha/izquierda: cambia umbral +/- 0.010 mm")
print("  Marcar area: pulsa el boton y haz 4 clicks sobre la zona de actuacion")
print("  CSV/DXF: exportan SOLO esa zona si hay area marcada")
print("  CSV/DXF: incluyen coordenadas scanner y coordenadas corregidas Samlight")
print("  DXF: relleno interno por lineas, no solo contorno")
print("  DXF: capas separadas por pin -> PIN_1_NIVEL_2, PIN_2_NIVEL_3, PIN_3_NIVEL_4")

plt.show()
