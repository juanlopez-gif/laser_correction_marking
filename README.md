# Laser Correction — SAMLight Height Profiler

Python tools for working with VR-6000 heightmap CSVs from the Keyence profilometer:
calibrating the scanner → SAMLight coordinate transform, generating DXF trajectories,
and interactively comparing PRE/POST laser correction profiles.

---

## SAMLight Pen Mapping and SVG Export

The current implementation exports both DXF and SVG. The DXF is kept for
traceability and CAD inspection, but the **validated SAMLight import path is the
SVG**, because SAMLight assigns pens much more reliably from SVG RGB colors than
from DXF layer names or DXF color indices.

Height levels are still named `N2`, `N3`, `N4`, etc., but each level now has a
**Pen 1-14 selector** in the Qt calibration UI. The height threshold and the
SAMLight pen assignment are independent.

Default level assignment:

| Level | Default pen | Export layer example |
|---|---:|---|
| N2 | Pen 1 | `PEN_1_NIVEL_2` |
| N3 | Pen 2 | `PEN_2_NIVEL_3` |
| N4 | Pen 3 | `PEN_3_NIVEL_4` |

The dropdowns let you override those defaults. The calibration UI also exposes
pen selectors for:

| Entity type | UI control | Default |
|---|---|---:|
| Height-level profile segments | Pen dropdown in each N-level row | N2 -> Pen 1, N3 -> Pen 2, ... |
| Calibration crosses | `Pen cruces calibracion` | Pen 4 |
| Work-point crosses | `Pen cruces trabajo` | Pen 6 |

For profile and cross exports, the Qt tool writes paired files:

```text
Perfil_Samlight_{stem}_{P1}_{P2}_{timestamp}.dxf
Perfil_Samlight_{stem}_{P1}_{P2}_{timestamp}.svg

Perfiles_Samlight_{stem}_{timestamp}.dxf
Perfiles_Samlight_{stem}_{timestamp}.svg

Cruces_Calibracion_{stem}_{timestamp}.dxf
Cruces_Calibracion_{stem}_{timestamp}.svg
```

The RGB table is stored in:

- `calibracion/samlight_pens_rgb.csv`, as the human-readable calibration record.
- `SAMLIGHT_PEN_RGB` in `interfaz_calibracion_manual_qt.py`, used by the exporter.

Validated SAMLight pen RGB table:

| Pen | R | G | B |
|---:|---:|---:|---:|
| 1 | 255 | 0 | 0 |
| 2 | 0 | 255 | 0 |
| 3 | 0 | 0 | 255 |
| 4 | 0 | 128 | 255 |
| 5 | 255 | 170 | 0 |
| 6 | 0 | 170 | 170 |
| 7 | 85 | 85 | 85 |
| 8 | 255 | 85 | 0 |
| 9 | 0 | 170 | 0 |
| 10 | 255 | 255 | 0 |
| 11 | 255 | 0 | 255 |
| 12 | 0 | 0 | 85 |
| 13 | 255 | 170 | 255 |
| 14 | 170 | 225 | 255 |

Important: **Pen 10 uses `255,255,0`**. Earlier cyan/teal candidates were confused
by SAMLight with Pen 4 or Pen 6.

### Implementation Notes

The implementation is intentionally simple:

- `SAMLIGHT_PEN_RGB` stores the validated pen colors.
- `make_pen_combo()` builds the Pen 1-14 dropdowns.
- `level_pen_assignments` stores the selected pen per height level.
- `dxf_config_for_level()` creates the export layer name and attaches the selected
  pen RGB.
- `write_svg_lines()` writes the parallel SVG file using the RGB color that
  SAMLight should map back to the selected pen.

DXF colors are still written, but they are secondary. The important file for
automatic SAMLight pen assignment is the SVG.

### Critical SVG `-Y` Correction

This is the important geometry detail: **SVG Y must be negated**.

DXF/SAMLight coordinates are treated as normal Cartesian millimeters. SVG's native
coordinate system has Y increasing downward. If the exporter writes SAMLight Y
directly into SVG, the imported geometry appears vertically inverted in SAMLight.

The exporter therefore writes:

```text
SVG x = SAMLight x
SVG y = -SAMLight y
```

The SVG `viewBox` is also created with the inverted Y range:

```text
viewBox = x_min, -y_max, width, height
```

This lives in `write_svg_lines()` in `interfaz_calibracion_manual_qt.py`.
Do not remove the `-Y` conversion unless the SAMLight SVG import settings are
changed and the orientation is revalidated with a known test file.

Useful validation files:

| File | Purpose |
|---|---|
| `salidas/dxf/Test_Pens_14_Lineas_RGB_reales.svg` | One line per pen using the current validated RGB table |
| `salidas/dxf/Test_Pens_14_Lineas_RGB_reales_pen10_255_255_0.svg` | Pen 10 yellow validation test |

---

## Requirements

| Requirement | Version |
|---|---|
| Python | **3.11** (required — Anaconda ships an old pyqtgraph incompatible with PySide6) |
| PySide6 | >= 6.4 |
| pyqtgraph | >= 0.13 |
| numpy, scipy | any recent |

Install into your Python 3.11 environment:

```powershell
pip install PySide6 pyqtgraph numpy scipy
```

> **Important:** Use the Python 3.11 executable directly or the `.bat` launchers provided.
> Anaconda's default `python` ships pyqtgraph < 0.13 which crashes on startup.

---

## Project Structure

```text
laser_correction_marking-main/
  interfaz_calibracion_manual_qt.py   ← main calibration & DXF tool
  compare_heights.py                  ← PRE/POST alignment pipeline (no GUI)
  perfil_interactivo.bat              ← launcher for compare_perfil_interactivo

  csv_entrada/                        ← input heightmap CSVs from profilometer
  calibracion/                        ← saved calibration files
  salidas/
    dxf/                              ← exported DXF files for SAMLight
    csv/                              ← exported profile CSVs
    imagenes/                         ← exported PNG captures

  comparacion_experimento1/
    compare_perfil_interactivo.py     ← interactive PRE/POST profile comparator
    perfil_interactivo.bat            ← launcher using Python 3.11
    compare_heights.py                ← alignment pipeline (shared)
    parte_1/ parte_2/ parte_3/ ...    ← per-experiment input CSVs
    resultados/                       ← comparison outputs
```

---

## Tool 1 — `interfaz_calibracion_manual_qt.py`

**Purpose:** Load a heightmap CSV, place calibration points by clicking on the map,
compute the profilometer → SAMLight affine transform, and export DXF correction
trajectories by height level.

### Launch

```powershell
C:\Users\mss\AppData\Local\Programs\Python\Python311\python.exe .\interfaz_calibracion_manual_qt.py .\csv_entrada\prueba1_steel_Height.csv
```

You can also reload a previously saved calibration:

```powershell
python .\interfaz_calibracion_manual_qt.py .\csv_entrada\prueba1.csv .\calibracion\calibracion_manual_prueba1_20260629_120000.csv
```

### Window Layout

The window opens as a single panel:
- **Left (large):** interactive heightmap — clickable, zoomable, pannable.
- **Right:** three tabs + status log at the bottom.

---

### Tab 1 — Heightmap

Adjust the colour range to bring out the surface detail you care about.

| Control | What it does |
|---|---|
| **Min (µm) / Max (µm)** spinboxes | Set exact colour limits |
| **Min / Max sliders** | Fast visual sweep of the range |

The colour map goes blue (low) → green → yellow → red (high).
Changing these values never touches the DXF or calibration — display only.

---

### Tab 2 — Calibration / DXF

This is the main working tab. No origin pixel needs to be defined; the calibration
is entirely defined by point pairs (profilometer location ↔ SAMLight coordinate).

#### Step-by-step workflow

1. **Enter the SAMLight coordinates** of the first point:
   - Fill in **ID**, **X SAMLight (mm)**, **Y SAMLight (mm)**.

2. **Click "Nuevo punto"** — the cursor changes to a crosshair.
   Click on the corresponding peak or mark on the heightmap.
   The point appears as a coloured circle with its ID label.

3. **Repeat** for at least 3 points (5+ spread across the surface gives a more
   robust affine transform). You can **drag** any point on the map to fine-tune
   its position.

4. The **calibration error table** (columns: ID, X laser, Y laser, X perfil,
   Y perfil, err µm) updates live. The residual error per point is shown in µm.

5. **"Guardar calib."** saves two files to `calibracion/`:
   - `calibracion_manual_{stem}_{timestamp}.csv` — reloadable by this tool
   - `calibracion_affine_{stem}_{timestamp}.csv` — compatible with `interfazmejorada.py`

6. **"DXF cruces"** exports a DXF with 0.5 × 0.5 mm crosses centred at each
   SAMLight coordinate you entered — useful for verifying alignment by burning
   test marks.

#### Profile between two points

7. Select **Perfil A** and **Perfil B** from the dropdowns (populated from your
   calibration points), then click **"Ver perfil"**.
   The height profile between those two points appears in the plot below.

8. **Height levels** (N2 / N3 / N4 by default) define the threshold bands for the
   DXF. You can **"Añadir nivel"** or **"Quitar ultimo"** to change the number of
   layers. Each level maps to one DXF layer / SAMLight pin:

   | Level | DXF layer | SAMLight pin |
   |---|---|---|
   | N2 | `PIN_1_NIVEL_2` | PIN 1 |
   | N3 | `PIN_2_NIVEL_3` | PIN 2 |
   | N4 | `PIN_3_NIVEL_4` | PIN 3 |

9. **"Exportar DXF perfil"** / **"Exportar CSV perfil"** write the profile
   trajectory to `salidas/dxf/` and `salidas/csv/` respectively.
   The exported CSV includes both coordinate systems:
   `distance_mm, height_mm, level, x_profile_mm, y_profile_mm, x_samlight_mm, y_samlight_mm`

> **DXF import note:** import into SAMLight at 1:1, no auto-centering, no scaling,
> no "fit to field". The coordinates are already in SAMLight mm.

---

### Tab 3 — COMSOL

Extract a 1D height profile for finite-element simulation input (no Matplotlib needed).

| Control | What it does |
|---|---|
| **Longitud mm** | Length of the profile segment (default 1.000 mm) |
| **Puntos perfil** | Number of sample points along the segment |
| **"Marcar inicio"** | Click on the heightmap to set the start point (x = 0) |
| **"Marcar fin"** | Click to set the direction; endpoint is snapped to the exact length |
| **"Ver COMSOL"** | Plots the interpolated height profile |
| **"COMSOL CSV"** | Saves `salidas/csv/Perfil_COMSOL_{stem}_{timestamp}.csv` (x_mm, y_mm) |
| **"COMSOL TXT"** | Saves `salidas/txt/Perfil_COMSOL_{stem}_{timestamp}.txt` (two columns, no header) |

Both endpoints are shown as orange handles on the map and can be dragged.
The TXT format is the simplest for direct COMSOL table/interpolation import.

---

### Status Log

The bottom-right panel shows all operations, calibration residuals, file paths
saved, and any warnings. Read it to confirm exports completed successfully.

---

### Output File Summary — `interfaz_calibracion_manual_qt.py`

| Output | Location | Description |
|---|---|---|
| `calibracion_manual_{stem}_{ts}.csv` | `calibracion/` | Reloadable calibration point pairs |
| `calibracion_affine_{stem}_{ts}.csv` | `calibracion/` | Affine matrix for `interfazmejorada.py` |
| `Cruces_Calibracion_{stem}_{ts}.dxf` | `salidas/dxf/` | DXF crosses at SAMLight coordinates |
| `Cruces_Calibracion_{stem}_{ts}.svg` | `salidas/dxf/` | SVG crosses with validated pen RGB colors and `-Y` correction |
| `Perfil_Samlight_{stem}_{P1}_{P2}_{ts}.dxf` | `salidas/dxf/` | DXF height-level trajectory |
| `Perfil_Samlight_{stem}_{P1}_{P2}_{ts}.svg` | `salidas/dxf/` | SVG height-level trajectory for SAMLight pen-color import |
| `Perfiles_Samlight_{stem}_{ts}.dxf` | `salidas/dxf/` | DXF export for all queued profiles |
| `Perfiles_Samlight_{stem}_{ts}.svg` | `salidas/dxf/` | SVG export for all queued profiles |
| `Perfil_{stem}_{ts}.csv` | `salidas/csv/` | Profile with both coordinate systems |
| `Perfil_COMSOL_{stem}_{ts}.csv` | `salidas/csv/` | COMSOL profile (x_mm, height_mm) |
| `Perfil_COMSOL_{stem}_{ts}.txt` | `salidas/txt/` | COMSOL profile (plain two-column) |

---

---

## Tool 2 — `compare_perfil_interactivo.py`

**Purpose:** Interactively compare height profiles between a PRE-correction and
POST-correction scan. Click two points on the DELTA map to extract a cross-section;
multiple profiles can be overlaid simultaneously.

### Launch

Use the provided batch file (it calls Python 3.11 automatically):

```bat
comparacion_experimento1\perfil_interactivo.bat <pre.csv> <post.csv>
```

Example:

```bat
comparacion_experimento1\perfil_interactivo.bat csv_entrada\prueba1_steel_Height.csv csv_entrada\postlinea1_steel_Height.csv
```

Optional output directory:

```bat
perfil_interactivo.bat pre.csv post.csv --out resultados\experimento1\
```

Or call Python 3.11 directly:

```powershell
C:\Users\mss\AppData\Local\Programs\Python\Python311\python.exe comparacion_experimento1\compare_perfil_interactivo.py pre.csv post.csv
```

---

### Alignment Pipeline

On startup the tool automatically aligns PRE and POST:

1. **Coarse + fine Phase-Only Correlation (POC)** — sub-pixel lateral shift between scans.
2. **In-plane rotation search** — corrects small angular misalignment.
3. **Z-offset (`dZ`)** — global height offset between the two scans.
4. **Tilt correction** — removes residual X/Y tilt from the POST scan.

The overlap region (pixels present in both scans after alignment) is the working area.
Alignment parameters are printed to the console: `Y, X shift`, `dZ`, `RMS delta`.

---

### SAMLight Coordinate Calibration (Automatic)

The tool converts click positions on the map to SAMLight mm so that profiles can be
positioned exactly as in the calibration interface.

Calibration is detected **automatically** — no `--cal` argument needed:

1. Looks for `calibracion/calibracion_manual_{pre_stem}_*.csv`
   (saved by `interfaz_calibracion_manual_qt.py` → "Guardar calib.").
2. If not found, looks for `salidas/csv/Perfil_{pre_stem}_*.csv`
   (a profile CSV exported from the calibration interface — it contains both
   coordinate systems in its columns, so the two endpoints give a full
   similarity transform: scale + rotation + translation).

If neither is found, profile positions are shown in profilometer coordinates
(still correct for measuring lengths and comparing heights).

---

### Window Layout

```
┌─────────────────────────────┬─────────────────────────────────┐
│                             │  Profile 1  [X]                 │
│   DELTA MAP (clickable)     │  ┌──────────────────────────┐   │
│                             │  │ PRE vs POST heights (mm) │   │
│   Click P1 → click P2       │  └──────────────────────────┘   │
│   to extract a profile      │  ┌──────────────────────────┐   │
│                             │  │ Delta (µm) fill chart    │   │
│   [Map mode dropdown]       │  └──────────────────────────┘   │
│   [Delete all] [Save all]   │  Profile 2  [X]                 │
│                             │  ...                            │
└─────────────────────────────┴─────────────────────────────────┘
  Status bar: x_profile_mm, y_profile_mm   (or x_samlight_mm if calibrated)
```

---

### How to Add a Profile

1. The map starts in **DELTA** mode (POST − PRE after alignment, in µm).
   You can switch to **PRE** or **POST** view using the dropdown.

2. **Left-click** on the first point (P1) — a green marker appears and the
   status bar shows the coordinates.

3. **Left-click** on the second point (P2) — the profile is immediately computed
   and a card appears in the right panel.

   > Tip: slight mouse movement during a click is handled correctly — both pure
   > clicks and very short drags register as point selections.

4. Repeat for as many profiles as needed. Each profile gets a distinct colour
   (red, blue, green, purple, orange, …).

---

### Profile Card

Each profile card shows:

- **Top plot:** height in mm for PRE (solid) and POST (dashed) along the profile
  distance axis.
- **Bottom plot:** delta = POST − PRE in µm, with positive excursions filled red
  and negative excursions filled blue.
- **[X] button:** removes that individual profile from the map and the panel.

---

### Toolbar Buttons

| Button | Action |
|---|---|
| Map mode dropdown | Switch between DELTA / PRE / POST display |
| **Borrar todos** | Remove all profiles |
| **Guardar todos** | Export all profile cards as PNG to the output folder |

---

### Output Files — `compare_perfil_interactivo.py`

Saved to the `--out` directory (default: `comparacion_experimento1/resultados/`):

```text
Perfil_1_{timestamp}.png
Perfil_2_{timestamp}.png
...
```

---

### Coordinate Convention

Both tools share the same coordinate system derived from the profilometer CSV:

```
x ∈ [col_offset × px,  (col_offset + ncols) × px]      (mm, left to right)
y ∈ [-(row_offset + nrows) × px,  -row_offset × px]     (mm, top is negative)
```

Where `px` is the pixel size read from the CSV header (`XY Calibration` field ÷ 1000).
This matches the axis labels in `interfaz_calibracion_manual_qt.py` exactly, so
coordinates can be copied between the two tools.

---

## Calibration File Format

Files saved by the calibration interface (`calibracion/calibracion_manual_*.csv`):

```csv
id,x_profile_mm,y_profile_mm,x_samlight_mm,y_samlight_mm,use_for_affine,...
P1,4.064494,-3.143519,-50.006488,-0.000709,yes,...
P2,2.100000,-4.500000,-51.900000,-1.200000,yes,...
```

- **`use_for_affine`**: set to `no` to exclude a point from the transform while keeping it as a reference.
- At least **3 non-collinear points** are needed for the full affine transform.
- The `compare_perfil_interactivo.py` tool only needs **2 points** (it uses a similarity
  transform — scale + rotation + translation — which is sufficient for well-calibrated scans).

---

## Typical Full Workflow

```
1. Scan PRE  →  profilometer CSV  →  csv_entrada/prueba1_steel_Height.csv
2. Run interfaz_calibracion_manual_qt.py with the PRE CSV
   → Place 3–5 calibration points, save calibration
   → (Optional) export DXF crosses, verify alignment on part
3. Apply laser correction in SAMLight
4. Scan POST  →  csv_entrada/postlinea1_steel_Height.csv
5. Run compare_perfil_interactivo.bat with PRE + POST CSVs
   → Tool auto-detects calibration from step 2
   → Click profiles across corrected lines to verify depth/shape
   → Save PNG exports
```

---

## Legacy Tools

| File | Status | Notes |
|---|---|---|
| `interfazmejorada.py` | Available, not primary | Full-area node extraction; had origin/offset issues |
| `interfaz_calibracion_manual.py` | Old (Matplotlib) | Replaced by the Qt version |

Old prototypes and outputs were archived to `archivo_viejo/20260625_limpieza/`.
