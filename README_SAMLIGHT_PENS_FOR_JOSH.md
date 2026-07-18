# SAMLight Pen RGB Mapping and SVG Export Notes

This document describes the pen-color export behavior that needs to be implemented
in any script that generates geometry for SAMLight.

The short version:

1. Keep generating DXF if needed for CAD/debugging.
2. Also generate an SVG with the same geometry.
3. Use RGB colors in the SVG to make SAMLight assign objects to the correct pens.
4. Negate Y coordinates in SVG: `svg_y = -samlight_y`.

## Why SVG

SAMLight did not reliably assign pens from DXF layer names or DXF color indices.
SVG import worked reliably when each entity used the exact RGB color associated
with the target SAMLight pen.

So the validated workflow is:

```text
script geometry -> DXF for reference
script geometry -> SVG for SAMLight import
```

The SVG is the important file for automatic pen assignment.

## Validated SAMLight Pen RGB Table

Use these exact RGB values:

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

Important: Pen 10 must be `255,255,0`. Earlier cyan/teal values were confused by
SAMLight with Pen 4 or Pen 6.

Recommended constant:

```python
SAMLIGHT_PEN_RGB = {
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
    4: (0, 128, 255),
    5: (255, 170, 0),
    6: (0, 170, 170),
    7: (85, 85, 85),
    8: (255, 85, 0),
    9: (0, 170, 0),
    10: (255, 255, 0),
    11: (255, 0, 255),
    12: (0, 0, 85),
    13: (255, 170, 255),
    14: (170, 225, 255),
}
```

## SVG Coordinate Convention

This is critical.

SAMLight/DXF coordinates are normal Cartesian millimeters. SVG's native coordinate
system has Y increasing downward. If the same Y values are written directly to SVG,
the geometry imports vertically inverted.

Therefore:

```text
svg_x = samlight_x
svg_y = -samlight_y
```

The SVG `viewBox` must also use the inverted Y range:

```text
viewBox = x_min, -y_max, width, height
```

Do not remove this `-Y` correction unless SAMLight import settings are changed and
orientation is re-tested with known geometry.

## Minimal SVG Writer Pattern

The script should collect line segments like this:

```python
segments = [
    {
        "layer": "PEN_1_NIVEL_2",
        "pen": 1,
        "x0": -50.0,
        "y0": 14.0,
        "x1": -48.0,
        "y1": 12.0,
    },
]
```

Then write SVG lines with the selected pen RGB and negated Y:

```python
def write_svg_lines(path, segments):
    xs = [v for s in segments for v in (s["x0"], s["x1"])]
    ys = [v for s in segments for v in (s["y0"], s["y1"])]

    margin = 1.0
    x_min = min(xs) - margin
    x_max = max(xs) + margin
    y_min = min(ys) - margin
    y_max = max(ys) + margin
    width = x_max - x_min
    height = y_max - y_min

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg"',
        f'     width="{width:.6f}mm" height="{height:.6f}mm"',
        f'     viewBox="{x_min:.6f} {-y_max:.6f} {width:.6f} {height:.6f}">',
        '  <g fill="none" stroke-width="0.02" stroke-linecap="round">',
    ]

    for seg in segments:
        r, g, b = SAMLIGHT_PEN_RGB[int(seg["pen"])]
        lines.append(
            f'    <line id="{seg["layer"]}" '
            f'x1="{seg["x0"]:.6f}" y1="{-seg["y0"]:.6f}" '
            f'x2="{seg["x1"]:.6f}" y2="{-seg["y1"]:.6f}" '
            f'stroke="rgb({r},{g},{b})" />'
        )

    lines.extend(["  </g>", "</svg>"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
```

## Suggested Entity Naming

Layer/entity names are not what assigns the SAMLight pen, but they are useful for
debugging.

Recommended names:

```text
PEN_1_NIVEL_2
PEN_2_NIVEL_3
PEN_10_NIVEL_11
CALIB_CRUCES_PEN_4_0p5mm
TRABAJO_CRUCES_PEN_6_0p5mm
```

The actual pen assignment should come from the SVG stroke RGB.

## Import Checklist in SAMLight

When importing the SVG into SAMLight:

1. Use the SVG file, not the DXF, for automatic pen assignment.
2. Preserve scale, 1:1 in millimeters.
3. Do not auto-center or fit-to-field unless deliberately changing coordinates.
4. Make sure SAMLight is using pen/color import behavior.
5. Verify orientation with known geometry, especially Y direction.

## Validation Files

Useful existing test files:

```text
salidas/dxf/Test_Pens_14_Lineas_RGB_reales.svg
salidas/dxf/Test_Pens_14_Lineas_RGB_reales_pen10_255_255_0.svg
```

The first file contains one line per pen using the current validated RGB table.
The second was used to confirm Pen 10 as yellow `255,255,0`.
