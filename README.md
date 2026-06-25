# Laser Correction / Samlight

Herramienta Python para convertir un CSV de altura del profilometro en:

- Mapa de alturas ajustable.
- Nodos por bandas de altura.
- CSV de nodos con coordenadas scanner y coordenadas Samlight corregidas.
- DXF para Samlight con relleno interno por lineas, separado por capas/pines.

La herramienta principal es:

```powershell
python .\interfazmejorada.py
```

## Estructura Del Proyecto

```text
laser_correction/
  interfazmejorada.py
  README.md

  csv_entrada/
    test2.csv
    prueba1_recto_template.csv

  calibracion/
    calibracion_test2.csv
    plantilla_marcas_calibracion.csv
    plantilla_puntos_referencia.csv
    plantilla_repetibilidad.csv

  salidas/
    dxf/
    csv/
    imagenes/

  referencias/
    prueba.PNG

  documentacion/
    protocolo_referencia_samlight.md

  archivo_viejo/
    20260625_limpieza/
```

## Uso Normal

1. Mete el CSV del profilometro en `csv_entrada/`.
2. Mete o edita el CSV de calibracion en `calibracion/`.
3. Ejecuta:

```powershell
python .\interfazmejorada.py
```

Por defecto lee:

```text
csv_entrada/test2.csv
calibracion/calibracion_test2.csv
```

Tambien puedes indicar archivos concretos:

```powershell
python .\interfazmejorada.py .\csv_entrada\mi_pieza.csv .\calibracion\mi_calibracion.csv
```

Si pasas solo el nombre del CSV, tambien lo buscara dentro de `csv_entrada/`:

```powershell
python .\interfazmejorada.py mi_pieza.csv
```

## Ventanas

La aplicacion abre dos ventanas.

### 1. Height Range

Sirve para ajustar solo la visualizacion del mapa de altura.

- `Min color`: altura minima del color.
- `Max color`: altura maxima del color.
- El cero queda centrado en verde.
- Negativo va a azul.
- Positivo va a amarillo/naranja/rojo.

Esto no cambia el DXF directamente; solo ayuda a ver la pieza.

### 2. Nodos

Sirve para crear las bandas de altura que se exportaran.

- `N4 Negro`: picos mas altos.
- `N3 Rosa`: banda intermedia.
- `N2 Verde`: banda baja.

Controles de teclado:

```text
4 / 3 / 2              selecciona el nivel activo
flecha arriba/abajo    cambia el umbral +/- 0.001 mm
flecha derecha/izq.    cambia el umbral +/- 0.010 mm
```

Botones:

```text
Marcar area     permite hacer 4 clicks para definir el area de actuacion
Borrar area     elimina el area y vuelve a procesar todo el scan
Exportar CSV    guarda datos de nodos en salidas/csv/
Exportar DXF    guarda trayectorias Samlight en salidas/dxf/
```

Si no marcas area, se exporta todo el scan. Si marcas area, solo se exporta lo que cae dentro del poligono.

## Salidas

Los archivos generados se guardan automaticamente aqui:

```text
salidas/dxf/       DXF para Samlight
salidas/csv/       CSV de nodos y coordenadas
salidas/imagenes/  captura PNG de la ventana de nodos
```

Los nombres incluyen el nombre del CSV de entrada y la fecha:

```text
Nodos_Samlight_test2_YYYYMMDD_HHMMSS.dxf
Niveles_Nodos_test2_YYYYMMDD_HHMMSS.csv
Figura_Nodos_test2_YYYYMMDD_HHMMSS.png
```

## DXF Para Samlight

El DXF se exporta ya en coordenadas Samlight corregidas.

Cada pixel/nodo pasa por:

```text
pixel del profilometro -> coordenada scanner mm -> coordenada Samlight mm
```

El DXF no debe importarse en Samlight con auto-centrado, auto-escalado ni "fit to field". Debe importarse 1:1 en mm.

Capas actuales:

```text
PIN_1_NIVEL_2
PIN_2_NIVEL_3
PIN_3_NIVEL_4
```

Cada capa puede asignarse a un pin/proceso distinto en Samlight.

El DXF no exporta solo contornos. Exporta lineas internas de relleno dentro de cada nodo. La separacion entre lineas se controla en `interfazmejorada.py`:

```python
DXF_HATCH_SPACING_MM = 0.025
```

Valores tipicos:

```text
0.015 mm  mas denso
0.025 mm  normal
0.050 mm  menos denso
```

## Calibracion

La calibracion se lee desde un CSV como:

```csv
id,x_scanner_rel_mm,y_scanner_rel_mm,x_samlight_mm,y_samlight_mm,use_for_affine
T1,4.331,-4.493,-55,35,yes
T2,14.347,-4.497,-45,35,yes
T3,4.283,-14.531,-55,25,yes
```

Columnas:

```text
id                  nombre del punto
x_scanner_rel_mm    coordenada X medida en el profilometro/scanner
y_scanner_rel_mm    coordenada Y medida en el profilometro/scanner
x_samlight_mm       coordenada X ordenada/esperada en Samlight
y_samlight_mm       coordenada Y ordenada/esperada en Samlight
use_for_affine      yes/no para usar o ignorar el punto
```

Se necesitan al menos 3 puntos activos.

- Con 3 puntos: transformacion afin exacta.
- Con 4 o mas puntos: ajuste afin por minimos cuadrados.

Para una pieza nueva, lo recomendable es usar 4 o 5 marcas repartidas en la zona util.

## Que Cambiar Para Otra Pieza

1. Copia el nuevo CSV del profilometro a `csv_entrada/`.
2. Crea una nueva calibracion en `calibracion/`, por ejemplo:

```text
calibracion/mi_pieza_calibracion.csv
```

3. En ese CSV pon las marcas medidas:

```text
scanner X/Y reales medidos con el profilometro
Samlight X/Y que se ordenaron al laser
```

4. Ejecuta:

```powershell
python .\interfazmejorada.py .\csv_entrada\mi_pieza.csv .\calibracion\mi_pieza_calibracion.csv
```

5. Ajusta el height range para visualizar.
6. Ajusta N2/N3/N4 hasta que los nodos representen bien los picos.
7. Marca el area de actuacion con 4 clicks si no quieres procesar todo el scan.
8. Exporta CSV y DXF.
9. Importa el DXF en Samlight 1:1, sin centrar ni escalar.

## Archivos Viejos

Los prototipos HTML/JS/BAT, DXF antiguos y salidas antiguas se guardaron en:

```text
archivo_viejo/20260625_limpieza/
```

No se han borrado; simplemente se han apartado para que no se confundan con el flujo actual.

