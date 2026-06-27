# Laser Correction / Samlight

Herramientas Python para trabajar con CSVs de altura del profilometro,
calibracion manual laser/Samlight, perfiles 2D y DXF.

## Estado Actual

Por ahora, la herramienta que usamos como flujo principal es:

```powershell
python .\interfaz_calibracion_manual_qt.py .\csv_entrada\test5.csv
```

Esta es la herramienta principal del proyecto en el estado actual. No necesita
que definas un origen fisico manualmente: la referencia queda definida por los
puntos que marcas en el heightmap y sus coordenadas reales de Samlight.

Este archivo abre una unica ventana con el heightmap a la izquierda y pestanas
de trabajo a la derecha. Sirve para:

- Cargar un CSV de altura del profilometro.
- Ajustar el rango visual del heightmap.
- Colocar puntos manualmente sobre el mapa y asignarles coordenadas reales de
  Samlight/laser.
- Calcular la calibracion afin perfilometro -> Samlight con esos puntos.
- Generar un DXF de cruces de calibracion.
- Seleccionar un perfil entre dos puntos, ver la altura y generar DXF por
  niveles/capas/pines.
- Anadir o quitar niveles de altura para el DXF.
- Seleccionar un perfil de 1 mm para COMSOL y exportarlo como CSV/TXT `x_mm,y_mm`.

La herramienta `interfazmejorada.py` sigue disponible, pero ahora mismo no es
el flujo principal:

```powershell
python .\interfazmejorada.py
```

Su objetivo es el flujo final deseado: dada una pieza, seleccionar un area,
filtrar picos por bandas de altura y generar nodos/rellenos para Samlight. Pero
por ahora no lo usamos como herramienta principal porque aparecieron problemas
de referencia/origen/offset y algunas salidas quedaban descuadradas respecto a
la realidad. Hasta resolver completamente esa referencia global, para trabajo
fiable usamos `interfaz_calibracion_manual_qt.py`.

La version Matplotlib queda como referencia antigua:

```powershell
python .\interfaz_calibracion_manual.py .\mi_archivo.csv
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
    plantilla_calibracion_con_origen.csv
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

## Uso De Interfazmejorada

Esta seccion documenta `interfazmejorada.py`, pero no es el flujo principal
actual. Esta herramienta queda como objetivo/futuro para procesar areas
completas cuando resolvamos del todo la referencia global.

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

## Ventanas De Interfazmejorada

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
Marcar origen   permite hacer click en el cruce fisico de las lineas
Marcar area     permite hacer 4 clicks para definir el area de actuacion
Borrar area     elimina el area y vuelve a procesar todo el scan
Exportar CSV    guarda datos de nodos en salidas/csv/
Exportar DXF    guarda trayectorias Samlight en salidas/dxf/
```

Antes de exportar un DXF desde el heightmap, marca el origen si el cruce de
las lineas no coincide exactamente con el pixel `(0,0)` del CSV. Si no se marca
o no se define en el CSV de calibracion, el script asume que el origen fisico
esta en la esquina superior izquierda del archivo, lo que produce un offset.

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

## Calibracion Manual Por Puntos Reales Del Laser

La herramienta nueva es:

```powershell
python .\interfaz_calibracion_manual.py .\pretest5.csv
```

Version rapida recomendada:

```powershell
python .\interfaz_calibracion_manual_qt.py .\pretest5.csv
```

Para abrir directamente `test5.csv`:

```powershell
python .\interfaz_calibracion_manual_qt.py .\csv_entrada\test5.csv
```

Tambien puedes cargar una calibracion manual guardada:

```powershell
python .\interfaz_calibracion_manual.py .\pretest5.csv .\calibracion\calibracion_manual_pretest5_YYYYMMDD_HHMMSS.csv
```

Flujo:

La version Qt se abre en una sola ventana: pieza/heightmap a la izquierda y
pestanas de trabajo a la derecha.

Nota importante: en esta herramienta no se define origen. Cada punto marcado
en el heightmap se empareja con su coordenada real de Samlight, y con esos pares
se calcula la transformacion. Con 3 puntos ya hay transformacion afin; con 5 o
mas puntos repartidos suele ser mas robusta.

1. En la pestana `Heightmap`, ajusta el `height range` con los sliders `Min` y `Max`.
   Tambien puedes escribir valores exactos en `Min exacto` y `Max exacto`.
2. En `Calibracion / DXF`, escribe `ID`, `X`, `Y` reales del laser/Samlight.
3. Pulsa `Nuevo punto` y haz click sobre el punto correspondiente del heightmap.
4. Puedes arrastrar cualquier punto para corregirlo.
5. Con 3 o mas puntos se calcula una transformacion afin perfilometro -> Samlight.
6. `DXF cruces` genera cruces de `0.5 x 0.5 mm` centradas en las coordenadas reales introducidas.
7. Escribe dos IDs en `Perfil A` y `Perfil B`, pulsa `Ver perfil`, ajusta niveles de altura y exporta el DXF.

Los niveles empiezan como `N2`, `N3` y `N4`, pero puedes usar `Anadir nivel`
o `Quitar ultimo`. Cada nivel exportado crea una capa/pin independiente en el
DXF.

### Perfil 2D Para COMSOL

En la pestana `COMSOL` de `interfaz_calibracion_manual_qt.py` puedes sacar un
perfil 2D `x,y` sin usar Matplotlib:

1. Ajusta el height range para ver bien la zona.
2. En `Longitud COMSOL mm`, deja `1.000000` para un segmento de 1 mm.
3. Pulsa `Marcar inicio` y haz click en el primer punto del perfil.
   Ese punto se exporta como `x=0`.
4. Pulsa `Marcar fin` y haz click hacia el extremo/direccion del segmento.
   La herramienta fuerza el extremo a la longitud exacta indicada, por ejemplo
   `x=1 mm`.
5. Puedes arrastrar los dos puntos naranjas sobre el heightmap.
6. Pulsa `Ver COMSOL` para ver el perfil.
7. Exporta con `COMSOL CSV` o `COMSOL TXT`.

Salidas:

```text
salidas/csv/Perfil_COMSOL_...csv   columnas x_mm,y_mm con cabecera
salidas/txt/Perfil_COMSOL_...txt   dos columnas sin cabecera
```

El `TXT` es el formato mas simple para importar como tabla/interpolacion en
COMSOL. La primera columna va siempre de `0` a la longitud indicada y la segunda
columna es la altura en mm.

El DXF del perfil se genera como recta por tramos de altura y separa capas/pines:

```text
PIN_1_NIVEL_2
PIN_2_NIVEL_3
PIN_3_NIVEL_4
```

Para evitar trazos absurdamente pequenos, la herramienta ignora tramos menores
que el diametro aproximado del haz:

```text
BEAM_DIAMETER_MM = 0.055
```

Al guardar calibracion, se crean dos archivos:

- `calibracion_manual_...csv`: editable y recargable por esta herramienta.
- `calibracion_affine_...csv`: compatible con `interfazmejorada.py` en modo `affine` para futuros DXF.

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
id,x_scanner_rel_mm,y_scanner_rel_mm,x_samlight_mm,y_samlight_mm,use_for_affine,origin_x_px,origin_y_px,profile_x_sign,profile_y_sign,origin_samlight_x_mm,origin_samlight_y_mm
T1,4.331,-4.493,-55,35,yes,0,0,1,-1,-59.551,39.559
T2,14.347,-4.497,-45,35,yes,,,,,,
T3,4.283,-14.531,-55,25,yes,,,,,,
```

Columnas:

```text
id                  nombre del punto
x_scanner_rel_mm    coordenada X medida en el profilometro/scanner
y_scanner_rel_mm    coordenada Y medida en el profilometro/scanner
x_samlight_mm       coordenada X ordenada/esperada en Samlight
y_samlight_mm       coordenada Y ordenada/esperada en Samlight
use_for_affine      yes/no para usar o ignorar el punto
origin_x_px         pixel X del origen fisico en el CSV del profilometro
origin_y_px         pixel Y del origen fisico en el CSV del profilometro
profile_x_sign      normalmente 1
profile_y_sign      1 si Y crece hacia abajo; -1 si quieres Y negativa hacia abajo
origin_samlight_x_mm coordenada X Samlight del origen fisico
origin_samlight_y_mm coordenada Y Samlight del origen fisico
```

Se necesitan al menos 3 puntos activos para validar la calibracion y, si dejas
`origin_samlight_x_mm/origin_samlight_y_mm` vacios, estimar el origen como:

```text
origen_samlight = media(Samlight_medido - desplazamiento_scanner)
```

En el modo actual no se aplica rotacion ni cambio de signo escondido: se usa
`Samlight = origen_samlight + dx/dy`.

Para una pieza nueva, lo recomendable es usar 4 o 5 marcas repartidas en la zona util.

### Origen

El origen tambien se define por pieza en el CSV de calibracion.

La opcion recomendada es usar:

```text
origin_x_px
origin_y_px
```

Estos son los pixeles del CSV/imagen del profilometro donde esta el origen fisico, por ejemplo el cruce de las lineas del soporte. Solo hace falta ponerlos en una fila; normalmente en la primera.

Con ese origen, el script convierte cualquier pixel asi:

```text
dx = (x_px - origin_x_px) * pixel_size * profile_x_sign
dy = (y_px - origin_y_px) * pixel_size * profile_y_sign
```

Y luego exporta a Samlight asi:

```text
X_samlight = origin_samlight_x_mm + dx
Y_samlight = origin_samlight_y_mm + dy
```

No hay cambio de signo escondido en la suma final. Si quieres que un desplazamiento en Y sea negativo, lo defines con `profile_y_sign=-1` o introduces `dy` negativo en tus marcas.

Tambien se puede definir el origen directamente en mm crudos con:

```text
origin_x_mm
origin_y_mm
```

No mezcles `origin_x_px/origin_y_px` con `origin_x_mm/origin_y_mm` en el mismo archivo.

## Que Cambiar Para Otra Pieza

1. Copia el nuevo CSV del profilometro a `csv_entrada/`.
2. Crea una nueva calibracion en `calibracion/`, por ejemplo:

```text
calibracion/mi_pieza_calibracion.csv
```

3. En ese CSV pon las marcas medidas:

```text
origin_x_px/origin_y_px del origen fisico de esa pieza
scanner X/Y reales medidos respecto a ese origen
Samlight X/Y que se ordenaron al laser
```

4. Ejecuta:

```powershell
python .\interfazmejorada.py .\csv_entrada\mi_pieza.csv .\calibracion\mi_pieza_calibracion.csv
```

5. Ajusta el height range para visualizar.
6. Pulsa `Marcar origen` y haz click en el cruce fisico de las lineas si no esta ya definido en la calibracion.
7. Ajusta N2/N3/N4 hasta que los nodos representen bien los picos.
8. Marca el area de actuacion con 4 clicks si no quieres procesar todo el scan.
9. Exporta CSV y DXF.
10. Importa el DXF en Samlight 1:1, sin centrar ni escalar.

## Archivos Viejos

Los prototipos HTML/JS/BAT, DXF antiguos y salidas antiguas se guardaron en:

```text
archivo_viejo/20260625_limpieza/
```

No se han borrado; simplemente se han apartado para que no se confundan con el flujo actual.
