# Protocolo de referencia y correccion para Samlight

Version: 1.0

## Objetivo

Definir un sistema de coordenadas fisico y repetible para piezas medidas con el profilometro y procesadas con laser en Samlight.

El cruce de las dos lineas del soporte se usa como origen fisico `O = (0,0)`. La linea larga del soporte define la direccion principal `X`. El eje `Y` no se toma de la segunda linea si esta no forma 90 deg; se construye como la perpendicular matematica a `X`, pasando por `O`.

La segunda linea del soporte se usa para encontrar y verificar el origen, y para medir el error angular del soporte.

## Convencion de coordenadas

- `O`: interseccion entre la linea larga del soporte y la segunda linea.
- `X`: direccion de la linea larga ajustada.
- `+X`: desde `O` hacia la zona util de trabajo sobre la pieza.
- `Y`: perpendicular exacta a `X`.
- `+Y`: hacia el interior de la pieza o hacia la zona donde se vaya a trabajar.
- Unidades: mm.

Para cualquier punto medido por el profilometro:

```text
P = punto medido en coordenadas del profilometro, en mm
u = vector unitario de la linea larga ajustada
v = vector unitario perpendicular a u, orientado hacia +Y
O = origen fisico

x_ref = dot(P - O, u)
y_ref = dot(P - O, v)
```

Estas coordenadas `(x_ref, y_ref)` son las coordenadas de referencia de la pieza.

## Preparacion

1. Fijar la pieza en el soporte sin cambiar la orientacion entre medicion de profilometro y proceso laser.
2. Limpiar visualmente las lineas de referencia del soporte para que sean detectables.
3. Elegir una zona de la pieza donde se puedan hacer marcas de calibracion con potencia minima. Si la superficie final es critica, usar margenes o zonas fuera del area funcional.
4. Registrar el archivo de profilometro usado como referencia inicial. En el ejemplo actual, el CSV indica `XY Calibration = 11.814 um/px`.

## Paso 1: definir el origen y el eje X

1. Medir o seleccionar al menos 8 puntos sobre la linea larga del soporte.
2. Distribuir los puntos a lo largo de la mayor longitud visible de la linea.
3. Evitar esquinas danadas, sombras, rebabas o zonas donde la linea tenga mucho grosor irregular.
4. Ajustar una recta a esos puntos. La direccion de esta recta es `X`.
5. Medir o seleccionar al menos 5 puntos sobre la segunda linea del soporte.
6. Ajustar una recta a la segunda linea.
7. Calcular `O` como la interseccion entre ambas rectas ajustadas.
8. Calcular el angulo entre ambas rectas y registrar:

```text
angulo_soporte_deg
error_90_deg = abs(90 - angulo_soporte_deg)
```

La segunda linea no define `Y`; solo ayuda a ubicar `O` y a cuantificar cuanto se desvia el soporte de 90 deg.

## Paso 2: crear las marcas de calibracion en Samlight

Crear 5 marcas pequenas con el laser en la pieza a testear:

- `M0`: marca central en la zona de calibracion.
- `M1`: esquina o margen inferior-izquierdo de la zona util.
- `M2`: esquina o margen inferior-derecho.
- `M3`: esquina o margen superior-derecho.
- `M4`: esquina o margen superior-izquierdo.

Reglas:

- Las 5 marcas no deben estar alineadas.
- Deben cubrir la mayor area razonable de la zona que se va a pulir.
- Usar la minima energia que deje una marca medible por el profilometro.
- Registrar las coordenadas ordenadas en Samlight para cada marca.
- No mover la pieza entre la medicion inicial, el marcado y la medicion posterior.

## Paso 3: medir las marcas con el profilometro

1. Medir la pieza despues de marcar.
2. Localizar el centro de cada marca `M0` a `M4`.
3. Convertir cada centro medido al sistema `(O, X, Y)` usando la convencion anterior.
4. Registrar cada punto medido como `(measured_x_ref_mm, measured_y_ref_mm)`.

## Paso 4: calcular la correccion para Samlight

Para corregir futuras trayectorias, se debe obtener una transformacion que convierta coordenadas de referencia de la pieza a coordenadas de comando en Samlight:

```text
S = T(R)

R = punto deseado en coordenadas de referencia de la pieza
S = punto que debe enviarse a Samlight
```

Usar las marcas medidas asi:

- `R_i`: posicion real medida de la marca `i` en el sistema `(O, X, Y)`.
- `S_i`: posicion que se ordeno a Samlight para esa marca.
- Ajustar una transformacion `T` que lleve `R_i -> S_i`.

Modelo recomendado:

- Primero probar rotacion + traslacion.
- Si el error residual es mayor que la tolerancia, probar transformacion afin.
- Usar transformacion afin si aparece cizalla, escala distinta por eje o efecto de no perpendicularidad.

Forma afin:

```text
Sx = a*x_ref + b*y_ref + tx
Sy = c*x_ref + d*y_ref + ty
```

Con 5 marcas hay redundancia suficiente para estimar la transformacion y medir el error residual.

## Criterio de aceptacion

Tolerancia inicial recomendada: `0.05 mm`, salvo que el proceso de laser requiera otro valor.

Aceptar la calibracion si:

- El error maximo de las marcas de calibracion es menor o igual que la tolerancia.
- El error RMS tambien es menor o igual que la tolerancia.
- La validacion con una linea o rectangulo pequeno cae dentro de tolerancia.
- La repetibilidad se mantiene durante al menos 3 ciclos.

Si rotacion + traslacion cumple, usar ese modelo por ser mas simple.

Si rotacion + traslacion no cumple pero afin si cumple, usar afin.

Si afin no cumple, repetir el ciclo completo y revisar:

- Pieza movida entre mediciones.
- Puntos de linea mal seleccionados.
- Marca laser mal detectada.
- Cambio de escala o unidades entre profilometro y Samlight.
- Zona marcada demasiado pequena para estimar bien la transformacion.

## Paso 5: prueba de repetibilidad

Repetir al menos 3 ciclos:

```text
profilometro -> laser -> profilometro
```

Para cada ciclo:

1. Registrar la captura de profilometro antes del laser.
2. Crear o validar marcas de calibracion en posiciones conocidas.
3. Medir las marcas despues del laser.
4. Calcular error maximo y error RMS.
5. Registrar si el modelo usado fue rigido o afin.

Si no conviene remarcar exactamente encima de las mismas marcas, repetir el patron con un pequeno desplazamiento en una zona de margen.

## Paso 6: validacion antes de polishing

Antes de lanzar un patron de polishing completo:

1. Generar una trayectoria simple en una zona no critica: una linea corta o un rectangulo pequeno.
2. Transformar esa trayectoria con la correccion aceptada.
3. Enviar la trayectoria corregida a Samlight.
4. Medirla con el profilometro.
5. Confirmar que el error maximo esta dentro de tolerancia.

Solo despues de esta validacion se debe usar el mismo modelo para trayectorias de polishing completas.

## Archivos de registro incluidos

- `plantilla_puntos_referencia.csv`: puntos usados para ajustar la linea larga y la segunda linea.
- `plantilla_marcas_calibracion.csv`: marcas ordenadas en Samlight y medidas en el profilometro.
- `plantilla_repetibilidad.csv`: resumen de errores por ciclo.

