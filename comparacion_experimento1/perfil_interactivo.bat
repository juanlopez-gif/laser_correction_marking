@echo off
rem Comparador interactivo de perfiles PRE/POST
rem La calibracion SAMLight se detecta automaticamente si has exportado
rem un perfil CSV desde el interface de calibracion manual.
rem
rem Uso:
rem   perfil_interactivo.bat <pre.csv> <post.csv>
rem
rem Ejemplo:
rem   perfil_interactivo.bat ..\csv_entrada\prueba1_steel_Height.csv ..\csv_entrada\postlinea1_steel_Height.csv

set PY311=C:\Users\mss\AppData\Local\Programs\Python\Python311\python.exe
"%PY311%" "%~dp0compare_perfil_interactivo.py" %*
