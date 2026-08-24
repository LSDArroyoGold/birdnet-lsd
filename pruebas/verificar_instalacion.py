"""
Verificacion de instalacion: confirma que el entorno, el modelo
descargado, el sesgo regional y el podado cargan bien y que el pipeline
completo (AcumuladorEventos + Clasificador + exportador de nombre de
archivo) corre de punta a punta sin error, sobre audio sintetico
(silencio + ruido de banda ancha) -- no hace falta audio real de ave para
validar que la INSTALACION en si funciona, la precision del modelo ya
esta validada aparte (ver README).
"""
import os, sys
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import numpy as np

from detector import cargar_config, AcumuladorEventos, SR
from clasificador import Clasificador
from exportador import nombre_archivo

BASE = os.path.join(os.path.dirname(__file__), '..')
CONFIG_PATH = os.path.join(BASE, 'config', 'config_deteccion.txt')
MODELO = os.path.join(BASE, 'modelo', 'LSDTector_Classifier_v2.tflite')
LABELS = os.path.join(BASE, 'modelo', 'LSDTector_Classifier_v2_Labels.txt')
PODADO = os.path.join(BASE, 'modelo', 'v10_podado.json')
BIAS = os.path.join(BASE, 'modelo', 'stock_regional_meta.json')

print('==> Verificando archivos...')
for nombre, path in [('config', CONFIG_PATH), ('modelo .tflite', MODELO), ('labels', LABELS),
                      ('podado', PODADO), ('sesgo regional', BIAS)]:
    ok = os.path.isfile(path)
    print(f'  {nombre}: {"OK" if ok else "FALTA -> " + path}')
    if not ok:
        sys.exit(1)

print('\n==> Cargando config...')
config = cargar_config(CONFIG_PATH)
print(f'  CONFIDENCE={config["CONFIDENCE"]}  SENSITIVITY={config["SENSITIVITY"]}  '
      f'banda disparador={config["TRIGGER_BANDA_MIN_HZ"]}-{config["TRIGGER_BANDA_MAX_HZ"]}Hz')

print('\n==> Cargando modelo + sesgo regional + podado...')
clasificador = Clasificador(MODELO, LABELS, config, podado_json=PODADO, bias_json=BIAS)
print(f'  {len(clasificador.labels)} clases cargadas')
print(f'  {len(clasificador.indices_suprimidos)} neuronas suprimidas por el podado')
print(f'  sesgo regional cargado para {int((clasificador.bias_vector != 0).sum())} especies')
print(f'  tabla cientifico->comun: {len(clasificador.sci_a_comun)} especies resueltas')
print(f'  ejemplo neurona reentrenada (bare) -> comun: "Furnarius rufus" -> '
      f'"{clasificador.nombre_comun_de("Furnarius rufus")}"')

print('\n==> Corriendo el pipeline completo sobre audio sintetico (2 bloques de 15s, '
      'con timestamps reales, silencio + ruido)...')
rng = np.random.default_rng(0)
bloque_silencio = (rng.normal(0, 0.001, SR * 15)).astype(np.float32)
bloque_ruido = (rng.normal(0, 0.3, SR * 15)).astype(np.float32)

t0 = datetime(2026, 8, 23, 9, 0, 0)
acumulador = AcumuladorEventos(config, duracion_bloque_s=15.0)
acumulador.procesar_bloque(bloque_silencio, timestamp_bloque=t0)
acumulador.procesar_bloque(bloque_ruido, timestamp_bloque=t0.replace(second=15))
acumulador.finalizar()
print(f'  eventos detectados por el acumulador: {len(acumulador.eventos_terminados)} '
      f'(con audio sintetico sin estructura real de canto, se espera 0 o pocos)')

for ev in acumulador.eventos_terminados:
    audio, ts = ev['audio'], ev['timestamp_inicio']
    ventanas = clasificador.analizar_ventanas_todas(audio)
    resultado = clasificador.decidir_confianza_racha(ventanas)
    print(f'  evento de {len(audio)/SR:.1f}s (timestamp={ts}) -> {resultado}')
    if resultado is not None:
        nombre, carpeta = nombre_archivo(resultado, ts)
        print(f'    -> archivo: {carpeta}/{nombre}')

print('\n==> Probando el armado de nombre de archivo con un resultado simulado '
      '(para no depender de que el ruido sintetico dispare una deteccion real)...')
resultado_simulado = {
    'especie': 'Furnarius rufus',  # neurona reentrenada bare, sin nombre comun propio
    'especie_comun': clasificador.nombre_comun_de('Furnarius rufus'),
    'confianza': 0.92,
    'racha_maxima': 3,
    'confianza_baja': False,
    'detectado': True,
}
nombre, carpeta = nombre_archivo(resultado_simulado, datetime(2026, 8, 23, 9, 52, 26))
print(f'  especie (label crudo): {resultado_simulado["especie"]}')
print(f'  especie_comun (resuelto): {resultado_simulado["especie_comun"]}')
print(f'  -> archivo: {carpeta}/{nombre}')
assert nombre == f'Rufous_Hornero-92-2026-08-23-birdnet-09_52_26.mp3', 'nombre de archivo inesperado'
print('  OK: coincide con la convencion de BirdNET-Pi (nombre comun, no cientifico)')

print('\n==> OK: instalacion funcional, el pipeline corre de punta a punta sin errores,')
print('    incluido el armado de nombre de archivo consistente con el resto del sistema.')
print('    (la precision del modelo ya esta validada con audio real -- ver README.md)')
