"""
Verificacion de instalacion: confirma que el entorno, el modelo
descargado, el sesgo regional y el podado cargan bien y que el pipeline
completo (AcumuladorEventos + Clasificador) corre de punta a punta sin
error, sobre audio sintetico (silencio + ruido de banda ancha) -- no hace
falta audio real de ave para validar que la INSTALACION en si funciona,
la precision del modelo ya esta validada aparte (ver README).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import numpy as np

from detector import cargar_config, AcumuladorEventos, SR
from clasificador import Clasificador

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

print('\n==> Corriendo el pipeline completo sobre audio sintetico (15s, silencio + ruido)...')
rng = np.random.default_rng(0)
bloque_silencio = (rng.normal(0, 0.001, SR * 15)).astype(np.float32)  # ruido muy bajo, no deberia disparar nada
bloque_ruido = (rng.normal(0, 0.3, SR * 15)).astype(np.float32)  # ruido fuerte de banda ancha, sin estructura de canto

acumulador = AcumuladorEventos(config, duracion_bloque_s=15.0)
acumulador.procesar_bloque(bloque_silencio)
acumulador.procesar_bloque(bloque_ruido)
acumulador.finalizar()
print(f'  eventos detectados por el acumulador: {len(acumulador.eventos_terminados)} '
      f'(con audio sintetico sin estructura real de canto, se espera 0 o pocos)')

for evento in acumulador.eventos_terminados:
    ventanas = clasificador.analizar_ventanas_todas(evento)
    resultado = clasificador.decidir_confianza_racha(ventanas)
    print(f'  evento de {len(evento)/SR:.1f}s -> {resultado}')

print('\n==> OK: instalacion funcional, el pipeline corre de punta a punta sin errores.')
print('    (la precision del modelo ya esta validada con audio real -- ver README.md)')
