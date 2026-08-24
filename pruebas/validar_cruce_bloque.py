"""
Valida el motor nuevo (AcumuladorEventos + Clasificador) contra los 15
pares reales Hornero/Kestrel reconstruidos hoy, partiendo cada clip en dos
'bloques' artificiales para simular que el evento cruza un limite de
bloque de grabacion real -- exactamente el escenario que el diseño de hoy
tiene que resolver. Compara contra el baseline ingenuo (clasificar cada
mitad por separado, como hace BirdNET-Pi hoy sin ningun contexto).
"""
import sys, os, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import numpy as np
import librosa

from detector import cargar_config, AcumuladorEventos, SR
from clasificador import Clasificador

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'config_deteccion.txt')
config = cargar_config(CONFIG_PATH)

MODEL = os.path.expanduser('~/Desktop/Tector/LSDTector-BirdNET-custom-v1/model/BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite')
LABELS = os.path.expanduser('~/Desktop/Tector/LSDTector-BirdNET-custom-v1/model/BirdNET_GLOBAL_6K_V2.4_Model_FP16_Labels.txt')
clasificador = Clasificador(MODEL, LABELS, config)

carpeta = os.path.expanduser('~/Desktop/Tector/Datasets_prueba/hornero_kestrel_reconstruidos')
archivos = sorted(glob.glob(os.path.join(carpeta, '*.wav')))
print(f'{len(archivos)} pares reconstruidos, partiendo cada uno en 2 bloques artificiales\n')

correctos_baseline = 0
correctos_nuevo = 0
total = 0

for f in archivos:
    y, sr = librosa.load(f, sr=SR, mono=True)
    mitad = len(y) // 2
    bloque_a, bloque_b = y[:mitad], y[mitad:]

    # --- baseline ingenuo: cada mitad clasificada sola, como BirdNET-Pi hoy ---
    r_a = clasificador.clasificar_evento(bloque_a)
    r_b = clasificador.clasificar_evento(bloque_b)
    baseline_dice_hornero = any(
        r and 'Furnarius rufus' in r['especie'] and r['detectado']
        for r in (r_a, r_b)
    )

    # --- motor nuevo: acumulador cruzando el "bloque" artificial ---
    acumulador = AcumuladorEventos(config, duracion_bloque_s=len(bloque_a) / SR)
    acumulador.procesar_bloque(bloque_a)
    acumulador.procesar_bloque(bloque_b)
    acumulador.finalizar()  # por si el evento seguia abierto (no hubo silencio final)

    if not acumulador.eventos_terminados:
        nuevo_dice_hornero = False
        nuevo_r = None
    else:
        evento = max(acumulador.eventos_terminados, key=len)  # el evento principal
        nuevo_r = clasificador.clasificar_evento(evento)
        nuevo_dice_hornero = bool(nuevo_r and 'Furnarius rufus' in nuevo_r['especie'] and nuevo_r['detectado'])

    total += 1
    correctos_baseline += baseline_dice_hornero
    correctos_nuevo += nuevo_dice_hornero

    detalle_nuevo = ''
    if nuevo_r is not None:
        detalle_nuevo = f"{nuevo_r['especie']} {nuevo_r['confianza']:.3f}"

    nombre = os.path.basename(f)
    print(f'{nombre}')
    print(f'  baseline (cada mitad sola): {"HORNERO" if baseline_dice_hornero else "no detecta Hornero"}')
    print(f'  motor nuevo (evento completo, {len(acumulador.eventos_terminados)} evento(s), {detalle_nuevo}): '
          f'{"HORNERO" if nuevo_dice_hornero else "no detecta Hornero"}\n')

print(f'===== Resumen ({total} casos) =====')
print(f'Baseline (mitades sueltas, como hoy):  {correctos_baseline}/{total} ({100*correctos_baseline/total:.1f}%)')
print(f'Motor nuevo (evento completo cruzando bloque): {correctos_nuevo}/{total} ({100*correctos_nuevo/total:.1f}%)')
