"""
Valida el motor (AcumuladorEventos + clasificador) contra los 15 pares
reales Hornero/Kestrel reconstruidos, partiendo cada clip en dos
'bloques' artificiales para simular que el evento cruza un limite de
bloque de grabacion real -- exactamente el escenario que el diseño del
acumulador tiene que resolver. Compara contra el baseline ingenuo
(clasificar cada mitad por separado, como hace BirdNET-Pi sin ningun
contexto).

Actualizado el 29/08/2026 para usar ClasificadorTectorNet (Perch2+BirdSet
ONNX) en vez del Clasificador BirdNET tflite retirado -- el mecanismo que
se prueba aca (AcumuladorEventos cruzando bloques) no cambio con el
cambio de modelo, sigue siendo agnostico al clasificador.
"""
import sys, os, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import librosa
import huggingface_hub

from detector import cargar_config, AcumuladorEventos, SR
from clasificador_tectornet import ClasificadorTectorNet, cargar_config_tectornet

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'config_deteccion.txt')
config = cargar_config(CONFIG_PATH)
config.update(cargar_config_tectornet(CONFIG_PATH))

perch_path = huggingface_hub.hf_hub_download(repo_id='justinchuby/Perch-onnx', filename='perch_v2_no_dft.onnx')
clasificador = ClasificadorTectorNet(
    perch_onnx_path=perch_path,
    perch_labels_csv=os.path.join(BASE_DIR, 'modelo/perch2_labels.csv'),
    birdset_onnx_path=os.path.join(BASE_DIR, 'modelo/birdset_efficientnetb1.onnx'),
    birdset_config_json=os.path.join(BASE_DIR, 'modelo/birdset_efficientnetb1_config.json'),
    ebird_taxonomy_json=os.path.join(BASE_DIR, 'modelo/eBird_taxonomy_codes_2024E.json'),
    config=config,
    bias_json=os.path.join(BASE_DIR, 'config/delta_b_reforzado_v2.json'),
    escala=config['ESCALA'],
)

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
    r_a = clasificador.clasificar_evento(bloque_a, paso_s=config['PASO_VENTANA_S'])
    r_b = clasificador.clasificar_evento(bloque_b, paso_s=config['PASO_VENTANA_S'])
    baseline_dice_hornero = any(
        r and r.get('especie') and 'Furnarius rufus' in r['especie'] and r['detectado']
        for r in (r_a, r_b)
    )

    # --- motor: acumulador cruzando el "bloque" artificial ---
    acumulador = AcumuladorEventos(config, duracion_bloque_s=len(bloque_a) / SR)
    acumulador.procesar_bloque(bloque_a)
    acumulador.procesar_bloque(bloque_b)
    acumulador.finalizar()  # por si el evento seguia abierto (no hubo silencio final)

    if not acumulador.eventos_terminados:
        nuevo_dice_hornero = False
        nuevo_r = None
    else:
        evento = max(acumulador.eventos_terminados, key=lambda e: len(e['audio']))['audio']  # el evento principal
        nuevo_r = clasificador.clasificar_evento(evento, paso_s=config['PASO_VENTANA_S'])
        nuevo_dice_hornero = bool(nuevo_r and nuevo_r.get('especie') and 'Furnarius rufus' in nuevo_r['especie'] and nuevo_r['detectado'])

    total += 1
    correctos_baseline += baseline_dice_hornero
    correctos_nuevo += nuevo_dice_hornero

    detalle_nuevo = ''
    if nuevo_r is not None and nuevo_r.get('especie'):
        detalle_nuevo = f"{nuevo_r['especie']} {nuevo_r['confianza']:.3f}"

    nombre = os.path.basename(f)
    print(f'{nombre}')
    print(f'  baseline (cada mitad sola): {"HORNERO" if baseline_dice_hornero else "no detecta Hornero"}')
    print(f'  motor (evento completo, {len(acumulador.eventos_terminados)} evento(s), {detalle_nuevo}): '
          f'{"HORNERO" if nuevo_dice_hornero else "no detecta Hornero"}\n')

print(f'===== Resumen ({total} casos) =====')
print(f'Baseline (mitades sueltas, como hoy):  {correctos_baseline}/{total} ({100*correctos_baseline/total:.1f}%)')
print(f'Motor (evento completo cruzando bloque): {correctos_nuevo}/{total} ({100*correctos_nuevo/total:.1f}%)')
