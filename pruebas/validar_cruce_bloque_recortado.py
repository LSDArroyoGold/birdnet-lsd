"""
Repite la validacion de cruce de bloque, pero esta vez recortando cada uno
de los dos clips originales a su tramo de señal activa real (mismo metodo
ya validado: pasa-banda + umbral relativo al piso de ruido propio) antes
de pasarlos como 'bloques' al acumulador -- para no arrastrar el silencio
interno de cada clip ya recortado por BirdNET-Pi, que es lo que causaba
el cierre prematuro en la prueba anterior (validar_cruce_bloque.py).

Actualizado el 29/08/2026 para usar ClasificadorTectorNet (Perch2+BirdSet
ONNX) en vez del Clasificador BirdNET tflite retirado -- el recorte a
señal activa y el acumulador no cambiaron con el cambio de modelo.
"""
import sys, os, glob, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from datetime import datetime
import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt
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

# ---- reencontrar los mismos 15 pares (misma logica que buscar_consecutivos.py) ----
BASE = os.path.expanduser('~/Desktop/Tector/Datasets_prueba/BirdNET_Detecciones')
patron = re.compile(r'^(.+)-(\d+)-(\d{4}-\d{2}-\d{2})-birdnet-(\d{2})[_:](\d{2})[_:](\d{2})\.mp3$')

def parsear(path):
    m = patron.match(os.path.basename(path))
    if not m:
        return None
    especie, conf, fecha, hh, mm, ss = m.groups()
    dt = datetime.strptime(f'{fecha} {hh}:{mm}:{ss}', '%Y-%m-%d %H:%M:%S')
    return {'path': path, 'especie': especie, 'conf': int(conf), 'dt': dt}

detecciones = []
for especie in ['American_Kestrel', 'Rufous_Hornero']:
    for f in glob.glob(os.path.join(BASE, '*', especie, '*.mp3')):
        r = parsear(f)
        if r:
            detecciones.append(r)
detecciones.sort(key=lambda r: r['dt'])

pares = []
for i in range(len(detecciones) - 1):
    a, b = detecciones[i], detecciones[i + 1]
    if a['especie'] == b['especie']:
        continue
    gap = (b['dt'] - a['dt']).total_seconds()
    if 0 < gap <= 10:
        pares.append((a, b))

print(f'{len(pares)} pares encontrados\n')

# ---- recorte a señal activa (mismo metodo validado) ----
FRAME, HOP = 1024, 256
sos = butter(4, [config['TRIGGER_BANDA_MIN_HZ'], config['TRIGGER_BANDA_MAX_HZ']], btype='band', fs=SR, output='sos')

def recortar_a_senal_activa(y, margen_s=0.15, percentil_piso=20, margen_db=8):
    filtrado = sosfiltfilt(sos, y)
    rms = librosa.feature.rms(y=filtrado, frame_length=FRAME, hop_length=HOP)[0]
    rms_db = 20 * np.log10(np.maximum(rms, 1e-10))
    piso = np.percentile(rms_db, percentil_piso)
    activo = rms_db > (piso + margen_db)
    if not activo.any():
        return y  # no se detecto nada, dejar como esta
    tiempos = librosa.frames_to_time(np.arange(len(activo)), sr=SR, hop_length=HOP)
    ini = max(0, tiempos[np.argmax(activo)] - margen_s)
    fin = min(len(y) / SR, tiempos[len(activo) - 1 - np.argmax(activo[::-1])] + (HOP / SR) + margen_s)
    return y[int(ini * SR):int(fin * SR)]

base_hornero = base_kestrel = base_ambas = 0
nuevo_hornero = nuevo_kestrel = nuevo_ninguna = 0
total = 0

def es(r, nombre_cientifico):
    return bool(r and r.get('especie') and nombre_cientifico in r['especie'] and r['detectado'])

for a, b in pares:
    ya, _ = librosa.load(a['path'], sr=SR, mono=True)
    yb, _ = librosa.load(b['path'], sr=SR, mono=True)
    ya_rec = recortar_a_senal_activa(ya)
    yb_rec = recortar_a_senal_activa(yb)

    r_a = clasificador.clasificar_evento(ya_rec, paso_s=config['PASO_VENTANA_S'])
    r_b = clasificador.clasificar_evento(yb_rec, paso_s=config['PASO_VENTANA_S'])
    baseline_hornero = any(es(r, 'Furnarius rufus') for r in (r_a, r_b))
    baseline_kestrel = any(es(r, 'Falco sparverius') for r in (r_a, r_b))

    acumulador = AcumuladorEventos(config, duracion_bloque_s=len(ya_rec) / SR)
    acumulador.procesar_bloque(ya_rec)
    acumulador.procesar_bloque(yb_rec)
    acumulador.finalizar()

    if not acumulador.eventos_terminados:
        nuevo_r = None
    else:
        evento = max(acumulador.eventos_terminados, key=lambda e: len(e['audio']))['audio']
        nuevo_r = clasificador.clasificar_evento(evento, paso_s=config['PASO_VENTANA_S'])

    nuevo_es_hornero = es(nuevo_r, 'Furnarius rufus')
    nuevo_es_kestrel = es(nuevo_r, 'Falco sparverius')

    total += 1
    base_hornero += baseline_hornero
    base_kestrel += baseline_kestrel
    base_ambas += (baseline_hornero and baseline_kestrel)
    nuevo_hornero += nuevo_es_hornero
    nuevo_kestrel += nuevo_es_kestrel
    nuevo_ninguna += not (nuevo_es_hornero or nuevo_es_kestrel)

    detalle = f"{nuevo_r['especie']} {nuevo_r['confianza']:.3f}" if (nuevo_r and nuevo_r.get('especie')) else ''
    print(f"{a['dt']}  {a['especie']}(c{a['conf']}) + {b['especie']}(c{b['conf']})  "
          f"[recortados a {len(ya_rec)/SR:.2f}s + {len(yb_rec)/SR:.2f}s]")
    print(f"  baseline (piezas sueltas): Hornero={'si' if baseline_hornero else 'no'}  Kestrel={'si' if baseline_kestrel else 'no'}"
          f"{'  <- las DOS a la vez' if (baseline_hornero and baseline_kestrel) else ''}")
    print(f"  motor ({detalle}): "
          f"{'HORNERO' if nuevo_es_hornero else ('KESTREL' if nuevo_es_kestrel else 'ninguna')}\n")

print(f'===== Resumen ({total} casos) — lo ideal seria Hornero=15, Kestrel=0 =====')
print(f'Baseline (piezas sueltas, como hoy):')
print(f'  Hornero detectado: {base_hornero}/{total}   Kestrel (falso positivo) detectado: {base_kestrel}/{total}   ambas a la vez: {base_ambas}/{total}')
print(f'Motor (evento completo, una sola respuesta):')
print(f'  Hornero detectado: {nuevo_hornero}/{total}   Kestrel (falso positivo) detectado: {nuevo_kestrel}/{total}   ninguna: {nuevo_ninguna}/{total}')
