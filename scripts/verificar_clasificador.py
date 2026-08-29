"""
Smoke-test del clasificador real, usado por chequear_salud() en
actualizar_birdnet_lsd.sh (ver ahi el porque: "el servicio esta activo y
arecord esta corriendo" no detecta un clasificador colgado o demasiado
lento, que es exactamente el modo de falla nuevo que introduce
TectorNet -- ver motor.py, docstring del modulo).

Carga el clasificador real (mismos archivos de modelo/config que usa
motor.py) y corre UNA clasificacion completa sobre un audio de prueba
real y corto, ya commiteado en el repo (scripts/test_audio/muestra_salud.mp3,
2.1s, no depende de que exista una deteccion previa en el disco del
dispositivo). Se invoca con "timeout 60" desde bash, no con try/except
interno -- un try/except no protege contra un cuelgue real de
onnxruntime.run(), solo contra una excepcion; hace falta el timeout
externo del proceso para agarrar ambos casos.

Sale con exit code 0 si la clasificacion se completo sin excepcion
(no importa que especie haya "ganado" -- esto valida que el pipeline
corre, no la exactitud del modelo). Cualquier excepcion, o que el
proceso ni siquiera llegue a imprimir "OK" antes de que timeout lo mate,
cuenta como fallo.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'scripts'))

import librosa
import huggingface_hub

from clasificador_tectornet import ClasificadorTectorNet, cargar_config_tectornet
from detector import cargar_config
from motor import PERCH2_REPO, PERCH2_CHECKPOINT

SR_MOTOR = 48000
AUDIO_PRUEBA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_audio', 'muestra_salud.mp3')


def main():
    config = cargar_config(os.path.join(BASE_DIR, 'config/config_deteccion.txt'))
    config.update(cargar_config_tectornet(os.path.join(BASE_DIR, 'config/config_deteccion.txt')))

    perch_path = huggingface_hub.hf_hub_download(repo_id=PERCH2_REPO, filename=PERCH2_CHECKPOINT)
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

    audio_48k, _ = librosa.load(AUDIO_PRUEBA, sr=SR_MOTOR, mono=True)
    resultado = clasificador.clasificar_evento(audio_48k, paso_s=config['PASO_VENTANA_S'])
    assert 'detectado' in resultado, 'formato de resultado invalido'

    print(f"OK -- clasificacion de prueba completa (especie de prueba: {resultado.get('especie_comun')})")


if __name__ == '__main__':
    main()
