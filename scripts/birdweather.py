"""
POST a la API publica de BirdWeather -- misma logica real que
bird_weather() en scripts/utils/reporting.py de BirdNET-Pi (confirmado
leyendo su codigo el 22/08): dos POST HTTP planos a un endpoint publico,
sin nada especifico de BirdNET-Pi ni de su base de datos. Primero se sube
el audio del evento como "soundscape" (formato FLAC), y con el id que
devuelve esa respuesta se postea la deteccion en si.
"""
import io
import requests
import soundfile as sf

BASE_URL = 'https://app.birdweather.com/api/v1/stations'


def enviar_deteccion(config_bw, resultado, timestamp_inicio, audio, sr, timeout_soundscape=30, timeout_deteccion=20):
    """config_bw: dict con BIRDWEATHER_ID, LATITUDE, LONGITUDE (ver
    config/config_birdweather.txt.ejemplo). resultado: el dict que
    devuelve Clasificador.decidir_confianza_racha (necesita 'especie',
    'especie_comun', 'confianza'). audio/sr: el audio del evento ya
    delimitado por AcumuladorEventos -- el mismo que se guarda como
    evidencia, no hace falta generar nada aparte.

    Devuelve None si no hay BIRDWEATHER_ID configurado (estacion no
    conectada a BirdWeather, comportamiento identico al de BirdNET-Pi).
    Lanza RuntimeError si BirdWeather rechaza el soundscape."""
    station_id = (config_bw.get('BIRDWEATHER_ID') or '').strip()
    if not station_id:
        return None

    buf = io.BytesIO()
    sf.write(buf, audio, sr, format='FLAC')
    flac_data = buf.getvalue()

    iso_inicio = timestamp_inicio.isoformat()
    soundscape_url = f'{BASE_URL}/{station_id}/soundscapes?timestamp={iso_inicio}'
    resp_soundscape = requests.post(
        soundscape_url, data=flac_data, timeout=timeout_soundscape,
        headers={'Content-Type': 'audio/flac'},
    )
    sdata = resp_soundscape.json()
    if not sdata.get('success'):
        raise RuntimeError(f'BirdWeather rechazo el soundscape: {sdata.get("message")}')
    soundscape_id = sdata['soundscape']['id']

    especie_cientifica = resultado['especie'].split('_', 1)[0]
    detection_url = f'{BASE_URL}/{station_id}/detections'
    payload = {
        'timestamp': iso_inicio,
        'lat': config_bw['LATITUDE'],
        'lon': config_bw['LONGITUDE'],
        'soundscapeId': soundscape_id,
        'soundscapeStartTime': 0,
        'soundscapeEndTime': len(audio) / sr,
        'commonName': resultado['especie_comun'],
        'scientificName': especie_cientifica,
        'algorithm': 'lsd-birdnet-lsd-v10',
        'confidence': resultado['confianza'],
    }
    resp_deteccion = requests.post(detection_url, json=payload, timeout=timeout_deteccion)
    return resp_deteccion
