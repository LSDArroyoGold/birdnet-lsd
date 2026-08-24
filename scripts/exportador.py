"""
Arma el nombre de archivo y la carpeta de destino para una deteccion,
usando exactamente la misma convencion que BirdNET-Pi (extract_detection()
en scripts/utils/reporting.py) para que el motor nuevo sea un reemplazo
directo -- todo lo que ya cuenta/sincroniza detecciones parseando ese
patron (cierre_amanecer.sh, cierre_atardecer.sh, sincronizar_detecciones.sh)
sigue funcionando sin tocar nada.
"""
import os


def nombre_archivo(resultado, timestamp_inicio, extension='mp3'):
    """resultado: el dict que devuelve Clasificador.decidir_confianza_racha
    (necesita 'especie_comun' y 'confianza'). timestamp_inicio: datetime
    real del inicio del evento (viene de AcumuladorEventos, propagado desde
    procesar_bloque(..., timestamp_bloque=...)).

    Devuelve (nombre_archivo, carpeta_relativa), ej:
    ('Rufous_Hornero-92-2026-08-23-birdnet-09_52_26.mp3',
     'By_Date/2026-08-23/Rufous_Hornero')
    """
    if timestamp_inicio is None:
        raise ValueError(
            'timestamp_inicio_evento es None -- procesar_bloque() se llamo '
            'sin timestamp_bloque. En produccion siempre hace falta pasarlo '
            '(la hora real de inicio de cada bloque de grabacion).'
        )

    nombre_comun_seguro = resultado['especie_comun'].replace("'", "").replace(' ', '_')
    confianza_pct = round(resultado['confianza'] * 100)
    fecha = timestamp_inicio.strftime('%Y-%m-%d')
    hora = timestamp_inicio.strftime('%H_%M_%S')

    nombre = f'{nombre_comun_seguro}-{confianza_pct}-{fecha}-birdnet-{hora}.{extension}'
    carpeta_relativa = os.path.join('By_Date', fecha, nombre_comun_seguro)
    return nombre, carpeta_relativa
