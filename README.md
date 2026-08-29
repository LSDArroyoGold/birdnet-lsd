# birdnet-lsd

Motor de deteccion propio para el LSD-Tector, en reemplazo del loop de
analisis de BirdNET-Pi.

## Cambio de modelo (29/08/2026)

El clasificador de especies cambio por completo. **BirdNET reentrenado
(v10, tflite)** -- descripto en el resto de este README como el modelo
original -- fue retirado: revision manual encontro que sobre-disparaba de
forma estructural (etiquetaba practicamente cualquier cosa como Hornero,
no un problema de calibracion fina que el podado/sesgo pudieran arreglar
del todo). El codigo completo del modelo viejo sigue disponible en el
historial de git (tag `pre-tectornet`) y en el repo espejo
[`birdnet-lsd-v1`](https://github.com/LSDArroyoGold/birdnet-lsd-v1).

Reemplazo: **TectorNet** -- Perch 2.0 ONNX (decisor, 14795 especies) +
BirdSet EfficientNetB1 ONNX (filtro de confirmacion cruzada, sin sesgo),
ambos cargados directo con `onnxruntime`, sin TensorFlow ni PyTorch.
Prototipo y arquitectura original en
[`TectorNet`](https://github.com/LSDArroyoGold/TectorNet) (de un
colaborador nuevo del equipo); portado e integrado aca con ajustes
adicionales de rendimiento encontrados al validar en hardware real
(tector1, Pi4B) el mismo dia -- ver `scripts/clasificador_tectornet.py`
para el detalle completo de la arquitectura y las notas de por que cada
decision se tomo asi. Las secciones "Como funciona" (puntos 3-5) y
"Validacion" mas abajo describen el modelo VIEJO -- se mantienen como
registro historico, no como estado actual. Validacion del modelo nuevo
en curso, a medida que tector1 acumula detecciones reales con el.

## Por que existe

Revisando el codigo real de BirdNET-Pi (`scripts/utils/analysis.py`) se
confirmo que no tiene ningun mecanismo de confirmacion entre ventanas: la
primera ventana de 3 segundos que cruza el umbral de confianza queda
logueada como deteccion final, sin mirar las ventanas vecinas ni el resto
del canto. Esto genera confusiones reales y reproducibles (documentadas a
fondo el 22-23/08 en el repo de reentreno, caso Hornero/Kestrel) cuando una
sola fase de un canto largo se parece acusticamente a otra especie.

Ademas, `extract_safe()` (el audio que se guarda como evidencia de cada
deteccion) esta limitado al mismo bloque de grabacion de 15s en el que cayo
la deteccion, y BirdNET-Pi borra cada bloque crudo apenas lo procesa -- no
hay forma de recuperar el canto completo si empieza cerca del final de un
bloque y sigue en el siguiente.

## Como funciona

En vez de clasificar cada ventana fija de forma aislada, el flujo es:

1. **Disparador liviano** (`DetectorActividad`): un chequeo de energia en
   banda 1.5-8kHz (rango que maximiza el contraste canto/ruido, medido
   empiricamente el 23/08 sobre 195 especies de Xeno-canto vs. ruido de
   campo real), relativo al piso de ruido propio del momento -- no corre
   el modelo, es barato, puede correr sobre todo el audio entrante.
2. **Acumulacion del evento completo** (`AcumuladorEventos`): una vez
   disparado, se junta el audio real (con buffer rodante de los bloques
   anteriores, para reconstruir el inicio del canto y cruzar el limite de
   bloque si hace falta) hasta que hay silencio sostenido o se llega a un
   tope de seguridad (30s, calibrado con la distribucion real de duracion
   de canto de 195 especies).
3. **Clasificacion una sola vez** (`Clasificador.decidir_confianza_racha`):
   ventanas de 3s solapadas (paso 1s) sobre el evento completo. Cada racha
   de ventanas consecutivas ganadas por la misma especie se resume en un
   promedio ponderado por posicion dentro de la racha (la ultima ventana
   de una racha larga pesa mas que la primera). El umbral de CONFIDENCE se
   aplica reciEn sobre esa confianza ya agregada -- un canto real de una
   sola ventana (menos de ~6s) sigue pudiendo confirmarse solo, con su
   propia confianza, sin necesitar compania.
4. **Sesgo regional aplicado en codigo, no horneado en el modelo**: el
   modelo v10 no tiene el sesgo regional (eso vive en el .tflite de
   `custom-v1`, un archivo distinto). Se suma `delta_b` (de
   `stock_regional_meta.json`) al logit crudo de cada especie antes del
   sigmoide, por nombre cientifico -- aplica igual a la neurona
   reentrenada y a la original.
5. **v10 podado** (`modelo/v10_podado.json`): 4 de las 14 neuronas
   reentrenadas de v10 mostraron sobreajuste al audio de campo (buen
   recall en campo, caida real en Xeno-canto externo, ver validacion mas
   abajo) -- para esas 4 especies se suprime la neurona reentrenada
   (siempre score 0) y se usa la neurona original de BirdNET sin tocar,
   que ya convive en el mismo modelo Append. No requiere modificar el
   `.tflite`, es una mascara aplicada en cada corrida.

## Validacion (22-23/08/2026) -- modelo VIEJO, ver nota al principio del README

Motor completo (acumulador + ventanas solapadas + confianza_racha + sesgo
+ podado) contra:

- **15 pares reales Hornero/Kestrel** (el caso mas dificil encontrado,
  canto real partido en dos ventanas por BirdNET-Pi, cada mitad leida
  como una especie distinta): **15/15 Hornero, 0/15 Kestrel** (con las
  neuronas de v10; el modelo stock+sesgo, sin las neuronas reentrenadas,
  nunca supero un empate en el mismo test, sin importar el criterio de
  decision probado -- ver el detalle completo en el repo de reentreno).
- **14 especies reentrenadas, campo real (BirdNET_Detecciones) + Xeno-canto
  externo (AMBA_test, ground truth independiente de cualquier modelo)**:
  resultados consistentes en ambos datasets para las 10 especies
  mantenidas reentrenadas; las 4 podadas mostraron mejora real en campo
  pero caida en Xeno-canto, señal de sobreajuste al sitio especifico (ver
  motivo detallado por especie en `modelo/v10_podado.json`).

## Sincronizacion (BirdWeather + Drive, event-driven)

Ni BirdWeather ni Drive se sincronizan por ciclo periodico -- las dos
subidas pasan a ser parte de la misma cadena de eventos que dispara cada
deteccion, en `scripts/motor.py`: clasificar -> guardar mp3 -> avisar a
BirdWeather -> subir a Drive, en ese orden, apenas se cierra el evento.
No hay que esperar a que un cron corra de nuevo ni a que termine la
ventana de grabacion.

Esto reemplaza dos mecanismos periodicos anteriores:

- `sincronizar_detecciones.sh` de LSD-Tector1.1, corriendo cada 5 minutos
  por crontab mientras hay ventana activa (`rclone copy` del arbol
  completo, filtrado a `*.mp3`).
- El `rclone copy` del arbol completo al cierre de cada ventana en
  LSD-Tector2.0 (`cierre_amanecer.sh`/`cierre_atardecer.sh`).

`scripts/drive.py` sube un solo archivo por vez (`rclone copyto`, no un
barrido del arbol entero) al mismo destino que ya usan esos scripts
(`gdrive:$DRIVE_PATH/Detecciones/<fecha>/<especie>/...` -- sin "By_Date",
que es una carpeta puramente local: `rclone copy .../By_Date/ destino`
copia el contenido de esa carpeta, no la carpeta en si, asi que nunca
llegaba a Drive), asi que si algun barrido periodico viejo sigue
corriendo por fuera no rompe nada -- encuentra el archivo ya arriba y no
hace nada de mas (`rclone copy` es idempotente).
Mismo timeout defensivo de 90s que el resto de las llamadas a rclone del
proyecto, por la cuota compartida de Google (ver nota al respecto en el
README de LSD-Tector1.1).

## Que falta (no incluido todavia)

- Nada del loop de deteccion en si -- motor, exportacion, BirdWeather y
  Drive ya estan integrados end-to-end en `scripts/motor.py`.
- Rotacion/limpieza de `motor.log` (hoy crece sin limite -- si el
  dispositivo corre meses seguidos conviene sumar logrotate).
- Sacar el barrido periodico redundante de LSD-Tector1.1/2.0 una vez que
  se confirme que la sincronizacion event-driven es estable en campo (por
  ahora conviven sin conflicto, ver seccion anterior).

## Instalacion

```bash
git clone https://github.com/LSDArroyoGold/birdnet-lsd.git
cd birdnet-lsd
bash instalar.sh
```

Arma un venv dedicado e instala las dependencias (sin TensorFlow ni
PyTorch). BirdSet ya viene commiteado en `modelo/` (no tiene host publico
propio todavia); Perch2 se descarga solo del cache de HuggingFace la
primera vez que corre. El propio `instalar.sh` verifica que el
clasificador carga y clasifica de punta a punta al final (mismo chequeo
que usa `scripts/actualizar_birdnet_lsd.sh` para decidir si un update
rompio algo).

Completar `config/config_birdweather.txt` y `config/config_sincronizacion.txt`
a partir de sus `.ejemplo` (datos del dispositivo, no se commitean).

Para que corra en segundo plano y sobreviva reinicios:

```bash
bash instalar_servicio.sh
sudo systemctl start birdnet-lsd.service
journalctl -u birdnet-lsd.service -f
```

## Estructura

- `scripts/detector.py` -- `DetectorActividad` (disparador) y
  `AcumuladorEventos` (acumulacion cruzando bloques). Sin cambios con el
  cambio de modelo -- es agnostico al clasificador.
- `scripts/clasificador_tectornet.py` -- `ClasificadorTectorNet`: Perch2
  decide (racha ponderada, mismo mecanismo que el `Clasificador` viejo),
  BirdSet confirma sin sesgo sobre el mismo tramo de audio de la racha
  ganadora. Ver el docstring del modulo para el detalle completo.
- `scripts/preprocess_birdset.py` -- audio -> espectrograma mel de
  BirdSet, reimplementado en `librosa` puro (sin torch).
- `scripts/motor.py` -- loop principal en vivo: captura con `arecord` en
  un hilo, clasificacion (Perch2+BirdSet, mas pesada que el tflite viejo)
  en OTRO hilo aparte via una cola -- ver el docstring del modulo, seccion
  "Cambio de arquitectura", para por que esto dejo de ser un solo hilo.
- `scripts/verificar_clasificador.py` -- smoke-test del clasificador real
  sobre `scripts/test_audio/muestra_salud.mp3` (audio real corto,
  commiteado). Usado por `instalar.sh` y por el chequeo de salud de
  `actualizar_birdnet_lsd.sh` -- no valida precision, valida que el
  pipeline corre sin crashear ni colgarse.
- `scripts/audio_io.py` -- escritura del audio del evento a mp3 (320kbps,
  sin lowpass) via `ffmpeg`.
- `scripts/exportador.py` -- nombre de archivo y carpeta, mismo patron
  que BirdNET-Pi (`extract_detection()`), para que el resto del sistema
  que ya cuenta/parsea detecciones siga funcionando sin tocar nada.
- `scripts/birdweather.py` / `scripts/drive.py` -- POST a BirdWeather y
  subida a Drive de una deteccion, ambos event-driven (ver seccion de
  sincronizacion mas arriba).
- `scripts/actualizar_birdnet_lsd.sh` -- git pull + reinstalar
  dependencias si `requirements.txt` cambio + chequeo de salud real
  (`verificar_clasificador.py`, con timeout) + rollback automatico si
  algo rompio. Corre en cada ventana de amanecer/atardecer.
- `config/config_deteccion.txt` -- todos los parametros ajustables (banda
  del disparador, umbrales, tope de duracion, buffer de bloques, captura,
  mas ESCALA/PASO_VENTANA_S especificos de TectorNet).
- `config/delta_b_reforzado_v2.json` -- sesgo regional de Perch2 (por
  nombre cientifico directo, ALPHA=0.7).
- `config/config_birdweather.txt.ejemplo` / `config/config_sincronizacion.txt.ejemplo`
  -- plantillas de datos especificos del dispositivo (no se commitean).
- `systemd/birdnet-lsd.service` + `instalar_servicio.sh` -- arranque
  automatico y reinicio solo (`Restart=always`) si el proceso muere.
- `modelo/` -- `birdset_efficientnetb1.onnx` (commiteado, 72MB, sin host
  publico propio todavia), `birdset_efficientnetb1_config.json`,
  `perch2_labels.csv`, `eBird_taxonomy_codes_2024E.json`. Perch2 en si
  (`perch_v2_no_dft.onnx`) NO esta commiteado -- se descarga solo del
  cache de HuggingFace la primera vez que corre `motor.py`.
- `pruebas/` -- `validar_cruce_bloque.py`/`validar_cruce_bloque_recortado.py`
  (prueban el `AcumuladorEventos`, agnostico al modelo -- actualizados
  para usar `ClasificadorTectorNet`) y `comparar_criterios_decision.py`
  (RETIRADO, no se porto -- comparaba 4 formas de decision especificas
  del `Clasificador` viejo que `ClasificadorTectorNet` no expone, ver su
  docstring).
