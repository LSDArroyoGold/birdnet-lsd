# birdnet-lsd

Motor de deteccion propio para el LSD-Tector, en reemplazo del loop de
analisis de BirdNET-Pi. No es una red nueva -- usa el mismo modelo v10 de
[`LSDTector-BirdNET-retrain-bsas`](https://github.com/LSDArroyoGold/LSDTector-BirdNET-retrain-bsas)
mas el sesgo regional de [`LSDTector-BirdNET-custom-v1`](https://github.com/LSDArroyoGold/LSDTector-BirdNET-custom-v1) --
lo que cambia es la logica de decision alrededor del modelo.

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

## Validacion (22-23/08/2026)

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

## Que falta (no incluido todavia)

Esto es el motor de decision, validado con audio ya grabado. Falta:

- Integracion con la grabacion real (hoy simulado con bloques de audio
  pasados directamente a `AcumuladorEventos.procesar_bloque()`) -- en
  produccion séria via `arecord`/`ffmpeg` como hace BirdNET-Pi, o un loop
  propio.
- Extraccion y guardado del audio del evento como evidencia (el mecanismo
  ya resuelve el problema de raiz -- el evento completo ya esta delimitado
  correctamente -- falta el paso de escribirlo a disco con el nombre y la
  carpeta que espera el resto del sistema).
- POST a BirdWeather (es simple: dos POST HTTP a la API publica, ver
  `bird_weather()` en `scripts/utils/reporting.py` de BirdNET-Pi como
  referencia -- no depende de nada interno de BirdNET-Pi).
- systemd / arranque automatico / integracion con el ciclo de
  encendido-apagado del dispositivo.

## Instalacion

```bash
git clone https://github.com/LSDArroyoGold/birdnet-lsd.git
cd birdnet-lsd
bash instalar.sh
```

Descarga el modelo v10 y el sesgo regional desde los repos correspondientes
(no estan commiteados aca, pesan demasiado) y arma un venv dedicado.
Verificar que quedo bien instalado:

```bash
source venv/bin/activate
python3 pruebas/verificar_instalacion.py
```

## Estructura

- `scripts/detector.py` -- `DetectorActividad` (disparador) y
  `AcumuladorEventos` (acumulacion cruzando bloques).
- `scripts/clasificador.py` -- `Clasificador`: ventanas solapadas, sesgo
  regional, podado, y las distintas formas de decision probadas (pico,
  mayoria, confianza acumulada, racha, confianza+racha -- la usada).
- `config/config_deteccion.txt` -- todos los parametros ajustables (banda
  del disparador, umbrales, tope de duracion, buffer de bloques).
- `modelo/v10_podado.json` -- que neuronas de v10 quedan suprimidas y por
  que (commiteado; el resto de `modelo/` se descarga con `instalar.sh`).
- `pruebas/` -- scripts de validacion (algunos referencian datasets
  locales del desarrollo, no pensados para correr fuera de esa maquina;
  `verificar_instalacion.py` si es autocontenido y sirve para cualquier
  instalacion nueva).
