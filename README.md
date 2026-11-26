# Lectura de matrículas

Este proyecto entrena **un solo modelo: HOG + SVM lineal** para reconocer los
caracteres anotados en las imágenes de UC3M-LP. **OCR** viene de *Optical
Character Recognition*: reconocimiento óptico de caracteres, es decir, convertir
texto de una imagen en texto digital.

El programa usa las cajas de matrícula y de caracteres del dataset. Evalúa el
reconocimiento, pero todavía no detecta matrículas por sí solo en una foto nueva.

## Ejecutarlo

Desde esta carpeta:

```bash
../../entorno/bin/python license_plate_ocr.py
```

El script carga los splits oficiales, recorta los caracteres, los normaliza a
24×32 píxeles, calcula HOG, entrena el SVM con `train` y evalúa las lecturas con
`test`.

## Cómo leer los resultados

En `outputs/`:

- `paso_a_paso.png`: ejemplo de una **matrícula entera**. Muestra la foto, el
  recorte de matrícula, los caracteres reconocidos y la lectura final. Se elige
  automáticamente del test, normalmente entre las lecturas con algún error.
- `matriz_confusion.png`: muestra qué caracteres se confunden; la diagonal son
  aciertos y cada fila suma 100%.
- `ejemplos_lecturas.png`: recortes de caracteres de test, primero los errores.
- `frecuencia_caracteres.png` y `ejemplos_caracteres.png`: distribución y muestras
  del dataset.
- `lecturas_matriculas.txt`: las 491 lecturas de test, con anotación, predicción y
  resultado, además del tiempo total.

El porcentaje por carácter cuenta caracteres correctos; una matrícula completa
solo cuenta como correcta si todos sus caracteres lo son. En esta ejecución se
obtuvieron 95,57% por carácter y 421 de 491 matrículas completas correctas
(85,74%). Tardó 6 min 44 s en este ordenador; el tiempo aparece al final de la
ejecución y en el TXT, y puede variar según el equipo.

El modelo se guarda en `artifacts/plate_character_svm.joblib`. `outputs/` y
`artifacts/` están excluidas por `.gitignore` y se regeneran al ejecutar.
