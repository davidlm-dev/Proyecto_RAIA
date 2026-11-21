"""Reconocimiento de caracteres de matrículas usando UC3M-LP, HOG y SVM."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

from plate_ocr_visualization import (
    plot_dataset_info,
    plot_example_steps,
    plot_evaluation,
)


PROJECT_DIR = Path(__file__).parent
DATASET_DIR = PROJECT_DIR.parent / "datasets" / "UC3M-LP"
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_PATH = PROJECT_DIR / "artifacts" / "plate_character_svm.joblib"

CHAR_WIDTH = 24
CHAR_HEIGHT = 32
CELL_SIZE = 4
ORIENTATION_BINS = 9


def hog_features(character: Image.Image) -> np.ndarray:
    """Calcula HOG: histogramas de gradientes por celda y bloques normalizados."""
    gray = ImageOps.grayscale(character).resize(
        (CHAR_WIDTH, CHAR_HEIGHT), Image.Resampling.LANCZOS
    )
    pixels = np.asarray(gray, dtype=np.float32) / 255.0
    gradient_y, gradient_x = np.gradient(pixels)
    magnitude = np.hypot(gradient_x, gradient_y)
    angle = np.degrees(np.arctan2(gradient_y, gradient_x)) % 180

    cell_rows = CHAR_HEIGHT // CELL_SIZE
    cell_columns = CHAR_WIDTH // CELL_SIZE
    cell_y, cell_x = np.indices((CHAR_HEIGHT, CHAR_WIDTH))
    bins = np.floor(angle * ORIENTATION_BINS / 180).astype(int)
    bins = np.minimum(bins, ORIENTATION_BINS - 1)
    histogram_indices = (
        (cell_y // CELL_SIZE * cell_columns + cell_x // CELL_SIZE)
        * ORIENTATION_BINS
        + bins
    )
    histograms = np.bincount(
        histogram_indices.ravel(),
        weights=magnitude.ravel(),
        minlength=cell_rows * cell_columns * ORIENTATION_BINS,
    ).reshape(cell_rows, cell_columns, ORIENTATION_BINS)

    blocks: list[np.ndarray] = []
    for row in range(cell_rows - 1):
        for column in range(cell_columns - 1):
            block = histograms[row : row + 2, column : column + 2].ravel().copy()
            block /= np.sqrt(np.dot(block, block) + 1e-6)
            block = np.minimum(block, 0.2)
            block /= np.sqrt(np.dot(block, block) + 1e-6)
            blocks.append(block)
    return np.concatenate(blocks)


def read_split(split: str) -> dict[str, object]:
    """Carga las cajas de caracteres anotadas en un split y extrae sus HOG."""
    split_dir = DATASET_DIR / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"No existe la carpeta del split: {split_dir}")

    features: list[np.ndarray] = []
    labels: list[str] = []
    crops: list[np.ndarray] = []
    plate_groups: list[dict[str, object]] = []
    skipped_images = 0
    plate_count = 0

    annotation_paths = sorted(split_dir.glob("*.json"))
    print(f"\nCargando {split}: {len(annotation_paths)} anotaciones...", flush=True)
    for annotation_path in annotation_paths:
        # En esta copia de UC3M-LP, el JPG correcto comparte nombre con el JSON;
        # el campo imagePath puede señalar al siguiente fichero.
        image_path = annotation_path.with_suffix(".jpg")
        if not image_path.is_file():
            print(f"Aviso: falta la imagen asociada: {image_path}")
            skipped_images += 1
            continue

        with annotation_path.open("r", encoding="utf-8") as file:
            annotation = json.load(file)

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            for plate_index, plate in enumerate(annotation.get("lps", [])):
                ordered_characters = sorted(
                    plate.get("characters", []),
                    key=lambda character: sum(
                        point[0] for point in character["bbox_coord"]
                    )
                    / len(character["bbox_coord"]),
                )
                group_indices: list[int] = []
                group_labels: list[str] = []
                group_boxes: list[tuple[int, int, int, int]] = []

                for character in ordered_characters:
                    points = character["bbox_coord"]
                    x1 = max(0, min(point[0] for point in points))
                    y1 = max(0, min(point[1] for point in points))
                    x2 = min(image.width, max(point[0] for point in points))
                    y2 = min(image.height, max(point[1] for point in points))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = image.crop((x1, y1, x2, y2))
                    label = str(character["char_id"]).upper()
                    group_indices.append(len(features))
                    group_labels.append(label)
                    group_boxes.append((x1, y1, x2, y2))
                    gray_crop = ImageOps.grayscale(crop).resize(
                        (CHAR_WIDTH, CHAR_HEIGHT), Image.Resampling.LANCZOS
                    )
                    features.append(hog_features(crop))
                    labels.append(label)
                    crops.append(np.asarray(gray_crop))

                if group_indices:
                    plate_groups.append(
                        {
                            "indices": group_indices,
                            "truth": "".join(group_labels),
                            "image": annotation_path.stem,
                            "image_path": str(image_path),
                            "number": plate_index + 1,
                            "boxes": group_boxes,
                            "polygon": [
                                (int(point[0]), int(point[1]))
                                for point in plate.get("poly_coord", [])
                            ],
                        }
                    )
                    plate_count += 1

    if not features:
        raise ValueError(f"No se han encontrado caracteres etiquetados en {split_dir}")

    print(
        f"{split}: {len(annotation_paths)} anotaciones, "
        f"{plate_count} matrículas, {len(labels)} caracteres, "
        f"{skipped_images} imágenes no encontradas."
    )
    return {
        "features": np.asarray(features, dtype=np.float32),
        "labels": np.asarray(labels),
        "crops": np.asarray(crops),
        "plate_groups": plate_groups,
    }


def evaluate_model(
    model: LinearSVC,
    test: dict[str, object],
) -> dict[str, object]:
    y_test = test["labels"]
    predictions = model.predict(test["features"])
    plate_results: list[dict[str, object]] = []
    for group in test["plate_groups"]:
        plate_prediction = "".join(
            predictions[index] for index in group["indices"]
        )
        plate_results.append(
            {
                "image": group["image"],
                "number": group["number"],
                "truth": group["truth"],
                "prediction": plate_prediction,
                "correct": plate_prediction == group["truth"],
            }
        )

    return {
        "predictions": predictions,
        "character_accuracy": accuracy_score(y_test, predictions),
        "plate_accuracy": np.mean(
            [result["correct"] for result in plate_results]
        ),
        "plate_results": plate_results,
    }


def save_plate_predictions(
    result: dict[str, object],
    elapsed_seconds: float,
) -> None:
    plate_results = result["plate_results"]
    correct_count = sum(item["correct"] for item in plate_results)
    minutes, seconds = divmod(elapsed_seconds, 60)
    with (OUTPUT_DIR / "lecturas_matriculas.txt").open(
        "w", encoding="utf-8"
    ) as file:
        file.write("LECTURAS DE MATRÍCULAS — UC3M-LP TEST\n")
        file.write("=" * 66 + "\n")
        file.write(
            "La lectura se construye con los caracteres anotados en el dataset.\n"
            "REAL contiene los caracteres visibles con caja y etiqueta.\n\n"
        )
        file.write(
            f"Total: {len(plate_results)} | Correctas: {correct_count} | "
            f"Incorrectas: {len(plate_results) - correct_count} | "
            f"Acierto: {result['plate_accuracy']:.2%}\n\n"
        )
        file.write(f"Tiempo total de ejecución: {int(minutes)} min {seconds:.1f} s\n\n")
        file.write(f"{'IMAGEN':<14} {'REAL':<16} {'PREDICCIÓN':<16} RESULTADO\n")
        file.write("-" * 66 + "\n")
        for item in plate_results:
            status = "CORRECTA" if item["correct"] else "INCORRECTA"
            image_name = f"{item['image']}#{item['number']}"
            file.write(
                f"{image_name:<14} {item['truth']:<16} "
                f"{item['prediction']:<16} {status}\n"
            )
def main() -> None:
    start_time = perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 66)
    print("LECTURA DE MATRÍCULAS — UC3M-LP")
    print("=" * 66)
    print(f"Dataset: {DATASET_DIR}")
    train = read_split("train")
    test = read_split("test")
    print(f"Clases de caracteres: {len(set(train['labels']))}")
    print(f"Representación: recortes en gris de {CHAR_WIDTH} × {CHAR_HEIGHT} píxeles.")
    plot_dataset_info(train, test, OUTPUT_DIR)

    model = LinearSVC(
        C=1.0, class_weight="balanced", max_iter=5000, random_state=21
    )
    print("\nEntrenando el modelo HOG + SVM lineal...", flush=True)
    model.fit(train["features"], train["labels"])
    result = evaluate_model(model, test)
    print("\n" + "=" * 58)
    print("RESULTADOS EN TEST — HOG + SVM LINEAL")
    print("=" * 58)
    print(f"Acierto por carácter: {result['character_accuracy']:.2%}")
    print(
        f"Matrículas completas correctas: "
        f"{sum(item['correct'] for item in result['plate_results'])}/"
        f"{len(result['plate_results'])} ({result['plate_accuracy']:.2%})"
    )
    plot_evaluation(test, result, OUTPUT_DIR)
    plot_example_steps(test, result, OUTPUT_DIR)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "image_size": (CHAR_WIDTH, CHAR_HEIGHT),
            "feature": "HOG",
        },
        MODEL_PATH,
    )
    print(f"Modelo guardado en: {MODEL_PATH}")
    print(f"Gráficas guardadas en: {OUTPUT_DIR}")
    elapsed_seconds = perf_counter() - start_time
    minutes, seconds = divmod(elapsed_seconds, 60)
    save_plate_predictions(result, elapsed_seconds)
    print(f"Listado de lecturas: {OUTPUT_DIR / 'lecturas_matriculas.txt'}")
    print(f"Tiempo total de ejecución: {int(minutes)} min {seconds:.1f} s")


if __name__ == "__main__":
    main()
