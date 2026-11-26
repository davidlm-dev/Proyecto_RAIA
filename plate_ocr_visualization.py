"""Gráficas de datos y evaluación para el ejemplo de reconocimiento de matrículas."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from sklearn.metrics import ConfusionMatrixDisplay, classification_report


def save_and_show(
    figure: plt.Figure,
    filename: str,
    output_dir: Path,
) -> None:
    figure.tight_layout()
    figure.savefig(output_dir / filename, dpi=150)
    if plt.get_backend().lower() not in {
        "agg",
        "cairo",
        "pdf",
        "ps",
        "svg",
        "template",
    }:
        plt.show()
    plt.close(figure)


def plot_dataset_info(
    train: dict[str, object],
    test: dict[str, object],
    output_dir: Path,
) -> None:
    train_counts = Counter(str(label) for label in train["labels"])
    test_counts = Counter(str(label) for label in test["labels"])
    characters = sorted(set(train_counts) | set(test_counts))
    positions = np.arange(len(characters))

    figure, axis = plt.subplots(figsize=(13, 5))
    axis.bar(
        positions - 0.2,
        [train_counts[char] for char in characters],
        width=0.4,
        label="Train",
    )
    axis.bar(
        positions + 0.2,
        [test_counts[char] for char in characters],
        width=0.4,
        label="Test",
    )
    axis.set_xticks(positions, characters)
    axis.set_title("Distribución de caracteres: train y test")
    axis.set_xlabel("Carácter")
    axis.set_ylabel("Número de recortes")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    save_and_show(figure, "frecuencia_caracteres.png", output_dir)

    labels = train["labels"]
    figure, axes = plt.subplots(3, 8, figsize=(13, 5))
    for axis, crop, label in zip(axes.flat, train["crops"][:24], labels[:24]):
        axis.imshow(crop, cmap="gray")
        axis.set_title(str(label))
        axis.axis("off")
    for axis in axes.flat[min(24, len(labels)) :]:
        axis.axis("off")
    figure.suptitle("Recortes de caracteres usados para entrenar")
    save_and_show(figure, "ejemplos_caracteres.png", output_dir)


def plot_evaluation(
    test: dict[str, object],
    result: dict[str, object],
    output_dir: Path,
) -> None:
    y_test = test["labels"]
    predictions = result["predictions"]
    classes = sorted(str(label) for label in set(y_test))
    figure, axis = plt.subplots(figsize=(11, 9))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=classes,
        display_labels=classes,
        normalize="true",
        values_format=".0%",
        cmap="Blues",
        colorbar=False,
        ax=axis,
    )
    axis.set_title("Matriz de confusión normalizada — HOG + SVM (test)")
    axis.set_xlabel("Carácter predicho")
    axis.set_ylabel("Carácter real")
    save_and_show(figure, "matriz_confusion.png", output_dir)

    report_labels = sorted(str(label) for label in set(y_test))
    print("\nDetalle por carácter (precision, recall y F1):")
    print(
        classification_report(
            y_test,
            predictions,
            labels=report_labels,
            target_names=report_labels,
            digits=3,
            zero_division=0,
        )
    )

    plate_results = result["plate_results"]
    errors = [item for item in plate_results if not item["correct"]]
    print(f"\nLecturas completas incorrectas: {len(errors)} / {len(plate_results)}")
    if errors:
        print(f"{'Imagen':<12} {'Real':<14} {'Predicción':<14}")
        for item in errors[:10]:
            print(
                f"{item['image'] + '#' + str(item['number']):<12} "
                f"{item['truth']:<14} {item['prediction']:<14}"
            )
        if len(errors) > 10:
            print(f"... y {len(errors) - 10} errores más.")

    wrong_characters = np.flatnonzero(y_test != predictions)
    correct_characters = np.flatnonzero(y_test == predictions)
    selected_indices = np.concatenate(
        (wrong_characters[:12], correct_characters[:12])
    )
    figure, axes = plt.subplots(4, 6, figsize=(13, 7))
    for axis, index in zip(axes.flat, selected_indices):
        axis.imshow(test["crops"][index], cmap="gray")
        axis.set_title(
            f"Real: {y_test[index]} | Pred: {predictions[index]}",
            color="green" if y_test[index] == predictions[index] else "red",
            fontsize=8,
        )
        axis.axis("off")
    for axis in axes.flat[len(selected_indices) :]:
        axis.axis("off")
    figure.suptitle("Caracteres de test: primero errores y luego aciertos")
    save_and_show(figure, "ejemplos_lecturas.png", output_dir)


def _plate_crop(source: Image.Image, polygon: list[tuple[int, int]]) -> Image.Image:
    if len(polygon) != 4:
        x_values = [point[0] for point in polygon]
        y_values = [point[1] for point in polygon]
        return source.crop(
            (min(x_values), min(y_values), max(x_values), max(y_values))
        )

    top_left, top_right, bottom_right, bottom_left = polygon
    width = max(
        np.hypot(top_right[0] - top_left[0], top_right[1] - top_left[1]),
        np.hypot(bottom_right[0] - bottom_left[0], bottom_right[1] - bottom_left[1]),
    )
    height = max(
        np.hypot(bottom_left[0] - top_left[0], bottom_left[1] - top_left[1]),
        np.hypot(bottom_right[0] - top_right[0], bottom_right[1] - top_right[1]),
    )
    quad = (
        top_left[0],
        top_left[1],
        bottom_left[0],
        bottom_left[1],
        bottom_right[0],
        bottom_right[1],
        top_right[0],
        top_right[1],
    )
    return source.transform(
        (max(1, round(width)), max(1, round(height))),
        Image.Transform.QUAD,
        quad,
        resample=Image.Resampling.BICUBIC,
    )


def plot_example_steps(
    test: dict[str, object],
    result: dict[str, object],
    output_dir: Path,
) -> None:
    plate_results = result["plate_results"]
    example_index = next(
        (
            index
            for index, item in enumerate(plate_results)
            if not item["correct"]
        ),
        0,
    )
    plate_result = plate_results[example_index]
    group = test["plate_groups"][example_index]
    predictions = result["predictions"]
    char_indices = group["indices"]
    polygon = group["polygon"]

    with Image.open(group["image_path"]) as image_file:
        source = image_file.convert("RGB")
        scale = min(1400 / source.width, 900 / source.height, 1)
        preview = source.resize(
            (round(source.width * scale), round(source.height * scale)),
            Image.Resampling.LANCZOS,
        )
        draw = ImageDraw.Draw(preview)
        if polygon:
            scaled_polygon = [
                (round(x * scale), round(y * scale)) for x, y in polygon
            ]
            draw.polygon(scaled_polygon, outline="red", width=5)
        for number, (x1, y1, x2, y2) in enumerate(group["boxes"], start=1):
            rectangle = tuple(round(value * scale) for value in (x1, y1, x2, y2))
            draw.rectangle(rectangle, outline="yellow", width=3)
            draw.text(
                (rectangle[0], max(0, rectangle[1] - 18)),
                str(number),
                fill="yellow",
            )
        plate_image = _plate_crop(source, polygon) if polygon else source.crop(
            (
                min(box[0] for box in group["boxes"]),
                min(box[1] for box in group["boxes"]),
                max(box[2] for box in group["boxes"]),
                max(box[3] for box in group["boxes"]),
            )
        )

    figure = plt.figure(figsize=(14, 9))
    grid = figure.add_gridspec(3, 2, height_ratios=(1.1, 0.9, 0.55))
    photo_axis = figure.add_subplot(grid[0, 0])
    plate_axis = figure.add_subplot(grid[0, 1])
    photo_axis.imshow(preview)
    photo_axis.set_title("1. Foto: la caja roja marca la matrícula")
    photo_axis.axis("off")
    plate_axis.imshow(plate_image)
    plate_axis.set_title("2. Recorte de la matrícula completa")
    plate_axis.axis("off")

    character_grid = grid[1, :].subgridspec(1, len(char_indices))
    for position, char_index in enumerate(char_indices):
        axis = figure.add_subplot(character_grid[0, position])
        axis.imshow(test["crops"][char_index], cmap="gray")
        truth = test["labels"][char_index]
        prediction = predictions[char_index]
        axis.set_title(
            f"{truth} → {prediction}",
            color="green" if truth == prediction else "red",
        )
        axis.axis("off")

    result_axis = figure.add_subplot(grid[2, 0])
    result_axis.axis("off")
    result_axis.text(
        0,
        0.9,
        "3. HOG + SVM clasifica cada carácter recortado",
        fontsize=12,
        weight="bold",
    )
    result_axis.text(
        0,
        0.45,
        "4. Se unen las predicciones, de izquierda a derecha",
        fontsize=11,
    )

    plate_axis_result = figure.add_subplot(grid[2, 1])
    plate_axis_result.axis("off")
    plate_axis_result.text(
        0,
        0.9,
        f"Imagen {plate_result['image']} — matrícula {plate_result['number']}",
        fontsize=11,
        weight="bold",
    )
    plate_axis_result.text(
        0,
        0.45,
        f"Anotada: {plate_result['truth']}    Predicha: {plate_result['prediction']}",
        fontsize=11,
        color="green" if plate_result["correct"] else "darkred",
    )
    figure.suptitle(
        "Ejemplo completo de reconocimiento (las cajas vienen del dataset)",
        fontsize=15,
    )
    save_and_show(figure, "paso_a_paso.png", output_dir)
