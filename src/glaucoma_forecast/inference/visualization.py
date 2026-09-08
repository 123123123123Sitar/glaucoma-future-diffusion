"""Inference visualization helpers."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def make_contact_sheet(image_paths: list[str], output_path: str | Path, tile_size: int = 192) -> None:
    images = [Image.open(path).convert("RGB").resize((tile_size, tile_size)) for path in image_paths]
    if not images:
        raise ValueError("No images supplied")
    sheet = Image.new("RGB", (tile_size * len(images), tile_size), (0, 0, 0))
    for idx, image in enumerate(images):
        sheet.paste(image, (idx * tile_size, 0))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
