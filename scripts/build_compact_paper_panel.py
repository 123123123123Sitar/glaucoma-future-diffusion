from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/review/sigf_multitask_papila_contours_20260816"
OUTPUT = ROOT / "docs/figures/sigf_two_eyes_16_papila_contours.png"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    path = Path("/System/Library/Fonts/Supplemental") / name
    return ImageFont.truetype(str(path), size)


def main() -> None:
    tile_size = 230
    label_width = 145
    top = 54
    row_height = 270
    gap = 12
    width = label_width + 4 * tile_size + 3 * gap + 24
    height = top + 4 * row_height + 30
    canvas = Image.new("RGB", (width, height), "#f6f8fb")
    draw = ImageDraw.Draw(canvas)

    headings = ["Baseline", "Year 1", "Year 3", "Year 5"]
    for index, heading in enumerate(headings):
        x = label_width + index * (tile_size + gap) + tile_size // 2
        draw.text((x, 18), heading, fill="#172033", font=font(22, True), anchor="ma")

    rows = [
        ("C02", "REAL", 106),
        ("C02", "GENERATED", 476),
        ("C04", "REAL", 106),
        ("C04", "GENERATED", 476),
    ]
    for row_index, (case_id, kind, source_y) in enumerate(rows):
        source = Image.open(SOURCE / case_id / "real_vs_generated_rows.png").convert("RGB")
        y = top + row_index * row_height
        draw.rounded_rectangle((10, y, label_width - 12, y + tile_size), radius=14, fill="#e2ecfa")
        draw.text((label_width // 2 - 1, y + 82), f"EYE {1 if case_id == 'C02' else 2}",
                  fill="#1f5b9f", font=font(23, True), anchor="mm")
        draw.text((label_width // 2 - 1, y + 126), kind,
                  fill="#172033", font=font(18, True), anchor="mm")
        for column in range(4):
            source_x = 184 + column * 314
            crop = source.crop((source_x, source_y, source_x + 300, source_y + 300))
            crop = crop.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
            x = label_width + column * (tile_size + gap)
            canvas.paste(crop, (x, y))
            draw.rectangle((x, y, x + tile_size - 1, y + tile_size - 1), outline="#ccd4df", width=1)
        note = "Observed follow-ups" if kind == "REAL" else "Model forecasts from baseline"
        draw.text((label_width, y + tile_size + 7), note, fill="#566273", font=font(15))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, quality=95)
    print(OUTPUT)


if __name__ == "__main__":
    main()
