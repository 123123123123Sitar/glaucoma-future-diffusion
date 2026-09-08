from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glaucoma_forecast.inference.contracts import BaselineExam
from glaucoma_forecast.inference.progression_predictor import predict_progression_3y


def _fundus(path: Path) -> None:
    image = Image.new("RGB", (768, 768), "black")
    draw = ImageDraw.Draw(image)
    draw.ellipse((30, 30, 738, 738), fill=(140, 65, 40))
    draw.ellipse((490, 310, 590, 410), fill=(230, 180, 100))
    image.save(path)


def test_baseline_exam_requires_61_visual_field_values() -> None:
    exam = BaselineExam(
        fundus_image="image.png", known_glaucoma=True, visual_field=[1.0] * 60
    )
    assert exam.validate() == ["visual_field_must_have_61_values"]


def test_missing_checkpoint_is_an_explicit_abstention() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        image = Path(temporary) / "fundus.png"
        _fundus(image)
        report = predict_progression_3y(
            BaselineExam(
                fundus_image=str(image),
                known_glaucoma=True,
                visual_field=[20.0] * 61,
            ),
            Path(temporary) / "out",
            checkpoints=[],
        )
        assert report["model_status"] == "progression_checkpoint_required"
        assert report["progression_probability_3y"] is None
        saved = json.loads(
            (Path(temporary) / "out" / "report.json").read_text()
        )
        assert saved["schema_version"] == "3.0"
