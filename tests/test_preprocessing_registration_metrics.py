from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glaucoma_forecast.data.preprocessing import detect_retinal_fov, locate_optic_disc, mirror_left_eye, optic_disc_crop, preprocess_image
from glaucoma_forecast.data.registration import AffineTransform, apply_translation, estimate_translation
from glaucoma_forecast.evaluation.anatomy_metrics import vcdr_from_masks
from glaucoma_forecast.evaluation.image_metrics import psnr


class PreprocessingRegistrationMetricTests(unittest.TestCase):
    def test_fov_detection_and_left_mirror(self) -> None:
        image = Image.new("RGB", (100, 100), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.ellipse((20, 10, 90, 90), fill=(120, 80, 60))
        box = detect_retinal_fov(image)
        self.assertLessEqual(box[0], 20)
        mirrored, did_mirror = mirror_left_eye(image, "L")
        self.assertTrue(did_mirror)
        self.assertEqual(mirrored.size, image.size)
        _, did_mirror_os = mirror_left_eye(image, "OS")
        self.assertTrue(did_mirror_os)

    def test_disc_location_and_crop(self) -> None:
        image = Image.new("RGB", (100, 100), (30, 20, 10))
        draw = ImageDraw.Draw(image)
        draw.ellipse((65, 35, 85, 55), fill=(250, 240, 180))
        center = locate_optic_disc(image)
        self.assertLess(abs(center[0] - 75), 8)
        self.assertLess(abs(center[1] - 45), 8)
        self.assertEqual(optic_disc_crop(image, center=center, output_size=32).size, (32, 32))

    def test_phase_translation(self) -> None:
        reference = Image.new("RGB", (64, 64), (0, 0, 0))
        draw = ImageDraw.Draw(reference)
        draw.ellipse((24, 20, 38, 36), fill=(230, 180, 120))
        moving = apply_translation(reference, 5, -3)
        shift_x, shift_y, confidence = estimate_translation(reference, moving, max_shift=10)
        aligned = apply_translation(moving, shift_x, shift_y)
        self.assertLess(np.mean(np.abs(np.asarray(reference, dtype=float) - np.asarray(aligned, dtype=float))), 3.0)
        self.assertGreater(confidence, 1.0)

    def test_preprocess_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "img.png"
            Image.new("RGB", (64, 64), (40, 30, 20)).save(path)
            result = preprocess_image(path, output_size=32, laterality="R")
            self.assertEqual(result.image.size, (32, 32))

    def test_affine_inverse(self) -> None:
        transform = AffineTransform(1, 0, 5, 0, 1, -2)
        inverse = transform.inverse()
        self.assertEqual(inverse.as_tuple(), (1.0, -0.0, -5.0, -0.0, 1.0, 2.0))

    def test_metrics(self) -> None:
        arr = np.zeros((8, 8), dtype=np.uint8)
        self.assertEqual(psnr(arr, arr), float("inf"))
        disc = np.zeros((10, 10), dtype=np.uint8)
        cup = np.zeros((10, 10), dtype=np.uint8)
        disc[2:8, 3:7] = 1
        cup[3:6, 4:6] = 1
        self.assertAlmostEqual(vcdr_from_masks(cup, disc), 3 / 6)


if __name__ == "__main__":
    unittest.main()
