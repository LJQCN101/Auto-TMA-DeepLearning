from __future__ import annotations

from pathlib import Path
import unittest

import cv2
import numpy as np

from auto_tma.vision import detect_blue_bearing_lines, detect_candidate_lines, reduce_lines


def _make_synthetic_blue_lines_image() -> np.ndarray:
    img = np.zeros((300, 420, 3), dtype=np.uint8)
    # Draw several distinct blue-ish lines (BGR)
    for i in range(6):
        x0 = 30 + i * 22
        y0 = 260 - i * 14
        cv2.line(img, (x0, y0), (x0 + 95, y0 - 140), (200 + i * 4, 50 + i * 2, 20), 2)
    return img


class VisionPipelineTests(unittest.TestCase):
    def test_sample_image_has_reduced_lines(self) -> None:
        image_path = Path(__file__).resolve().parents[1] / "TMA_opencv" / "sc_review20.jpg"
        if not image_path.exists():
            self.skipTest(f"sample image not present: {image_path}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        self.assertIsNotNone(image)
        lines = detect_candidate_lines(image)
        reduced = reduce_lines(lines)

        self.assertGreater(len(lines), 0)
        self.assertGreater(len(reduced), 0)
        self.assertLessEqual(len(reduced), len(lines))

    def test_blue_mask_detects_synthetic_blue_lines(self) -> None:
        img = _make_synthetic_blue_lines_image()
        # baseline should find some
        base_lines = detect_candidate_lines(img)
        self.assertGreater(len(base_lines), 0)

        # blue path should find comparable or better targeted set
        blue_lines = detect_candidate_lines(img, blue_mask=True, min_line_length=30, hough_threshold=40)
        self.assertGreaterEqual(len(blue_lines), 1)

        reduced = reduce_lines(blue_lines)
        self.assertGreater(len(reduced), 0)
        self.assertLessEqual(len(reduced), len(blue_lines))

    def test_detect_blue_bearing_lines_wrapper(self) -> None:
        img = _make_synthetic_blue_lines_image()
        lines = detect_blue_bearing_lines(img)
        self.assertIsInstance(lines, list)
        reduced = reduce_lines(lines)
        self.assertGreaterEqual(len(reduced), 1)


if __name__ == "__main__":
    unittest.main()