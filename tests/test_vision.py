from __future__ import annotations

from pathlib import Path
import unittest

import cv2

from auto_tma.vision import detect_candidate_lines, reduce_lines


class VisionPipelineTests(unittest.TestCase):
    def test_sample_image_has_reduced_lines(self) -> None:
        image_path = Path(__file__).resolve().parents[1] / "TMA_opencv" / "sc_review20.jpg"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        self.assertIsNotNone(image)
        lines = detect_candidate_lines(image)
        reduced = reduce_lines(lines)

        self.assertGreater(len(lines), 0)
        self.assertGreater(len(reduced), 0)
        self.assertLessEqual(len(reduced), len(lines))


if __name__ == "__main__":
    unittest.main()