import os
import unittest
from unittest.mock import patch

from scripts.trial import check_gpu


class GPUPreflightTests(unittest.TestCase):
    def test_samples_again_after_last_context_exits(self):
        samples = ["A800, uuid, 550, 81920, 1, 100", "", "A800, uuid, 550, 81920, 1, 0", ""]
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="4"), patch("scripts.trial.smi", side_effect=samples), patch("scripts.trial.time.sleep") as wait:
            self.assertEqual(check_gpu(4)["name"], "A800")
            wait.assert_called_once_with(1)

    def test_existing_compute_process_blocks_acquisition(self):
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="4"), patch("scripts.trial.smi", side_effect=["A800, uuid, 550, 81920, 50, 0", "123"]), patch("scripts.trial.time.sleep") as wait:
            with self.assertRaisesRegex(RuntimeError, "already in use"):
                check_gpu(4)
            wait.assert_not_called()
