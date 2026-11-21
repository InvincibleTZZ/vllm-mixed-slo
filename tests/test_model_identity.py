from pathlib import Path
import tempfile
import unittest

from scripts.model_identity import model_identity


class ModelIdentityTests(unittest.TestCase):
    def test_changed_weights_are_visible_even_when_config_is_identical(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config.json").write_text("{}")
            with self.assertRaises(ValueError):
                model_identity(root)
            weight = root / "model.safetensors"
            weight.write_bytes(b"first")
            before = model_identity(root)
            weight.write_bytes(b"other")
            after = model_identity(root)
            self.assertEqual(before["files"]["config.json"], after["files"]["config.json"])
            self.assertNotEqual(before["sha256"], after["sha256"])
