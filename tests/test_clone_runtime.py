"""Private environment construction, with no dependency on GPU libraries or CUDA."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from engine.install import RELATIVE, UPSTREAM_SHA
from scripts.clone_runtime import build, copy_tree, destination


class CloneRuntimeTests(unittest.TestCase):
    def test_overlap_existing_and_relative_python_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            public = root / "public"
            public.mkdir()
            shared = dict(prefix=str(public), base_prefix=str(public), package=str(public / "site/vllm"),
                          metadata=str(public / "site/wheel"), sites=[str(public / "site")])
            python = public / "bin/python"
            for target in [public, public / "private", root]:
                with self.subTest(target=target), self.assertRaises(ValueError):
                    destination(python, target, shared)
            with self.assertRaisesRegex(ValueError, "absolute"):
                destination(Path("bin/python"), root / "private", shared)
            self.assertEqual(destination(python, root / "private", shared), (root / "private").resolve())
            self.assertFalse((root / "private").exists())

    def test_regular_copies_and_bytecode_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "in", Path(tmp) / "out"
            src.mkdir()
            (src / "policy.py").write_text("original")
            (src / "__pycache__").mkdir()
            (src / "__pycache__/old.pyc").write_bytes(b"unrelated")
            copy_tree(src, dst)
            (dst / "policy.py").write_text("changed private copy")
            self.assertEqual((src / "policy.py").read_text(), "original")
            self.assertFalse((dst / "__pycache__").exists())
            self.assertFalse(os.path.samefile(src / "policy.py", dst / "policy.py"))

    @unittest.skipUnless(sys.platform.startswith("linux") and os.environ.get("MIXED_SLO_UPSTREAM_FIXTURE"),
                         "Linux and the SHA-pinned upstream scheduler are required")
    def test_real_source_venv_private_resolution_and_readonly_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            public, private = root / "public", root / "private"
            subprocess.run([sys.executable, "-I", "-B", "-m", "venv", "--without-pip", str(public)], check=True)
            python = public / "bin/python"
            site = Path(subprocess.check_output([str(python), "-I", "-B", "-c",
                        'import sysconfig; print(sysconfig.get_path("purelib"))'], text=True).strip())
            pkg = site / "vllm"
            scheduler = pkg / RELATIVE
            scheduler.parent.mkdir(parents=True)
            (pkg / "__init__.py").write_text("tag='synthetic-test-package'\n")
            scheduler.write_bytes(Path(os.environ["MIXED_SLO_UPSTREAM_FIXTURE"]).read_bytes())
            meta = site / "vllm-0.10.2.dist-info"
            meta.mkdir()
            (meta / "METADATA").write_text("Metadata-Version: 2.1\nName: vllm\nVersion: 0.10.2\n")
            (meta / "RECORD").write_text(meta.name + "/METADATA,,\n" + meta.name + "/RECORD,,\n")
            (site / "shared_clone_test_dep.py").write_text("marker=True\n")
            with self.assertRaisesRegex(ValueError, "disjoint"):
                build(python, public / "nested")
            manifest = build(python, private)
            self.assertEqual(manifest["source_prefix"], str(public))
            self.assertNotEqual(manifest["source_base_prefix"], manifest["source_prefix"])
            self.assertEqual(manifest["source_scheduler_sha256_after"], UPSTREAM_SHA)
            self.assertEqual(hashlib.sha256(scheduler.read_bytes()).hexdigest(), UPSTREAM_SHA)
            self.assertNotEqual(manifest["private_scheduler_sha256"], UPSTREAM_SHA)
            info = json.loads(subprocess.check_output([manifest["private_python"], "-c",
                "import json, os, sys, shared_clone_test_dep as dep, vllm; "
                "print(json.dumps([dep.__file__, vllm.__file__, sys.dont_write_bytecode, os.environ['TRITON_CACHE_DIR']]))"],
                text=True, env={**os.environ, "PYTHONPATH":str(site)}))
            self.assertEqual(Path(info[0]).parent, site)
            self.assertEqual(Path(info[1]).parent, Path(manifest["private_package"]))
            self.assertTrue(info[2])
            self.assertTrue(Path(info[3]).is_relative_to(private))
            self.assertFalse(list(site.rglob("shared_clone_test_dep*.pyc")))
            with self.assertRaisesRegex(ValueError, "already exists"):
                build(python, private)


if __name__ == "__main__":
    unittest.main()
