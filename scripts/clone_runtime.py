"""Create a private patched vLLM package that reads dependencies from a shared Linux interpreter."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.install import RELATIVE, UPSTREAM_SHA, operate, render


# Probe the selected interpreter itself. Resolving bin/python on a real venv
# would follow its symlink into the base installation and lose its site paths.
PROBE = r"""
import importlib.metadata as metadata, importlib.util, json, pathlib, sys, sysconfig
P = pathlib.Path
spec = importlib.util.find_spec("vllm")
if not spec or not spec.origin:
    sys.exit("vllm is missing from this interpreter's isolated import path")
dist = metadata.distribution("vllm")
metadata_files = [f for f in dist.files or []
                  if f.name == "METADATA" and f.parent.name.endswith(".dist-info")]
if len(metadata_files) != 1:
    sys.exit("expected a wheel installation of vllm with .dist-info/RECORD")
paths = sysconfig.get_paths()
stdlib = {P(paths[k]).resolve() for k in ("stdlib", "platstdlib")}
stdlib.update(p/"lib-dynload" for p in list(stdlib))
deps = [str(P(p).resolve()) for p in sys.path if p and P(p).is_absolute()
        and P(p).is_dir() and P(p).resolve() not in stdlib]
print(json.dumps(dict(prefix=sys.prefix, base_prefix=sys.base_prefix, purelib=paths["purelib"],
    package=str(P(spec.origin).resolve().parent), metadata=str(P(dist.locate_file(metadata_files[0])).resolve().parent),
    version=dist.version, sites=list(dict.fromkeys(deps)), dont_write_bytecode=sys.dont_write_bytecode)))
"""

CACHES = {
    "XDG_CACHE_HOME":"xdg", "PIP_CACHE_DIR":"pip", "TORCH_HOME":"torch", "TORCH_EXTENSIONS_DIR":"torch-extensions",
    "TORCHINDUCTOR_CACHE_DIR":"inductor", "TRITON_CACHE_DIR":"triton", "TRITON_HOME":"triton-home",
    "CUDA_CACHE_PATH":"cuda", "VLLM_CACHE_ROOT":"vllm", "VLLM_CONFIG_ROOT":"vllm-config", "HF_HOME":"hf",
    "HF_HUB_CACHE":"hf/hub", "HUGGINGFACE_HUB_CACHE":"hf/hub", "TRANSFORMERS_CACHE":"hf/hub",
    "HF_MODULES_CACHE":"hf/modules", "HF_ASSETS_CACHE":"hf/assets", "NUMBA_CACHE_DIR":"numba",
    "MPLCONFIGDIR":"matplotlib", "TMPDIR":"tmp",
}


def cache_environment(root):
    paths = {key:str(root / folder) for key, folder in CACHES.items()}
    for path in set(paths.values()):
        Path(path).mkdir(parents=True, exist_ok=True)
    return {**paths, "PYTHONDONTWRITEBYTECODE":"1"}


def invoke(python, args, overrides, capture=False):
    env = dict(os.environ)
    for key in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "__PYVENV_LAUNCHER__"):
        env.pop(key, None)
    return subprocess.run([str(python), *args], env={**env, **overrides}, check=True, text=True,
                          stdout=subprocess.PIPE if capture else None).stdout


def probe(python, env):
    result = json.loads(invoke(python, ["-I", "-B", "-c", PROBE], env, capture=True))
    if result["version"].split("+")[0] != "0.10.2":
        raise ValueError(f"requires vllm==0.10.2, found {result['version']}")
    return result


def destination(python, out, source):
    if not python.is_absolute():
        raise ValueError("--python must be an absolute interpreter path")
    if os.path.lexists(out):
        raise ValueError(f"--out already exists: {out}")
    target = out.resolve()
    protected = [source["prefix"], source["base_prefix"], source["package"], source["metadata"], *source["sites"]]
    for name in protected:
        shared = Path(name).resolve()
        if target.is_relative_to(shared) or shared.is_relative_to(target):
            raise ValueError(f"source and target must be disjoint: {target}, {shared}")
    return target


def copy_tree(source, target):
    # New files, never hardlinks. Private directories remain writable, including
    # when the original installation has been chmod'ed read-only.
    target.mkdir()
    def unreadable(error):
        raise error

    for base, dirs, names in os.walk(source, onerror=unreadable, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        parent = Path(base)
        if any((parent / d).is_symlink() for d in dirs):
            raise ValueError(f"directory symlink in the vllm installation: {parent}")
        new_parent = target / parent.relative_to(source)
        new_parent.mkdir(exist_ok=True)
        for name in sorted(names):
            if name.endswith((".pyc", ".pyo")):
                continue
            old, new = parent / name, new_parent / name
            if not old.is_file():
                raise ValueError(f"expected a regular installed file: {old}")
            shutil.copyfile(old, new)
            new.chmod((stat.S_IMODE(old.stat().st_mode) & 0o777) | stat.S_IWUSR | stat.S_IRUSR)


def record_private_site(site, source_sites, env):
    # Private startup also runs in vLLM's child Python processes.
    module = site / "_mixed_slo_private_site.py"
    module.write_text(
        "import os, sys\n"
        f"private = {str(site)!r}\n"
        "if private in sys.path:\n    sys.path.remove(private)\n"
        "sys.path.insert(0, private)\n"
        f"for path in {source_sites!r}:\n    if path not in sys.path:\n        sys.path.append(path)\n"
        "sys.dont_write_bytecode = True\n"
        f"os.environ.update({env!r})\n", encoding="utf-8")
    (site / "00_mixed_slo_private.pth").write_text("import _mixed_slo_private_site\n", encoding="utf-8")


def build(python, out):
    if not sys.platform.startswith("linux"):
        raise ValueError("requires Linux")
    if not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("--python must name an executable absolute interpreter path")
    if os.path.lexists(out):
        raise ValueError(f"--out already exists: {out}")
    # Avoid accidental cache writes even in ordinary startup .pth hooks of the trusted source.
    with tempfile.TemporaryDirectory(prefix="mixed-slo-source-probe-", dir="/tmp") as temp:
        source = probe(python, cache_environment(Path(temp)))
    target = destination(python, out, source)
    package = Path(source["package"])
    meta = Path(source["metadata"])
    if package.name != "vllm" or package.parent not in map(Path, source["sites"]):
        raise ValueError("expected a regular vllm wheel package on the selected interpreter's site path")
    scheduler = package / RELATIVE
    original = scheduler.read_bytes()
    if hashlib.sha256(original).hexdigest() != UPSTREAM_SHA:
        raise ValueError("shared scheduler is not the pinned unmodified vLLM 0.10.2 source")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(mode=0o700, exist_ok=False)
    try:
        env = cache_environment(target / "cache")
        # Build at the final absolute path. venv embeds it in activation scripts.
        invoke(python, ["-I", "-B", "-m", "venv", "--without-pip", str(target)], env)
        private_py = target / "bin" / "python"
        site_info = json.loads(invoke(private_py, ["-I", "-B", "-c",
            'import json,sys,sysconfig; print(json.dumps([sys.prefix,sysconfig.get_path("purelib")]))'],
            env, capture=True))
        private_site = Path(site_info[1]).resolve()
        if Path(site_info[0]).resolve() != target or not private_site.is_relative_to(target):
            raise ValueError("venv prefix or private site-packages escaped the new target")
        clone = private_site / "vllm"
        copy_tree(package, clone)
        copy_tree(meta, private_site / meta.name)
        operate("apply", clone)
        patched = (clone / RELATIVE).read_bytes()
        if patched != render(original):
            raise ValueError("private scheduler does not match the expected patch")
        record_private_site(private_site, source["sites"], env)
        private = probe(private_py, {})
        if (Path(private["package"]) != clone or Path(private["prefix"]).resolve() != target
                or Path(private["metadata"]) != private_site / meta.name or private["version"] != source["version"]):
            raise ValueError("private Python is importing vllm or wheel metadata from the wrong location")
        # Also check ordinary venv startup without -B and without a cache override from the caller.
        runtime_check = json.loads(invoke(private_py, ["-I", "-c",
            'import json,os,sys; print(json.dumps([sys.dont_write_bytecode,os.environ["XDG_CACHE_HOME"]]))'],
            {}, capture=True))
        if runtime_check != [True, env["XDG_CACHE_HOME"]]:
            raise ValueError("private bytecode/cache settings did not take effect")
        after = hashlib.sha256(scheduler.read_bytes()).hexdigest()
        if after != UPSTREAM_SHA:
            raise ValueError("public scheduler SHA changed during clone")
        evidence = dict(schema=1, version=source["version"], source_python=str(python),
                        source_prefix=source["prefix"], source_base_prefix=source["base_prefix"],
                        source_package=str(package), source_dependency_paths=source["sites"],
                        source_scheduler_sha256_before=UPSTREAM_SHA, source_scheduler_sha256_after=after,
                        private_python=str(private_py), private_package=private["package"],
                        private_metadata=private["metadata"], private_scheduler_sha256=hashlib.sha256(patched).hexdigest(),
                        cache_root=str(target / "cache"))
        (target / "clone-manifest.json").write_text(json.dumps(evidence, indent=2)+"\n", encoding="utf-8")
        return evidence
    except BaseException:
        # This process created target exclusively after checking all source boundaries.
        shutil.rmtree(target)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True, type=Path, help="Absolute Python in the trusted public environment")
    parser.add_argument("--out", required=True, type=Path, help="New private environment; must not already exist")
    args = parser.parse_args(argv)
    try:
        manifest = build(args.python, args.out)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"clone_runtime: {exc}\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
