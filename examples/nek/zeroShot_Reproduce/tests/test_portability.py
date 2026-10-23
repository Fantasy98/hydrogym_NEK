"""Standard-library tests with tiny fixtures; no containers, network or Slurm."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import channel_restarts, export_sources, submit
from common.config import load_settings


def clean_environment():
    return {key: value for key, value in os.environ.items()
            if not key.startswith(("ZS_", "TCF_", "WING_", "HG_"))}


class PortableSettings(unittest.TestCase):
    def test_cluster_layouts_and_case_paths(self):
        for site, cores, channel_nodes, wing_nodes in (("juwels", 48, 11, 86), ("jureca", 128, 5, 33)):
            with patch.dict(os.environ, {**clean_environment(), "ZS_SITE": site}, clear=True):
                for stage, nodes in (("train", channel_nodes), ("evaluate", channel_nodes), ("deploy", wing_nodes)):
                    settings = load_settings(stage)
                    command, case = submit.command_for(stage, settings)
                    self.assertIn(f"--nodes={nodes}", command)
                    self.assertIn(f"--ntasks-per-node={cores}", command)
                    self.assertTrue(case.is_relative_to(ROOT))
                    self.assertEqual(settings["ZS_ROOT"], str(ROOT))

    def test_settings_work_outside_repository(self):
        with tempfile.TemporaryDirectory(prefix="zs-code-fixture-") as temp:
            relocated = Path(temp) / "renamed_workflow"
            for relative in ("common/settings.sh", "sites/juwels.sh", "sites/jureca.sh",
                             "0_setup/00_settings.sh", "1_tcf_train/00_settings.sh",
                             "2_naca0012_deploy/00_settings.sh"):
                target = relocated / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, target)
            for case, variable in (("0_setup", "HG_CASE_DIR"), ("1_tcf_train", "TCF_CASE_DIR"),
                                   ("2_naca0012_deploy", "WING_CASE_DIR")):
                result = subprocess.run(["bash", "-c", 'set -eu; source "$1"; printf "%s" "${!2}"',
                                         "fixture", str(relocated / case / "00_settings.sh"), variable],
                                        env=clean_environment(), check=True, text=True, capture_output=True)
                self.assertEqual(result.stdout, str(relocated / case))

    def test_previews_never_invoke_sbatch(self):
        with tempfile.TemporaryDirectory(prefix="zs-launch-fixture-") as temp:
            directory = Path(temp)
            marker = directory / "called"
            launcher = directory / "sbatch"
            launcher.write_text('#!/bin/sh\nprintf called > "$ZS_TEST_MARKER"\nexit 99\n')
            launcher.chmod(0o755)
            env = {**clean_environment(), "PATH": f"{directory}:{os.environ['PATH']}",
                   "ZS_TEST_MARKER": str(marker), "PYTHONDONTWRITEBYTECODE": "1"}
            for stage in submit.SCRIPTS:
                result = subprocess.run([sys.executable, "-B", str(ROOT / "tools/submit.py"), stage],
                                        env=env, check=True, text=True, capture_output=True)
                self.assertIn("PLAN ONLY", result.stdout)
                self.assertFalse(marker.exists())


class HostPython(unittest.TestCase):
    def test_ignores_path_python_and_inherited_conda_python_settings(self):
        with tempfile.TemporaryDirectory(prefix="zs-host-python-") as temp:
            directory = Path(temp)
            marker = directory / "wrong-python-called"
            fake_python = directory / "python3"
            fake_python.write_text('#!/bin/sh\ntouch "$ZS_TEST_MARKER"\nexit 99\n')
            fake_python.chmod(0o755)
            (directory / "json.py").write_text('raise RuntimeError("Inherited PYTHONPATH was used")\n')
            env = {"PATH": f"{directory}:/usr/bin:/bin", "PYTHONHOME": str(directory),
                   "PYTHONPATH": str(directory), "CONDA_PREFIX": str(directory),
                   "ZS_TEST_MARKER": str(marker)}
            result = subprocess.run([str(ROOT / "tools/host_python.sh"), "-c",
                                     "import json, sys; print(json.dumps(sys.executable))"],
                                    env=env, check=True, text=True, capture_output=True)
            self.assertEqual(json.loads(result.stdout), "/usr/bin/python3")
            for launcher in ("1_tcf_train/04_submit_chain.sh", "2_naca0012_deploy/07_submit_deploy.sh"):
                result = subprocess.run([str(ROOT / launcher)], env=env, check=True,
                                        text=True, capture_output=True)
                self.assertIn("PLAN ONLY", result.stdout)
            self.assertFalse(marker.exists())

    def test_invalid_interpreter_override_has_clear_error(self):
        for interpreter in ("python3", "/nonexistent/zero-shot-test-python"):
            env = {"PATH": "/usr/bin:/bin", "ZS_HOST_PYTHON": interpreter}
            result = subprocess.run([str(ROOT / "tools/host_python.sh"), "--version"],
                                    env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("Set ZS_HOST_PYTHON", result.stderr)


class RestartImport(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="zs-restart-fixture-")
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.source = self.base / "source"
        self.destination = self.base / "destination"
        self.records = []
        for index in range(1, 3):
            relative = f"init_{index}/rs6tcf0.f00001"
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            data = f"original fixture {index}".encode()
            path.write_bytes(data)
            self.records.append({"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})

    def test_copy_is_exact_and_independent(self):
        channel_restarts.install(self.source, self.destination, self.records)
        channel_restarts.verify(self.destination, self.records, independent=True)
        source = self.source / self.records[0]["path"]
        destination = self.destination / self.records[0]["path"]
        self.assertFalse(os.path.samefile(source, destination))
        destination.write_bytes(b"modified fixture")
        channel_restarts.verify(self.source, self.records)

    def test_bad_source_does_not_publish_destination(self):
        (self.source / self.records[0]["path"]).write_bytes(b"corrupt")
        with self.assertRaisesRegex(RuntimeError, "checksum"):
            channel_restarts.install(self.source, self.destination, self.records)
        self.assertFalse(self.destination.exists())

    def test_existing_bad_destination_is_not_overwritten(self):
        channel_restarts.install(self.source, self.destination, self.records)
        path = self.destination / self.records[0]["path"]
        path.write_bytes(b"keep this failed copy")
        with self.assertRaises(RuntimeError):
            channel_restarts.install(self.source, self.destination, self.records)
        self.assertEqual(path.read_bytes(), b"keep this failed copy")

    def test_hardlinks_and_symlinks_are_rejected_for_destination(self):
        for kind in ("hardlink", "symlink"):
            destination = self.base / kind
            for record in self.records:
                target = destination / record["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                if kind == "hardlink":
                    os.link(self.source / record["path"], target)
                else:
                    target.symlink_to(self.source / record["path"])
            with self.assertRaisesRegex(RuntimeError, "independent"):
                channel_restarts.verify(destination, self.records, independent=True)


class SourceExport(unittest.TestCase):
    def test_only_manifest_files_are_exported(self):
        with tempfile.TemporaryDirectory(prefix="zs-export-fixture-") as temp:
            source, destination = Path(temp) / "source", Path(temp) / "destination"
            (source / "manifests").mkdir(parents=True)
            (source / "data").mkdir()
            (source / "data/never-export").write_text("generated data")
            content = b"reviewed source\n"
            (source / "example.py").write_bytes(content)
            manifest = {"files": [{"path": "example.py", "sha256": hashlib.sha256(content).hexdigest()}]}
            (source / "manifests/bundle_files.json").write_text(json.dumps(manifest))
            with patch.object(export_sources, "ROOT", source), patch.object(sys, "argv", ["export", "--destination", str(destination)]), contextlib.redirect_stdout(io.StringIO()):
                export_sources.main()
            self.assertEqual((destination / "example.py").read_bytes(), content)
            self.assertFalse((destination / "data").exists())
            with patch.object(export_sources, "ROOT", source), patch.object(sys, "argv", ["export", "--destination", str(destination)]):
                with self.assertRaisesRegex(RuntimeError, "empty"):
                    export_sources.main()


if __name__ == "__main__":
    unittest.main()
