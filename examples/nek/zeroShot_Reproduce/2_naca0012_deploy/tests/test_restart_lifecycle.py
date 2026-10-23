"""Small-file restart/archiving regression tests; no MPI or CFD jobs launched.

Run with the deployment Python environment:
    python -m unittest discover -s tests -v
"""

import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("stage", ROOT / "02_stage_solver.py")
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


def checkpoint(path, step, time):
    header = f"#std 8 8 8 8 1 1 {time:.8f} {step} 0 1 XUP"
    path.write_bytes(header.encode().ljust(132) + b"x" * 124)


class RestartLifecycle(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="wing-test-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.environment = self.base / "data" / "wing"
        self.seed = self.environment / "restart_files" / "init_1"
        self.seed.mkdir(parents=True)
        self.workspace = self.base / "w"
        self.workspace.mkdir()
        self.solver = self.base / "r" / "s"
        self.config = self.base / "runtime.yml"
        for name in ("naca_wing.re2", "naca_wing.map", "naca_wing.wall",
                     "mask_naca_wing0.f00002", "stat_pts.in", "forparam.i",
                     "environment_config.yaml"):
            (self.environment / name).write_text("fixture\n")
        (self.environment / "naca_wing.restart").write_text("1\n")
        (self.environment / "naca_wing.rea").write_text("".join(
            f"1 {token} fixture\n" for token in stage.V17_PARAMETERS.values()))
        simulation = dict.fromkeys(stage.V17_PARAMETERS, 1)
        simulation.update(CASENAME="naca_wing", solver_version="v17")
        OmegaConf.save(OmegaConf.create({"simulation": simulation,
                                        "runner": {"random_init": -2, "rank": 1}}), self.config)
        sizes = {}
        for index in range(1, 9):
            path = self.seed / f"rs8naca_wing0.f{index:05d}"
            checkpoint(path, 100 + index, 10 + index / 100)
            sizes[f"restart_files/init_1/{path.name}"] = path.stat().st_size
        (self.workspace / "PREPARED.json").write_text(json.dumps({"files": sizes}))
        (self.workspace / "nek5000").write_text("fixture executable\n")
        (self.workspace / "SIZE").write_text("fixture size\n")

    def run_command(self, args, *, success=True, env=None):
        result = subprocess.run(args, text=True, capture_output=True, env=env)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def stage(self, *extra, success=True):
        return self.run_command([sys.executable, str(ROOT / "02_stage_solver.py"),
                                 "--config", str(self.config), "--environment-dir", str(self.environment),
                                 "--workspace", str(self.workspace), "--solver-dir", str(self.solver),
                                 *map(str, extra)], success=success)

    def test_seed_and_reuse_preserve_committed_state_and_hf_source(self):
        self.stage()
        name = "rs8naca_wing0.f00005"
        original = (self.seed / name).read_bytes()
        self.assertFalse((self.solver / name).is_symlink())
        self.assertTrue((self.solver / "naca_wing.re2").is_symlink())
        for index in range(1, 5):
            checkpoint(self.solver / f"rs8naca_wing0.f{index:05d}", 5000 + index, 20 + index / 100)
        (self.solver / "naca_wing.restart").write_text("0\n")
        # An interrupted write to the INACTIVE set must not prevent resume.
        (self.solver / name).write_bytes(b"unfinished")
        self.stage()
        info = json.loads((self.solver / "restart-input.json").read_text())
        self.assertEqual(info["mode"], "reuse")
        self.assertEqual(info["selector"], 0)
        self.assertEqual(info["members"][0]["nek_step"], 5001)
        self.assertEqual((self.seed / name).read_bytes(), original)
        self.assertEqual((self.environment / "naca_wing.restart").read_text(), "1\n")

    def test_incomplete_selected_checkpoint_is_rejected(self):
        self.stage()
        (self.solver / "rs8naca_wing0.f00005").write_bytes(b"unfinished")
        self.assertIn("Incomplete checkpoint", self.stage(success=False).stderr)

    def test_mutable_symlinks_and_hardlinks_are_rejected(self):
        self.stage()
        path = self.solver / "rs8naca_wing0.f00001"
        path.unlink()
        path.symlink_to(self.seed / path.name)
        self.assertIn("independent regular file", self.stage(success=False).stderr)
        path.unlink()
        os.link(self.seed / path.name, path)
        self.assertIn("independent regular file", self.stage(success=False).stderr)

    def test_import_only_initializes_a_new_solver(self):
        imported = self.base / "import"
        imported.mkdir()
        for index in range(1, 9):
            checkpoint(imported / f"rs8naca_wing0.f{index:05d}", 200 + index, 30 + index / 100)
        (imported / "naca_wing.restart").write_text("0\n")
        self.stage("--restart-from", imported)
        self.assertEqual(json.loads((self.solver / "restart-input.json").read_text())["mode"], "import")
        self.stage("--restart-from", self.base / "does-not-exist")
        self.assertEqual((self.solver / "naca_wing.restart").read_text(), "0\n")

    def test_invalid_seed_never_publishes_solver(self):
        (self.seed / "rs8naca_wing0.f00005").write_bytes(b"unfinished")
        self.stage(success=False)
        self.assertFalse(self.solver.exists())
        self.assertEqual(list(self.solver.parent.glob(".s-seed-*")), [])

    def job_environment(self):
        bindir = self.base / "bin"
        bindir.mkdir()
        apptainer = bindir / "apptainer"
        # Tests run inside the deployment container. Execute the actual inner
        # shell command on fixtures without nesting a container or starting MPI.
        apptainer.write_text(f"#!{sys.executable}\n" + '''import subprocess, sys
i = sys.argv.index('bash')
raise SystemExit(subprocess.call(sys.argv[i:]))
''')
        srun = bindir / "srun"
        srun.write_text(f"#!{sys.executable}\n" + '''import os
from pathlib import Path
s = Path(os.environ['WING_SOLVER_DIR'])
for i in range(1, 5):
    header = f'#std 8 8 8 8 1 1 {20 + i/100:.8f} {5000 + i} 0 1 XUP'
    (s / f'rs8naca_wing0.f{i:05d}').write_bytes(header.encode().ljust(132) + b'y'*124)
(s / 'naca_wing.restart').write_text('0\\n')
(s / 'naca_wing0.f00001').write_text('field from ' + os.environ['SLURM_JOB_ID'])
''')
        apptainer.chmod(0o755)
        srun.chmod(0o755)
        image = self.base / "image.sif"
        image.touch()
        checkpoints = self.base / "checkpoints"
        checkpoints.mkdir()
        # The launcher checks the selected model pair before staging.
        for name in ("model_1_steps.zip", "vecnormalize_1_steps.pkl"):
            (checkpoints / name).write_text("fixture checkpoint\n")
        env = os.environ.copy()
        env.update(PATH=f"{bindir}:{env['PATH']}", WING_CASE_DIR=str(ROOT),
                   WING_NEK_MODULE="Nek5000/1.0-gompi-2024a-SystemCUDA-NACA0012",
                   WING_PYTHON_VENV="/home/easybuild/venvs/hydrogym_cpu",
                   WING_IMAGE=str(image), WING_RUNTIME_CONFIG=str(self.config),
                   WING_WORK_DIR=str(self.workspace), WING_DATA_ROOT=str(self.base / "data"),
                   WING_ENV_NAME="wing", WING_RUNS_DIR=str(self.base / "r"),
                   WING_SOLVER_DIR=str(self.solver), WING_RESULTS_DIR=str(self.base / "results"),
                   WING_TCF_CHECKPOINT_DIR=str(checkpoints), WING_MODEL_STEP="1", WING_RESTART_FROM="",
                   WING_MV_DATA="true", SLURM_JOB_NODELIST="fixture", SLURM_JOB_ID="1")
        return env

    def test_job_handoff_archives_copies_and_handles_missing_finish_marker(self):
        env = self.job_environment()
        command = ["bash", str(ROOT / "05_naca0012_deploy.slurm")]
        self.run_command([*command, "--mv-data=false"], env=env)
        self.assertFalse((self.base / "results").exists())
        first_job = self.base / "r/j-1"
        self.assertEqual((first_job / "s").resolve(), self.solver)
        self.assertEqual(json.loads((first_job / "restart-input.json").read_text())["mode"], "initial")
        env["SLURM_JOB_ID"] = "2"
        self.run_command(command, env=env)
        job2 = self.base / "r" / "j-2"
        info = json.loads((job2 / "restart-input.json").read_text())
        self.assertEqual(info["mode"], "reuse")
        self.assertEqual(info["selector"], 0)
        result = self.base / "results" / "res_1"
        name = "rs8naca_wing0.f00001"
        archived = (result / "rs_data" / name).read_bytes()
        self.assertEqual((result / "rs_data" / "naca_wing.restart").read_text(), "0\n")
        self.assertEqual((result / "reg_data" / "naca_wing0.f00001").read_text(), "field from 1")
        self.assertTrue((self.solver / name).is_file())
        (self.solver / name).write_bytes(archived[:-1] + b"z")
        self.assertEqual((result / "rs_data" / name).read_bytes(), archived)
        (job2 / "RUN_FINISHED").unlink()
        env["SLURM_JOB_ID"] = "3"
        self.run_command(command, env=env)
        status = json.loads((self.base / "results/res_2/archive_status.json").read_text())
        self.assertEqual(status["solver_exit_code"], 1)

    def test_launcher_forwards_initial_restart_source(self):
        self.stage()
        imported = self.base / "import"
        self.solver.rename(imported)
        (imported / "naca_wing.restart").write_text("0\n")
        env = self.job_environment()
        env["WING_RESTART_FROM"] = str(imported)
        self.run_command(["bash", str(ROOT / "05_naca0012_deploy.slurm")], env=env)
        info = json.loads((self.base / "r/j-1/restart-input.json").read_text())
        self.assertEqual(info["mode"], "import")
        self.assertEqual(info["source"], str(imported))
        self.assertEqual(info["selector"], 0)

    def test_job_specific_solver_leaves_production_state_unchanged(self):
        self.stage()
        before = {p.name: p.read_bytes() for p in self.solver.glob("rs8*")}
        env = self.job_environment()
        env["WING_SOLVER_DIR"] = str(self.base / "r/j-1/s")
        self.run_command(["bash", str(ROOT / "05_naca0012_deploy.slurm"), "--mv-data=false"], env=env)
        self.assertFalse((self.base / "r/j-1/s").is_symlink())
        self.assertEqual(json.loads((self.base / "r/j-1/restart-input.json").read_text())["mode"], "initial")
        self.assertEqual((self.solver / "naca_wing.restart").read_text(), "1\n")
        self.assertEqual({p.name: p.read_bytes() for p in self.solver.glob("rs8*")}, before)
        self.assertFalse(Path(str(self.solver) + ".last-run").exists())

    def test_job_and_manual_archive_refuse_locked_solver(self):
        env = self.job_environment()
        self.stage()
        job = self.base / "r/j-old"
        job.mkdir()
        (job / "s").symlink_to(self.solver)
        with Path(str(self.solver) + ".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.run_command(["bash", str(ROOT / "05_naca0012_deploy.slurm")], env=env, success=False)
            self.assertIn("Another job or archiver", result.stderr)
            result = self.run_command(["bash", str(ROOT / "06_archive_nek_data.sh"),
                                       "--run-root", str(job), "--results-root", str(self.base / "results")], success=False)
            self.assertIn("solver is in use", result.stderr)

    def test_manual_archive_refuses_stale_job_link(self):
        self.stage()
        old = self.base / "r/j-old"
        old.mkdir()
        (old / "s").symlink_to(self.solver)
        Path(str(self.solver) + ".last-run").write_text(str(self.base / "r/j-new") + "\n")
        result = self.run_command(["bash", str(ROOT / "06_archive_nek_data.sh"),
                                   "--run-root", str(old), "--results-root", str(self.base / "results")], success=False)
        self.assertIn("outputs belong to", result.stderr)


if __name__ == "__main__":
    unittest.main()
