"""The one-line installer (install.sh) and the app agree on the model file, the UMA versions and the command."""

import re
import subprocess
from pathlib import Path

from catalyst_engine.design import screening

REPO = Path(__file__).resolve().parents[1]
SCRIPT = (REPO / "install.sh").read_text()


def _var(name: str) -> str:
    return re.search(rf'^{name}="([^"]*)"', SCRIPT, re.M).group(1)


def test_installer_is_valid_posix_shell():
    assert subprocess.run(["sh", "-n", str(REPO / "install.sh")]).returncode == 0


def test_installer_fetches_the_checkpoint_the_app_uses(monkeypatch):
    monkeypatch.delenv("CE_UMA_CHECKPOINT", raising=False)
    assert Path(screening.uma_setup()["checkpoint"]).name == _var("MODEL_NAME") == "uma-s-1p1.pt"
    assert re.fullmatch(r"[0-9a-f]{64}", _var("MODEL_SHA256"))


def test_uma_versions_are_the_checked_ones():
    pins = dict(line.split("==") for line in (REPO / "uma-requirements.txt").read_text().splitlines()
                if line and not line.startswith("#"))
    assert pins["fairchem-core"] == "2.23.0"  # the version every check in experiments/ ran with
    assert pins["torch"] == "2.13.0"


def test_install_command_points_at_this_script():
    assert screening.INSTALL_COMMAND.endswith("/main/install.sh | sh")
    assert screening.INSTALL_COMMAND.split()[-3].startswith("https://raw.githubusercontent.com/BTONGK/catalyst-engine/")


def test_uma_python_follows_the_environment_variable(monkeypatch, tmp_path):
    monkeypatch.setenv("CE_UMA_PYTHON", str(tmp_path / "python"))
    assert screening.uma_setup()["python"] == str(tmp_path / "python")
