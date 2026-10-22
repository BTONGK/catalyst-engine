import pytest


@pytest.fixture(autouse=True)
def _private_data_dir(tmp_path_factory, monkeypatch):
    # Pages and commands that open a store use a throwaway folder, never the repository's data/.
    monkeypatch.setenv("CE_DATA_DIR", str(tmp_path_factory.getbasetemp() / "ce-data"))
