# Общие фикстуры тестов: конфиг проекта.

from __future__ import annotations

from pathlib import Path

import pytest

from core.config import Config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@pytest.fixture(scope="session")
def cfg() -> Config:
    return Config.from_yaml(CONFIG_PATH)


@pytest.fixture(scope="session")
def raw_config() -> dict:
    import yaml

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)

