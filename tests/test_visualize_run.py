# Тесты сводки и отрисовки визуализации.

from __future__ import annotations

from pathlib import Path

import pytest

from core.config import Config
from tests.test_pipeline import _frames

viz_module = pytest.importorskip("tools.visualize_run")
pytest.importorskip("matplotlib")
pytest.importorskip("PIL")

VIZ = Path(__file__).resolve().parents[1] / "configs" / "viz.yaml"


def _source(cfg: Config, n: int):
    for index, (cloud, intensity, ring, t_rel, stamp) in enumerate(
            _frames(cfg, n, with_body=True)):
        yield index, stamp, cloud, intensity, ring, t_rel


def test_renders_png_and_gif_from_a_synthetic_run(cfg: Config, tmp_path: Path) -> None:
    viz = viz_module.load_viz(VIZ)
    top = viz["top_view"]
    region = (tuple(top["x_range_m"]), tuple(top["y_range_m"]))
    views = viz_module.collect(cfg, _source(cfg, 4), 1, 3, 4, region)
    assert [v.index for v in views] == [1, 2, 3]
    assert all(v.points.shape[1] == 3 and v.points.size for v in views)

    summary = viz_module.summarise(views)
    assert summary["frames"] == 3 and summary["unchecked"] == 0

    paths = viz_module.render(views, cfg.gauge, viz, tmp_path / "frames", "синтетика")
    gif = tmp_path / "run.gif"
    viz_module.write_gif(paths, gif, viz["drawing"]["fps"])
    assert len(paths) == 3 and all(p.stat().st_size > 0 for p in paths)
    assert gif.stat().st_size > 0


def test_summary_does_not_depend_on_point_thinning(cfg: Config) -> None:
    viz = viz_module.load_viz(VIZ)
    top = viz["top_view"]
    region = (tuple(top["x_range_m"]), tuple(top["y_range_m"]))
    coarse = viz_module.collect(cfg, _source(cfg, 3), 0, 2, 16, region)
    fine = viz_module.collect(cfg, _source(cfg, 3), 0, 2, 1, region)
    assert viz_module.summarise(coarse) == viz_module.summarise(fine)
    assert coarse[-1].points.shape[0] < fine[-1].points.shape[0]


def test_viz_config_rejects_unknown_keys(tmp_path: Path) -> None:
    text = VIZ.read_text(encoding="utf-8") + "\nsurprise: 1\n"
    broken = tmp_path / "viz.yaml"
    broken.write_text(text, encoding="utf-8")
    with pytest.raises(viz_module.VizConfigError):
        viz_module.load_viz(broken)

