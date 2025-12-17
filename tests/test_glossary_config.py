from pathlib import Path

from src.gui.config import GuiConfig
from src.utils.glossary import DEFAULT_GLOSSARY_TERMS


def test_gui_config_glossary_roundtrip(monkeypatch, tmp_path: Path):
    # ~/.flowcut/config.json を触らないように home を差し替える
    import src.gui.config as gui_config_module

    monkeypatch.setattr(gui_config_module.Path, "home", lambda: tmp_path)

    cfg = GuiConfig()
    assert cfg.get_glossary_terms() == DEFAULT_GLOSSARY_TERMS

    cfg.set_glossary_terms(["  菅義偉  ", "", "菅義偉", "公明党"])
    assert cfg.get_glossary_terms() == ["菅義偉", "公明党"]

    # ファイル保存→再読込できること
    cfg2 = GuiConfig()
    assert cfg2.get_glossary_terms() == ["菅義偉", "公明党"]

