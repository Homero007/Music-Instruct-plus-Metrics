"""Pruebas sin dependencias pesadas (solo stdlib). Ejecutar: python -m pytest -q
o directamente: python tests/test_captions.py"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hybrid_music_engine.instruct import caption_builder as cb  # noqa: E402


def test_caption_structured():
    meta = cb.TrackMeta(
        genre="Jazz Latino",
        instruments=["Trompeta", "Congas", "Piano"],
        tempo_bpm=120,
        style="Sincopado, en vivo",
    )
    caption = cb.build_caption(meta, language="es")
    assert "Género: Jazz Latino" in caption
    assert "Trompeta" in caption and "Congas" in caption
    assert "120 BPM" in caption
    assert "Sincopado" in caption


def test_caption_from_dict_and_extra():
    caption = cb.build_caption(
        {"genre": "Techno", "instruments": "kick, hats", "tempo_bpm": 130, "label": "Berlin"}
    )
    assert "130 BPM" in caption
    assert "Berlin" in caption  # campo extra


def test_default_genre_caption_is_rich():
    caption = cb.default_genre_caption("reggaeton")
    assert caption.count("|") >= 3  # varios campos, no una sola etiqueta


def test_build_instruction_variants():
    assert cb.build_instruction("add", "a piano solo") == "Add a piano solo"
    assert cb.build_instruction("remove", "the drums", keep="the bassline") == (
        "Remove the drums but keep the bassline"
    )
    assert cb.build_instruction(
        "replace", "the lead synth", replacement="an electric guitar"
    ) == "Replace the lead synth with an electric guitar"


def test_unknown_action_raises():
    try:
        cb.build_instruction("teleport", "the bass")
    except ValueError:
        return
    raise AssertionError("debió lanzar ValueError")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("TODAS LAS PRUEBAS PASARON")
