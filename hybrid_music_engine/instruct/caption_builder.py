"""
caption_builder.py — Construcción de entradas ricas para el codificador T5.

Punto 1 de la estrategia: T5 es un modelo de lenguaje natural, así que la calidad
semántica de sus estados ocultos depende directamente de cuán descriptivo sea el
texto de entrada. Pasarle solo "Jazz" desperdicia su capacidad contextual.

Este módulo construye dos tipos de texto:

  build_caption(...)       → descripción densa para condicionar generación
                             "Género: Jazz Latino | Instrumentos: Trompeta, Congas,
                              Piano | Tempo: 120 BPM | Estilo: Sincopado, en vivo"

  build_instruction(...)   → comando de edición para Instruct-MusicGen
                             "Add a piano solo over the existing track"
                             "Remove the drums but keep the bassline"

No tiene dependencias pesadas (solo stdlib), por lo que puede usarse en cualquier
etapa del pipeline sin cargar PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

# ── Plantilla de descripción ──────────────────────────────────────────────────
#
# El orden importa: T5 lee de izquierda a derecha y la posición influye en la
# atención. Ponemos primero lo más identitario (género) y luego los detalles
# concretos (instrumentos, tempo) que el modelo puede "anclar" como tokens
# individuales: "Trompeta" y "120 BPM" sobreviven como unidades semánticas.

CAPTION_FIELDS_ES = [
    ("genre", "Género"),
    ("instruments", "Instrumentos"),
    ("tempo_bpm", "Tempo"),
    ("key", "Tonalidad"),
    ("mood", "Ambiente"),
    ("style", "Estilo"),
    ("energy", "Energía"),
]

CAPTION_FIELDS_EN = [
    ("genre", "Genre"),
    ("instruments", "Instruments"),
    ("tempo_bpm", "Tempo"),
    ("key", "Key"),
    ("mood", "Mood"),
    ("style", "Style"),
    ("energy", "Energy"),
]


@dataclass
class TrackMeta:
    """Metadatos de una pista. Todos opcionales salvo el género."""

    genre: str
    instruments: Sequence[str] = field(default_factory=list)
    tempo_bpm: float | int | None = None
    key: str | None = None
    mood: str | None = None
    style: str | None = None
    energy: str | None = None
    extra: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "TrackMeta":
        known = {f for f, _ in CAPTION_FIELDS_EN}
        extra = {k: str(v) for k, v in data.items() if k not in known and v is not None}
        return cls(
            genre=str(data.get("genre", "music")),
            instruments=_as_list(data.get("instruments")),
            tempo_bpm=data.get("tempo_bpm"),
            key=_opt_str(data.get("key")),
            mood=_opt_str(data.get("mood")),
            style=_opt_str(data.get("style")),
            energy=_opt_str(data.get("energy")),
            extra=extra,
        )


def _as_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, Iterable):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value)]


def _opt_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _format_value(field_name: str, value: object) -> str | None:
    if value is None:
        return None
    if field_name == "instruments":
        items = _as_list(value)
        return ", ".join(items) if items else None
    if field_name == "tempo_bpm":
        try:
            return f"{round(float(value))} BPM"
        except (TypeError, ValueError):
            return None
    text = str(value).strip()
    return text or None


def build_caption(
    meta: TrackMeta | Mapping[str, object],
    *,
    language: str = "es",
    include_extra: bool = True,
    separator: str = " | ",
) -> str:
    """
    Construye un caption estructurado y denso a partir de metadatos.

    >>> build_caption(TrackMeta(genre="Jazz Latino",
    ...                         instruments=["Trompeta", "Congas", "Piano"],
    ...                         tempo_bpm=120, style="Sincopado, en vivo"))
    'Género: Jazz Latino | Instrumentos: Trompeta, Congas, Piano | Tempo: 120 BPM | Estilo: Sincopado, en vivo'
    """
    if not isinstance(meta, TrackMeta):
        meta = TrackMeta.from_dict(meta)

    fields = CAPTION_FIELDS_EN if language == "en" else CAPTION_FIELDS_ES
    parts: list[str] = []
    for attr, label in fields:
        formatted = _format_value(attr, getattr(meta, attr))
        if formatted:
            parts.append(f"{label}: {formatted}")

    if include_extra and meta.extra:
        for key, value in meta.extra.items():
            label = key.replace("_", " ").capitalize()
            parts.append(f"{label}: {value}")

    if not parts:
        return f"{meta.genre} music"
    return separator.join(parts)


# ── Captions por defecto enriquecidos por género ──────────────────────────────
#
# Reemplazan los DEFAULT_GENRE_PROMPTS de una sola línea del script original por
# descripciones multi-campo. Sirven de fallback cuando no hay metadatos por pista.

DEFAULT_GENRE_META: dict[str, TrackMeta] = {
    "classical": TrackMeta(
        genre="Classical",
        instruments=["piano", "strings", "violin", "cello", "woodwinds"],
        mood="elegant, expressive",
        style="orchestral, symphonic",
        energy="dynamic",
    ),
    "electronic": TrackMeta(
        genre="Electronic",
        instruments=["synthesizers", "drum machine", "bass synth", "pads"],
        tempo_bpm=128,
        mood="energetic",
        style="EDM, four-on-the-floor",
        energy="high",
    ),
    "reggaeton": TrackMeta(
        genre="Reggaeton",
        instruments=["dembow drums", "urban percussion", "808 bass", "synth lead"],
        tempo_bpm=95,
        mood="danceable",
        style="latin urban, syncopated",
        energy="high",
    ),
}


def default_genre_caption(genre: str, *, language: str = "en") -> str:
    """Caption rico por defecto para un género (inglés recomendado para T5)."""
    meta = DEFAULT_GENRE_META.get(genre)
    if meta is None:
        meta = TrackMeta(genre=genre)
    return build_caption(meta, language=language)


# ── Instrucciones de edición (Instruct-MusicGen) ──────────────────────────────
#
# Cuando el usuario edita en vez de describir, el texto es un comando. T5 fue
# preentrenado en tareas text-to-text (instrucciones, traducción, resumen), por
# lo que entiende los verbos de acción y sus complementos musicales con precisión.
# Mantenemos un vocabulario canónico de acciones para consistencia en el dataset
# de fine-tuning del editor.

EDIT_ACTIONS = {
    "add": "Add {target}",
    "remove": "Remove {target}",
    "extract": "Extract only {target}",
    "replace": "Replace {target} with {replacement}",
    "emphasize": "Make {target} more prominent",
    "soften": "Make {target} quieter and softer",
    "restyle": "Make the track sound more {target}",
}


def build_instruction(
    action: str,
    target: str,
    *,
    replacement: str | None = None,
    keep: str | None = None,
) -> str:
    """
    Construye un comando de edición canónico en inglés (alineado con MusicGen).

    >>> build_instruction("add", "a piano solo")
    'Add a piano solo'
    >>> build_instruction("remove", "the drums", keep="the bassline")
    'Remove the drums but keep the bassline'
    >>> build_instruction("replace", "the lead synth", replacement="an electric guitar")
    'Replace the lead synth with an electric guitar'
    """
    action_key = action.lower().strip()
    template = EDIT_ACTIONS.get(action_key)
    if template is None:
        raise ValueError(
            f"Acción de edición desconocida: {action!r}. "
            f"Válidas: {sorted(EDIT_ACTIONS)}"
        )
    text = template.format(target=target.strip(), replacement=(replacement or "").strip())
    if keep:
        text = f"{text} but keep {keep.strip()}"
    return text


if __name__ == "__main__":
    # Mini-demostración ejecutable
    demo = TrackMeta(
        genre="Jazz Latino",
        instruments=["Trompeta", "Congas", "Piano"],
        tempo_bpm=120,
        style="Sincopado, en vivo",
    )
    print("Caption ES :", build_caption(demo, language="es"))
    print("Caption EN :", build_caption(demo, language="en"))
    print("Default    :", default_genre_caption("reggaeton"))
    print("Instrucción:", build_instruction("remove", "the drums", keep="the bassline"))
