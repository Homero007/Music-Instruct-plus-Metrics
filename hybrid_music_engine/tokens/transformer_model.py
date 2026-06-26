from __future__ import annotations

import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from hybrid_music_engine.core.config import EngineConfig
from hybrid_music_engine.core.ids import create_id

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"


def train_token_transformer_model(
    config: EngineConfig,
    *,
    token_manifest_path: Path,
    model_name: str = "token_transformer",
    sequence_length: int = 128,
    epochs: int = 8,
    batch_size: int = 16,
    embedding_dim: int = 128,
    num_layers: int = 3,
    num_heads: int = 4,
    feedforward_dim: int = 256,
    learning_rate: float = 3e-4,
    seed: int = 42,
) -> dict[str, Any]:
    torch, nn, data = _torch_modules()
    if sequence_length < 8:
        raise RuntimeError("sequence_length debe ser mayor o igual a 8.")
    if embedding_dim % num_heads != 0:
        raise RuntimeError("embedding_dim debe ser divisible entre num_heads.")

    random.seed(seed)
    torch.manual_seed(seed)
    token_sequences = _load_token_sequences(token_manifest_path)
    vocab = _build_vocab(token_sequences)
    encoded = [[vocab[BOS_TOKEN], *[vocab[token] for token in tokens], vocab[EOS_TOKEN]] for tokens in token_sequences]
    windows = _build_windows(encoded, sequence_length=sequence_length, pad_id=vocab[PAD_TOKEN])
    if not windows:
        raise RuntimeError("No hay suficientes tokens para entrenar el Transformer.")

    dataset = TokenWindowDataset(windows, torch=torch)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = TokenTransformer(
        vocab_size=len(vocab),
        sequence_length=sequence_length,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        feedforward_dim=feedforward_dim,
        pad_id=vocab[PAD_TOKEN],
        nn=nn,
        torch=torch,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab[PAD_TOKEN])

    model.train()
    metrics: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        batches = 0
        for inputs, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu().item())
            batches += 1
        metrics.append(
            {
                "epoch": epoch,
                "train_loss": round(total_loss / max(batches, 1), 6),
                "batches": batches,
            }
        )

    model_id = create_id(model_name, prefix="tokenmodel")
    output_dir = config.data_dir / "models" / "tokens" / model_id
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    model_path = output_dir / "model.json"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab": vocab,
            "config": {
                "sequence_length": sequence_length,
                "embedding_dim": embedding_dim,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "feedforward_dim": feedforward_dim,
                "pad_id": vocab[PAD_TOKEN],
            },
        },
        checkpoint_path,
    )
    metadata = {
        "schema_version": "token-transformer-model-v1",
        "model_type": "transformer",
        "model_id": model_id,
        "model_name": model_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "token_manifest_path": str(Path(token_manifest_path).expanduser().resolve()),
        "token_files": len(token_sequences),
        "training_windows": len(windows),
        "vocab_size": len(vocab),
        "sequence_length": sequence_length,
        "epochs": epochs,
        "batch_size": batch_size,
        "embedding_dim": embedding_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "feedforward_dim": feedforward_dim,
        "learning_rate": learning_rate,
        "metrics": metrics,
        "checkpoint_path": str(checkpoint_path),
        "path": str(model_path),
    }
    model_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def generate_tokens_from_transformer_model(
    config: EngineConfig,
    *,
    model_path: Path,
    duration_seconds: float,
    output_name: str = "generated_transformer",
    seed: int | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.9,
    top_k: int | None = 50,
    top_p: float | None = 0.95,
    condition_genre: str | None = None,
    feature_tokens: list[str] | None = None,
    embedding_path: Path | None = None,
    export_layers: bool = False,
) -> dict[str, Any]:
    torch, nn, _data = _torch_modules()
    if duration_seconds <= 0:
        raise RuntimeError("duration_seconds debe ser mayor que cero.")
    resolved_model_path = Path(model_path).expanduser().resolve()
    metadata = json.loads(resolved_model_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != "token-transformer-model-v1":
        raise RuntimeError("El modelo indicado no es un Transformer de tokens.")
    checkpoint_path = _resolve_checkpoint_path(metadata, resolved_model_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    vocab = {str(key): int(value) for key, value in checkpoint["vocab"].items()}
    inv_vocab = {value: key for key, value in vocab.items()}
    model_config = checkpoint["config"]
    model = TokenTransformer(
        vocab_size=len(vocab),
        sequence_length=int(model_config["sequence_length"]),
        embedding_dim=int(model_config["embedding_dim"]),
        num_layers=int(model_config["num_layers"]),
        num_heads=int(model_config["num_heads"]),
        feedforward_dim=int(model_config["feedforward_dim"]),
        pad_id=int(model_config["pad_id"]),
        nn=nn,
        torch=torch,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
    target_tokens = max_tokens or max(int(duration_seconds * 32), 32)
    condition_tokens = _condition_tokens(
        condition_genre,
        [*(feature_tokens or []), *_embedding_feature_tokens(embedding_path)],
    )
    generated_ids = [vocab[BOS_TOKEN]]
    for token in condition_tokens:
        if token in vocab:
            generated_ids.append(vocab[token])
    with torch.no_grad():
        while len(generated_ids) < target_tokens + 1:
            context = generated_ids[-model.sequence_length :]
            if len(context) < model.sequence_length:
                context = [vocab[PAD_TOKEN]] * (model.sequence_length - len(context)) + context
            inputs = torch.tensor([context], dtype=torch.long)
            logits = model(inputs)[0, -1, :] / max(float(temperature), 0.05)
            next_id = _sample_next_id(logits, torch=torch, top_k=top_k, top_p=top_p)
            if next_id == vocab[EOS_TOKEN]:
                break
            generated_ids.append(next_id)

    generated_tokens = [
        inv_vocab[token_id]
        for token_id in generated_ids
        if inv_vocab.get(token_id) not in {PAD_TOKEN, BOS_TOKEN, EOS_TOKEN}
    ]
    output_id = create_id(output_name, prefix="generated")
    output_dir = config.data_dir / "generated" / output_id
    output_dir.mkdir(parents=True, exist_ok=True)
    token_path = output_dir / "tokens.json"
    midi_path = output_dir / f"{output_name}.mid"
    payload = {
        "schema_version": "generated-token-json-v1",
        "generator": "token-transformer",
        "generation_id": output_id,
        "model_path": str(Path(model_path).expanduser().resolve()),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "duration_seconds_requested": duration_seconds,
        "seed": seed,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "condition_genre": condition_genre,
        "feature_tokens": feature_tokens or [],
        "embedding_path": str(Path(embedding_path).expanduser().resolve()) if embedding_path else None,
        "token_count": len(generated_tokens),
        "tokens": generated_tokens,
        "path": str(token_path),
        "midi_path": str(midi_path),
    }
    token_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    from hybrid_music_engine.quality.midi_metrics import analyze_midi_quality
    from hybrid_music_engine.tokens.generative_model import tokens_to_layered_midis, tokens_to_midi

    tokens_to_midi(generated_tokens, midi_path, duration_seconds=duration_seconds)
    payload["metrics"] = analyze_midi_quality(midi_path)
    if export_layers:
        payload["layer_midis"] = tokens_to_layered_midis(
            generated_tokens,
            output_dir / "layers",
            duration_seconds=duration_seconds,
        )
        payload["layer_metrics"] = {
            name: analyze_midi_quality(Path(path))
            for name, path in payload["layer_midis"].items()
        }
        token_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        token_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _resolve_checkpoint_path(metadata: dict[str, Any], model_path: Path) -> Path:
    raw_path = metadata.get("checkpoint_path")
    candidates: list[Path] = []
    if raw_path:
        candidates.append(Path(str(raw_path)).expanduser())
    candidates.append(model_path.parent / "checkpoint.pt")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    shown = ", ".join(str(candidate) for candidate in candidates)
    raise RuntimeError(f"Checkpoint del Transformer no encontrado. Rutas probadas: {shown}")


class TokenWindowDataset:
    def __init__(self, windows: list[list[int]], *, torch) -> None:
        self.windows = windows
        self.torch = torch

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int):
        window = self.windows[index]
        values = self.torch.tensor(window, dtype=self.torch.long)
        return values[:-1], values[1:]


class TokenTransformer:
    def __init__(
        self,
        *,
        vocab_size: int,
        sequence_length: int,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        feedforward_dim: int,
        pad_id: int,
        nn,
        torch,
    ) -> None:
        self.sequence_length = sequence_length
        self._torch = torch
        self._nn = nn
        class _Module(nn.Module):
            def __init__(inner_self) -> None:
                super().__init__()
                inner_self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
                inner_self.position_embedding = nn.Embedding(sequence_length, embedding_dim)
                layer = nn.TransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=num_heads,
                    dim_feedforward=feedforward_dim,
                    dropout=0.1,
                    batch_first=True,
                    activation="gelu",
                )
                inner_self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
                inner_self.output = nn.Linear(embedding_dim, vocab_size)

            def forward(inner_self, inputs):
                batch, length = inputs.shape
                positions = torch.arange(length, device=inputs.device).unsqueeze(0).expand(batch, length)
                x = inner_self.token_embedding(inputs) + inner_self.position_embedding(positions)
                mask = torch.triu(
                    torch.ones(length, length, device=inputs.device, dtype=torch.bool),
                    diagonal=1,
                )
                x = inner_self.transformer(x, mask=mask)
                return inner_self.output(x)

        self.module = _Module()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self.module, name)


def _load_token_sequences(token_manifest_path: Path) -> list[list[str]]:
    manifest = json.loads(Path(token_manifest_path).expanduser().read_text(encoding="utf-8"))
    sequences: list[list[str]] = []
    for entry in manifest.get("entries", []):
        path = Path(str(entry.get("path", ""))).expanduser()
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        tokens = [str(token) for token in payload.get("tokens", [])]
        metadata_tokens = _condition_tokens(
            payload.get("genre"),
            _feature_tokens_from_payload(payload),
        )
        if len(tokens) >= 4:
            sequences.append([*metadata_tokens, *tokens])
    if not sequences:
        raise RuntimeError("El manifest no contiene secuencias de tokens válidas.")
    return sequences


def _condition_tokens(
    condition_genre: str | None,
    feature_tokens: list[str] | None = None,
) -> list[str]:
    tokens: list[str] = []
    if condition_genre:
        tokens.append(f"genre:{_normalize_token_value(condition_genre)}")
    for token in feature_tokens or []:
        value = _normalize_token_value(token)
        if value:
            tokens.append(value if value.startswith("feature:") else f"feature:{value}")
    return tokens


def _embedding_feature_tokens(embedding_path: Path | None, *, max_dims: int = 16) -> list[str]:
    if not embedding_path:
        return []
    payload = json.loads(Path(embedding_path).expanduser().read_text(encoding="utf-8"))
    values = payload.get("embedding", [])
    if not isinstance(values, list) or not values:
        return []
    numeric = [float(value) for value in values[:max_dims]]
    tokens: list[str] = []
    for index, value in enumerate(numeric):
        bucket = max(0, min(7, int((value + 1.0) * 4)))
        tokens.append(f"embedding:d{index}:b{bucket}")
    norm = sum(value * value for value in numeric) ** 0.5
    tokens.append(f"embedding:norm:b{max(0, min(7, int(norm)))}")
    return tokens


def _feature_tokens_from_payload(payload: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    duration = payload.get("duration_seconds")
    if isinstance(duration, int | float):
        if duration < 10:
            tokens.append("duration:short")
        elif duration < 30:
            tokens.append("duration:medium")
        else:
            tokens.append("duration:long")
    token_count = payload.get("token_count")
    if isinstance(token_count, int):
        if token_count < 128:
            tokens.append("density:low")
        elif token_count < 512:
            tokens.append("density:medium")
        else:
            tokens.append("density:high")
    return tokens


def _normalize_token_value(value: object) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace(",", "_")
        .replace(":", "_")
    )


def _build_vocab(token_sequences: list[list[str]]) -> dict[str, int]:
    vocab = {PAD_TOKEN: 0, BOS_TOKEN: 1, EOS_TOKEN: 2}
    for token in sorted({token for sequence in token_sequences for token in sequence}):
        vocab.setdefault(token, len(vocab))
    return vocab


def _build_windows(
    encoded_sequences: list[list[int]],
    *,
    sequence_length: int,
    pad_id: int,
) -> list[list[int]]:
    window_size = sequence_length + 1
    windows: list[list[int]] = []
    for sequence in encoded_sequences:
        if len(sequence) <= window_size:
            windows.append(sequence + [pad_id] * (window_size - len(sequence)))
            continue
        stride = max(sequence_length // 2, 1)
        for start in range(0, len(sequence) - window_size + 1, stride):
            windows.append(sequence[start : start + window_size])
    return windows


def _sample_next_id(logits, *, torch, top_k: int | None = 50, top_p: float | None = 0.95) -> int:
    if top_k is not None and top_k > 0 and top_k < logits.numel():
        logits, indices = torch.topk(logits, top_k)
    else:
        indices = torch.arange(logits.numel())
    probabilities = torch.softmax(logits, dim=-1)
    if top_p is not None and 0.0 < top_p < 1.0 and probabilities.numel() > 1:
        sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
        keep = cumulative <= float(top_p)
        keep[0] = True
        filtered_probabilities = sorted_probabilities[keep]
        filtered_indices = sorted_indices[keep]
        filtered_probabilities = filtered_probabilities / filtered_probabilities.sum()
        sampled = torch.multinomial(filtered_probabilities, num_samples=1)
        return int(indices[filtered_indices[sampled]].item())
    sampled = torch.multinomial(probabilities, num_samples=1)
    return int(indices[sampled].item())


def _torch_modules():
    try:
        import torch
        from torch import nn
        from torch.utils import data
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch es necesario para entrenar/generar con Transformer. "
            "Instala las dependencias opcionales con .venv/bin/python -m pip install -e '.[ml]'."
        ) from exc
    if not math.isfinite(1.0):
        raise RuntimeError("Estado numérico inválido.")
    return torch, nn, data
