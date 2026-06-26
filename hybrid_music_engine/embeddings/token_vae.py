from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from hybrid_music_engine.core.config import EngineConfig
from hybrid_music_engine.core.ids import create_id


def train_token_vae(
    config: EngineConfig,
    *,
    token_manifest_path: Path,
    latent_dim: int = 32,
    hidden_dim: int = 128,
    epochs: int = 80,
    learning_rate: float = 1e-3,
    beta: float = 0.001,
    seed: int = 42,
) -> dict[str, Any]:
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
    except ImportError as exc:
        raise RuntimeError("Token-VAE requiere PyTorch.") from exc

    manifest_metadata = _manifest_metadata(token_manifest_path)
    sequences = _load_token_payloads(token_manifest_path)
    vocab = _build_vocab(sequences)
    matrix = np.asarray([_token_histogram(tokens, vocab) for tokens in sequences], dtype=np.float32)
    if matrix.size == 0:
        raise RuntimeError("No hay tokens válidos para entrenar Token-VAE.")

    torch.manual_seed(seed)
    inputs = torch.tensor(matrix, dtype=torch.float32)
    model = TokenVAE(input_dim=inputs.shape[1], hidden_dim=hidden_dim, latent_dim=latent_dim, nn_module=nn)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(inputs)
        reconstruction_loss = F.binary_cross_entropy(reconstruction, inputs, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = reconstruction_loss + beta * kl_loss
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch == epochs or epoch % max(epochs // 10, 1) == 0:
            history.append(
                {
                    "epoch": epoch,
                    "loss": round(float(loss.detach()), 8),
                    "reconstruction_loss": round(float(reconstruction_loss.detach()), 8),
                    "kl_loss": round(float(kl_loss.detach()), 8),
                }
            )

    model_id = create_id("token_vae", prefix="tokenvae")
    output_dir = config.data_dir / "embeddings" / "token_vae" / model_id
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pt"
    metadata_path = output_dir / "metadata.json"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(inputs.shape[1]),
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "vocab": vocab,
        },
        model_path,
    )
    metadata = {
        "schema_version": "token-vae-metadata-v1",
        "model_id": model_id,
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "token_manifest_path": str(Path(token_manifest_path).expanduser().resolve()),
        "source_processing_mode": manifest_metadata.get("processing_mode"),
        "intended_model": manifest_metadata.get("intended_model"),
        "warnings": _token_vae_manifest_warnings(manifest_metadata),
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "beta": beta,
        "seed": seed,
        "vocab_size": len(vocab),
        "sequence_count": len(sequences),
        "history": history,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _manifest_metadata(token_manifest_path: Path) -> dict[str, Any]:
    try:
        return json.loads(Path(token_manifest_path).expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _token_vae_manifest_warnings(manifest: dict[str, Any]) -> list[str]:
    if manifest.get("processing_mode") == "token_vae_demucs":
        return []
    return [
        "El manifest no fue creado con processing_mode='token_vae_demucs'. "
        "Token-VAE funcionará, pero la separación por capas puede ser menos precisa."
    ]


def encode_token_vae_embedding(
    config: EngineConfig,
    *,
    token_source_path: Path,
    model_path: Path | None = None,
    output_name: str = "token_embedding",
) -> dict[str, Any]:
    tokens = _load_tokens_from_source(token_source_path)
    if not tokens:
        raise RuntimeError("La fuente no contiene tokens para codificar.")
    encoded = _encode_tokens(config, tokens=tokens, model_path=model_path)

    output_id = create_id(output_name, prefix="tokenembedding")
    output_dir = config.data_dir / "embeddings" / "token_vae" / "encoded"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_id}.json"
    payload = {
        "schema_version": "token-vae-embedding-v1",
        "embedding_id": output_id,
        "source_path": str(Path(token_source_path).expanduser().resolve()),
        "model_path": str(encoded["model_path"]),
        "path": str(output_path),
        "latent_dim": int(encoded["latent_dim"]),
        "embedding": encoded["embedding"],
        "logvar": encoded["logvar"],
        "reconstruction_error": encoded["reconstruction_error"],
        "token_count": len(tokens),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def encode_token_vae_genre_embeddings(
    config: EngineConfig,
    *,
    token_manifest_path: Path,
    model_path: Path | None = None,
    output_name: str = "genre_embeddings",
) -> dict[str, Any]:
    grouped = _load_token_payloads_by_genre(token_manifest_path)
    if not grouped:
        raise RuntimeError("El manifest no contiene géneros con tokens válidos.")

    run_id = create_id(output_name, prefix="genreembeddings")
    output_dir = config.data_dir / "embeddings" / "token_vae" / "genres" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "genre_embeddings.json"
    embeddings: list[dict[str, Any]] = []
    for genre, bundle in sorted(grouped.items()):
        encoded = _encode_tokens(config, tokens=bundle["tokens"], model_path=model_path)
        safe_genre = _safe_name(genre)
        output_path = output_dir / f"{safe_genre}.json"
        payload = {
            "schema_version": "token-vae-genre-embedding-v1",
            "embedding_id": f"{run_id}-{safe_genre}",
            "run_id": run_id,
            "genre": genre,
            "source_manifest_path": str(Path(token_manifest_path).expanduser().resolve()),
            "model_path": str(encoded["model_path"]),
            "path": str(output_path),
            "latent_dim": int(encoded["latent_dim"]),
            "embedding": encoded["embedding"],
            "logvar": encoded["logvar"],
            "reconstruction_error": encoded["reconstruction_error"],
            "token_count": int(bundle["token_count"]),
            "file_count": int(bundle["file_count"]),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        embeddings.append(
            {
                "genre": genre,
                "path": str(output_path),
                "latent_dim": payload["latent_dim"],
                "token_count": payload["token_count"],
                "file_count": payload["file_count"],
                "reconstruction_error": payload["reconstruction_error"],
            }
        )

    summary = {
        "schema_version": "token-vae-genre-embeddings-v1",
        "run_id": run_id,
        "output_name": output_name,
        "source_manifest_path": str(Path(token_manifest_path).expanduser().resolve()),
        "model_path": str(_resolve_token_vae_model(config, model_path)),
        "path": str(summary_path),
        "genre_count": len(embeddings),
        "genres": [item["genre"] for item in embeddings],
        "embeddings": embeddings,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


class TokenVAE:
    def __init__(self, *, input_dim: int, hidden_dim: int, latent_dim: int, nn_module: Any):
        nn = nn_module

        class _Model(nn.Module):
            def __init__(inner_self) -> None:
                super().__init__()
                inner_self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
                inner_self.mu = nn.Linear(hidden_dim, latent_dim)
                inner_self.logvar = nn.Linear(hidden_dim, latent_dim)
                inner_self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, input_dim),
                    nn.Sigmoid(),
                )

            def encode(inner_self, x):
                hidden = inner_self.encoder(x)
                return inner_self.mu(hidden), inner_self.logvar(hidden)

            def reparameterize(inner_self, mu, logvar):
                std = (0.5 * logvar).exp()
                epsilon = std.new_empty(std.shape).normal_()
                return mu + epsilon * std

            def decode(inner_self, z):
                return inner_self.decoder(z)

            def forward(inner_self, x):
                mu, logvar = inner_self.encode(x)
                z = inner_self.reparameterize(mu, logvar)
                return inner_self.decode(z), mu, logvar

        self.module = _Model()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self.module, name)


def _load_token_payloads(token_manifest_path: Path) -> list[list[str]]:
    manifest = json.loads(Path(token_manifest_path).expanduser().read_text(encoding="utf-8"))
    sequences: list[list[str]] = []
    for entry in manifest.get("entries", []):
        path = Path(str(entry.get("path", ""))).expanduser()
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        tokens = [str(token) for token in payload.get("tokens", [])]
        if len(tokens) >= 4:
            sequences.append(tokens)
    if not sequences:
        raise RuntimeError("El manifest no contiene tokens válidos para Token-VAE.")
    return sequences


def _load_token_payloads_by_genre(token_manifest_path: Path) -> dict[str, dict[str, Any]]:
    manifest_path = Path(token_manifest_path).expanduser()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    grouped: dict[str, dict[str, Any]] = {}
    for entry in manifest.get("entries", []):
        path = Path(str(entry.get("path", ""))).expanduser()
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        tokens = [str(token) for token in payload.get("tokens", [])]
        if len(tokens) < 4:
            continue
        genre = str(entry.get("genre") or payload.get("genre") or "unknown").strip() or "unknown"
        bundle = grouped.setdefault(genre, {"tokens": [], "file_count": 0, "token_count": 0})
        bundle["tokens"].extend(tokens)
        bundle["file_count"] += 1
        bundle["token_count"] += len(tokens)
    return grouped


def _load_tokens_from_source(path: Path) -> list[str]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if "entries" in payload:
        sequences = _load_token_payloads(path)
        return [token for sequence in sequences for token in sequence]
    return [str(token) for token in payload.get("tokens", [])]


def _build_vocab(sequences: list[list[str]], max_vocab: int = 4096) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sequence in sequences:
        for token in sequence:
            counts[token] = counts.get(token, 0) + 1
    tokens = sorted(counts, key=lambda token: (-counts[token], token))[:max_vocab]
    return {token: index for index, token in enumerate(tokens)}


def _token_histogram(tokens: list[str], vocab: dict[str, int]) -> np.ndarray:
    vector = np.zeros(len(vocab), dtype=np.float32)
    for token in tokens:
        index = vocab.get(token)
        if index is not None:
            vector[index] += 1.0
    total = float(vector.sum())
    if total > 0:
        vector /= total
    return vector


def _latest_token_vae_model(config: EngineConfig) -> Path:
    models = sorted((config.data_dir / "embeddings" / "token_vae").glob("*/model.pt"))
    if not models:
        raise RuntimeError("No hay modelo Token-VAE entrenado.")
    return models[-1]


def _resolve_token_vae_model(config: EngineConfig, model_path: Path | None = None) -> Path:
    return Path(model_path).expanduser().resolve() if model_path else _latest_token_vae_model(config).resolve()


def _encode_tokens(config: EngineConfig, *, tokens: list[str], model_path: Path | None = None) -> dict[str, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("Token-VAE requiere PyTorch.") from exc

    resolved_model = _resolve_token_vae_model(config, model_path)
    checkpoint = torch.load(resolved_model, map_location="cpu", weights_only=False)
    vocab = {str(key): int(value) for key, value in checkpoint["vocab"].items()}
    vector = _token_histogram(tokens, vocab)
    model = TokenVAE(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        latent_dim=int(checkpoint["latent_dim"]),
        nn_module=nn,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(vector.reshape(1, -1), dtype=torch.float32)
        mu, logvar = model.encode(tensor)
        reconstruction = model.decode(mu)
        reconstruction_error = torch.mean((reconstruction - tensor) ** 2)
    return {
        "model_path": resolved_model,
        "latent_dim": int(checkpoint["latent_dim"]),
        "embedding": [round(float(value), 8) for value in mu.numpy().reshape(-1)],
        "logvar": [round(float(value), 8) for value in logvar.numpy().reshape(-1)],
        "reconstruction_error": round(float(reconstruction_error), 8),
    }


def _safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.lower())
    return cleaned.strip("_") or "unknown"
