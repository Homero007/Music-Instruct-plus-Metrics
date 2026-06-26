from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hybrid_music_engine.core.config import EngineConfig
from hybrid_music_engine.storage.manifest import ProjectManifest, list_projects, project_path


@dataclass(frozen=True)
class FeatureMatrix:
    project_ids: list[str]
    feature_names: list[str]
    matrix: np.ndarray


def flatten_numeric_features(payload: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}

    def visit(prefix: str, item: Any) -> None:
        if isinstance(item, bool):
            values[prefix] = 1.0 if item else 0.0
        elif isinstance(item, int | float) and not isinstance(item, bool):
            number = float(item)
            if np.isfinite(number):
                values[prefix] = number
        elif isinstance(item, list):
            for index, value in enumerate(item):
                visit(f"{prefix}[{index}]", value)
        elif isinstance(item, dict):
            for key, value in sorted(item.items()):
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                visit(next_prefix, value)

    visit("", payload)
    return values


def vectorize_features(
    payload: dict[str, Any],
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str], dict[str, list[str]]]:
    flat = flatten_numeric_features(payload)
    names = feature_names or sorted(flat)
    vector = np.asarray([flat.get(name, 0.0) for name in names], dtype=np.float32)
    diagnostics = {
        "missing_features": [name for name in names if name not in flat],
        "extra_features": [name for name in sorted(flat) if name not in set(names)],
    }
    return vector, names, diagnostics


def load_feature_matrix(config: EngineConfig) -> FeatureMatrix:
    rows: list[tuple[str, dict[str, float]]] = []
    for project in list_projects(config):
        project_id = project.get("project_id")
        feature_path = project.get("features", {}).get("path")
        if not project_id or not feature_path:
            continue
        path = Path(str(feature_path))
        if not path.exists():
            continue
        features = json.loads(path.read_text(encoding="utf-8"))
        flat = flatten_numeric_features(features)
        if flat:
            rows.append((str(project_id), flat))

    if not rows:
        raise RuntimeError("No hay features disponibles. Ejecuta extract-features primero.")

    feature_names = sorted({name for _project_id, row in rows for name in row})
    matrix = np.asarray(
        [[row.get(name, 0.0) for name in feature_names] for _project_id, row in rows],
        dtype=np.float32,
    )
    return FeatureMatrix(
        project_ids=[project_id for project_id, _row in rows],
        feature_names=feature_names,
        matrix=matrix,
    )


def train_feature_vae(
    config: EngineConfig,
    *,
    latent_dim: int = 32,
    hidden_dim: int = 128,
    epochs: int = 200,
    learning_rate: float = 1e-3,
    beta: float = 0.001,
    seed: int = 42,
) -> dict[str, Any]:
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
    except ImportError as exc:
        raise RuntimeError(
            "El entrenamiento VAE requiere PyTorch. Instala: python -m pip install -e '.[ml]'"
        ) from exc

    feature_matrix = load_feature_matrix(config)
    torch.manual_seed(seed)
    np.random.seed(seed)

    matrix = feature_matrix.matrix
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std < 1e-6] = 1.0
    normalized = (matrix - mean) / std
    inputs = torch.tensor(normalized, dtype=torch.float32)

    model = FeatureVAE(
        input_dim=inputs.shape[1],
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        nn_module=nn,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(inputs)
        reconstruction_loss = F.mse_loss(reconstruction, inputs, reduction="mean")
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

    output_dir = config.data_dir / "embeddings" / "feature_vae"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pt"
    metadata_path = output_dir / "metadata.json"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(inputs.shape[1]),
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "feature_names": feature_matrix.feature_names,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "project_ids": feature_matrix.project_ids,
        },
        model_path,
    )
    metadata = {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "beta": beta,
        "seed": seed,
        "input_dim": int(inputs.shape[1]),
        "project_count": len(feature_matrix.project_ids),
        "project_ids": feature_matrix.project_ids,
        "history": history,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def encode_project_embedding(
    manifest: ProjectManifest,
    config: EngineConfig,
    *,
    model_path: Path | None = None,
) -> dict[str, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "La codificación VAE requiere PyTorch. Instala: python -m pip install -e '.[ml]'"
        ) from exc

    resolved_model_path = model_path or config.data_dir / "embeddings" / "feature_vae" / "model.pt"
    if not resolved_model_path.exists():
        raise RuntimeError("No hay modelo VAE entrenado. Ejecuta train-vae primero.")

    feature_path = manifest.features.get("path")
    if not feature_path:
        raise RuntimeError("El proyecto no tiene features. Ejecuta extract-features primero.")
    features = json.loads(Path(str(feature_path)).read_text(encoding="utf-8"))

    checkpoint = torch.load(resolved_model_path, map_location="cpu", weights_only=False)
    feature_names = list(checkpoint["feature_names"])
    vector, _names, diagnostics = vectorize_features(features, feature_names)
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    normalized = (vector - mean) / std

    model = FeatureVAE(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        latent_dim=int(checkpoint["latent_dim"]),
        nn_module=nn,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with torch.no_grad():
        tensor = torch.tensor(normalized.reshape(1, -1), dtype=torch.float32)
        mu, logvar = model.encode(tensor)
        reconstruction = model.decode(mu)
        reconstruction_error = torch.mean((reconstruction - tensor) ** 2)

    output_dir = project_path(config, manifest.project_id) / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "embedding.json"
    payload = {
        "project_id": manifest.project_id,
        "model_path": str(resolved_model_path),
        "path": str(output_path),
        "latent_dim": int(checkpoint["latent_dim"]),
        "embedding": [round(float(value), 8) for value in mu.numpy().reshape(-1)],
        "logvar": [round(float(value), 8) for value in logvar.numpy().reshape(-1)],
        "reconstruction_error": round(float(reconstruction_error), 8),
        "feature_count": len(feature_names),
        "diagnostics": diagnostics,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


class FeatureVAE:
    def __init__(self, *, input_dim: int, hidden_dim: int, latent_dim: int, nn_module: Any):
        nn = nn_module

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
                self.mu = nn.Linear(hidden_dim, latent_dim)
                self.logvar = nn.Linear(hidden_dim, latent_dim)
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, input_dim),
                )

            def encode(self, inputs):
                hidden = self.encoder(inputs)
                return self.mu(hidden), self.logvar(hidden)

            def reparameterize(self, mu, logvar):
                std = (0.5 * logvar).exp()
                eps = std.new_empty(std.shape).normal_()
                return mu + eps * std

            def decode(self, latent):
                return self.decoder(latent)

            def forward(self, inputs):
                mu, logvar = self.encode(inputs)
                latent = self.reparameterize(mu, logvar)
                return self.decode(latent), mu, logvar

        self._model = _Model()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._model(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._model, name)
