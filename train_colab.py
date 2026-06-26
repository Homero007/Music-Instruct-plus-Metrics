#!/usr/bin/env python3
"""
Script CLI para entrenar el Transformer en Colab sin interfaz web.

Uso:
    python train_colab.py --epochs 100 --batch-size 8 --save-every 10
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader


class TokenDataset(Dataset):
    """Dataset de tokens sintéticos para demostración.

    Reproducibilidad robusta: cada ítem usa un generador propio sembrado con
    ``seed + idx``, así ``dataset[i]`` es determinista e independiente del orden
    de acceso (no depende del estado global de NumPy/torch).
    """

    def __init__(self, num_samples=100, seq_len=128, vocab_size=2048, seed=42):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.seed = seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed + int(idx))
        tokens = rng.integers(0, self.vocab_size, self.seq_len)
        return torch.tensor(tokens, dtype=torch.long)


class SimpleTransformer(nn.Module):
    """Transformer minimalista para tokens."""

    def __init__(self, vocab_size=2048, d_model=256, nhead=4, num_layers=2, seq_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.01)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512, batch_first=True, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoder[:, :x.shape[1], :]
        x = self.encoder(x)
        x = self.head(x)
        return x


def train_epoch(model, dataloader, optimizer, loss_fn, device):
    """Entrena una época."""
    model.train()
    total_loss = 0

    for batch in dataloader:
        batch = batch.to(device)
        x = batch[:, :-1]
        y = batch[:, 1:]

        logits = model(x)
        vocab_size = logits.shape[-1]
        loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Guarda checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat(),
    }, path)
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Entrena Transformer en Colab/Kaggle'
    )
    parser.add_argument('--epochs', type=int, default=50, help='Número de epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save-every', type=int, default=10, help='Guardar checkpoint cada N epochs')
    parser.add_argument('--num-samples', type=int, default=100, help='Muestras del dataset')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Directorio de checkpoints')
    parser.add_argument('--results-dir', type=str, default='./results', help='Directorio de resultados')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Reproducibilidad: siembra los RNG globales (init del modelo + barajado del
    # DataLoader). El dataset ya es determinista por índice de forma propia.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = Path(args.checkpoint_dir)
    results_dir = Path(args.results_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔥 Device: {device}")
    print(f"📁 Checkpoints: {checkpoint_dir}")
    print(f"📊 Resultados: {results_dir}")

    # Dataset y modelo
    dataset = TokenDataset(num_samples=args.num_samples, seed=args.seed)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = SimpleTransformer().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🤖 Modelo: {total_params:,} parámetros")
    print(f"⚙️  Configuración:")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Checkpoint cada: {args.save_every} epochs\n")

    # Entrenamiento
    metrics = {'epoch': [], 'loss': [], 'timestamp': []}
    start_time = datetime.now()

    try:
        for epoch in range(args.epochs):
            loss = train_epoch(model, dataloader, optimizer, loss_fn, device)
            metrics['epoch'].append(epoch + 1)
            metrics['loss'].append(float(loss))
            metrics['timestamp'].append(datetime.now().isoformat())

            if (epoch + 1) % args.save_every == 0:
                ckpt_path = checkpoint_dir / f'model_ep{epoch+1:03d}.pt'
                save_checkpoint(model, optimizer, epoch, loss, ckpt_path)
                print(f"Epoch {epoch+1:03d}/{args.epochs} | Loss: {loss:.4f} | 💾 Checkpoint guardado")
            elif (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1:03d}/{args.epochs} | Loss: {loss:.4f}")

        elapsed_min = (datetime.now() - start_time).total_seconds() / 60
        print(f"\n✅ Entrenamiento completado en {elapsed_min:.1f} minutos")

        # Guarda métricas
        metrics_path = results_dir / 'training_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"📊 Métricas guardadas: {metrics_path}")

        metrics_csv_path = results_dir / 'training_metrics.csv'
        with open(metrics_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'loss', 'timestamp'])
            writer.writeheader()
            for index, epoch_value in enumerate(metrics['epoch']):
                writer.writerow({
                    'epoch': epoch_value,
                    'loss': metrics['loss'][index],
                    'timestamp': metrics['timestamp'][index],
                })
        print(f"📊 Métricas CSV: {metrics_csv_path}")

        # Resumen
        summary = {
            'total_epochs': args.epochs,
            'final_loss': metrics['loss'][-1],
            'best_loss': min(metrics['loss']),
            'total_parameters': total_params,
            'num_samples': args.num_samples,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'save_every': args.save_every,
            'seed': args.seed,
            'training_time_minutes': elapsed_min,
            'training_metrics_csv': str(metrics_csv_path),
            'timestamp': datetime.now().isoformat(),
        }
        summary_path = results_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"📝 Resumen: {summary_path}")

        return 0

    except KeyboardInterrupt:
        print("\n⚠️  Entrenamiento interrumpido por usuario")
        # Salva checkpoint de emergencia
        emergency_path = checkpoint_dir / 'model_INTERRUPTED.pt'
        save_checkpoint(model, optimizer, epoch, loss, emergency_path)
        print(f"💾 Checkpoint de emergencia: {emergency_path}")
        return 1

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == '__main__':
    sys.exit(main())
