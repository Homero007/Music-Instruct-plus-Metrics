"""
adapter.py — MusicGenInstructAdapter: el wrapper de integración.

Responsabilidades:

  1. Cargar `MusicgenForCausalLM` (con o sin pesos preentrenados).
  2. Para cada capa del decoder:
        - Parchear con AudioFusionModule (insertado tras la cross-attn de texto).
        - Inyectar LoRA en `encoder_attn` (cross-attn de texto)
          y en `audio_fusion.cross_attn` (cross-attn de audio).
  3. Congelar TODO menos los parámetros del adaptador (LoRA + gates).
  4. Forward con teacher forcing que:
        - aplica el delay pattern,
        - mete `(audio_embed, audio_mask)` por contexto,
        - delega en MusicGen.

  5. Guardar/cargar SOLO los adaptadores y la config (el modelo base se
     re-construye o se recarga desde HF separately).

Espera que `transformers` esté instalado. La API se reduce a lo que necesita
el bucle de entrenamiento (`finetune.train_loop.train`), así que el adapter
es plug-in: misma interfaz que `InstructEditableModel` (forward + save/load
adapter + trainable_parameters).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn

from hybrid_music_engine.instruct.lora import (
    adapter_parameters,
    count_parameters,
    inject_lora,
    mark_only_lora_trainable,
)

from .delay_pattern import LABEL_IGNORE, prepare_teacher_forcing_inputs
from .locate import DecoderRefs, cross_attn_modules, locate_decoder
from .patch_layer import audio_memory, patch_decoder_layer

log = logging.getLogger(__name__)


@dataclass
class MusicGenAdapterConfig:
    """Configuración del adapter (no de la arquitectura MusicGen subyacente).

    El modelo base puede venir de tres fuentes:
      - pretrained_name: descarga "facebook/musicgen-small" (requiere red)
      - state_dict_path: pesos locales en disco
      - config_only_for_test: construye el modelo aleatorio (solo para tests)
    """
    pretrained_name: str | None = None       # ej. "facebook/musicgen-small"
    state_dict_path: str | None = None       # alternativa: ruta a state_dict
    config_only_for_test: dict | None = None # para tests: dict con DecoderConfig de juguete

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    inject_text_cross: bool = True
    inject_audio_fusion: bool = True

    # AudioFusionModule
    d_audio: int = 128                       # dim del embedding EnCodec (encodec_32khz = 128)
    audio_fusion_n_heads: int | None = None  # None → mismo que el decoder
    audio_fusion_dropout: float = 0.0


class MusicGenInstructAdapter(nn.Module):
    """
    Envuelve un MusicgenForCausalLM y lo prepara para fine-tuning de edición.

    Uso:
        adapter = MusicGenInstructAdapter(MusicGenAdapterConfig(
            pretrained_name="facebook/musicgen-small", d_audio=128,
        ))
        adapter.prepare_for_finetuning()
        opt = torch.optim.AdamW(list(adapter.trainable_parameters()), lr=1e-4)

        # En el batch:
        loss = adapter.compute_loss(
            target_codes=codes,                 # (B, K, T)
            text_hidden=t5_seq,                 # (B, L, hidden_size)
            text_mask=t5_mask,                  # (B, L) bool
            audio_embed=src_embed,              # (B, S, d_audio)
            audio_mask=src_mask,                # (B, S) bool
        )
    """

    def __init__(self, cfg: MusicGenAdapterConfig):
        super().__init__()
        self.cfg = cfg
        self.model = _build_base_model(cfg)
        self.refs: DecoderRefs = locate_decoder(self.model)
        self._prepared = False

        # text_in: proyecta T5 (768) → hidden_size del decoder cuando no coinciden.
        # En MusicGen-small el text_encoder ya es T5-base (768) y el decoder es
        # 1024 → necesitamos esta proyección. La marcamos como NO entrenable
        # por consistencia con el espíritu de "solo adaptador entrena" — si
        # quieres entrenarla, llama `enable_text_projection_training()`.
        self.text_in: nn.Linear | None = None

    # ── Configuración pre-entrenamiento ──────────────────────────────────────

    def prepare_for_finetuning(self, t5_dim: int = 768) -> dict:
        """
        Parchea las capas con AudioFusionModule, inyecta LoRA en las cross-
        attentions, congela el resto. Idempotente.

        Args:
          t5_dim: dimensión de los hidden states de T5. 768 para T5-base
                  (el que usa MusicGen oficial).
        """
        if self._prepared:
            return self.parameter_summary()

        d_model = self.refs.hidden_size
        n_heads = self.cfg.audio_fusion_n_heads or self.refs.num_heads

        # 1) Patchear cada capa con AudioFusionModule (gate=0 → identidad inicial)
        patched = 0
        for layer in self.refs.layers:
            patch_decoder_layer(
                layer,
                d_audio=self.cfg.d_audio,
                n_heads=n_heads,
                dropout=self.cfg.audio_fusion_dropout,
            )
            patched += 1

        # 2) Inyectar LoRA en las cross-attentions
        injected_text = 0
        injected_audio = 0
        if self.cfg.inject_text_cross:
            for attn in cross_attn_modules(self.refs):
                injected_text += inject_lora(
                    attn,
                    target_names=("q_proj", "k_proj", "v_proj"),
                    r=self.cfg.lora_r,
                    alpha=self.cfg.lora_alpha,
                    dropout=self.cfg.lora_dropout,
                )
        if self.cfg.inject_audio_fusion:
            for layer in self.refs.layers:
                injected_audio += inject_lora(
                    layer.audio_fusion.cross_attn,
                    target_names=("q_proj", "k_proj", "v_proj"),
                    r=self.cfg.lora_r,
                    alpha=self.cfg.lora_alpha,
                    dropout=self.cfg.lora_dropout,
                )

        # 3) Proyección T5→decoder si las dimensiones no coinciden.
        #    En entornos donde T5 ya esté proyectado, t5_dim == d_model y
        #    text_in se queda como Identity (cero parámetros nuevos).
        if t5_dim != d_model:
            self.text_in = nn.Linear(t5_dim, d_model, bias=False)
            # NO entrenable por defecto: si quieres entrenarla, llama
            # `enable_text_projection_training()` después.
            for p in self.text_in.parameters():
                p.requires_grad = False
        else:
            self.text_in = nn.Identity()  # type: ignore[assignment]

        # 4) Congelar todo lo que no sea adaptador
        mark_only_lora_trainable(self.model, train_gates=True)
        self._prepared = True

        summary = self.parameter_summary()
        summary["patched_layers"] = patched
        summary["injected_linears"] = {"text_cross": injected_text, "audio_fusion": injected_audio}
        log.info(
            "MusicGen adapter listo: %d capas parcheadas | LoRA text=%d, audio=%d | "
            "entrenables=%d/%d (%.3f%%)",
            patched, injected_text, injected_audio,
            summary["trainable"], summary["total"], summary["trainable_pct"],
        )
        return summary

    def enable_text_projection_training(self) -> None:
        """
        Opcional: hace entrenable la proyección T5→decoder. Útil cuando los
        rangos numéricos de los hidden states de tu T5 (e.g. uno custom)
        difieren del T5 que MusicGen vio en preentrenamiento.
        """
        if isinstance(self.text_in, nn.Linear):
            for p in self.text_in.parameters():
                p.requires_grad = True

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        target_codes: torch.Tensor,            # (B, K, T) síncronos
        text_hidden: torch.Tensor,             # (B, L, t5_dim)
        text_mask: torch.Tensor | None = None, # (B, L) bool
        audio_embed: torch.Tensor | None = None, # (B, S, d_audio)
        audio_mask: torch.Tensor | None = None,  # (B, S) bool
    ) -> dict:
        """
        Forward de entrenamiento. Devuelve `{"loss": ..., "logits": ...}`.

        El delay pattern se aplica internamente. La pérdida es la que MusicGen
        calcula con `labels=` (cross-entropy con `ignore_index=LABEL_IGNORE`).
        """
        if not self._prepared:
            raise RuntimeError(
                "Llama prepare_for_finetuning() antes del forward."
            )

        # 1) Delay pattern → (B*K, T+K) para input_ids y para labels base
        input_ids, labels_flat = prepare_teacher_forcing_inputs(target_codes, self.refs)

        # MusicGen espera labels en forma (B, seq_len, num_codebooks). Nosotros
        # producimos (B*K, T_d) por compatibilidad con el embed_tokens del
        # decoder (que sí espera (B*K, T_d)). Reshape solo para los labels.
        B = target_codes.shape[0]
        K = target_codes.shape[1]
        T_d = labels_flat.shape[-1]
        # (B*K, T_d) → (B, K, T_d) → (B, T_d, K)  ←  el orden que pide HF
        labels = labels_flat.reshape(B, K, T_d).permute(0, 2, 1).contiguous()

        # 2) Proyección T5 si hace falta
        text_proj = self.text_in(text_hidden) if self.text_in is not None else text_hidden

        # 3) Forward con audio en contexto
        with audio_memory(audio_embed, audio_mask):
            out = self.model(
                input_ids=input_ids,
                encoder_hidden_states=text_proj,
                encoder_attention_mask=text_mask,
                labels=labels,
            )
        return {"loss": out.loss, "logits": out.logits, "labels": labels}

    def compute_loss(self, **kwargs) -> torch.Tensor:
        """Atajo: solo la pérdida (lo que el bucle de entrenamiento necesita)."""
        return self.forward(**kwargs)["loss"]

    # ── Introspección ────────────────────────────────────────────────────────

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Parámetros del adaptador: LoRA + gates (+ text_in si fue habilitada)."""
        # Reutilizamos el iterador del módulo lora.py para LoRA+gates dentro
        # del modelo, y añadimos manualmente text_in si está entrenable.
        for p in adapter_parameters(self.model):
            yield p
        if self.text_in is not None:
            for p in self.text_in.parameters():
                if p.requires_grad:
                    yield p

    def parameter_summary(self) -> dict:
        return count_parameters(self.model)

    # ── Persistencia ─────────────────────────────────────────────────────────

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        """Solo lo entrenable: LoRA + gates + text_in si está habilitada."""
        out: dict[str, torch.Tensor] = {}
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                out[f"model.{name}"] = p.detach().cpu().clone()
        if self.text_in is not None and isinstance(self.text_in, nn.Linear):
            for name, p in self.text_in.named_parameters():
                if p.requires_grad:
                    out[f"text_in.{name}"] = p.detach().cpu().clone()
        return out

    def save_adapter(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.cfg),
                "adapter_state_dict": self.adapter_state_dict(),
                "t5_dim": (self.text_in.in_features
                           if isinstance(self.text_in, nn.Linear) else None),
            },
            path,
        )
        return path

    @classmethod
    def load_adapter(
        cls,
        path: str | Path,
        device: str = "cpu",
    ) -> "MusicGenInstructAdapter":
        blob = torch.load(path, map_location=device, weights_only=False)
        cfg = MusicGenAdapterConfig(**blob["config"])
        adapter = cls(cfg)
        adapter.prepare_for_finetuning(t5_dim=blob.get("t5_dim") or 768)
        # Cargar los pesos del adaptador en su sitio
        own_model = dict(adapter.model.named_parameters())
        own_text = dict(adapter.text_in.named_parameters()) if isinstance(adapter.text_in, nn.Linear) else {}
        missing = []
        for name, t in blob["adapter_state_dict"].items():
            if name.startswith("model."):
                key = name[len("model."):]
                if key in own_model:
                    own_model[key].data.copy_(t.to(device))
                else:
                    missing.append(name)
            elif name.startswith("text_in."):
                key = name[len("text_in."):]
                if key in own_text:
                    own_text[key].data.copy_(t.to(device))
                else:
                    missing.append(name)
        if missing:
            log.warning("Adaptadores no cargados (%d): %s", len(missing), missing[:5])
        adapter.to(device)
        return adapter


# ── Construcción del modelo base ──────────────────────────────────────────────

def _build_base_model(cfg: MusicGenAdapterConfig) -> nn.Module:
    """Resuelve la fuente del modelo base según la config."""
    n_set = sum(x is not None for x in
                (cfg.pretrained_name, cfg.state_dict_path, cfg.config_only_for_test))
    if n_set != 1:
        raise ValueError(
            "Exactamente UNA de pretrained_name / state_dict_path / "
            "config_only_for_test debe estar definida."
        )

    from transformers import MusicgenDecoderConfig, MusicgenForCausalLM

    if cfg.pretrained_name:
        # Requiere red abierta. Carga pesos preentrenados de HF.
        # En la práctica MusicgenForConditionalGeneration es el más usado
        # porque incluye el T5 encoder; pero para fine-tuning de edición
        # usamos el ForCausalLM (decoder solo) y le pasamos hidden_states
        # de T5 ya pre-computados.
        log.info("Cargando MusicGen preentrenado: %s", cfg.pretrained_name)
        return MusicgenForCausalLM.from_pretrained(cfg.pretrained_name)

    if cfg.state_dict_path:
        log.info("Cargando state_dict local: %s", cfg.state_dict_path)
        # Hace falta saber la config para construir el modelo. Asumimos que
        # el state_dict incluye el config; si no, el usuario debe pasarlo.
        sd = torch.load(cfg.state_dict_path, map_location="cpu", weights_only=False)
        if "config" in sd:
            model_cfg = MusicgenDecoderConfig(**sd["config"])
            state = sd["state_dict"]
        else:
            raise RuntimeError(
                "state_dict_path debe contener {'config': ..., 'state_dict': ...}"
            )
        model = MusicgenForCausalLM(model_cfg)
        model.load_state_dict(state, strict=False)
        return model

    # config_only_for_test
    log.info("Construyendo MusicGen aleatorio para test")
    return MusicgenForCausalLM(MusicgenDecoderConfig(**cfg.config_only_for_test))
