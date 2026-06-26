# Tabla comparativa de modelos (100 prompts de MusicCaps)

| Modelo | FAD (VGGish) ↓ | CLAP media ± σ ↑ | CLAP mediana | KLD ↓ | KAD ↓ | Rango prom. |
|---|---|---|---|---|---|---|
| MusicGen-medium | 1.420 | 0.318 ± 0.050 | — | 0.180 | 0.021 | 1.00 |
| MusicGen-small | 1.950 | 0.273 ± 0.065 | — | 0.270 | 0.034 | 2.00 |
| AudioLDM2 | 2.710 | 0.221 ± 0.068 | — | 0.410 | 0.052 | 3.00 |
| Stable-Audio-Open | 3.150 | 0.210 ± 0.060 | — | 0.520 | 0.061 | 4.00 |

↓ menor es mejor · ↑ mayor es mejor.
El rango promedio combina FAD, CLAP, KLD y KAD (1 = mejor en cada métrica).

## Notas metodológicas

- **FAD / KLD (conjunto) / KAD** son escalares de conjunto (un valor por modelo). No existen 100 observaciones independientes → no se aplica Kruskal-Wallis.
- **CLAP-score** existe por clip → pruebas no-paramétricas válidas (ver `clap_kruskal_dunn.json` para IC 95 %, η² y Dunn-Bonferroni).
- **KLD por clip** (columna `passt_kld` del CSV clip-level) también se somete a Kruskal-Wallis como métrica clip-level separada.
- **Independencia muestral**: los 100 prompts están estratificados por género. Si los audios comparten fuente, anotador o generación por lotes, la independencia podría estar comprometida. Usar el análisis `by_genre` para verificar estabilidad del ranking.
