import numpy as np

from hybrid_music_engine.fusion.latent_blend import blend_embeddings


def test_blend_embeddings_midpoint():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)

    blended = blend_embeddings(a, b, 0.5)

    assert np.allclose(blended, np.array([0.5, 0.5], dtype=np.float32))
