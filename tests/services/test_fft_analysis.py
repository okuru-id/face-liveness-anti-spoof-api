import numpy as np
from app.services.fft_analysis import compute_fft_score

def test_fft_score_in_valid_range():
    fake_face = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
    score = compute_fft_score(fake_face)
    assert 0 <= score <= 1

def test_fft_score_high_for_noise():
    noise_image = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
    score = compute_fft_score(noise_image)
    assert score > 0

def test_fft_score_for_uniform():
    uniform_image = np.ones((80, 80, 3), dtype=np.uint8) * 128
    score = compute_fft_score(uniform_image)
    assert score >= 0
