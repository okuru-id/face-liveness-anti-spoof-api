import numpy as np
import cv2
from app.core.config import settings


class FFTService:
    def compute_fft_score(self, face_crop: np.ndarray) -> float:
        """Compute a multi-metric anti-spoof score.

        Returns [0, 1]:
        - Low  → real face (smooth skin, natural texture)
        - High → screen/replay spoof (Moiré patterns, sharp screen pixels, JPEG artifacts)
        """
        if face_crop.size == 0:
            return 0.0

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

        h_img, w_img = gray.shape
        min_dim = min(h_img, w_img)
        gray_sq = cv2.resize(gray, (min_dim, min_dim))

        # --- 1. FFT high-frequency ratio ------------------------------------
        f = np.fft.fft2(gray_sq.astype(np.float32))
        fshift = np.fft.fftshift(f)
        mag = np.log(1 + np.abs(fshift))
        H, W = mag.shape
        cy, cx = H // 2, W // 2
        Y, X = np.ogrid[:H, :W]
        dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)

        low_mask = (dist > 2) & (dist <= max(min_dim // 16, 4))
        high_mask = dist > max(min_dim // 4, 16)

        low_energy = float(mag[low_mask].mean()) if np.any(low_mask) else 0.0
        high_energy = float(mag[high_mask].mean()) if np.any(high_mask) else 0.0

        fft_ratio = high_energy / (low_energy + 1e-6)
        fft_score = float(np.clip(fft_ratio / (fft_ratio + 1.0), 0.0, 1.0))

        # --- 2. JPEG compression artifact --------------------------------------
        _, buf = cv2.imencode('.jpg', gray, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        decoded = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        jpeg_diff = float(np.mean(np.abs(gray.astype(float) - decoded.astype(float))))

        # --- 3. Edge density ---------------------------------------------------
        edges = cv2.Canny(gray, 80, 200)
        edge_density = float(edges.mean())

        # --- 4. Laplacian variance (sharpness / blur) ------------------------
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # --- Combine signals (simple weighted sum, all normalised to [0,1]) ---
        # fft_score  already [0,1]
        jpeg_norm = float(np.clip(jpeg_diff / 6.0, 0, 1))
        edge_norm = float(np.clip(edge_density / 8.0, 0, 1))
        # Laplacian: moderate values (~80-120) = live; extreme high (>300) or very low (<30) = spoof
        lap_norm = 0.0
        if lap_var < 50:
            lap_norm = 1.0 - (lap_var / 50.0)
        elif lap_var > 200:
            lap_norm = min((lap_var - 200.0) / 300.0, 1.0)
        else:
            lap_norm = 0.0

        # Weighted combination: FFT dominant, edge secondary, jpeg/laplacian auxiliary
        score = (0.45 * fft_score + 0.30 * edge_norm + 0.15 * jpeg_norm + 0.10 * lap_norm)
        return float(np.clip(score, 0.0, 1.0))


_fft_service = None

def get_fft_service():
    global _fft_service
    if _fft_service is None:
        _fft_service = FFTService()
    return _fft_service
