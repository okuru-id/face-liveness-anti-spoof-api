from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    app_name: str = "Liveness Detection API"
    app_version: str = "1.0.0"
    model_version: str = "liveness-v1.0"

    api_keys: list[str] = ["dev-test-key-001"]
    max_image_size_bytes: int = 5 * 1024 * 1024
    min_resolution: int = 320
    min_face_size: int = 100
    blur_threshold: float = 100.0
    brightness_min: float = 40.0
    brightness_max: float = 220.0
    threshold_profile: str = "dev"
    live_threshold: float = 0.6
    spoof_threshold: float = 0.4
    prod_live_threshold: float = 0.65
    prod_spoof_threshold: float = 0.35

    rate_limit_per_minute: int = 60

    anti_spoof_model_path: str = "models/active/antispoof/best_2.7_80x80_MiniFASNetV2.onnx,models/active/antispoof/best_4_0_0_80x80_MiniFASNetV1SE.onnx"
    fft_enabled: bool = True
    fft_weight: float = 0.4
    fas_weight: float = 0.6
    fft_log_scale: bool = True
    fft_spoof_override_threshold: float = 0.6
    blur_fft_spoof_threshold: float = 0.6
    blurry_high_fft_spoof_threshold: float = 0.46
    blurry_low_fft_spoof_threshold: float = 0.34
    blurry_high_blur_min: float = 40.0
    blurry_low_blur_max: float = 25.0
    prod_fft_spoof_override_threshold: float = 0.62
    prod_blur_fft_spoof_threshold: float = 0.62
    prod_blurry_high_fft_spoof_threshold: float = 0.48
    prod_blurry_low_fft_spoof_threshold: float = 0.33
    prod_blurry_high_blur_min: float = 45.0
    prod_blurry_low_blur_max: float = 22.0
    rppg_model_path: str = "models/active/rppg/UBFC-rPPG_PhysNet_DiffNormalized.onnx"
    retinaface_deploy_path: str = "models/active/face_detector/deploy.prototxt"
    retinaface_caffemodel_path: str = "models/active/face_detector/Widerface-RetinaFace.caffemodel"

    onnx_intra_op_threads: int = 2
    onnx_inter_op_threads: int = 1
    stream_window_ms: int = 3000
    stream_min_frames: int = 6
    stream_frame_rate: float = 6.0
    stream_session_ttl_ms: int = 120000
    stream_fast_reject_spoof_confidence: float = 0.95
    stream_quality_gate_enabled: bool = False
    fusion_live_threshold: float = 0.7
    fusion_spoof_threshold: float = 0.3

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def is_prod_threshold_profile(self) -> bool:
        return str(self.threshold_profile).strip().lower() == "prod"

    @property
    def effective_live_threshold(self) -> float:
        return self.prod_live_threshold if self.is_prod_threshold_profile else self.live_threshold

    @property
    def effective_spoof_threshold(self) -> float:
        return self.prod_spoof_threshold if self.is_prod_threshold_profile else self.spoof_threshold

    @property
    def effective_fft_spoof_override_threshold(self) -> float:
        return self.prod_fft_spoof_override_threshold if self.is_prod_threshold_profile else self.fft_spoof_override_threshold

    @property
    def effective_blur_fft_spoof_threshold(self) -> float:
        return self.prod_blur_fft_spoof_threshold if self.is_prod_threshold_profile else self.blur_fft_spoof_threshold

    @property
    def effective_blurry_high_fft_spoof_threshold(self) -> float:
        return self.prod_blurry_high_fft_spoof_threshold if self.is_prod_threshold_profile else self.blurry_high_fft_spoof_threshold

    @property
    def effective_blurry_low_fft_spoof_threshold(self) -> float:
        return self.prod_blurry_low_fft_spoof_threshold if self.is_prod_threshold_profile else self.blurry_low_fft_spoof_threshold

    @property
    def effective_blurry_high_blur_min(self) -> float:
        return self.prod_blurry_high_blur_min if self.is_prod_threshold_profile else self.blurry_high_blur_min

    @property
    def effective_blurry_low_blur_max(self) -> float:
        return self.prod_blurry_low_blur_max if self.is_prod_threshold_profile else self.blurry_low_blur_max


settings = Settings()
