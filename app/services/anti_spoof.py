from enum import Enum
from typing import NamedTuple
import cv2
import numpy as np
from pathlib import Path
from app.core.config import settings
from app.core.errors import ModelUnavailableError
from app.vendor.silent_face_utility import get_kernel, parse_model_name
from app.vendor.mini_fasnet import MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE': MiniFASNetV1SE,
    'MiniFASNetV2SE': MiniFASNetV2SE,
}


class SpoofLabel(str, Enum):
    LIVE = 'LIVE'
    SPOOF = 'SPOOF'


class AntiSpoofResult(NamedTuple):
    label: SpoofLabel
    confidence: float
    debug: dict | None = None


class AntiSpoofService:
    def __init__(self, model_path: str | None = None):
        raw_paths = model_path or settings.anti_spoof_model_path
        self.model_paths = [Path(part.strip()) for part in raw_paths.split(',') if part.strip()]
        self._models: list[dict] = []

    def _load(self):
        if self._models:
            return
        try:
            import torch

            device = torch.device('cpu')
            for model_path in self.model_paths:
                model_name = model_path.name
                h_input, w_input, model_type, scale = parse_model_name(model_name)
                kernel_size = get_kernel(h_input, w_input)
                model = MODEL_MAPPING[model_type](conv6_kernel=kernel_size).to(device)
                state_dict = torch.load(str(model_path), map_location=device)
                keys = iter(state_dict)
                first_layer_name = next(keys)
                if first_layer_name.find('module.') >= 0:
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for key, value in state_dict.items():
                        new_state_dict[key[7:]] = value
                    model.load_state_dict(new_state_dict)
                else:
                    model.load_state_dict(state_dict)
                model.eval()
                self._models.append({
                    'path': model_path,
                    'model': model,
                    'scale': scale,
                    'out_h': h_input,
                    'out_w': w_input,
                    'device': device,
                })
        except Exception as e:
            raise ModelUnavailableError(f'Failed to load anti-spoof model: {e}')

    def predict(self, face_crop: np.ndarray, bbox: tuple[int, int, int, int] | None = None, full_image: np.ndarray | None = None) -> AntiSpoofResult:
        self._load()
        if not self._models:
            raise ModelUnavailableError('No anti-spoof model configured')

        import torch
        prediction = np.zeros((1, 3), dtype=np.float32)
        per_model_debug = []
        for model_cfg in self._models:
            if full_image is not None and bbox is not None and model_cfg['scale'] is not None:
                crop = self._scale_crop(full_image, bbox, scale=model_cfg['scale'])
            elif full_image is not None and bbox is not None:
                crop = cv2.resize(full_image, (model_cfg['out_w'], model_cfg['out_h']))
            else:
                crop = face_crop

            if crop.size == 0:
                continue
            resized = cv2.resize(crop, (model_cfg['out_w'], model_cfg['out_h']), interpolation=cv2.INTER_LINEAR)
            img = resized.astype(np.float32)
            img = np.transpose(img, (2, 0, 1))
            tensor = torch.from_numpy(img).unsqueeze(0).to(model_cfg['device'])
            with torch.no_grad():
                out = model_cfg['model'](tensor)
                probs = torch.softmax(out, dim=1).cpu().numpy()
            prediction += probs
            per_model_debug.append({
                'model': model_cfg['path'].name,
                'scale': model_cfg['scale'],
                'probs': [round(float(x), 6) for x in probs[0].tolist()],
                'pred_label': int(np.argmax(probs[0])),
            })

        pred_label = int(np.argmax(prediction))
        confidence = float(prediction[0][pred_label] / max(len(self._models), 1))
        label = SpoofLabel.LIVE if pred_label == 1 else SpoofLabel.SPOOF
        debug = {
            'summed_probs': [round(float(x), 6) for x in prediction[0].tolist()],
            'avg_probs': [round(float(x / max(len(self._models), 1)), 6) for x in prediction[0].tolist()],
            'pred_label': pred_label,
            'pred_label_name': 'LIVE' if pred_label == 1 else 'SPOOF',
            'models': per_model_debug,
        }
        return AntiSpoofResult(label=label, confidence=confidence, debug=debug)

    def _scale_crop(self, image: np.ndarray, bbox: tuple[int, int, int, int], scale: float = 2.7) -> np.ndarray:
        x, y, box_w, box_h = bbox
        src_h, src_w = image.shape[:2]
        scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))
        new_width = box_w * scale
        new_height = box_h * scale
        center_x, center_y = box_w / 2 + x, box_h / 2 + y
        left_top_x = center_x - new_width / 2
        left_top_y = center_y - new_height / 2
        right_bottom_x = center_x + new_width / 2
        right_bottom_y = center_y + new_height / 2
        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0
        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0
        if right_bottom_x > src_w - 1:
            left_top_x -= right_bottom_x - src_w + 1
            right_bottom_x = src_w - 1
        if right_bottom_y > src_h - 1:
            left_top_y -= right_bottom_y - src_h + 1
            right_bottom_y = src_h - 1
        left_top_x = int(left_top_x)
        left_top_y = int(left_top_y)
        right_bottom_x = int(right_bottom_x)
        right_bottom_y = int(right_bottom_y)
        return image[left_top_y:right_bottom_y + 1, left_top_x:right_bottom_x + 1]


anti_spoof_service = AntiSpoofService()
