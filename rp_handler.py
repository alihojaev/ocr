import base64
import io
from typing import Any, Dict, List, Optional

import runpod
from PIL import Image
import numpy as np
import cv2

_ocr = None


def get_ocr(lang: str = "en"):
    global _ocr
    # Keep single instance per cold start; ignore lang change for serverless simplicity
    if _ocr is None:
        from paddleocr import PaddleOCR  # type: ignore
        _ocr = PaddleOCR(use_angle_cls=True, lang=lang)
    return _ocr


def _strip_data_url(b64: str) -> str:
    if b64.startswith("data:") and ";base64," in b64:
        return b64.split(",", 1)[1]
    return b64


def _b64_to_bytes(data_b64: str) -> bytes:
    b64_clean = _strip_data_url(data_b64).strip()
    padding = len(b64_clean) % 4
    if padding:
        b64_clean += "=" * (4 - padding)
    return base64.b64decode(b64_clean)


def _bytes_to_image(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _pil_to_cv2_bgr(img: Image.Image) -> "np.ndarray":
    arr = np.array(img)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    try:
        payload = (event or {}).get("input", {})
        image_field: Optional[str] = payload.get("image")
        lang: str = payload.get("lang", "en")

        if not image_field:
            return {"error": "Missing 'image' (base64)"}

        image_bytes = _b64_to_bytes(image_field)
        image = _bytes_to_image(image_bytes)
        image_bgr = _pil_to_cv2_bgr(image)

        ocr = get_ocr(lang)
        result = ocr.ocr(image_bgr, cls=True)

        lines: List[dict] = []
        for page in result:
            for det in page:
                bbox = det[0]
                text = det[1][0]
                score = float(det[1][1])
                lines.append({"bbox": bbox, "text": text, "score": score})

        return {"results": lines}
    except Exception as e:
        return {"error": f"OCR failed: {e}"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})


