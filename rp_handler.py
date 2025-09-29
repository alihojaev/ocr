import base64
import io
from typing import Any, Dict, List, Optional, Tuple

import runpod
from PIL import Image
import numpy as np
import cv2

_ocr_cache: Dict[Tuple, object] = {}


def get_ocr(
    *,
    lang: str = "en",
    det_limit_side_len: int = 1920,
    det_db_box_thresh: float = 0.4,
    det_db_unclip_ratio: float = 1.8,
    drop_score: float = 0.3,
    use_angle_cls: bool = True,
):
    key = (
        lang,
        int(det_limit_side_len),
        float(det_db_box_thresh),
        float(det_db_unclip_ratio),
        float(drop_score),
        bool(use_angle_cls),
    )
    inst = _ocr_cache.get(key)
    if inst is not None:
        return inst
    from paddleocr import PaddleOCR  # type: ignore
    inst = PaddleOCR(
        use_angle_cls=use_angle_cls,
        lang=lang,
        det_limit_side_len=det_limit_side_len,
        det_db_box_thresh=det_db_box_thresh,
        det_db_unclip_ratio=det_db_unclip_ratio,
        drop_score=drop_score,
    )
    _ocr_cache[key] = inst
    return inst


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


def _preprocess(
    bgr: "np.ndarray",
    *,
    scale: float = 1.0,
    apply_clahe: bool = True,
    denoise: bool = True,
    binarize: bool = False,
    gamma: float = 0.0,
) -> "np.ndarray":
    out = bgr
    if scale and scale != 1.0:
        h, w = out.shape[:2]
        out = cv2.resize(out, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    if gamma and gamma > 0:
        g = gamma
        table = ((np.arange(256) / 255.0) ** (1.0 / g) * 255.0).astype(np.uint8)
        out = cv2.LUT(out, table)
    if apply_clahe:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        lab2 = cv2.merge((l2, a, b))
        out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    if denoise:
        out = cv2.bilateralFilter(out, d=7, sigmaColor=75, sigmaSpace=75)
    if binarize:
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        thr = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11
        )
        out = cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)
    return out


def _polygon_area(poly: List[List[float]]) -> float:
    area = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def _filter_results(
    results: List[Dict],
    image_shape: Tuple[int, int, int],
    *,
    min_area_ratio: float = 0.0002,
    min_height_px: int = 12,
    min_text_len: int = 2,
    filter_punct: bool = True,
) -> List[Dict]:
    h, w = image_shape[:2]
    area_img = float(h * w)
    out: List[Dict] = []
    for r in results:
        bbox = r["bbox"]
        text = (r["text"] or "").strip()
        ymin = min(p[1] for p in bbox)
        ymax = max(p[1] for p in bbox)
        height = float(ymax - ymin)
        area = _polygon_area(bbox)
        if area < max(1.0, min_area_ratio * area_img):
            continue
        if height < float(min_height_px):
            continue
        if min_text_len > 0 and len(text) < min_text_len:
            if filter_punct and not any(ch.isalnum() for ch in text):
                continue
        out.append(r)
    return out


def _parse_hex_color_to_bgr(color_hex: str) -> tuple:
    s = color_hex.strip().lstrip("#")
    if len(s) == 6:
        r = int(s[0:2], 16); g = int(s[2:4], 16); b = int(s[4:6], 16)
    elif len(s) == 3:
        r = int(s[0]*2, 16); g = int(s[1]*2, 16); b = int(s[2]*2, 16)
    else:
        r, g, b = 255, 0, 0
    return (b, g, r)

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    try:
        payload = (event or {}).get("input", {})
        image_field: Optional[str] = payload.get("image")
        lang: str = payload.get("lang", "en")
        lang_alt: Optional[str] = payload.get("lang_alt")

        # Preprocess params
        scale = float(payload.get("scale", 1.0))
        apply_clahe = bool(payload.get("apply_clahe", True))
        denoise = bool(payload.get("denoise", True))
        binarize = bool(payload.get("binarize", False))
        gamma = float(payload.get("gamma", 0.0))

        # Detector params
        det_limit_side_len = int(payload.get("det_limit_side_len", 1920))
        det_db_box_thresh = float(payload.get("det_db_box_thresh", 0.4))
        det_db_unclip_ratio = float(payload.get("det_db_unclip_ratio", 1.8))
        drop_score = float(payload.get("drop_score", 0.3))

        # Filters
        min_area_ratio = float(payload.get("min_area_ratio", 0.0002))
        min_height_px = int(payload.get("min_height_px", 12))
        min_text_len = int(payload.get("min_text_len", 2))
        filter_punct = bool(payload.get("filter_punct", True))

        # Overlay options
        return_overlay = bool(payload.get("return_overlay", False))
        color = str(payload.get("color", "#00FF00"))
        alpha = float(payload.get("alpha", 0.35))
        show_text = bool(payload.get("show_text", True))

        if not image_field:
            return {"error": "Missing 'image' (base64)"}

        image_bytes = _b64_to_bytes(image_field)
        image = _bytes_to_image(image_bytes)
        bgr = _pil_to_cv2_bgr(image)
        bgr = _preprocess(
            bgr,
            scale=scale,
            apply_clahe=apply_clahe,
            denoise=denoise,
            binarize=binarize,
            gamma=gamma,
        )

        ocr = get_ocr(
            lang=lang,
            det_limit_side_len=det_limit_side_len,
            det_db_box_thresh=det_db_box_thresh,
            det_db_unclip_ratio=det_db_unclip_ratio,
            drop_score=drop_score,
        )
        result = ocr.ocr(bgr, cls=True)

        if lang_alt:
            ocr_alt = get_ocr(
                lang=lang_alt,
                det_limit_side_len=det_limit_side_len,
                det_db_box_thresh=det_db_box_thresh,
                det_db_unclip_ratio=det_db_unclip_ratio,
                drop_score=drop_score,
            )
            result += ocr_alt.ocr(bgr, cls=True)

        flat: List[dict] = []
        for page in result:
            for det in page:
                pts = det[0]
                text = det[1][0]
                score = float(det[1][1])
                flat.append({"bbox": pts, "text": text, "score": score})

        flat = _filter_results(
            flat,
            bgr.shape,
            min_area_ratio=min_area_ratio,
            min_height_px=min_height_px,
            min_text_len=min_text_len,
            filter_punct=filter_punct,
        )

        out: Dict[str, Any] = {"results": flat}

        if return_overlay:
            overlay = bgr.copy()
            bgr_color = _parse_hex_color_to_bgr(color)
            for r in flat:
                pts = np.array(r["bbox"], dtype=np.int32)
                cv2.fillPoly(overlay, [pts], bgr_color)
                cv2.polylines(overlay, [pts], isClosed=True, color=bgr_color, thickness=2)
                if show_text:
                    x = int(min(p[0] for p in r["bbox"]))
                    y = int(min(p[1] for p in r["bbox"])) - 5
                    label = f"{r['text']} ({r['score']:.2f})"
                    cv2.putText(overlay, label, (x, max(y, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            rgb = cv2.cvtColor(cv2.addWeighted(overlay, max(0.0, min(1.0, alpha)), bgr, 1-alpha, 0), cv2.COLOR_BGR2RGB)
            import io as _io
            from PIL import Image as _Image
            buf = _io.BytesIO()
            _Image.fromarray(rgb).save(buf, format="PNG")
            out["image"] = base64.b64encode(buf.getvalue()).decode("utf-8")

        return out
    except Exception as e:
        return {"error": f"OCR failed: {e}"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})


