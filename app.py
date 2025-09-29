import base64
import io
from typing import List, Optional, Tuple, Dict

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import cv2

app = FastAPI(title="PaddleOCR API")

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
    # Cache by config to avoid re-initializing heavy models
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


def _b64_to_bytes(b64: str) -> bytes:
    b64_clean = b64
    if b64_clean.startswith("data:") and ";base64," in b64_clean:
        b64_clean = b64_clean.split(",", 1)[1]
    b64_clean = b64_clean.strip()
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
    # PIL is RGB; convert to BGR for OpenCV/PaddleOCR typical pipeline
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


def _rect_iou(a: List[List[float]], b: List[List[float]]) -> float:
    # Approximate polygon as bounding rectangles and compute IoU
    ax1 = min(p[0] for p in a); ay1 = min(p[1] for p in a)
    ax2 = max(p[0] for p in a); ay2 = max(p[1] for p in a)
    bx1 = min(p[0] for p in b); by1 = min(p[1] for p in b)
    bx2 = max(p[0] for p in b); by2 = max(p[1] for p in b)
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1); ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return float(inter / union)


def _polygon_area(poly: List[List[float]]) -> float:
    # Shoelace formula
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
    filtered: List[Dict] = []
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
            # allow if it's alnum like single letter/number; otherwise drop
            if filter_punct and not any(ch.isalnum() for ch in text):
                continue
        filtered.append(r)
    return filtered


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ocr")
async def ocr_endpoint(
    image: Optional[UploadFile] = File(default=None),
    image_b64: Optional[str] = Form(default=None),
    lang: str = Form(default="en"),
    lang_alt: Optional[str] = Form(default=None),
    # Preprocess
    scale: float = Form(default=1.0),
    apply_clahe: bool = Form(default=True),
    denoise: bool = Form(default=True),
    binarize: bool = Form(default=False),
    gamma: float = Form(default=0.0),
    # Detector params (PP-OCR DB)
    det_limit_side_len: int = Form(default=1920),
    det_db_box_thresh: float = Form(default=0.4),
    det_db_unclip_ratio: float = Form(default=1.8),
    drop_score: float = Form(default=0.3),
    # Post filters
    min_area_ratio: float = Form(default=0.0002),
    min_height_px: int = Form(default=12),
    min_text_len: int = Form(default=2),
    filter_punct: bool = Form(default=True),
) -> JSONResponse:
    try:
        if image is not None:
            content = await image.read()
        elif image_b64 is not None:
            content = _b64_to_bytes(image_b64)
        else:
            raise HTTPException(status_code=400, detail="Provide 'image' file or 'image_b64'.")

        img = _bytes_to_image(content)
        img_bgr = _pil_to_cv2_bgr(img)
        img_bgr = _preprocess(
            img_bgr,
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
        result_primary = ocr.ocr(img_bgr, cls=True)

        # Optional second pass with alternative language and merge
        results_merged = []
        for page in result_primary:
            for det in page:
                bbox = det[0]; text = det[1][0]; score = float(det[1][1])
                results_merged.append({"bbox": bbox, "text": text, "score": score})

        if lang_alt:
            ocr_alt = get_ocr(
                lang=lang_alt,
                det_limit_side_len=det_limit_side_len,
                det_db_box_thresh=det_db_box_thresh,
                det_db_unclip_ratio=det_db_unclip_ratio,
                drop_score=drop_score,
            )
            result_alt = ocr_alt.ocr(img_bgr, cls=True)
            for page in result_alt:
                for det in page:
                    bbox_b = det[0]; text_b = det[1][0]; score_b = float(det[1][1])
                    # suppress duplicates by IoU
                    dup = False
                    for r in results_merged:
                        if _rect_iou(bbox_b, r["bbox"]) > 0.5:
                            dup = True
                            if score_b > r["score"]:
                                r.update({"bbox": bbox_b, "text": text_b, "score": score_b})
                            break
                    if not dup:
                        results_merged.append({"bbox": bbox_b, "text": text_b, "score": score_b})

        # Normalize output
        results_filtered = _filter_results(
            results_merged,
            img_bgr.shape,
            min_area_ratio=min_area_ratio,
            min_height_px=min_height_px,
            min_text_len=min_text_len,
            filter_punct=filter_punct,
        )
        return JSONResponse({"results": results_filtered})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")


def _parse_hex_color_to_bgr(color_hex: str) -> tuple:
    s = color_hex.strip().lstrip("#")
    if len(s) == 6:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
    elif len(s) == 3:
        r = int(s[0] * 2, 16)
        g = int(s[1] * 2, 16)
        b = int(s[2] * 2, 16)
    else:
        r, g, b = 255, 0, 0
    return (b, g, r)


@app.post("/ocr_overlay")
async def ocr_overlay_endpoint(
    image: Optional[UploadFile] = File(default=None),
    image_b64: Optional[str] = Form(default=None),
    lang: str = Form(default="en"),
    lang_alt: Optional[str] = Form(default=None),
    color: str = Form(default="#FF0000"),
    alpha: float = Form(default=0.35),
    thickness: int = Form(default=2),
    show_text: bool = Form(default=True),
    # Preprocess
    scale: float = Form(default=1.0),
    apply_clahe: bool = Form(default=True),
    denoise: bool = Form(default=True),
    binarize: bool = Form(default=False),
    gamma: float = Form(default=0.0),
    # Detector params
    det_limit_side_len: int = Form(default=1920),
    det_db_box_thresh: float = Form(default=0.4),
    det_db_unclip_ratio: float = Form(default=1.8),
    drop_score: float = Form(default=0.3),
    # Post filters
    min_area_ratio: float = Form(default=0.0002),
    min_height_px: int = Form(default=12),
    min_text_len: int = Form(default=2),
    filter_punct: bool = Form(default=True),
) -> JSONResponse:
    try:
        if image is not None:
            content = await image.read()
        elif image_b64 is not None:
            content = _b64_to_bytes(image_b64)
        else:
            raise HTTPException(status_code=400, detail="Provide 'image' file or 'image_b64'.")

        pil = _bytes_to_image(content)
        bgr = _pil_to_cv2_bgr(pil)
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
            result_alt = ocr_alt.ocr(bgr, cls=True)
            result += result_alt

        overlay = bgr.copy()
        bgr_color = _parse_hex_color_to_bgr(color)

        # Flatten and apply filters
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

        for r in flat:
            pts = r["bbox"]
            text = r["text"]
            score = r["score"]

            poly = np.array(pts, dtype=np.int32)
            cv2.fillPoly(overlay, [poly], bgr_color)
            cv2.polylines(overlay, [poly], isClosed=True, color=bgr_color, thickness=thickness)

            if show_text:
                x = int(min(p[0] for p in pts))
                y = int(min(p[1] for p in pts)) - 5
                label = f"{text} ({score:.2f})"
                cv2.putText(overlay, label, (x, max(y, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        alpha = max(0.0, min(1.0, float(alpha)))
        blended = cv2.addWeighted(overlay, alpha, bgr, 1 - alpha, 0)

        rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        out_pil = Image.fromarray(rgb)
        buf = io.BytesIO()
        out_pil.save(buf, format="PNG")
        out_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return JSONResponse({"image": out_b64})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Overlay failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7861, reload=False)


