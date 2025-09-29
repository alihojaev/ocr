from paddleocr import PaddleOCR
import numpy as np
import time

LANGS = ["japan", "korean", "ch", "chinese_cht", "en"]

def ensure(lang: str, tries: int = 3) -> bool:
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    for i in range(1, tries + 1):
        try:
            print(f"[preload] ({i}/{tries}) lang={lang} init...")
            ocr = PaddleOCR(use_angle_cls=True, lang=lang, ocr_version="PP-OCRv4")
            _ = ocr.ocr(dummy, cls=True)
            print(f"[preload] OK: {lang}")
            return True
        except Exception as e:
            print(f"[preload] WARN {lang}: {e}")
            time.sleep(3)
    return False


def main() -> None:
    for lang in LANGS:
        ensure(lang)


if __name__ == "__main__":
    main()


