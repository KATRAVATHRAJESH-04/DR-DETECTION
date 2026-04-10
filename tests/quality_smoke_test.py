import io
import os
import random

import numpy as np
import requests
from PIL import Image, ImageFilter


def pil_to_bytes(img, fmt="JPEG", quality=95):
    buf = io.BytesIO()
    # JPEG reduces size and simulates real upload artifacts a bit.
    if fmt.upper() == "JPEG":
        img.save(buf, format="JPEG", quality=quality)
    else:
        img.save(buf, format=fmt)
    return buf.getvalue()


def add_noise(pil_img, sigma=25.0):
    arr = np.array(pil_img).astype(np.float32)
    noise = np.random.normal(0.0, sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def low_res_down_up(pil_img, scale_min=0.25, scale_max=0.4):
    w, h = pil_img.size
    scale = random.uniform(scale_min, scale_max)
    new_w = max(16, int(w * scale))
    new_h = max(16, int(h * scale))
    small = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    return small.resize((w, h), Image.Resampling.BILINEAR)


def main():
    api_url = os.environ.get("API_URL", "http://127.0.0.1:8000")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_eye_path = os.path.join(base_dir, "test_eye.png")

    if not os.path.exists(test_eye_path):
        raise FileNotFoundError(f"Missing {test_eye_path}. Expected a sample image for smoke testing.")

    base_img = Image.open(test_eye_path).convert("RGB")

    # Create a few intentionally degraded uploads.
    degraded = [
        ("blur", base_img.filter(ImageFilter.GaussianBlur(radius=3))),
        ("lowres", low_res_down_up(base_img)),
        ("noise", add_noise(base_img, sigma=25.0)),
    ]

    results = []
    for tag, img in degraded:
        img_bytes = pil_to_bytes(img, fmt="JPEG")
        files = {"file": (f"{tag}.jpg", img_bytes, "image/jpeg")}

        res = requests.post(f"{api_url}/predict", files=files, timeout=60)
        res.raise_for_status()
        data = res.json()

        q = data.get("quality") or {}
        results.append((tag, q.get("quality_label"), q.get("quality_score")))
        print(f"{tag}: prediction={data.get('prediction')}, quality_label={q.get('quality_label')}, quality_score={q.get('quality_score')}")

    # Basic sanity check: not all degraded images should be marked as high-quality.
    labels = [label for _, label, _ in results]
    if all(l == "high" for l in labels if l is not None):
        raise RuntimeError(f"Quality estimator seems too permissive. Labels: {labels}")

    print("Smoke test passed.")


if __name__ == "__main__":
    main()

