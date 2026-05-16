import base64
import json
import logging
import os
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen, urlretrieve

import cv2
import numpy as np
from ultralytics import YOLO

from label_studio_ml.model import LabelStudioMLBase

logger = logging.getLogger(__name__)

WEIGHTS_PATH = os.getenv(
    "MODEL_WEIGHTS",
    str(Path(__file__).parent.parent / "runs/detect/runs/detect/fire_smoke/weights/yolo8n_300.pt"),
)

CLASSES = ["fire", "smoke"]
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
IOU_THRESHOLD  = float(os.getenv("IOU_THRESHOLD",  "0.45"))

_pat_cache: dict = {"refresh": None, "access": None, "expiry": 0.0}


def jwt_exp_ts(token: str) -> float:
    try:
        payload_b64 = token.split(".")[1]
        pad = "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64 + pad))
        return float(payload.get("exp", 0))
    except Exception:
        return 0.0


def looks_like_jwt_pat(key: str) -> bool:
    return key.count(".") == 2 and key.startswith("eyJ")


def ls_refresh_access(refresh: str, base: str) -> str | None:
    global _pat_cache
    now = time.time()
    base = base.rstrip("/")
    if (
        _pat_cache["refresh"] == refresh
        and _pat_cache["access"]
        and now < _pat_cache["expiry"] - 30
    ):
        return _pat_cache["access"]
    body = json.dumps({"refresh": refresh}).encode()
    req = Request(
        f"{base}/api/token/refresh",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        logger.error(f"Label Studio /api/token/refresh failed: {e}")
        return None
    access = data.get("access")
    if not access:
        logger.error("Label Studio token refresh returned no access")
        return None
    exp = jwt_exp_ts(access)
    if exp <= 0:
        exp = now + 300
    _pat_cache["refresh"] = refresh
    _pat_cache["access"] = access
    _pat_cache["expiry"] = exp
    return access


class YOLOFireSmokeDetector(LabelStudioMLBase):

    def __init__(self, label_config=None, train_output=None, **kwargs):
        super().__init__(label_config=label_config, train_output=train_output, **kwargs)
        self.model = YOLO(WEIGHTS_PATH)
        logger.info(f"Loaded model from {WEIGHTS_PATH}")

    def predict(self, tasks: list[dict], context: dict | None = None, **kwargs) -> list[dict]:
        predictions = []

        for task in tasks:
            image_url = task["data"].get("image") or task["data"].get("img")
            if not image_url:
                predictions.append({"result": [], "score": 0.0})
                continue

            img_path = self.resolve_image(image_url)
            if img_path is None:
                logger.warning(f"Cannot resolve image: {image_url}")
                predictions.append({"result": [], "score": 0.0})
                continue

            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Cannot read image: {img_path}")
                predictions.append({"result": [], "score": 0.0})
                continue

            img_h, img_w = img.shape[:2]

            results = self.model.predict(
                source=img_path,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                verbose=False,
            )[0]

            from_name, to_name = self.get_label_config_names()

            result = []
            scores = []
            for box in results.boxes:
                cls_id = int(box.cls)
                conf   = float(box.conf)
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                result.append({
                    "type": "rectanglelabels",
                    "from_name": from_name,
                    "to_name": to_name,
                    "original_image_width": img_w,
                    "original_image_height": img_h,
                    "value": {
                        "x":      x1 / img_w * 100,
                        "y":      y1 / img_h * 100,
                        "width":  (x2 - x1) / img_w * 100,
                        "height": (y2 - y1) / img_h * 100,
                        "rotation": 0,
                        "rectanglelabels": [CLASSES[cls_id]],
                    },
                    "score": conf,
                })
                scores.append(conf)

            avg_score = float(np.mean(scores)) if scores else 0.0
            predictions.append({"result": result, "score": avg_score})

        return predictions

    def get_label_config_names(self):
        parsed = self.parsed_label_config or {}
        for from_name, cfg in parsed.items():
            if cfg.get("type") == "RectangleLabels":
                to_names = cfg.get("to_name") or []
                to_name = to_names[0] if to_names else "image"
                return from_name, to_name
        return "label", "image"

    def try_local_upload_path(self, url: str) -> str | None:
        if not url.startswith("/data/upload"):
            return None
        try:
            from label_studio_tools.core.utils.io import get_data_dir
        except ImportError:
            return None
        parts = [p for p in url.split("/") if p]
        if len(parts) < 4 or parts[0] != "data" or parts[1] != "upload":
            return None
        project_id = parts[-2]
        fname = parts[-1]
        upload_root = os.path.join(get_data_dir(), "media", "upload")
        local_p = os.path.join(upload_root, project_id, fname)
        if os.path.isfile(local_p):
            return local_p
        return None

    def resolve_image(self, url: str) -> str | None:
        local = self.try_local_upload_path(url)
        if local:
            return local

        if url.startswith("/data/local-files"):
            local_root = os.getenv("LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT", "/")
            rel = url.replace("/data/local-files/?d=", "", 1)
            return os.path.join(local_root, rel)

        if url.startswith("/"):
            base = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080").rstrip("/")
            url = f"{base}{url}"

        if url.startswith("http://") or url.startswith("https://"):
            import tempfile

            suffix = Path(urlparse(url).path).suffix or ".jpg"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            try:
                ls_key = (
                    os.getenv("LABEL_STUDIO_API_KEY", "").strip()
                    or (getattr(self, "access_token", None) or "").strip()
                )
                path_part = urlparse(url).path or ""
                if ls_key and path_part.startswith("/data/"):
                    base = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080").rstrip(
                        "/"
                    )
                    if looks_like_jwt_pat(ls_key):
                        access = ls_refresh_access(ls_key, base)
                        if not access:
                            return None
                        auth = {"Authorization": f"Bearer {access}"}
                    else:
                        auth = {"Authorization": f"Token {ls_key}"}
                    try:
                        req = Request(url, headers=auth)
                        with urlopen(req) as r:
                            tmp.write(r.read())
                        return tmp.name
                    except HTTPError as e:
                        if not looks_like_jwt_pat(ls_key):
                            try:
                                req = Request(
                                    url,
                                    headers={"Authorization": f"Bearer {ls_key}"},
                                )
                                with urlopen(req) as r:
                                    tmp.write(r.read())
                                return tmp.name
                            except Exception:
                                pass
                        logger.error(f"Failed to download {url}: {e}")
                        return None
                urlretrieve(url, tmp.name)
                return tmp.name
            except Exception as e:
                logger.error(f"Failed to download {url}: {e}")
                return None

        if os.path.isfile(url):
            return url

        return None
