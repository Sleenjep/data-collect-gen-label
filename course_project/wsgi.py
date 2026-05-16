import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

from label_studio_ml.api import init_app
from model import YOLOFireSmokeDetector

backend_root = Path(__file__).resolve().parent
model_dir = Path(os.getenv("ML_BACKEND_MODEL_DIR") or (backend_root / "model_data"))
model_dir.mkdir(parents=True, exist_ok=True)

app = init_app(model_class=YOLOFireSmokeDetector, model_dir=str(model_dir))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9090)
    args = parser.parse_args()

    app.run(host=args.host, port=args.port, debug=False)
