import shutil
import tarfile
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

import config
from config import get_models_dir
from services.Logging import LoggingService
from services.S3Bucket import S3Bucket


class EmbeddingModel:
    logger = LoggingService.get_logger("EmbeddingModel")

    def __init__(self, model_name: str):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dir = Path(get_models_dir()).resolve() / model_name

        if not model_dir.exists():
            self.logger.info(f"Model not found at {model_dir}. Downloading from S3.")
            self._download_model_from_s3(model_name)
        else:
            self.logger.info(f"Using local model from {model_dir}")

        self.model = SentenceTransformer(str(model_dir), trust_remote_code=True)
        self.model.to(device)
        self.model.eval()
        self.logger.info(
            f"Model on {device} | {self.model.__class__.__name__} | {self.model.get_sentence_embedding_dimension()} embedding dimension")

    def _download_model_from_s3(self, model_name: str):
        model_bucket = S3Bucket("kaneel-sagemaker-testing")
        key = f"embedding-model-artifacts/{model_name}/output/model.tar.gz"

        model_base_dir = Path(get_models_dir()).resolve()
        model_base_dir.mkdir(parents=True, exist_ok=True)

        model_dir = model_base_dir / model_name
        archive_path = model_base_dir / f"{model_name}.tar.gz"

        self.logger.info(f"Downloading model artifact s3://{model_bucket.bucket_name}/{key}")
        next_percent_to_log = 10

        def _log_progress(downloaded_bytes: int, total_bytes: int):
            nonlocal next_percent_to_log
            if total_bytes <= 0:
                return

            current_percent = int((downloaded_bytes * 100) / total_bytes)
            while current_percent >= next_percent_to_log and next_percent_to_log <= 100:
                self.logger.info(
                    f"Downloading {model_name}: {next_percent_to_log}% "
                    f"({downloaded_bytes}/{total_bytes} bytes)"
                )
                next_percent_to_log += 10

        total_bytes = model_bucket.download_to_file(key, archive_path, _log_progress)
        if total_bytes > 0 and next_percent_to_log <= 100:
            downloaded_bytes = archive_path.stat().st_size
            self.logger.info(
                f"Downloading {model_name}: 100% ({downloaded_bytes}/{total_bytes} bytes)"
            )

        if model_dir.exists():
            shutil.rmtree(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        with tarfile.open(archive_path, "r:gz") as tar:
            # Guard against path traversal in tar members.
            for member in tar.getmembers():
                target_path = (model_dir / member.name).resolve()
                if target_path != model_dir and model_dir not in target_path.parents:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
            tar.extractall(model_dir)

        archive_path.unlink(missing_ok=True)
        self.logger.info(f"Downloaded and extracted model to {model_dir}")
        return str(model_dir)

    def get_embedding(self, text: str):
        return self.model.encode(text, normalize_embeddings=True, convert_to_numpy=True)


embedding_model = EmbeddingModel(config.get_embedding_model_id())
