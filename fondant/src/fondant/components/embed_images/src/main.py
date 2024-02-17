"""This component that embeds images using a model from the Hugging Face hub."""
import io
import logging
import os
import typing as t

import numpy as np
import pandas as pd
import torch
from dask.distributed import Client, get_worker
from dask_cuda import LocalCUDACluster
from fondant.component import PandasTransformComponent
from PIL import Image
from transformers import BatchEncoding, CLIPProcessor, CLIPVisionModelWithProjection

logger = logging.getLogger(__name__)

os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"


class EmbedImagesComponent(PandasTransformComponent):
    """Component that embeds images using a CLIP model from the Hugging Face hub."""

    def __init__(
        self,
        *,
        model_id: str,
        batch_size: int,
    ):
        """
        Args:
            model_id: id of the model on the Hugging Face hub
            batch_size: batch size to use.
        """
        self.model_id = model_id
        self.batch_size = batch_size

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using device '%s'", self.device)

        super().__init__()

    def dask_client(self) -> Client:
        if self.device == "cuda":
            cluster = LocalCUDACluster()
            return Client(cluster)

        return super().dask_client()

    def process_image_batch(self, images: np.ndarray) -> t.List[torch.Tensor]:
        """
        Process image in batches to a list of tensors.

        Args:
            images: The input images as a numpy array containing byte strings.
        """
        worker = get_worker()

        if hasattr(worker, "processor"):
            processor = worker.processor
        else:
            logger.info(
                "Initializing processor for '%s' on worker '%s",
                self.model_id,
                worker,
            )
            processor = CLIPProcessor.from_pretrained(self.model_id)
            worker.processor = processor

        def load(img: bytes) -> Image:
            """Load the bytestring as an image."""
            bytes_ = io.BytesIO(img)
            return Image.open(bytes_).convert("RGB")

        def transform(img: Image) -> BatchEncoding:
            """Transform the image to a tensor using a clip processor and move it to the specified
            device.
            """
            # Edge case: https://github.com/huggingface/transformers/issues/21638
            if img.width == 1 or img.height == 1:
                img = img.resize((224, 224))

            return processor(images=img, return_tensors="pt").to(self.device)

        return [transform(load(image))["pixel_values"] for image in images]

    @torch.no_grad()
    def embed_image_batch(
        self,
        image_batch: t.List[torch.Tensor],
        *,
        index: pd.Series,
    ) -> pd.Series:
        """Embed a batch of images."""
        worker = get_worker()

        if hasattr(worker, "model"):
            model = worker.model
        else:
            logger.info("Initializing model '%s' on worker '%s", self.model_id, worker)
            model = CLIPVisionModelWithProjection.from_pretrained(self.model_id).to(
                self.device,
            )
            worker.model = model

        input_batch = torch.cat(image_batch)
        output_batch = model(input_batch)
        embeddings_batch = output_batch.image_embeds.cpu().tolist()
        return pd.Series(embeddings_batch, index=index)

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        images = dataframe["image"]

        results: t.List[pd.Series] = []
        for batch in np.split(
            images,
            np.arange(self.batch_size, len(images), self.batch_size),
        ):
            if not batch.empty:
                image_tensors = self.process_image_batch(
                    batch,
                )
                embeddings = self.embed_image_batch(
                    image_tensors,
                    index=batch.index,
                ).T
                results.append(embeddings)

        return pd.concat(results).to_frame(name="embedding")
