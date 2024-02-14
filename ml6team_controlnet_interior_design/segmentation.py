import logging
from typing import List, Tuple, Dict

import streamlit as st
import torch
import gc
import numpy as np
from PIL import Image

from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

from palette import ade_palette

LOGGING = logging.getLogger(__name__)


def flush():
    gc.collect()
    torch.cuda.empty_cache()

@st.cache_resource(max_entries=5)
def get_segmentation_pipeline() -> Tuple[AutoImageProcessor, UperNetForSemanticSegmentation]:
    """Method to load the segmentation pipeline
    Returns:
        Tuple[AutoImageProcessor, UperNetForSemanticSegmentation]: segmentation pipeline
    """
    image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
    image_segmentor = UperNetForSemanticSegmentation.from_pretrained(
        "openmmlab/upernet-convnext-small")
    return image_processor, image_segmentor


@torch.inference_mode()
@torch.autocast('cuda')
def segment_image(image: Image) -> Image:
    """Method to segment image
    Args:
        image (Image): input image
    Returns:
        Image: segmented image
    """
    image_processor, image_segmentor = get_segmentation_pipeline()
    pixel_values = image_processor(image, return_tensors="pt").pixel_values
    with torch.no_grad():
        outputs = image_segmentor(pixel_values)

    seg = image_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]])[0]
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    palette = np.array(ade_palette())
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color
    color_seg = color_seg.astype(np.uint8)
    seg_image = Image.fromarray(color_seg).convert('RGB')
    return seg_image