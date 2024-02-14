"""Preprocessing methods"""
import logging
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageFilter
import streamlit as st

from config import COLOR_RGB, WIDTH, HEIGHT
# from enhance_config import ENHANCE_SETTINGS

LOGGING = logging.getLogger(__name__)


def preprocess_seg_mask(canvas_seg, real_seg: Image.Image = None) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess the segmentation mask.
    Args:
        canvas_seg: segmentation canvas
        real_seg (Image.Image, optional): segmentation mask. Defaults to None.
    Returns:
        Tuple[np.ndarray, np.ndarray]: segmentation mask, segmentation mask with overlay
    """
    # get unique colors in the segmentation
    image_seg = canvas_seg.image_data.copy()[:, :, :3]

    # average the colors of the segmentation masks
    average_color = np.mean(image_seg, axis=(2))
    mask = average_color[:, :] > 0
    if mask.sum() > 0:
        mask = mask * 1

    unique_colors = np.unique(image_seg.reshape(-1, image_seg.shape[-1]), axis=0)
    unique_colors = [tuple(color) for color in unique_colors]

    unique_colors = [color for color in unique_colors if np.sum(
        np.all(image_seg == color, axis=-1)) > 100]

    unique_colors_exact = [color for color in unique_colors if color in COLOR_RGB]

    if real_seg is not None:
        overlay_seg = np.array(real_seg)

        unique_colors = np.unique(overlay_seg.reshape(-1, overlay_seg.shape[-1]), axis=0)
        unique_colors = [tuple(color) for color in unique_colors]

        for color in unique_colors_exact:
            if color != (255, 255, 255) and color != (0, 0, 0):
                overlay_seg[np.all(image_seg == color, axis=-1)] = color
        image_seg = overlay_seg

    return mask, image_seg


def get_mask(image_mask: np.ndarray) -> np.ndarray:
    """Get the mask from the segmentation mask.
    Args:
        image_mask (np.ndarray): segmentation mask
    Returns:
        np.ndarray: mask
    """
    # average the colors of the segmentation masks
    average_color = np.mean(image_mask, axis=(2))
    mask = average_color[:, :] > 0
    if mask.sum() > 0:
        mask = mask * 1
    return mask


def get_image() -> np.ndarray:
    """Get the image from the session state.
    Returns:
        np.ndarray: image
    """
    if 'initial_image' in st.session_state and st.session_state['initial_image'] is not None:
        initial_image = st.session_state['initial_image']
        if isinstance(initial_image, Image.Image):
            return np.array(initial_image.resize((WIDTH, HEIGHT)))
        else:
            return np.array(Image.fromarray(initial_image).resize((WIDTH, HEIGHT)))
    else:
        return None


# def make_enhance_config(segmentation, objects=None):
    """Make the enhance config for the segmentation image.
    """
    info = ENHANCE_SETTINGS[objects]

    segmentation = np.array(segmentation)

    if 'replace' in info:
        replace_color = info['replace']
        mask = np.zeros(segmentation.shape)
        for color in info['colors']:
            mask[np.all(segmentation == color, axis=-1)] = [1, 1, 1]
            segmentation[np.all(segmentation == color, axis=-1)] = replace_color

    if info['inverse'] is False:
        mask = np.zeros(segmentation.shape)
        for color in info['colors']:
            mask[np.all(segmentation == color, axis=-1)] = [1, 1, 1]
    else:
        mask = np.ones(segmentation.shape)
        for color in info['colors']:
            mask[np.all(segmentation == color, axis=-1)] = [0, 0, 0]

    st.session_state['positive_prompt'] = info['positive_prompt']
    st.session_state['negative_prompt'] = info['negative_prompt']

    if info['inpainting'] is True:
        mask = mask.astype(np.uint8)
        mask = Image.fromarray(mask)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=13))
        mask = mask.filter(ImageFilter.MaxFilter(size=9))
        mask = np.array(mask)

        mask[mask < 0.1] = 0
        mask[mask >= 0.1] = 1
        mask = mask.astype(np.uint8)

        conditioning = dict(
            mask_image=mask,
            positive_prompt=info['positive_prompt'],
            negative_prompt=info['negative_prompt'],
        )
    else:
        conditioning = dict(
            mask_image=mask,
            controlnet_conditioning_image=segmentation,
            positive_prompt=info['positive_prompt'],
            negative_prompt=info['negative_prompt'],
            strength=info['strength']
        )
    return conditioning, info['inpainting']