"""This file contains methods for inference and image generation."""
import logging
from typing import List, Tuple, Dict

import streamlit as st
import torch
import gc
import time
import numpy as np
from PIL import Image
from PIL import ImageFilter

from diffusers import ControlNetModel, UniPCMultistepScheduler

from config import WIDTH, HEIGHT
from palette import ade_palette
from stable_diffusion_controlnet_inpaint_img2img import StableDiffusionControlNetInpaintImg2ImgPipeline
from helpers import flush, postprocess_image_masking, convolution
from pipelines import ControlNetPipeline, SDPipeline, get_inpainting_pipeline, get_controlnet

LOGGING = logging.getLogger(__name__)


@torch.inference_mode()
def make_image_controlnet(image: np.ndarray,
                          mask_image: np.ndarray,
                          controlnet_conditioning_image: np.ndarray,
                          positive_prompt: str, negative_prompt: str,
                          seed: int = 2356132) -> List[Image.Image]:
    """Method to make image using controlnet
    Args:
        image (np.ndarray): input image
        mask_image (np.ndarray): mask image
        controlnet_conditioning_image (np.ndarray): conditioning image
        positive_prompt (str): positive prompt string
        negative_prompt (str): negative prompt string
        seed (int, optional): seed. Defaults to 2356132.
    Returns:
        List[Image.Image]: list of generated images
    """

    pipe = get_controlnet()
    flush()

    image = Image.fromarray(image).convert("RGB")
    controlnet_conditioning_image = Image.fromarray(controlnet_conditioning_image).convert("RGB")#.filter(ImageFilter.GaussianBlur(radius = 9))
    mask_image = Image.fromarray((mask_image * 255).astype(np.uint8)).convert("RGB")
    mask_image_postproc = convolution(mask_image)


    st.success(f"{pipe.queue_size} images in the queue, can take up to {(pipe.queue_size+1) * 10} seconds")
    generated_image = pipe(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        strength=1.00,
        guidance_scale=7.0,
        generator=[torch.Generator(device="cuda").manual_seed(seed)],
        image=image,
        mask_image=mask_image,
        controlnet_conditioning_image=controlnet_conditioning_image,
    ).images[0]
    generated_image = postprocess_image_masking(generated_image, image, mask_image_postproc)

    return generated_image


@torch.inference_mode()
def make_inpainting(positive_prompt: str,
                    image: Image,
                    mask_image: np.ndarray,
                    negative_prompt: str = "") -> List[Image.Image]:
    """Method to make inpainting
    Args:
        positive_prompt (str): positive prompt string
        image (Image): input image
        mask_image (np.ndarray): mask image
        negative_prompt (str, optional): negative prompt string. Defaults to "".
    Returns:
        List[Image.Image]: list of generated images
    """
    pipe = get_inpainting_pipeline()
    mask_image = Image.fromarray((mask_image * 255).astype(np.uint8))
    mask_image_postproc = convolution(mask_image)

    flush()
    st.success(f"{pipe.queue_size} images in the queue, can take up to {(pipe.queue_size+1) * 10} seconds")
    generated_image = pipe(image=image,
                    mask_image=mask_image,
                    prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=50,
                    height=HEIGHT,
                    width=WIDTH,
                    ).images[0]
    generated_image = postprocess_image_masking(generated_image, image, mask_image_postproc)

    return generated_image
