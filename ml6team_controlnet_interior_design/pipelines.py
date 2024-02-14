import logging
from typing import List, Tuple, Dict

import streamlit as st
import torch
import gc
import time
import numpy as np
from PIL import Image
from time import perf_counter
from contextlib import contextmanager
from scipy.signal import fftconvolve
from PIL import ImageFilter

from diffusers import ControlNetModel, UniPCMultistepScheduler
from diffusers import StableDiffusionInpaintPipeline

from config import WIDTH, HEIGHT
from stable_diffusion_controlnet_inpaint_img2img import StableDiffusionControlNetInpaintImg2ImgPipeline
from helpers import flush

LOGGING = logging.getLogger(__name__)

class ControlNetPipeline:
    def __init__(self):
        self.in_use = False
        self.controlnet = ControlNetModel.from_pretrained(
        "BertChristiaens/controlnet-seg-room", torch_dtype=torch.float16)

        self.pipe = StableDiffusionControlNetInpaintImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            controlnet=self.controlnet,
            safety_checker=None,
            torch_dtype=torch.float16
        )

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe = self.pipe.to("cuda")
        
        self.waiting_queue = []
        self.count = 0
    
    @property
    def queue_size(self):
        return len(self.waiting_queue)
    
    def __call__(self, **kwargs):
        self.count += 1
        number = self.count

        self.waiting_queue.append(number)
        
        # wait until the next number in the queue is the current number
        while self.waiting_queue[0] != number:
            print(f"Wait for your turn {number} in queue {self.waiting_queue}")
            time.sleep(0.5)
            pass

        # it's your turn, so remove the number from the queue
        # and call the function
        print("It's the turn of", self.count)
        results = self.pipe(**kwargs)
        self.waiting_queue.pop(0)
        flush()
        return results
    
class SDPipeline:
    def __init__(self):
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
            safety_checker=None,
        )

        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe = self.pipe.to("cuda")
        
        self.waiting_queue = []
        self.count = 0
    
    @property
    def queue_size(self):
        return len(self.waiting_queue)
    
    def __call__(self, **kwargs):
        self.count += 1
        number = self.count

        self.waiting_queue.append(number)
        
        # wait until the next number in the queue is the current number
        while self.waiting_queue[0] != number:
            print(f"Wait for your turn {number} in queue {self.waiting_queue}")
            time.sleep(0.5)
            pass

        # it's your turn, so remove the number from the queue
        # and call the function
        print("It's the turn of", self.count)
        results = self.pipe(**kwargs)
        self.waiting_queue.pop(0)
        flush()
        return results



@st.cache_resource(max_entries=5)
def get_controlnet():
    """Method to load the controlnet model
    Returns:
        ControlNetModel: controlnet model
    """
    pipe = ControlNetPipeline()
    return pipe



@st.cache_resource(max_entries=5)
def get_inpainting_pipeline():
    """Method to load the inpainting pipeline
    Returns:
        StableDiffusionInpaintPipeline: inpainting pipeline
    """
    pipe = SDPipeline()
    return pipe
