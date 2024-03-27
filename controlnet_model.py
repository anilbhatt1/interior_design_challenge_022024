from typing import Tuple, Union, List
import os
import cv2

import numpy as np
from PIL import Image

import torch
from diffusers import ControlNetModel
# from diffusers.pipelines.controlnet import MultiControlNetModel, StableDiffusionControlNetInpaintPipeline, StableDiffusionControlNetPipeline
from diffusers.pipelines.controlnet import StableDiffusionControlNetInpaintPipeline 
from diffusers import ControlNetModel, UniPCMultistepScheduler
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation, CLIPTextModel, CLIPTokenizer, DataCollatorWithPadding
# from controlnet_aux.processor import Processor
# from controlnet_aux import HEDdetector
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInpaintPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor

from models.colors import ade_palette
from models.utils import map_colors_rgb

text_encoder = CLIPTextModel.from_pretrained(
    "models/runwayml--stable-diffusion-inpainting",
    subfolder="text_encoder") 

tokenizer = CLIPTokenizer.from_pretrained(
    "models/runwayml--stable-diffusion-inpainting",
    subfolder="tokenizer")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def filter_items(
    colors_list: Union[List, np.ndarray],
    items_list: Union[List, np.ndarray],
    items_to_retain: Union[List, np.ndarray]
) -> Tuple[Union[List, np.ndarray], Union[List, np.ndarray]]:
    """
    Filters items and their corresponding colors from given lists, excluding
    specified items.

    Args:
        colors_list: A list or numpy array of colors corresponding to items.
        items_list: A list or numpy array of items.
        items_to_remove: A list or numpy array of items to be removed.

    Returns:
        A tuple of two lists or numpy arrays: filtered colors and filtered
        items.
    """
    filtered_colors = []
    filtered_items = []
    for color, item in zip(colors_list, items_list):
        if item in items_to_retain:
            filtered_colors.append(color)
            filtered_items.append(item)
    return filtered_colors, filtered_items

def filter_items_mask(colors_list,items_list,items_to_mask):
    """
    Filters items and their corresponding colors from given lists, excluding
    specified items.

    Args:
        colors_list: A list or numpy array of colors corresponding to items.
        items_list: A list or numpy array of items.
        items_to_remove: A list or numpy array of items to be removed.

    Returns:
        A tuple of two lists or numpy arrays: filtered colors and filtered
        items.
    """
    filtered_colors = []
    filtered_items = []
    for color, item in zip(colors_list, items_list):
        if item not in items_to_mask:
            filtered_colors.append(color)
            filtered_items.append(item)
    return filtered_colors, filtered_items

def filter_items_retain(colors_list,items_list,items_to_retain):
    """
    Filters items and their corresponding colors from given lists, excluding
    specified items.

    Args:
        colors_list: A list or numpy array of colors corresponding to items.
        items_list: A list or numpy array of items.
        items_to_remove: A list or numpy array of items to be removed.

    Returns:
        A tuple of two lists or numpy arrays: filtered colors and filtered
        items.
    """
    filtered_colors = []
    filtered_items = []
    for color, item in zip(colors_list, items_list):
        if item in items_to_retain:
            filtered_colors.append(color)
            filtered_items.append(item)
    return filtered_colors, filtered_items

def get_segmentation_pipeline(
) -> Tuple[AutoImageProcessor, UperNetForSemanticSegmentation]:
    """Method to load the segmentation pipeline
    Returns:
        Tuple[AutoImageProcessor, UperNetForSemanticSegmentation]: segmentation pipeline
    """
    image_processor = AutoImageProcessor.from_pretrained(
        "models/openmmlab--upernet-convnext-large"
    )
    image_segmentor = UperNetForSemanticSegmentation.from_pretrained(
        "models/openmmlab--upernet-convnext-large"
    )
    return image_processor, image_segmentor


@torch.inference_mode()
@torch.autocast('cuda')
def segment_image(
        image: Image,
        image_processor: AutoImageProcessor,
        image_segmentor: UperNetForSemanticSegmentation
) -> Image:
    """
    Segments an image using a semantic segmentation model.

    Args:
        image (Image): The input image to be segmented.
        image_processor (AutoImageProcessor): The processor to prepare the
            image for segmentation.
        image_segmentor (UperNetForSemanticSegmentation): The semantic
            segmentation model used to identify different segments in the image.

    Returns:
        Image: The segmented image with each segment colored differently based
            on its identified class.
    """
    # image_processor, image_segmentor = get_segmentation_pipeline()
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

def resize_dimensions(dimensions, target_size):
    """ 
    Resize PIL to target size while maintaining aspect ratio 
    If smaller than target size leave it as is
    """
    width, height = dimensions

    # Check if both dimensions are smaller than the target size
    if width < target_size and height < target_size:
        return dimensions

    # Determine the larger side
    if width > height:
        # Calculate the aspect ratio
        aspect_ratio = height / width
        # Resize dimensions
        return (target_size, int(target_size * aspect_ratio))
    else:
        # Calculate the aspect ratio
        aspect_ratio = width / height
        # Resize dimensions
        return (int(target_size * aspect_ratio), target_size)

def tokenize_function(caption):
    return tokenizer(caption, truncation=False)

def do_encode(inputs, text_encoder, device, max_seq_len=75):
    embeddings = []
    tokens = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    num_chunks = (tokens.size(1) + max_seq_len - 1) // max_seq_len

    text_encoder = text_encoder.to(device)
    tokens = tokens.to(device)
    attention_mask = attention_mask.to(device)
    
    for i in range(num_chunks):
        start_idx = i * max_seq_len
        end_idx = start_idx + max_seq_len
        chunk_tokens = tokens[:, start_idx:end_idx]
        # chunk_attention_mask = attention_mask[:, start_idx:end_idx]

        chunk_embeddings = text_encoder.text_model.embeddings.token_embedding(chunk_tokens)

        chunk_size = chunk_tokens.size(1)
        position_ids = torch.arange(start_idx, start_idx + chunk_size, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).expand(chunk_tokens.size(0), chunk_size)

        position_ids = torch.clamp(position_ids.to(device), max=text_encoder.text_model.embeddings.position_embedding.num_embeddings - 1)
        position_embeddings = text_encoder.text_model.embeddings.position_embedding(position_ids)
        chunk_embeddings += position_embeddings

        embeddings.append(chunk_embeddings)

    concatenated_embeddings = torch.cat(embeddings, dim=1)
    attention_mask_expanded = attention_mask.unsqueeze(1).unsqueeze(2).repeat(1, 1, attention_mask.shape[1], 1)
    encoder_outputs = text_encoder.text_model.encoder(concatenated_embeddings, attention_mask=attention_mask_expanded)
    return (encoder_outputs.last_hidden_state)
    # return encoder_outputs[0]

def get_pipeline_embeds_mod(input_ids, negative_ids):

    max_length = tokenizer.model_max_length

    shape_max_length = max(input_ids.shape[-1], negative_ids.shape[-1])                                 

    concat_embeds = []
    neg_embeds = []
    for i in range(0, shape_max_length, max_length):
        concat_embeds.append(text_encoder(input_ids[:, i: i + max_length])[0])
        neg_embeds.append(text_encoder(negative_ids[:, i: i + max_length])[0])

    return torch.cat(concat_embeds, dim=1), torch.cat(neg_embeds, dim=1)

# def get_pipeline_embeds(prompt, negative_prompt):
#     max_length = tokenizer.model_max_length

#     # simple way to determine length of tokens
#     count_prompt = len(prompt.split(" "))
#     count_negative_prompt = len(negative_prompt.split(" "))
#     print(f' count_prompt : {count_prompt}, count_negative_prompt : {count_negative_prompt}')

#     # create the tensor based on which prompt is longer
#     if count_prompt >= count_negative_prompt:
#         input_ids = tokenizer(prompt, return_tensors="pt", truncation=False).input_ids
#         shape_max_length = input_ids.shape[-1]
#         negative_ids = tokenizer(negative_prompt, truncation=False, padding="max_length",
#                                           max_length=shape_max_length, return_tensors="pt").input_ids

#     else:
#         negative_ids = tokenizer(negative_prompt, return_tensors="pt", truncation=False).input_ids
#         shape_max_length = negative_ids.shape[-1]
#         input_ids = tokenizer(prompt, return_tensors="pt", truncation=False, padding="max_length",
#                                        max_length=shape_max_length).input_ids

#     print(f'shape_max_length : {shape_max_length}')                                    

#     concat_embeds = []
#     neg_embeds = []
#     for i in range(0, shape_max_length, max_length):
#         concat_embeds.append(text_encoder(input_ids[:, i: i + max_length])[0])
#         neg_embeds.append(text_encoder(negative_ids[:, i: i + max_length])[0])

#     return torch.cat(concat_embeds, dim=1), torch.cat(neg_embeds, dim=1)
    
# def get_controlnet_cond_image(img_input):
#     processor = HEDdetector.from_pretrained("models/lllyasviel-Annotators")
#     hed_image = processor(img_input, to_pil=True)
#     controlnet_condition_img = hed_image
#     return controlnet_condition_img    
    
# class ControlNetDesignModel_baseline:
#     """ Produces random noise images """
#     def __init__(self):
#         """ Initialize your model(s) here """

#         os.environ['HF_HUB_OFFLINE'] = "True"
#         controlnet_seg = ControlNetModel.from_pretrained(
#             "models/BertChristiaens--controlnet-seg-room/", torch_dtype=torch.float16)

#         self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
#             "models/runwayml--stable-diffusion-inpainting",
#             controlnet=controlnet_seg,
#             safety_checker=None,
#             torch_dtype=torch.float16
#         )

#         self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
#         self.pipe.enable_xformers_memory_efficient_attention()
#         self.pipe = self.pipe.to("cuda")

#         self.seed = 2
#         self.neg_prompt = "lowres, watermark, banner, logo, watermark, contactinfo, text, deformed, blurry, blur, out of focus, out of frame, surreal, ugly"
#         self.control_items = ["floor;flooring", "rug;carpet;carpeting", "wall", "ceiling"]
#         self.additional_quality_suffix = "interior design, 4K, high resolution"
#         self.seg_image_processor, self.image_segmentor = get_segmentation_pipeline()

#     def generate_design(self, empty_room_image: Image, prompt: str) -> Image:
#         """
#         Given an image of an empty room and a prompt
#         generate the designed room according to the prompt
#         Inputs - 
#             empty_room_image - An RGB PIL Image of the empty room
#             prompt - Text describing the target design elements of the room
#         Returns - 
#             design_image - PIL Image of the same size as the empty room image
#                            If the size is not the same the submission will fail.
#         """
#         print(prompt)               
   
#         pos_prompt = prompt + f', {self.additional_quality_suffix}'

#         orig_w, orig_h = empty_room_image.size
#         new_width, new_height = resize_dimensions(empty_room_image.size, 768)
#         input_image = empty_room_image.resize((new_width, new_height))
#         print((orig_w, orig_h), (new_width, new_height))
#         real_seg = np.array(segment_image(input_image,
#                                           self.seg_image_processor,
#                                           self.image_segmentor))
#         unique_colors = np.unique(real_seg.reshape(-1, real_seg.shape[2]), axis=0)
#         unique_colors = [tuple(color) for color in unique_colors]
#         segment_items = [map_colors_rgb(i) for i in unique_colors]
#         chosen_colors, segment_items = filter_items(
#             colors_list=unique_colors,
#             items_list=segment_items,
#             items_to_retain=self.control_items
#         )
#         mask = np.zeros_like(real_seg)
#         for color in chosen_colors:
#             color_matches = (real_seg == color).all(axis=2)
#             mask[color_matches] = 1

#         image_np = np.array(input_image)
#         image = Image.fromarray(image_np).convert("RGB")
#         segmentation_cond_image = Image.fromarray(real_seg).convert("RGB")
#         mask_image = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")
        
#         generated_image = self.pipe(
#             prompt=pos_prompt,
#             negative_prompt=self.neg_prompt,
#             num_inference_steps=50,
#             strength=1,
#             guidance_scale=7,
#             generator=[torch.Generator(device="cuda").manual_seed(self.seed)],
#             image=image,
#             mask_image=mask_image,
#             control_image=segmentation_cond_image,
#         ).images[0]

#         design_image = generated_image.resize(
#             (orig_w, orig_h), Image.Resampling.LANCZOS
#         )
        
#         return design_image

# class ControlNetDesignModel:
#     """ Produces random noise images """
#     def __init__(self):
#         """ Initialize your model(s) here """

#         os.environ['HF_HUB_OFFLINE'] = "True"
#         controlnet_seg = ControlNetModel.from_pretrained(
#             "models/BertChristiaens--controlnet-seg-room/", torch_dtype=torch.float16)

#         self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
#             "models/runwayml--stable-diffusion-inpainting",
#             controlnet=controlnet_seg,
#             safety_checker=None,
#             torch_dtype=torch.float16
#         )

#         self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
#         self.pipe.enable_xformers_memory_efficient_attention()
#         self.pipe = self.pipe.to("cuda")

#         self.seed = 323
#         self.neg_prompt = "lowres, watermark, banner, logo, watermark, contactinfo, text, deformed, blurry, blur, out of focus, out of frame, surreal, ugly"
#         self.control_items = ["windowpane;window"]
#         self.additional_quality_suffix = "interior design, 4K, high resolution"

#         self.seg_image_processor, self.image_segmentor = get_segmentation_pipeline()

#     def generate_design(self, empty_room_image: Image, prompt: str) -> Image:
#         """
#         Given an image of an empty room and a prompt
#         generate the designed room according to the prompt
#         Inputs - 
#             empty_room_image - An RGB PIL Image of the empty room
#             prompt - Text describing the target design elements of the room
#         Returns - 
#             design_image - PIL Image of the same size as the empty room image
#                            If the size is not the same the submission will fail.
#         """
        
#         pos_prompt = prompt + f', {self.additional_quality_suffix}'

#         orig_w, orig_h = empty_room_image.size
#         new_width, new_height = resize_dimensions(empty_room_image.size, 768)
#         input_image = empty_room_image.resize((new_width, new_height))
#         print((orig_w, orig_h), (new_width, new_height))
#         real_seg = np.array(segment_image(input_image,
#                                           self.seg_image_processor,
#                                           self.image_segmentor))
#         unique_colors = np.unique(real_seg.reshape(-1, real_seg.shape[2]), axis=0)
#         unique_colors = [tuple(color) for color in unique_colors]
#         segment_items = [map_colors_rgb(i) for i in unique_colors]
#         chosen_colors, segment_items = filter_items(
#             colors_list=unique_colors,
#             items_list=segment_items,
#             items_to_remove=self.control_items
#         )
#         mask = np.zeros_like(real_seg)
#         for color in chosen_colors:
#             color_matches = (real_seg == color).all(axis=2)
#             mask[color_matches] = 1

#         image_np = np.array(input_image)
#         image = Image.fromarray(image_np).convert("RGB")
#         segmentation_cond_image = Image.fromarray(real_seg).convert("RGB")
#         mask_image = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")

#         generated_image = self.pipe(
#             prompt=pos_prompt,
#             negative_prompt=self.neg_prompt,
#             num_inference_steps=50,
#             strength=1,
#             guidance_scale=7,
#             generator=[torch.Generator(device="cuda").manual_seed(self.seed)],
#             image=image,
#             mask_image=mask_image,
#             control_image=segmentation_cond_image,
#         ).images[0]

#         design_image = generated_image.resize(
#             (orig_w, orig_h), Image.Resampling.LANCZOS
#         )
        
#         return design_image
    
###### additional imports, new model ########

# from transformers import DPTImageProcessor, DPTForDepthEstimation
# import numpy as np 

# class ControlNetDesignModel_Submission2:
#     """ Produces random noise images """
#     def __init__(self):
#         """ Initialize your model(s) here """

#         os.environ['HF_HUB_OFFLINE'] = "True"

#         self.depth_processor = DPTImageProcessor.from_pretrained(
#             "models/intel--dpt-large")
#         self.depth_model = DPTForDepthEstimation.from_pretrained(
#             "models/intel--dpt-large")

#         controlnet_depth = ControlNetModel.from_pretrained(
#             "models/lllyasviel--sd-controlnet-depth", torch_dtype=torch.float16)

#         self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
#             "models/runwayml--stable-diffusion-inpainting",
#             controlnet=controlnet_depth,
#             safety_checker=None,
#             torch_dtype=torch.float16
#         )

#         self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
#         # self.pipe.enable_xformers_memory_efficient_attention()
#         self.pipe = self.pipe.to("cuda")

#         self.seed = 323
#         self.neg_prompt = "lowres, watermark, banner, logo, watermark, contactinfo, text, deformed, blurry, blur, out of focus, out of frame, surreal, ugly"
#         self.control_items = ["windowpane;window"]
#         self.additional_quality_suffix = "interior design, 4K, high resolution"

#     def generate_design(self, empty_room_image: Image, prompt: str) -> Image:
#         """
#         Given an image of an empty room and a prompt
#         generate the designed room according to the prompt
#         Inputs -
#             empty_room_image - An RGB PIL Image of the empty room
#             prompt - Text describing the target design elements of the room
#         Returns -
#             design_image - PIL Image of the same size as the empty room image
#                            If the size is not the same the submission will fail.
#         """
#         print("Input prompt:", prompt)

#         pos_prompt = prompt + f', {self.additional_quality_suffix}'

#         orig_w, orig_h = empty_room_image.size
#         new_width, new_height = resize_dimensions(empty_room_image.size, 768)
#         input_image = empty_room_image.resize((new_width, new_height))
#         print("Shapes orig, new", (orig_w, orig_h), (new_width, new_height))

#         image_np = np.array(input_image)
#         input_image = Image.fromarray(image_np).convert("RGB")

#         inputs_depth = self.depth_processor(images=input_image, return_tensors="pt")

#         with torch.no_grad():
#             outputs = self.depth_model(**inputs_depth)
#             predicted_depth = outputs.predicted_depth

#         # interpolate to original size
#         depth_condition_image = torch.nn.functional.interpolate(
#             predicted_depth.unsqueeze(1),
#             size=input_image.size[::-1],
#             mode="bicubic",
#             align_corners=False,).squeeze(0).squeeze(0).numpy()

#         depth_condition_image = (depth_condition_image - np.min(depth_condition_image))/(np.max(depth_condition_image) - np.min(depth_condition_image))
#         depth_condition_image = depth_condition_image*255
#         depth_condition_image = depth_condition_image.astype(np.uint8)
#         depth_condition_image = depth_condition_image[:, :, None]
#         depth_condition_image = np.concatenate([depth_condition_image, depth_condition_image, depth_condition_image], axis=2)
#         depth_condition_image = Image.fromarray(depth_condition_image)

#         mask = np.zeros(np.shape(input_image))+1
#         mask_image = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")

#         generated_image = self.pipe(
#             prompt=pos_prompt,
#             negative_prompt=self.neg_prompt,
#             num_inference_steps=50,
#             strength=1,
#             guidance_scale=7,
#             generator=[torch.Generator(device="cuda").manual_seed(self.seed)],
#             image=input_image,
#             mask_image = mask_image,
#             control_image=depth_condition_image,
#         ).images[0]

#         design_image = generated_image.resize(
#             (orig_w, orig_h), Image.Resampling.LANCZOS
#         )

#         return design_image

# ### put other imports here, eg: MLSD
# from models.mlsd_main import MLSDdetector

# class ControlNetDesignModel_Submission3:
#     """ Produces random noise images """
#     def __init__(self):
#         """ Initialize your model(s) here """
        
#         self.mlsd = MLSDdetector("models/mlsd--detector/mlsd_large_512_fp32.pth")
        
#         controlnet_mlsd = ControlNetModel.from_pretrained(
#             "models/lllyasviel--sd-controlnet-mlsd", torch_dtype=torch.float16)

#         self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
#             "models/runwayml--stable-diffusion-inpainting",
#             controlnet=controlnet_mlsd,
#             safety_checker=None,
#             torch_dtype=torch.float16
#         )

#         self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
#         self.pipe.enable_xformers_memory_efficient_attention()
#         self.pipe = self.pipe.to("cuda")

#         self.seed = 1
#         self.neg_prompt = "lowres, watermark, banner, logo, watermark, contactinfo, text, deformed, blurry, blur, out of focus, out of frame, surreal, ugly"
#         self.control_items = ["windowpane;window"]
#         self.additional_quality_suffix = "interior design, 4K, high resolution"

#         self.seg_image_processor, self.image_segmentor = get_segmentation_pipeline()
#         self.cond_type = 'seg'

#     def generate_design(self, empty_room_image: Image, prompt: str) -> Image:
        
#         print("Input prompt", prompt)

#         pos_prompt = prompt + f', {self.additional_quality_suffix}'

#         orig_w, orig_h = empty_room_image.size
#         new_width, new_height = resize_dimensions(empty_room_image.size, 768)
#         input_image = empty_room_image.resize((new_width, new_height))
        
#         print("Shapes orig, new", (orig_w, orig_h), (new_width, new_height))
        
#         image_np = np.array(input_image)
#         image = Image.fromarray(image_np).convert("RGB")

#         real_seg = np.array(segment_image(input_image,
#                                           self.seg_image_processor,
#                                           self.image_segmentor))
#         unique_colors = np.unique(real_seg.reshape(-1, real_seg.shape[2]), axis=0)
#         unique_colors = [tuple(color) for color in unique_colors]
#         segment_items = [map_colors_rgb(i) for i in unique_colors]
        
#         chosen_colors, segment_items = filter_items(
#             colors_list=unique_colors,
#             items_list=segment_items,
#             items_to_remove=self.control_items
#         )
#         mask = np.zeros_like(real_seg)
#         for color in chosen_colors:
#             color_matches = (real_seg == color).all(axis=2)
#             mask[color_matches] = 1
#         mask_image = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB") 
                
#         mlsd_cond_image = Image.fromarray(self.mlsd(np.array(input_image))).convert("RGB")

#         generated_image = self.pipe(
#             prompt=pos_prompt,
#             negative_prompt=self.neg_prompt,
#             num_inference_steps=50,
#             strength=1,
#             guidance_scale=7,
#             generator=[torch.Generator(device="cuda").manual_seed(self.seed)],
#             image=image,
#             mask_image=mask_image,
#             control_image=mlsd_cond_image,
#         ).images[0]

#         design_image = generated_image.resize(
#             (orig_w, orig_h), Image.Resampling.LANCZOS
#         )

#         return design_image

# ### put other imports here, eg: MLSD
# from models.mlsd_main import MLSDdetector

# class ControlNetDesignModel_Submission4:
#     """ Produces random noise images """
#     def __init__(self):
#         """ Initialize your model(s) here """

#         controlnet_mlsd = ControlNetModel.from_pretrained(
#             "models/lllyasviel--sd-controlnet-mlsd", torch_dtype=torch.float16)

#         self.pipe1 = StableDiffusionControlNetPipeline.from_pretrained(
#             "models/runwayml--stable-diffusion-v1-5",
#             controlnet=controlnet_mlsd,
#             safety_checker=None,
#             torch_dtype=torch.float16
#         )
        
#         self.pipe2 = StableDiffusionControlNetInpaintPipeline.from_pretrained(
#             "models/runwayml--stable-diffusion-inpainting",
#             controlnet=controlnet_mlsd,
#             safety_checker=None,
#             torch_dtype=torch.float16
#         )
        
#         self.pipe1.scheduler = UniPCMultistepScheduler.from_config(self.pipe1.scheduler.config)
#         self.pipe1.enable_xformers_memory_efficient_attention()
#         self.pipe1 = self.pipe1.to("cuda")

#         self.pipe2.scheduler = UniPCMultistepScheduler.from_config(self.pipe2.scheduler.config)
#         self.pipe2.enable_xformers_memory_efficient_attention()
#         self.pipe2 = self.pipe2.to("cuda")

#         self.seed = 2
#         self.neg_prompt = "lowres, watermark, banner, logo, watermark, contactinfo, text, deformed, blurry, blur, out of focus, out of frame, surreal, ugly"
#         self.control_items = ["windowpane;window"]
#         self.additional_quality_suffix = "interior design, 4K, high resolution"

#         self.seg_image_processor, self.image_segmentor = get_segmentation_pipeline()
#         self.cond_type = 'seg'
        
#         self.mlsd = MLSDdetector("models/mlsd--detector/mlsd_large_512_fp32.pth")

#     def generate_design(self, empty_room_image: Image, prompt: str) -> Image:
        
#         print("Input prompt", prompt)

#         pos_prompt = prompt + f', {self.additional_quality_suffix}'

#         orig_w, orig_h = empty_room_image.size
#         new_width, new_height = resize_dimensions(empty_room_image.size, 768)
#         input_image = empty_room_image.resize((new_width, new_height))
        
#         print("Shapes orig, new", (orig_w, orig_h), (new_width, new_height))
        
#         mlsd_cond_image = Image.fromarray(self.mlsd(np.array(input_image))).convert("RGB")
        
#         generated_image1 = self.pipe1(
#             pos_prompt,
#             mlsd_cond_image,
#             negative_prompt=self.neg_prompt,
#             num_inference_steps=20,
#             generator=[torch.Generator(device="cuda").manual_seed(self.seed)]
#         ).images[0]
        
#         real_seg = np.array(segment_image(generated_image1,
#                                           self.seg_image_processor,
#                                           self.image_segmentor))
#         image_np = np.array(generated_image1)
#         image = Image.fromarray(image_np).convert("RGB")
        
#         unique_colors = np.unique(real_seg.reshape(-1, real_seg.shape[2]), axis=0)
#         unique_colors = [tuple(color) for color in unique_colors]
#         segment_items = [map_colors_rgb(i) for i in unique_colors]
#         chosen_colors, segment_items = filter_items(
#             colors_list=unique_colors,
#             items_list=segment_items,
#             items_to_remove=self.control_items
#         )
#         mask = np.zeros_like(real_seg)
#         for color in chosen_colors:
#             color_matches = (real_seg == color).all(axis=2)
#             mask[color_matches] = 1
#         mask_image = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")
        
#         # perform inpainting
#         generated_image2 = self.pipe2(
#             prompt=pos_prompt,
#             negative_prompt=self.neg_prompt,
#             num_inference_steps=50,
#             strength=1,
#             guidance_scale=7,
#             generator=[torch.Generator(device="cuda").manual_seed(self.seed)],
#             image=image,
#             mask_image=mask_image,
#             control_image=mlsd_cond_image,
#         ).images[0]

#         design_image = generated_image2.resize(
#             (orig_w, orig_h), Image.Resampling.LANCZOS
#         )
        
#         return design_image


# from models.stable_diffusion_controlnet_inpaint_img2img import StableDiffusionControlNetInpaintImg2ImgPipeline
# from models.mlsd_main import MLSDdetector

# class ControlNetDesignModel_Submission5:
#     """ Produces random noise images """
#     def __init__(self):
#         """ Initialize your model(s) here """

#         controlnet_mlsd = ControlNetModel.from_pretrained(
#             "models/lllyasviel--sd-controlnet-mlsd", torch_dtype=torch.float16)

#         self.pipe1 = StableDiffusionControlNetPipeline.from_pretrained(
#             "models/runwayml--stable-diffusion-v1-5",
#             controlnet=controlnet_mlsd,
#             safety_checker=None,
#             torch_dtype=torch.float16
#         )
        
#         self.pipe2 = StableDiffusionControlNetInpaintImg2ImgPipeline.from_pretrained(
#             "models/runwayml--stable-diffusion-inpainting",
#             controlnet=controlnet_mlsd,
#             safety_checker=None,
#             torch_dtype=torch.float16
#         )
        
#         self.pipe1.scheduler = UniPCMultistepScheduler.from_config(self.pipe1.scheduler.config)
#         self.pipe1.enable_xformers_memory_efficient_attention()
#         self.pipe1 = self.pipe1.to("cuda")

#         self.pipe2.scheduler = UniPCMultistepScheduler.from_config(self.pipe2.scheduler.config)
#         self.pipe2.enable_xformers_memory_efficient_attention()
#         self.pipe2 = self.pipe2.to("cuda")

#         self.seed = 2
#         self.neg_prompt = "lowres, watermark, banner, logo, watermark, contactinfo, text, deformed, blurry, blur, out of focus, out of frame, surreal, ugly"
#         self.control_items = ["windowpane;window"]
#         self.additional_quality_suffix = "interior design, 4K, high resolution"
#         self.constraint_1 = "Keep positions of step, stair, window, sunshade, door same as in reference image."
#         self.constraint_2 = "Retain existing ones but dont add extra step, stair, window, sunshade or doors."

#         self.seg_image_processor, self.image_segmentor = get_segmentation_pipeline()
#         self.cond_type = 'seg'
        
#         self.mlsd = MLSDdetector("models/mlsd--detector/mlsd_large_512_fp32.pth")

#     def generate_design(self, empty_room_image: Image, prompt: str) -> Image:
        
#         print("Input prompt", prompt)
        
#         prompt = prompt + '.' + self.constraint_1 + self.constraint_2
#         pos_prompt = prompt + f', {self.additional_quality_suffix}'

#         orig_w, orig_h = empty_room_image.size
#         new_width, new_height = resize_dimensions(empty_room_image.size, 768)
#         input_image = empty_room_image.resize((new_width, new_height))
        
#         print("Shapes orig, new", (orig_w, orig_h), (new_width, new_height))

#         mlsd_cond_image = Image.fromarray(self.mlsd(np.array(input_image))).convert("RGB")

#         generated_image1 = self.pipe1(
#             pos_prompt,
#             mlsd_cond_image,
#             negative_prompt=self.neg_prompt,
#             num_inference_steps=20,
#             generator=[torch.Generator(device="cuda").manual_seed(self.seed)]
#         ).images[0]
        
#         mlsd_cond_image_2 = Image.fromarray(self.mlsd(np.array(generated_image1))).convert("RGB")
        
#         real_seg = np.array(segment_image(generated_image1,
#                                           self.seg_image_processor,
#                                           self.image_segmentor))
#         image_np = np.array(generated_image1)
#         image = Image.fromarray(image_np).convert("RGB")
        
#         unique_colors = np.unique(real_seg.reshape(-1, real_seg.shape[2]), axis=0)
#         unique_colors = [tuple(color) for color in unique_colors]
#         segment_items = [map_colors_rgb(i) for i in unique_colors]
#         chosen_colors, segment_items = filter_items(
#             colors_list=unique_colors,
#             items_list=segment_items,
#             items_to_remove=self.control_items
#         )
#         mask = np.zeros_like(real_seg)
#         for color in chosen_colors:
#             color_matches = (real_seg == color).all(axis=2)
#             mask[color_matches] = 1
#         mask_image = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")
        
#         # perform inpainting
#         generated_image2 = self.pipe2(
#             prompt=pos_prompt,
#             negative_prompt=self.neg_prompt,
#             num_inference_steps=50,
#             strength=1,
#             guidance_scale=7,
#             generator=[torch.Generator(device="cuda").manual_seed(self.seed)],
#             image=generated_image1,
#             mask_image=mask_image,
#             controlnet_conditioning_image=mlsd_cond_image_2,
#         ).images[0]

#         design_image = generated_image2.resize(
#             (orig_w, orig_h), Image.Resampling.LANCZOS
#         )
        
#         return design_image
    
# class ControlNetDesignModel_Submission6:
#     """ Produces random noise images """
#     def __init__(self):
#         """ Initialize your model(s) here """

#         os.environ['HF_HUB_OFFLINE'] = "True"
#         controlnet_seg = ControlNetModel.from_pretrained(
#             "models/BertChristiaens--controlnet-seg-room/", torch_dtype=torch.float16)

#         self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
#             "models/runwayml--stable-diffusion-inpainting",
#             controlnet=controlnet_seg,
#             safety_checker=None,
#             torch_dtype=torch.float16
#         )

#         self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
#         self.pipe.enable_xformers_memory_efficient_attention()
#         self.pipe = self.pipe.to("cuda")

#         self.seed = 2
#         self.neg_prompt = "lowres, watermark, banner, logo, watermark, contactinfo, text, deformed, blurry, blur, out of focus, out of frame, surreal, ugly"
#         self.control_items = ["windowpane;window"]
#         self.additional_quality_suffix = "interior design, 4K, high resolution"
#         self.constraint1 = "Keep positions of step, staircase, window, sunshade, door same as reference image and dont add extra ones"

#         self.seg_image_processor, self.image_segmentor = get_segmentation_pipeline()

#     def generate_design(self, empty_room_image: Image, prompt: str) -> Image:
#         """
#         Given an image of an empty room and a prompt
#         generate the designed room according to the prompt
#         Inputs - 
#             empty_room_image - An RGB PIL Image of the empty room
#             prompt - Text describing the target design elements of the room
#         Returns - 
#             design_image - PIL Image of the same size as the empty room image
#                            If the size is not the same the submission will fail.
#         """
        
#         prompt = prompt + '.' + self.constraint1 
#         pos_prompt = prompt + f', {self.additional_quality_suffix}'

#         orig_w, orig_h = empty_room_image.size
#         new_width, new_height = resize_dimensions(empty_room_image.size, 768)
#         input_image = empty_room_image.resize((new_width, new_height))
#         real_seg = np.array(segment_image(input_image,
#                                           self.seg_image_processor,
#                                           self.image_segmentor))
#         unique_colors = np.unique(real_seg.reshape(-1, real_seg.shape[2]), axis=0)
#         unique_colors = [tuple(color) for color in unique_colors]
#         segment_items = [map_colors_rgb(i) for i in unique_colors]
#         chosen_colors, segment_items = filter_items(
#             colors_list=unique_colors,
#             items_list=segment_items,
#             items_to_remove=self.control_items
#         )
#         mask = np.zeros_like(real_seg)
#         for color in chosen_colors:
#             color_matches = (real_seg == color).all(axis=2)
#             mask[color_matches] = 1

#         image_np = np.array(input_image)
#         image = Image.fromarray(image_np).convert("RGB")
#         segmentation_cond_image = Image.fromarray(real_seg).convert("RGB")
#         mask_image = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")

#         generated_image = self.pipe(
#             prompt=pos_prompt,
#             negative_prompt=self.neg_prompt,
#             num_inference_steps=50,
#             strength=1,
#             guidance_scale=7,
#             generator=[torch.Generator(device="cuda").manual_seed(self.seed)],
#             image=image,
#             mask_image=mask_image,
#             control_image=segmentation_cond_image,
#         ).images[0]

#         design_image = generated_image.resize(
#             (orig_w, orig_h), Image.Resampling.LANCZOS
#         )
        
#         return design_image
    
# class ControlNetDesignModel_Submission10:
#     """ Produces random noise images """
#     def __init__(self):
#         """ Initialize your model(s) here """

#         os.environ['HF_HUB_OFFLINE'] = "True"
#         controlnet_seg = ControlNetModel.from_pretrained(
#             "models/BertChristiaens--controlnet-seg-room/", torch_dtype=torch.float16)

#         self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
#             "models/runwayml--stable-diffusion-inpainting",
#             controlnet=controlnet_seg,
#             safety_checker=None,
#             torch_dtype=torch.float16
#         )

#         self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
#         self.pipe.enable_xformers_memory_efficient_attention()
#         self.pipe = self.pipe.to("cuda")

#         self.seed = 1710
#         self.neg_prompt = "lowres, watermark, banner, logo, watermark, contactinfo, text, deformed, blurry, blur, out of focus, out of frame, surreal, ugly"
#         self.control_items = []
#         self.additional_quality_suffix = "interior design, 4K, high resolution"
#         self.constraint1 = "Keep positions of step, staircase, window, sunshade, door same as reference image and dont add extra ones"

#         self.seg_image_processor, self.image_segmentor = get_segmentation_pipeline()

#     def generate_design(self, empty_room_image: Image, prompt: str) -> Image:
#         """
#         Given an image of an empty room and a prompt
#         generate the designed room according to the prompt
#         Inputs - 
#             empty_room_image - An RGB PIL Image of the empty room
#             prompt - Text describing the target design elements of the room
#         Returns - 
#             design_image - PIL Image of the same size as the empty room image
#                            If the size is not the same the submission will fail.
#         """
        
#         prompt = prompt + '.' + self.constraint1 
#         pos_prompt = prompt + f', {self.additional_quality_suffix}'

#         orig_w, orig_h = empty_room_image.size
#         new_width, new_height = resize_dimensions(empty_room_image.size, 768)
#         input_image = empty_room_image.resize((new_width, new_height))
#         real_seg = np.array(segment_image(input_image,
#                                           self.seg_image_processor,
#                                           self.image_segmentor))
#         unique_colors = np.unique(real_seg.reshape(-1, real_seg.shape[2]), axis=0)
#         unique_colors = [tuple(color) for color in unique_colors]
#         segment_items = [map_colors_rgb(i) for i in unique_colors]
#         chosen_colors, segment_items = filter_items(
#             colors_list=unique_colors,
#             items_list=segment_items,
#             items_to_remove=self.control_items
#         )
#         mask = np.zeros_like(real_seg)
#         for color in chosen_colors:
#             color_matches = (real_seg == color).all(axis=2)
#             mask[color_matches] = 1

#         image_np = np.array(input_image)
#         image = Image.fromarray(image_np).convert("RGB")
#         segmentation_cond_image = Image.fromarray(real_seg).convert("RGB")
#         mask_image = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")

#         generated_image = self.pipe(
#             prompt=pos_prompt,
#             negative_prompt=self.neg_prompt,
#             num_inference_steps=50,
#             strength=1,
#             guidance_scale=7,
#             generator=[torch.Generator(device="cuda").manual_seed(self.seed)],
#             image=image,
#             mask_image=mask_image,
#             control_image=segmentation_cond_image,
#         ).images[0]

#         design_image = generated_image.resize(
#             (orig_w, orig_h), Image.Resampling.LANCZOS
#         )
        
#         return design_image
    

# class ControlNetDesignModel_HED2:
#     """ Produces random noise images """
#     def __init__(self):
#         """ Initialize your model(s) here """

#         os.environ['HF_HUB_OFFLINE'] = "True"
#         controlnet_hed = ControlNetModel.from_pretrained(
#             "models/thibaud--controlnet-sd21-hed/", torch_dtype=torch.float16)

#         self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
#             "models/stabilityai--stable-diffusion-v2-1-inpainting",
#             controlnet=controlnet_hed,
#             safety_checker=None,
#             torch_dtype=torch.float16
#         )

#         self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
#         self.pipe.enable_xformers_memory_efficient_attention()
#         self.pipe = self.pipe.to("cuda")

#         self.seed = 2
#         self.neg_prompt = "lowres, watermark, banner, logo, watermark, contactinfo, text, deformed, blurry, blur, out of focus, out of frame, surreal, ugly"
#         self.control_items = ["floor;flooring", "rug;carpet;carpeting", "wall", "ceiling"]
#         self.additional_quality_suffix = "interior design, 4K, high resolution"
#         self.seg_image_processor, self.image_segmentor = get_segmentation_pipeline()

#     def generate_design(self, empty_room_image: Image, prompt: str) -> Image:
#         """
#         Given an image of an empty room and a prompt
#         generate the designed room according to the prompt
#         Inputs - 
#             empty_room_image - An RGB PIL Image of the empty room
#             prompt - Text describing the target design elements of the room
#         Returns - 
#             design_image - PIL Image of the same size as the empty room image
#                            If the size is not the same the submission will fail.
#         """
#         print(prompt)               
   
#         pos_prompt = prompt + f', {self.additional_quality_suffix}'

#         orig_w, orig_h = empty_room_image.size
#         new_width, new_height = resize_dimensions(empty_room_image.size, 768)
#         input_image = empty_room_image.resize((new_width, new_height))
#         print((orig_w, orig_h), (new_width, new_height))
#         real_seg = np.array(segment_image(input_image,
#                                           self.seg_image_processor,
#                                           self.image_segmentor))
#         unique_colors = np.unique(real_seg.reshape(-1, real_seg.shape[2]), axis=0)
#         unique_colors = [tuple(color) for color in unique_colors]
#         segment_items = [map_colors_rgb(i) for i in unique_colors]
#         chosen_colors, segment_items = filter_items(
#             colors_list=unique_colors,
#             items_list=segment_items,
#             items_to_retain=self.control_items
#         )
#         mask = np.zeros_like(real_seg)
#         for color in chosen_colors:
#             color_matches = (real_seg == color).all(axis=2)
#             mask[color_matches] = 1

#         image_np = np.array(input_image)
#         image = Image.fromarray(image_np).convert("RGB")
#         segmentation_cond_image = Image.fromarray(real_seg).convert("RGB")
#         mask_image = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")
#         controlnet_cond_image = get_controlnet_cond_image(input_image)

#         generated_image = self.pipe(
#             prompt=pos_prompt,
#             negative_prompt=self.neg_prompt,
#             num_inference_steps=50,
#             strength=1,
#             guidance_scale=7,
#             generator=[torch.Generator(device="cuda").manual_seed(self.seed)],
#             image=image,
#             mask_image=mask_image,
#             control_image=segmentation_cond_image,
#         ).images[0]

#         design_image = generated_image.resize(
#             (orig_w, orig_h), Image.Resampling.LANCZOS
#         )
        
#         return design_image
    
# class ControlNetDesignModel_finetuned:
#     """ Produces random noise images """
#     def __init__(self):
#         """ Initialize your model(s) here """

#         os.environ['HF_HUB_OFFLINE'] = "True"
#         controlnet_seg = ControlNetModel.from_pretrained(
#             "models/BertChristiaens--controlnet-seg-room/", torch_dtype=torch.float16)
        
#         unet = UNet2DConditionModel.from_pretrained(
#             "models/runwayml--stable-diffusion-inpainting", subfolder="unet")
        
#         unet.requires_grad_(False)
#         weight_dtype = torch.float16
#         unet.to('cuda', dtype=weight_dtype)

#         lora_attn_procs = {}
#         for name in unet.attn_processors.keys():
#             # print(f'name in unet : {name}')
#             cross_attention_dim = (
#                 None
#                 if name.endswith("attn1.processor")
#                 else unet.config.cross_attention_dim
#             )
#             if name.startswith("mid_block"):
#                 hidden_size = unet.config.block_out_channels[-1]
#             elif name.startswith("up_blocks"):
#                 block_id = int(name[len("up_blocks.")])
#                 hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
#             elif name.startswith("down_blocks"):
#                 block_id = int(name[len("down_blocks.")])
#                 hidden_size = unet.config.block_out_channels[block_id]

#             lora_attn_procs[name] = LoRAAttnProcessor(
#                 hidden_size=hidden_size,
#                 cross_attention_dim=cross_attention_dim,
#                 rank=4,
#             )
#         unet.set_attn_processor(lora_attn_procs)
#         # lora_layers = AttnProcsLayers(unet.attn_processors)

#         self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
#             "models/runwayml--stable-diffusion-inpainting",
#             controlnet=controlnet_seg,
#             safety_checker=None,
#             torch_dtype=torch.float16
#         )

#         self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
#         self.pipe.enable_xformers_memory_efficient_attention()
#         self.pipe = self.pipe.to("cuda")

#         unet_weight_path = "models/unet_fine_tuned_weights/unet_finetuned.safetensors"
#         self.pipe.unet.load_attn_procs(unet_weight_path, use_safetensors=True)

#         self.seed = 2
#         self.neg_prompt = "lowres, watermark, banner, logo, watermark, contactinfo, text, deformed, blurry, blur, out of focus, out of frame, surreal, ugly"
#         self.control_items = ["floor;flooring", "rug;carpet;carpeting", "wall", "ceiling"]
#         self.additional_quality_suffix = "interior design, 4K, high resolution"
#         self.seg_image_processor, self.image_segmentor = get_segmentation_pipeline()

#     def generate_design(self, empty_room_image: Image, prompt: str) -> Image:
#         """
#         Given an image of an empty room and a prompt
#         generate the designed room according to the prompt
#         Inputs - 
#             empty_room_image - An RGB PIL Image of the empty room
#             prompt - Text describing the target design elements of the room
#         Returns - 
#             design_image - PIL Image of the same size as the empty room image
#                            If the size is not the same the submission will fail.
#         """
#         print(prompt)               
   
#         pos_prompt = prompt + f', {self.additional_quality_suffix}'

#         orig_w, orig_h = empty_room_image.size
#         new_width, new_height = resize_dimensions(empty_room_image.size, 768)
#         input_image = empty_room_image.resize((new_width, new_height))
#         print((orig_w, orig_h), (new_width, new_height))
#         real_seg = np.array(segment_image(input_image,
#                                           self.seg_image_processor,
#                                           self.image_segmentor))
#         unique_colors = np.unique(real_seg.reshape(-1, real_seg.shape[2]), axis=0)
#         unique_colors = [tuple(color) for color in unique_colors]
#         segment_items = [map_colors_rgb(i) for i in unique_colors]
#         chosen_colors, segment_items = filter_items(
#             colors_list=unique_colors,
#             items_list=segment_items,
#             items_to_retain=self.control_items
#         )
#         mask = np.zeros_like(real_seg)
#         for color in chosen_colors:
#             color_matches = (real_seg == color).all(axis=2)
#             mask[color_matches] = 1

#         image_np = np.array(input_image)
#         image = Image.fromarray(image_np).convert("RGB")
#         segmentation_cond_image = Image.fromarray(real_seg).convert("RGB")
#         mask_image = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")
        
#         generated_image = self.pipe(
#             prompt=pos_prompt,
#             negative_prompt=self.neg_prompt,
#             num_inference_steps=50,
#             strength=1,
#             guidance_scale=7,
#             generator=[torch.Generator(device="cuda").manual_seed(self.seed)],
#             image=image,
#             mask_image=mask_image,
#             control_image=segmentation_cond_image,
#         ).images[0]

#         design_image = generated_image.resize(
#             (orig_w, orig_h), Image.Resampling.LANCZOS
#         )
        
#         return design_image
    

# class ControlNetDesignModel_baseline_torch32:
#     """ Produces random noise images """
#     def __init__(self):
#         """ Initialize your model(s) here """

#         os.environ['HF_HUB_OFFLINE'] = "True"
#         controlnet_seg = ControlNetModel.from_pretrained(
#             "int_ch/models/BertChristiaens--controlnet-seg-room/", torch_dtype=torch.float32)

#         self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
#             "int_ch/models/runwayml--stable-diffusion-inpainting",
#             controlnet=controlnet_seg,
#             safety_checker=None,
#             torch_dtype=torch.float32
#         )

#         self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
#         self.pipe.enable_xformers_memory_efficient_attention()
#         self.pipe = self.pipe.to("cuda")

#         self.seed = 1
#         self.neg_prompt = "lowres, watermark, banner, logo, watermark, contactinfo, text, deformed, blurry, blur, out of focus, out of frame, surreal, ugly"
#         self.control_items = ["floor;flooring", "rug;carpet;carpeting", "wall", "ceiling"]
#         self.additional_quality_suffix = "interior design, real, 4K, high resolution"
#         self.seg_image_processor, self.image_segmentor = get_segmentation_pipeline()

#     def generate_design(self, empty_room_image: Image, prompt: str) -> Image:
#         """
#         Given an image of an empty room and a prompt
#         generate the designed room according to the prompt
#         Inputs - 
#             empty_room_image - An RGB PIL Image of the empty room
#             prompt - Text describing the target design elements of the room
#         Returns - 
#             design_image - PIL Image of the same size as the empty room image
#                            If the size is not the same the submission will fail.
#         """
#         print(prompt)               
   
#         pos_prompt = prompt + f', {self.additional_quality_suffix}'

#         orig_w, orig_h = empty_room_image.size
#         new_width, new_height = resize_dimensions(empty_room_image.size, 768)
#         input_image = empty_room_image.resize((new_width, new_height))
#         print((orig_w, orig_h), (new_width, new_height))
#         real_seg = np.array(segment_image(input_image,
#                                           self.seg_image_processor,
#                                           self.image_segmentor))
#         unique_colors = np.unique(real_seg.reshape(-1, real_seg.shape[2]), axis=0)
#         unique_colors = [tuple(color) for color in unique_colors]
#         segment_items = [map_colors_rgb(i) for i in unique_colors]
#         chosen_colors, segment_items = filter_items(
#             colors_list=unique_colors,
#             items_list=segment_items,
#             items_to_retain=self.control_items
#         )
#         mask = np.zeros_like(real_seg)
#         for color in chosen_colors:
#             color_matches = (real_seg == color).all(axis=2)
#             mask[color_matches] = 1

#         image_np = np.array(input_image)
#         image = Image.fromarray(image_np).convert("RGB")
#         segmentation_cond_image = Image.fromarray(real_seg).convert("RGB")
#         mask_image = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")
        
#         generated_image = self.pipe(
#             prompt=pos_prompt,
#             negative_prompt=self.neg_prompt,
#             num_inference_steps=100,
#             strength=1.0,
#             guidance_scale=7,
#             generator=[torch.Generator(device="cuda").manual_seed(self.seed)],
#             image=image,
#             mask_image=mask_image,
#             control_image=segmentation_cond_image,
#         ).images[0]

#         design_image = generated_image.resize(
#             (orig_w, orig_h), Image.Resampling.LANCZOS
#         )
        
#         return design_image

# class ControlNetDesignModel_prompt_embed:
#     """ Produces random noise images """
#     def __init__(self):
#         """ Initialize your model(s) here """

#         os.environ['HF_HUB_OFFLINE'] = "True"
#         controlnet_seg = ControlNetModel.from_pretrained(
#             "models/BertChristiaens--controlnet-seg-room/", torch_dtype=torch.float32)

#         self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
#             "models/runwayml--stable-diffusion-inpainting",
#             controlnet=controlnet_seg,
#             safety_checker=None,
#             torch_dtype=torch.float32
#         )

#         self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
#         self.pipe.enable_xformers_memory_efficient_attention()
#         self.pipe = self.pipe.to("cuda")

#         self.seed = 2
#         self.neg_prompt = "lowres, watermark, banner, logo, watermark, contactinfo, text, deformed, blurry, blur, out of focus, out of frame, surreal, ugly"
#         self.neg_const0 = "cluttered space, Low ceilings,Mismatched colors and patterns, Poor lighting, Empty and uninviting space"
#         self.neg_const1 = "furniture with illogical or non-functional designs, missing parts or unbalanced bases." 
#         self.neg_const2 = "plants growing inside furniture (e.g., stools, baskets), No plants defying gravity (e.g., floating or hanging unrealistically)."
#         self.neg_const3 = "furniture merging with other objects (e.g., sofa with table, bed with floor)."
#         self.control_items = ["floor;flooring", "rug;carpet;carpeting", "wall", "ceiling"]
#         self.additional_quality_suffix = "Photorealistic interior design, 4K, high resolution"
#         self.seg_image_processor, self.image_segmentor = get_segmentation_pipeline()
#         self.pos_const1 = "Sofas, tables, bed cots, chairs must have sturdy base."
#         self.pos_const2 = "Bed must be smooth and queen size. Bed cot Headboard must be smooth, simple."
#         self.pos_const3 = "Ceilings and walls must be smooth."
#         self.pos_const3 = "Table must have a level top with a flat, stable surface made of glass, wooden or marble tops."
#         self.pos_const4 = "Plants must be placed in proper terracotta pots."
#         self.pos_const5 = "Furnitures and objects must maintain their distinct boundaries."

#     def generate_design(self, empty_room_image: Image, prompt: str) -> Image:
#         """
#         Given an image of an empty room and a prompt
#         generate the designed room according to the prompt
#         Inputs - 
#             empty_room_image - An RGB PIL Image of the empty room
#             prompt - Text describing the target design elements of the room
#         Returns - 
#             design_image - PIL Image of the same size as the empty room image
#                            If the size is not the same the submission will fail.
#         """            
   
#         pos_prompt = prompt + f', {self.pos_const1}' + f', {self.pos_const2}' + f', {self.pos_const3}'+ f', {self.pos_const4}' + f', {self.pos_const5}' + f', {self.additional_quality_suffix}'        
#         neg_prompt = self.neg_prompt + f', {self.neg_const0}' + f', {self.neg_const1}' + f', {self.neg_const2}' + f', {self.neg_const3}'

#         orig_w, orig_h = empty_room_image.size
#         new_width, new_height = resize_dimensions(empty_room_image.size, 768)
#         input_image = empty_room_image.resize((new_width, new_height))
#         real_seg = np.array(segment_image(input_image,
#                                           self.seg_image_processor,
#                                           self.image_segmentor))
#         unique_colors = np.unique(real_seg.reshape(-1, real_seg.shape[2]), axis=0)
#         unique_colors = [tuple(color) for color in unique_colors]
#         segment_items = [map_colors_rgb(i) for i in unique_colors]
#         chosen_colors, segment_items = filter_items(
#             colors_list=unique_colors,
#             items_list=segment_items,
#             items_to_retain=self.control_items
#         )
#         mask = np.zeros_like(real_seg)
#         for color in chosen_colors:
#             color_matches = (real_seg == color).all(axis=2)
#             mask[color_matches] = 1

#         image_np = np.array(input_image)
#         image = Image.fromarray(image_np).convert("RGB")
#         segmentation_cond_image = Image.fromarray(real_seg).convert("RGB")
#         mask_image = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")

#         prompt_lst = [pos_prompt, neg_prompt]
#         prompt_token_lst = []
#         for prompt in prompt_lst:
#             prompt_dict = tokenize_function(prompt)
#             prompt_token_lst.append(prompt_dict)
#         prompt_tensors = data_collator(prompt_token_lst)
#         prompt_ids = prompt_tensors['input_ids']
#         # prompt_embeds = do_encode(prompt_tensors, text_encoder, 'cuda')
#         # print(f'prompt_embeds : {prompt_embeds.size()}')
#         # pos_prompt_embed = prompt_embeds[0, :, :].unsqueeze(0)
#         # print(f'pos_prompt_embed : {pos_prompt_embed.size()}')
#         # neg_prompt_embed = prompt_embeds[1, :, :].unsqueeze(0)
#         # print(f'neg_prompt_embed : {neg_prompt_embed.size()}')

#         pos_prompt_ids = prompt_ids[0, :].unsqueeze(0)
#         neg_prompt_ids = prompt_ids[1, :].unsqueeze(0)
#         pos_prompt_embed, neg_prompt_embed = get_pipeline_embeds_mod(pos_prompt_ids, neg_prompt_ids) 
       
#         generated_image = self.pipe(
#             prompt_embeds=pos_prompt_embed,
#             negative_prompt_embeds=neg_prompt_embed,
#             num_inference_steps=60,
#             strength=1.0,
#             guidance_scale=7.0,
#             generator=[torch.Generator(device="cuda").manual_seed(self.seed)],
#             image=image,
#             mask_image=mask_image,
#             control_image=segmentation_cond_image,
#         ).images[0]

#         design_image = generated_image.resize(
#             (orig_w, orig_h), Image.Resampling.LANCZOS
#         )
        
#         return design_image

# class ControlNetDesignModel_wall_mask:
#     """ Produces random noise images """
#     def __init__(self):
#         """ Initialize your model(s) here """

#         os.environ['HF_HUB_OFFLINE'] = "True"

#         unet = UNet2DConditionModel.from_pretrained(
#         "models/runwayml--stable-diffusion-inpainting", subfolder="unet")
                    
#         unet.requires_grad_(False)
#         weight_dtype = torch.float32
#         unet.to('cuda', dtype=weight_dtype)

#         lora_attn_procs = {}
#         for name in unet.attn_processors.keys():
#             # print(f'name in unet : {name}')
#             cross_attention_dim = (
#                 None
#                 if name.endswith("attn1.processor")
#                 else unet.config.cross_attention_dim
#             )
#             if name.startswith("mid_block"):
#                 hidden_size = unet.config.block_out_channels[-1]
#             elif name.startswith("up_blocks"):
#                 block_id = int(name[len("up_blocks.")])
#                 hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
#             elif name.startswith("down_blocks"):
#                 block_id = int(name[len("down_blocks.")])
#                 hidden_size = unet.config.block_out_channels[block_id]

#             lora_attn_procs[name] = LoRAAttnProcessor(
#                 hidden_size=hidden_size,
#                 cross_attention_dim=cross_attention_dim,
#                 rank=64,
#             )
            
#         unet.set_attn_processor(lora_attn_procs)

#         lora_layers = AttnProcsLayers(unet.attn_processors)    


#         controlnet_seg = ControlNetModel.from_pretrained(
#             "models/BertChristiaens--controlnet-seg-room/", torch_dtype=torch.float32)

#         self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
#             "models/runwayml--stable-diffusion-inpainting",
#             controlnet=controlnet_seg,
#             safety_checker=None,
#             torch_dtype=torch.float32
#         )

#         self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
#         self.pipe.enable_xformers_memory_efficient_attention()
#         self.pipe = self.pipe.to("cuda")

#         unet_weight_path = "models/unet_fine_tuned_weights/pytorch_lora_weights_run1403.safetensors"
#         self.pipe.unet.load_attn_procs(unet_weight_path, use_safetensors=True)

#         self.seg_image_processor, self.image_segmentor = get_segmentation_pipeline()

#         self.seed = 2
#         self.neg_prompt = "lowres, watermark, banner, logo, contactinfo, text, deformed, blurry, blur, \
#         out of focus, out of frame, surreal, ugly, distortion, low-res, poor quality, "
#         self.additional_quality_suffix = "interior design, 4K, high resolution"        
#         self.control_items = ["floor;flooring", "rug;carpet;carpeting", "wall", "ceiling"]
#         self.control_items_mask = ["stairs;steps", "step;stair", "stairway;staircase", "radiator", "screen;door;screen", "windowpane;window", "door;double;door", "countertop", "fireplace;hearth;open;fireplace","column;pillar"]
#         self.control_items_retain = ["floor;flooring", "rug;carpet;carpeting", "wall", "ceiling"]

#     def generate_design(self, empty_room_image: Image, prompt: str) -> Image:
#         """
#         Given an image of an empty room and a prompt
#         generate the designed room according to the prompt
#         Inputs - 
#             empty_room_image - An RGB PIL Image of the empty room
#             prompt - Text describing the target design elements of the room
#         Returns - 
#             design_image - PIL Image of the same size as the empty room image
#                            If the size is not the same the submission will fail.
#         """            
   
#         orig_w, orig_h = empty_room_image.size
#         new_width, new_height = resize_dimensions(empty_room_image.size, 768)
#         input_image = empty_room_image.resize((new_width, new_height))
#         real_seg = np.array(segment_image(input_image,
#                                           self.seg_image_processor,
#                                           self.image_segmentor))
#         unique_colors = np.unique(real_seg.reshape(-1, real_seg.shape[2]), axis=0)
#         unique_colors = [tuple(color) for color in unique_colors]
#         segment_items = [map_colors_rgb(i) for i in unique_colors]
#         chosen_colors, segment_items_1 = filter_items_mask(
#             colors_list=unique_colors,
#             items_list=segment_items,
#             items_to_mask=self.control_items_mask
#         )
#         mask = np.zeros_like(real_seg)
#         for color in chosen_colors:
#             color_matches = (real_seg == color).all(axis=2)
#             mask[color_matches] = 1

#         image_np = np.array(input_image)
#         image = Image.fromarray(image_np).convert("RGB")
#         segmentation_cond_image = Image.fromarray(real_seg).convert("RGB")
#         mask_image = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")

#         mask_0_array = (mask * 255).astype(np.uint8)
#         mask_1_image = Image.fromarray(mask_0_array).convert("L")
#         mask_1_array = np.array(mask_1_image)

#         object_items_2 = ["wall"]
#         chosen_colors_2, segment_items_2 = filter_items_mask(
#             colors_list=unique_colors,
#             items_list=segment_items,
#             items_to_mask=object_items_2,
#         )                
#         mask_2 = np.zeros_like(real_seg)
#         for color in chosen_colors_2:
#             color_matches = (real_seg == color).all(axis=2)
#             mask_2[color_matches] = 1   
            
#         mask_2_array = (mask_2 * 255).astype(np.uint8)
#         mask_2_image = Image.fromarray(mask_2_array).convert("L")

#         # Find the wall height for each column of the image
#         mask_3_array = np.array(mask_2_image)
#         wall_heights = []
#         for col in range(mask_3_array.shape[1]):
#             # Find the black pixelsfrom the top of the column
#             black_indices = np.nonzero(mask_3_array[:, col] == 0)[0]
#             if black_indices.size == 0:
#                 min_ = 0
#                 max_ = 6
#             else:
#                 max_ = max(black_indices)
#                 min_ = min(black_indices)            
#             tup = (min_, max_)
#             wall_heights.append(tup)
    
#         height, width = mask_3_array.shape
#         white_image_array = np.full((height, width), 255, dtype=np.uint8)
    
#         for col_idx, coords in enumerate(wall_heights):
#             min_, max_ = coords
#             wall_ht = max_ - min_
#             mask_wall_ht = int(0.15 * (wall_ht)) 
#             new_max_ = min_ + mask_wall_ht
#             for col in range(white_image_array.shape[1]):
#                 white_image_array[min_: new_max_, col_idx] = 0    
        
#         combined_mask_array = cv2.bitwise_and(mask_1_array, white_image_array)  
#         final_mask_image = Image.fromarray((combined_mask_array).astype(np.uint8)).convert("RGB")

#         pos_prompt = prompt + f', {self.additional_quality_suffix},'
      

#         prompt_lst = [pos_prompt, self.neg_prompt]
#         prompt_token_lst = []
#         for prompt in prompt_lst:
#             prompt_dict = tokenize_function(prompt)
#             prompt_token_lst.append(prompt_dict)
#         prompt_tensors = data_collator(prompt_token_lst)
#         prompt_ids = prompt_tensors['input_ids']
#         pos_prompt_ids = prompt_ids[0, :].unsqueeze(0)
#         neg_prompt_ids = prompt_ids[1, :].unsqueeze(0)
#         pos_prompt_embed, neg_prompt_embed = get_pipeline_embeds_mod(pos_prompt_ids, neg_prompt_ids) 
       
#         generated_image = self.pipe(
#             prompt_embeds=pos_prompt_embed,
#             negative_prompt_embeds=neg_prompt_embed,
#             num_inference_steps=50,
#             strength=1.0,
#             guidance_scale=7.0,
#             generator=[torch.Generator(device="cuda").manual_seed(self.seed)],
#             image=image,
#             mask_image=final_mask_image,
#             control_image=segmentation_cond_image,
#         ).images[0]

#         design_image = generated_image.resize(
#             (orig_w, orig_h), Image.Resampling.LANCZOS
#         )
        
#         return design_image

class ControlNetDesignModel_wall_window_mask:
    """ Produces random noise images """
    def __init__(self):
        """ Initialize your model(s) here """

        os.environ['HF_HUB_OFFLINE'] = "True"

        unet = UNet2DConditionModel.from_pretrained(
        "models/runwayml--stable-diffusion-inpainting", subfolder="unet")
                    
        unet.requires_grad_(False)
        weight_dtype = torch.float32
        unet.to('cuda', dtype=weight_dtype)

        lora_attn_procs = {}
        for name in unet.attn_processors.keys():
            # print(f'name in unet : {name}')
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=64,
            )
            
        unet.set_attn_processor(lora_attn_procs)

        lora_layers = AttnProcsLayers(unet.attn_processors)    


        controlnet_seg = ControlNetModel.from_pretrained(
            "models/BertChristiaens--controlnet-seg-room/", torch_dtype=torch.float32)

        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "models/runwayml--stable-diffusion-inpainting",
            controlnet=controlnet_seg,
            safety_checker=None,
            torch_dtype=torch.float32
        )

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe = self.pipe.to("cuda")

        unet_weight_path = "models/unet_fine_tuned_weights/pytorch_lora_weights_run1403.safetensors"
        self.pipe.unet.load_attn_procs(unet_weight_path, use_safetensors=True)

        self.seg_image_processor, self.image_segmentor = get_segmentation_pipeline()

        self.seed = 2
        self.neg_prompt = "lowres, watermark, banner, logo, contactinfo, text, deformed, blurry, blur, \
        out of focus, out of frame, surreal, ugly, distortion, low-res, poor quality, "
        self.additional_quality_suffix = "interior design, 4K, high resolution"        
        self.control_items = ["floor;flooring", "rug;carpet;carpeting", "wall", "ceiling"]
        self.control_items_mask = ["stairs;steps", "step;stair", "stairway;staircase", "radiator", "screen;door;screen", "windowpane;window", "door;double;door", "countertop", "fireplace;hearth;open;fireplace","column;pillar"]
        self.control_items_retain = ["floor;flooring", "rug;carpet;carpeting", "wall", "ceiling"]

        self.wall_ht_pct = 0.15
        self.win_area_threshold = 0.15
        self.win_border_colors = [(77, 77, 77)]
        self.win_border_thickness = 10

    def generate_design(self, empty_room_image: Image, prompt: str) -> Image:
        """
        Given an image of an empty room and a prompt
        generate the designed room according to the prompt
        Inputs - 
            empty_room_image - An RGB PIL Image of the empty room
            prompt - Text describing the target design elements of the room
        Returns - 
            design_image - PIL Image of the same size as the empty room image
                           If the size is not the same the submission will fail.
        """            
   
        orig_w, orig_h = empty_room_image.size
        new_width, new_height = resize_dimensions(empty_room_image.size, 768)
        image_area = new_width * new_height
        input_image = empty_room_image.resize((new_width, new_height))
        real_seg = np.array(segment_image(input_image,
                                          self.seg_image_processor,
                                          self.image_segmentor))
        unique_colors = np.unique(real_seg.reshape(-1, real_seg.shape[2]), axis=0)
        unique_colors = [tuple(color) for color in unique_colors]
        segment_items = [map_colors_rgb(i) for i in unique_colors]

        # Finding out area of window and its ratio with respect to image-area
        object_items_win = ["windowpane;window"]
        chosen_colors_win, segment_items_win = filter_items_retain(
            colors_list=unique_colors,
            items_list=segment_items,
            items_to_retain=object_items_win)
        win_color = np.array(chosen_colors_win)
        if win_color.shape == (0,):
            win_mask_area = 1
        else:
            win_mask = cv2.inRange(real_seg, win_color, win_color)
            win_mask_area = cv2.countNonZero(win_mask)  
        win_area_ratio = win_mask_area / image_area

        chosen_colors, segment_items_1 = filter_items_mask(
            colors_list=unique_colors,
            items_list=segment_items,
            items_to_mask=self.control_items_mask
        )
        mask = np.zeros_like(real_seg)
        for color in chosen_colors:
            color_matches = (real_seg == color).all(axis=2)
            mask[color_matches] = 1

        image_np = np.array(input_image)
        image = Image.fromarray(image_np).convert("RGB")
        segmentation_cond_image = Image.fromarray(real_seg).convert("RGB")
        mask_image = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")

        mask_0_array = (mask * 255).astype(np.uint8)
        mask_1_image = Image.fromarray(mask_0_array).convert("L")
        mask_1_array = np.array(mask_1_image)

        object_items_2 = ["wall"]
        chosen_colors_2, segment_items_2 = filter_items_mask(
            colors_list=unique_colors,
            items_list=segment_items,
            items_to_mask=object_items_2,
        )                
        mask_2 = np.zeros_like(real_seg)
        for color in chosen_colors_2:
            color_matches = (real_seg == color).all(axis=2)
            mask_2[color_matches] = 1   
            
        mask_2_array = (mask_2 * 255).astype(np.uint8)
        mask_2_image = Image.fromarray(mask_2_array).convert("L")

        # Find the wall height for each column of the image
        mask_3_array = np.array(mask_2_image)
        wall_heights = []
        for col in range(mask_3_array.shape[1]):
            # Find the black pixelsfrom the top of the column
            black_indices = np.nonzero(mask_3_array[:, col] == 0)[0]
            if black_indices.size == 0: # If no black indices found in a column keep minimal pixels as black
                min_ = 0
                max_ = 6
            else:
                max_ = max(black_indices)
                min_ = min(black_indices)            
            tup = (min_, max_)
            wall_heights.append(tup)
    
        height, width = mask_3_array.shape
        white_image_array = np.full((height, width), 255, dtype=np.uint8) # Create a white image 
        
        # Calculate the wall-ht & for a fixed pct of ht from ceiling, mask the wall as black
        for col_idx, coords in enumerate(wall_heights):
            min_, max_ = coords
            wall_ht = max_ - min_
            mask_wall_ht = int(self.wall_ht_pct * (wall_ht)) 
            new_max_ = min_ + mask_wall_ht
            for col in range(white_image_array.shape[1]):
                white_image_array[min_: new_max_, col_idx] = 0    
        
        # Combining the initial masking array with masked-wall (fixedpct)
        combined_mask_array = cv2.bitwise_and(mask_1_array, white_image_array)  
        combined_mask_image = Image.fromarray((combined_mask_array).astype(np.uint8)).convert("RGB")

        # If windows are too big, then completely mask them. Else mask only borders
        object_items_win = ["windowpane;window"]
        chosen_colors_win, segment_items_win = filter_items_retain(
            colors_list=unique_colors,
            items_list=segment_items,
            items_to_retain=object_items_win)        
        win_color = np.array(chosen_colors_win)

        # Mask the borders only for Smaller windows (eg: <0.15).Also, making sure window is infact detected (area >1)
        if win_area_ratio < self.win_area_threshold and win_mask_area > 1:
            # Threshold the mask image to extract the mask of the specified color
            win_mask = cv2.inRange(real_seg, win_color, win_color)
            # Find contours in the mask
            win_contours, _ = cv2.findContours(win_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

            # Draw a (77, 77, 77) around the outer contour
            bordered_win_mask = cv2.drawContours(real_seg, win_contours, -1, self.win_border_colors[0], thickness=self.win_border_thickness)
            bordered_win_mask_image = Image.fromarray(bordered_win_mask)

            # Create a mask with border areas only masked
            win_interim_mask = np.zeros_like(real_seg)
            for color in self.win_border_colors:
                color_matches = (real_seg == color).all(axis=2)
                win_interim_mask[color_matches] = 1 
            win_interim_array = (win_interim_mask * 255).astype(np.uint8)
            win_interim_array = cv2.bitwise_not(win_interim_array)   # Converting white borders to black & remaining white
            win_interim_image = Image.fromarray(win_interim_array).convert("L")
            win_interim_array = np.array(win_interim_image)

            # Combining the windows border array with initial array having masked wall-ht
            final_mask_array = cv2.bitwise_and(combined_mask_array, win_interim_array)
            final_mask_image = Image.fromarray((final_mask_array).astype(np.uint8)).convert("RGB") 
        # Bigger window OR window not there
        else:
            if win_color.shape == (0,):  # Window not detected
                final_mask_image = combined_mask_image
            else: # Bigger window, fully mask the window portion
                win_interim_mask = np.zeros_like(real_seg)
                for color in chosen_colors_win:
                    color_matches = (real_seg == color).all(axis=2)
                    win_interim_mask[color_matches] = 1 
                win_interim_array = (win_interim_mask * 255).astype(np.uint8)
                win_interim_array = cv2.bitwise_not(win_interim_array) # Converting white win areas to black & remaining white
                win_interim_image = Image.fromarray(win_interim_array).convert("L")
                win_interim_array = np.array(win_interim_image) 

                # Combining the windows mask array with initial array having masked wall-ht
                final_mask_array = cv2.bitwise_and(combined_mask_array, win_interim_array)
                final_mask_image = Image.fromarray((final_mask_array).astype(np.uint8)).convert("RGB") 

        pos_prompt = prompt + f', {self.additional_quality_suffix},'      

        prompt_lst = [pos_prompt, self.neg_prompt]
        prompt_token_lst = []
        for prompt in prompt_lst:
            prompt_dict = tokenize_function(prompt)
            prompt_token_lst.append(prompt_dict)
        prompt_tensors = data_collator(prompt_token_lst)
        prompt_ids = prompt_tensors['input_ids']
        pos_prompt_ids = prompt_ids[0, :].unsqueeze(0)
        neg_prompt_ids = prompt_ids[1, :].unsqueeze(0)
        pos_prompt_embed, neg_prompt_embed = get_pipeline_embeds_mod(pos_prompt_ids, neg_prompt_ids) 
       
        generated_image = self.pipe(
            prompt_embeds=pos_prompt_embed,
            negative_prompt_embeds=neg_prompt_embed,
            num_inference_steps=50,
            strength=1.0,
            guidance_scale=7.0,
            generator=[torch.Generator(device="cuda").manual_seed(self.seed)],
            image=image,
            mask_image=final_mask_image,
            control_image=segmentation_cond_image,
        ).images[0]

        design_image = generated_image.resize(
            (orig_w, orig_h), Image.Resampling.LANCZOS
        )
        
        return design_image

