import gc
import torch
from scipy.signal import fftconvolve
from PIL import Image
import numpy as np

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    


def convolution(mask: Image.Image, size=9) -> Image:
    """Method to blur the mask
    Args:
        mask (Image): masking image
        size (int, optional): size of the blur. Defaults to 9.
    Returns:
        Image: blurred mask
    """
    mask = np.array(mask.convert("L"))
    conv = np.ones((size, size)) / size**2
    mask_blended = fftconvolve(mask, conv, 'same')
    mask_blended = mask_blended.astype(np.uint8).copy()

    border = size

    # replace borders with original values
    mask_blended[:border, :] = mask[:border, :]
    mask_blended[-border:, :] = mask[-border:, :]
    mask_blended[:, :border] = mask[:, :border]
    mask_blended[:, -border:] = mask[:, -border:]

    return Image.fromarray(mask_blended).convert("L")


def postprocess_image_masking(inpainted: Image, image: Image, mask: Image) -> Image:
    """Method to postprocess the inpainted image
    Args:
        inpainted (Image): inpainted image
        image (Image): original image
        mask (Image): mask
    Returns:
        Image: inpainted image
    """
    final_inpainted = Image.composite(inpainted.convert("RGBA"), image.convert("RGBA"), mask)
    return final_inpainted.convert("RGB")
