"""File with configs"""
from palette import COLOR_MAPPING_, COLOR_MAPPING

HEIGHT = 512
WIDTH = 512

def to_rgb(color: str) -> tuple:
    """Convert hex color to rgb.
    Args:
        color (str): hex color
    Returns:
        tuple: rgb color
    """
    return tuple(int(color[i:i+2], 16) for i in (1, 3, 5))

COLOR_NAMES = list(COLOR_MAPPING.keys())
COLOR_RGB = [to_rgb(k) for k in COLOR_MAPPING_.keys()] + [(0, 0, 0), (255, 255, 255)]
INVERSE_COLORS = {v: to_rgb(k) for k, v in COLOR_MAPPING_.items()}
COLOR_MAPPING_RGB = {to_rgb(k): v for k, v in COLOR_MAPPING_.items()}

def map_colors(color: str) -> str:
    """Map color to hex value.
    Args:
        color (str): color name
    Returns:
        str: hex value
    """
    return COLOR_MAPPING[color]

def map_colors_rgb(color: tuple) -> str:
    return COLOR_MAPPING_RGB[color]


POS_PROMPT = "tree, sky, cloud, scenery, outdoors, grass, flowers, sunlight, beautiful, ultra detailed beautiful landscape, architectural renderings vegetation, high res, best high quality landscape, outdoor lighting, sunshine, 4k, 8k, realistic"
NEG_PROMPT= "lowres, deformed, blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, blurry, floating limbs, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, mutated hands and fingers, out of frame"
