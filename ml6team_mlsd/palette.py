"""This file contains color information"""
from typing import List, Dict
from colors import COLOR_MAPPING_, COLOR_MAPPING_CATEGORY_, ade_palette


def convert_hex_to_rgba(hex_code: str) -> str:
    """Convert hex code to rgba.
    Args:
        hex_code (str): hex string
    Returns:
        str: rgba string
    """
    # print(f'palette : convert_hex_to_rgba')
    hex_code = hex_code.lstrip('#')
    return "rgba(" + str(int(hex_code[0:2], 16)) + ", " + str(int(hex_code[2:4], 16)) + ", " + str(int(hex_code[4:6], 16)) + ", 1.0)"


def convert_dict_to_rgba(color_dict: Dict) -> Dict:
    """Convert hex code to rgba for all elements in a dictionary.
    Args:
        color_dict (Dict): color dictionary
    Returns:
        Dict: color dictionary with rgba values
    """
    # print(f'palette : convert_dict_to_rgba')
    updated_dict = {}
    for k, v in color_dict.items():
        updated_dict[convert_hex_to_rgba(k)] = v
    return updated_dict


def convert_nested_dict_to_rgba(nested_dict):
    # print(f'palette : convert_nested_dict_to_rgba')
    updated_dict = {}
    for k, v in nested_dict.items():
        updated_dict[k] = convert_dict_to_rgba(v)
    return updated_dict


COLOR_MAPPING = convert_dict_to_rgba(COLOR_MAPPING_)
COLOR_MAPPING_CATEGORY = convert_nested_dict_to_rgba(COLOR_MAPPING_CATEGORY_)