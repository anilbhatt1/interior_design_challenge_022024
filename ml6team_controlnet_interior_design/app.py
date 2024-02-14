import streamlit as st
# wide layout
st.set_page_config(layout="wide")

from streamlit_drawable_canvas import st_canvas
from PIL import Image
from typing import Union
import random
import numpy as np
import os
import time

from models import make_image_controlnet, make_inpainting
from segmentation import segment_image
from config import HEIGHT, WIDTH, POS_PROMPT, NEG_PROMPT, COLOR_MAPPING, map_colors, map_colors_rgb
from palette import COLOR_MAPPING_CATEGORY
from preprocessing import preprocess_seg_mask, get_image, get_mask
from explanation import make_inpainting_explanation, make_regeneration_explanation, make_segmentation_explanation


def on_upload() -> None:
    """Upload image to the canvas."""
    if 'input_image' in st.session_state and st.session_state['input_image'] is not None:
        image = Image.open(st.session_state['input_image']).convert('RGB')
        st.session_state['initial_image'] = image
        if 'seg' in st.session_state:
            del st.session_state['seg']
        if 'unique_colors' in st.session_state:
            del st.session_state['unique_colors']
        if 'output_image' in st.session_state:
            del st.session_state['output_image']

def make_image_row(image_0, image_1):
    col_0, col_1 = st.columns(2)
    with col_0:
        st.image(image_0, use_column_width=True)
    with col_1:
        st.image(image_1, use_column_width=True)


def check_reset_state() -> bool:
    """Check whether the UI elements need to be reset
    Returns:
        bool: True if the UI elements need to be reset, False otherwise
    """
    if ('reset_canvas' in st.session_state and st.session_state['reset_canvas']):
        st.session_state['reset_canvas'] = False
        return True
    st.session_state['reset_canvas'] = False
    return False


def move_image(source: Union[str, Image.Image],
               dest: str,
               rerun: bool = True,
               remove_state: bool = True) -> None:
    """Move image from source to destination.
    Args:
        source (Union[str, Image.Image]): source image
        dest (str): destination image location
        rerun (bool, optional): rerun streamlit. Defaults to True.
        remove_state (bool, optional): remove the canvas state. Defaults to True.
    """
    source_image = source if isinstance(source, Image.Image) else st.session_state[source]

    if remove_state:
        st.session_state['reset_canvas'] = True
        if 'seg' in st.session_state:
            del st.session_state['seg']
        if 'unique_colors' in st.session_state:
            del st.session_state['unique_colors']

    st.session_state[dest] = source_image
    st.session_state['dest'] = source_image
    if rerun:
        st.experimental_rerun()


def on_change_radio() -> None:
    """Reset the UI elements when the radio button is changed."""
    st.session_state['reset_canvas'] = True


def make_canvas_dict(canvas_color, brush, paint_mode, _reset_state):
    canvas_dict = dict(
        fill_color=canvas_color,
        stroke_color=canvas_color,
        background_color="#FFFFFF",
        background_image=st.session_state['initial_image'] if 'initial_image' in st.session_state else None,
        stroke_width=brush,
        initial_drawing={'version': '4.4.0', 'objects': []} if _reset_state else None,
        update_streamlit=True,
        height=512,
        width=512,
        drawing_mode=paint_mode,
        key="canvas",
    )
    return canvas_dict  

def make_prompt_row():
    col_0_0, col_0_1 = st.columns(2)
    with col_0_0:
        st.text_input(label="Positive prompt", value="a photograph of a room, interior design, 4k, high resolution", key='positive_prompt')
    with col_0_1:
        st.text_input(label="Negative prompt", value="lowres, watermark, banner, logo, watermark, contactinfo, text, deformed, blurry, blur, out of focus, out of frame, surreal, ugly", key='negative_prompt')

def make_sidebar():
    with st.sidebar:
        input_image = st.file_uploader("", type=["png", "jpg"], key='input_image', on_change=on_upload)
        generation_mode = st.selectbox("Generation mode", ["Regenerate",
                                                           "Segmentation",
                                                           "Inpainting"], on_change=on_change_radio)


        if generation_mode == "Segmentation":
            paint_mode = st.sidebar.selectbox("Painting mode", ("freedraw", "polygon"))
            if paint_mode == "freedraw":
                brush = st.slider("Stroke width", 5, 140, 100, key='slider_seg')
            else:
                brush = 5
    
            category_chooser = st.sidebar.selectbox("Filter on category", list(
                COLOR_MAPPING_CATEGORY.keys()), index=0, key='category_chooser')

            chosen_colors = list(COLOR_MAPPING_CATEGORY[category_chooser].keys())

            color_chooser = st.sidebar.selectbox(
                "Choose a color", chosen_colors, index=0, format_func=map_colors, key='color_chooser'
            )

        elif generation_mode == "Regenerate":
            color_chooser = "rgba(0, 0, 0, 0.0)"
            paint_mode = 'freedraw'
            brush = 0

        else:
            paint_mode = st.sidebar.selectbox("Painting mode", ("freedraw", "polygon"))
            if paint_mode == "freedraw":
                brush = st.slider("Stroke width", 5, 140, 100, key='slider_seg')
            else:
                brush = 5

            color_chooser = "#000000"
    return input_image, generation_mode, brush, color_chooser, paint_mode


def make_output_image():
    if 'output_image' in st.session_state:
        output_image = st.session_state['output_image']
        if isinstance(output_image, np.ndarray):
            output_image = Image.fromarray(output_image)

        if isinstance(output_image, Image.Image):
            output_image = output_image.resize((512, 512))
    else:
        output_image = Image.new('RGB', (512, 512), (255, 255, 255))

    st.write("#### Output image")
    st.image(output_image, width=512)
    if st.button("Move to input image"):
        move_image('output_image', 'initial_image', remove_state=True, rerun=True)

def make_editing_canvas(canvas_color, brush, _reset_state, generation_mode, paint_mode):
    st.write("#### Input image")
    canvas_dict = make_canvas_dict(
        canvas_color=canvas_color,
        paint_mode=paint_mode,
        brush=brush,
        _reset_state=_reset_state
    )
    if generation_mode == "Segmentation":
        canvas = st_canvas(
            **canvas_dict,
        )

        if st.button("generate image", key='generate_button'):
            image = get_image()
            print("Preparing image segmentation")
            real_seg = segment_image(Image.fromarray(image))
            mask, seg = preprocess_seg_mask(canvas, real_seg)

            with st.spinner(text="Generating image"):
                print("Making image")
                result_image = make_image_controlnet(image=image,
                                                        mask_image=mask,
                                                        controlnet_conditioning_image=seg,
                                                        positive_prompt=st.session_state['positive_prompt'],
                                                        negative_prompt=st.session_state['negative_prompt'],
                                                        seed=random.randint(0, 100000) # nosec
                                                        )
                if isinstance(result_image, np.ndarray):
                    result_image = Image.fromarray(result_image)
                st.session_state['output_image'] = result_image


    elif generation_mode == "Regenerate":
        canvas = st_canvas(
            **canvas_dict,
        )
        if 'seg' not in st.session_state:
            with st.spinner(text="Preparing image segmentation"):
                image = get_image()
                real_seg = np.array(segment_image(Image.fromarray(image)))
                st.session_state['seg'] = real_seg

        if 'unique_colors' not in st.session_state:
            real_seg = st.session_state['seg']
            unique_colors = np.unique(real_seg.reshape(-1, real_seg.shape[2]), axis=0)
            unique_colors = [tuple(color) for color in unique_colors]
            st.session_state['unique_colors'] = unique_colors

        with st.expander("Explanation", expanded=True):
            st.write("This mode allows you to choose which objects you want to re-generate in the image. "
                 "Use the selection dropdown to add or remove objects. If you are ready, press the generate button"
                 " to generate the image, which can take up to 30 seconds. If you want to improve the generated image, click"
                 " the 'move image to input' button."
                 )
            
        chosen_colors = st.multiselect(
            label="Choose which concepts you want to regenerate in the image",
            options=st.session_state['unique_colors'],
            key='chosen_colors',
            default=st.session_state['unique_colors'],
            format_func=map_colors_rgb,
        )

        if st.button("generate image", key='generate_button'):
            image = get_image()
            print(chosen_colors)

            segmentation = st.session_state['seg']
            mask = np.zeros_like(segmentation)
            for color in chosen_colors:
                # if the color is in the segmentation, set mask to 1
                mask[np.where((segmentation == color).all(axis=2))] = 1

            with st.spinner(text="Generating image"):
                result_image = make_image_controlnet(image=image,
                                                        mask_image=mask,
                                                        controlnet_conditioning_image=segmentation,
                                                        positive_prompt=st.session_state['positive_prompt'],
                                                        negative_prompt=st.session_state['negative_prompt'],
                                                        seed=random.randint(0, 100000) # nosec
                                                        )
                if isinstance(result_image, np.ndarray):
                    result_image = Image.fromarray(result_image)
                st.session_state['output_image'] = result_image

    elif generation_mode == "Inpainting":
        image = get_image()

        canvas = st_canvas(
            **canvas_dict,
        )

        if st.button("generate images", key='generate_button'):
            canvas_mask = canvas.image_data
            if not isinstance(canvas_mask, np.ndarray):
                canvas_mask = np.array(canvas_mask)
            mask = get_mask(canvas_mask)

            with st.spinner(text="Generating new images"):
                print("Making image")
                result_image = make_inpainting(positive_prompt=st.session_state['positive_prompt'],
                                                image=Image.fromarray(image),
                                                mask_image=mask,
                                                negative_prompt=st.session_state['negative_prompt'],
                                                )
                if isinstance(result_image, np.ndarray):
                    result_image = Image.fromarray(result_image)
                st.session_state['output_image'] = result_image

def main():
    # center text
    st.write("## Controlnet sprint - interior design", unsafe_allow_html=True)

    input_image, generation_mode, brush, color_chooser, paint_mode = make_sidebar()

    # check if there is an input_image
    if not ('initial_image' in st.session_state and st.session_state['initial_image'] is not None):
        st.success("Upload an image to start")
        st.write("Welcome to the interior design controlnet demo! "
                 "You can start by uploading a picture of your room, after which you will see "
                 "a good variety of options to edit your current room to generate the room of your dreams! "
                 "You can choose between inpainting, Segmentation and re-generating objects, which "
                 "use our custom trained controlnet model. The main idea is that you can iterate over the "
                 "generated images, because you will rarely get something perfect in one step (although it's possible). "
                 "We added functionality to load in the generated image into the input, so you can keep "
                 "changing the image until you are satisfied."
                 )
        with st.expander("Useful information", expanded=True):
            st.write("### About the dataset")
            st.write("To make this demo as good as possible, our team spend a lot of time training a custom model. "
                    "We used the LAION5B dataset to build our custom dataset, which contains 130k images of 15 types of rooms "
                    "in almost 30 design styles. After fetching all these images, we started adding metadata such as "
                    "captions (from the BLIP captioning model) and segmentation maps (from the HuggingFace UperNetForSemanticSegmentation model). "
                    )
            st.write("For the gathering and inference of the metadata we used the Fondant framework (https://github.com/ml6team/fondant) provided by ML6 (https://www.ml6.eu/), which is an open source "
                     "data centric framework for data preparation. The pipeline used for training this controlnet will soon be available as an "
                     "example pipeline within Fondant and can be easily adapted for building your own dataset."
                     )
            st.write("### About the model")
            st.write(
                "These were then used to train the controlnet model to generate quality interior design images by using "
                "the segmentation maps and prompts as conditioning information for the model. "
                "By training on segmentation maps, the enduser has a very finegrained control over which objects they "
                "want to place in their room. "
                "The resulting model is then used in a community pipeline that supports image2image and inpainting, "
                "so the user can keep elements of their room and change specific parts of the image."
                ""
            )
            
            st.write("### Trivia")
            st.write("The first time someone uses the demo after startup, the models still need to be loaded into memory. "
                    "After this initial load, the model is cached as a resource and can be used for all the users. "
                    "To avoid simultaneous requests, we have implemented a queueing mechanism that ensures that only one "
                    "user accesses the model at a time (similar to the Gradio framework).\n"
                    )
            st.write("To enable the features in the demo, we calculate the underlying segmentation maps and categories that "
                    "are present in the image. This allows us to hide some of the manual work for the user, and "
                    "by doing this, the users don't need to make a segmentation map in an external tool. Everything needed can be done within this demo."
                    )
            
            # st.write("### News: Fondant - an open source data-centric framework for Foundation model finetuning")
            # st.write("The ML6 team  is proud to announce that we are open sourcing our Fondant framework, which is a "
            #         "data-centric framework that allows you to prepare large scale multimodal datasets with ease. We have implemented the components "
            #         "that we used to train this controlnet model in Fondant as an example pipeline, and we are excited to see what you can do with it! In the future we will add a whole library of plug-and-play data preparation components, such as different ML models and filtering steps, in addition to dataset scraping components that connect to LAION5B."
            #         )
            # st.write("The framework is built on top of kubeflow pipelines and abstracts all the complexity of efficient storing and moving of large datasets, so you can focus on implemented just that piece of code that you need without worrying about the rest. We also build it to run on each Cloud provider or VM. You can find the code on our github page: https://github.com/ml6team/fondant.")

        st.write("### Testing images")
        st.write("If you don't have any pictures close, you can use one of these images to test the model by clicking on the 'use example X' buttons")
        
        st.session_state['example_image_0'] = Image.open("content/example_0.png")
        st.session_state['example_image_1'] = Image.open("content/example_1.jpg")
        st.session_state['example_image_2'] = Image.open("content/example_2.jpg")
        st.session_state['example_image_3'] = Image.open("content/example_3.jpg")
        
        col_im_0, col_im_1 = st.columns(2)
        
        with col_im_0:
            st.image(st.session_state['example_image_0'], caption="Example image 1", use_column_width=True)
            if st.button("Use example 1"):
                move_image('example_image_0', 'initial_image', remove_state=True, rerun=True)

            st.image(st.session_state['example_image_2'], caption="Example image 3", use_column_width=True)
            if st.button("Use example 3"):
                move_image('example_image_2', 'initial_image', remove_state=True, rerun=True)
        with col_im_1:
            st.image(st.session_state['example_image_1'], caption="Example image 2", use_column_width=True)
            if st.button("Use example 2"):
                move_image('example_image_1', 'initial_image', remove_state=True, rerun=True)

            st.image(st.session_state['example_image_3'], caption="Example image 4", use_column_width=True)
            if st.button("Use example 4"):
                move_image('example_image_3', 'initial_image', remove_state=True, rerun=True)

        st.write("## Generated examples")
        make_image_row(Image.open("content/output_1.png"), Image.open("content/regen_example.png"))
        make_image_row(Image.open("content/keep background 2.png"), Image.open("content/output_0.png"))
        make_image_row(Image.open("content/segmentation window.png"), Image.open("content/output_3.png"))
        
        st.write("## Example video")
        st.write("### Video 1")
        st.video(open('content/controlnet_sprint_demo.mp4', 'rb').read())
        st.write("### Video 2")
        st.video(open('content/controlnet_demo_video_2.mp4', 'rb').read())

    else:
        make_prompt_row()

        _reset_state = check_reset_state()

        if generation_mode == "Inpainting":
            make_inpainting_explanation()
        elif generation_mode == "Segmentation":
            make_segmentation_explanation()
        elif generation_mode == "Regenerate":
            make_regeneration_explanation()

        col1, col2 = st.columns(2)
        with col1:
            make_editing_canvas(canvas_color=color_chooser,
                                brush=brush,
                                _reset_state=_reset_state,
                                generation_mode=generation_mode,
                                paint_mode=paint_mode
                                )

        with col2:
            make_output_image()

if __name__ == "__main__":
    main()
    
    