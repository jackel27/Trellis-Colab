"""
app.py
Refined TRELLIS-based 3D Generation Interface

Author: Microsoft TRELLIS team & Contributors
Colab Setup & Notebook by: David Kaiser (https://github.com/jackel27/Trellis-Colab)

This script creates a Gradio Blocks interface (with a Dracula dark theme) for single- or multi-image
conversion into 3D assets (Gaussian + Mesh). It uses the TRELLIS pipeline, plus includes a real-time
video preview and GLB/Gaussian export functionality.
"""

import gradio as gr
import gradio.themes as themes
from gradio_litmodel3d import LitModel3D

import os
import shutil
from typing import List, Tuple, Literal
import torch
import numpy as np
import imageio
from easydict import EasyDict as edict
from PIL import Image

# TRELLIS components
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils


# ======================================
# Global Config & Constants
# ======================================
MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
os.makedirs(TMP_DIR, exist_ok=True)

# --------------------------------------
# Initialize the pipeline globally
#  (If you prefer lazy-loading or dynamic choice,
#   you can do so in __main__ or a function.)
# --------------------------------------
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()


# ======================================
# Session Management
# ======================================
def start_session(req: gr.Request):
    """Create a temporary user directory for this session."""
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)

def end_session(req: gr.Request):
    """Remove the temporary directory when the session ends."""
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)


# ======================================
# Preprocessing
# ======================================
def preprocess_image(image: Image.Image) -> Image.Image:
    """Preprocess a single image with the pipeline's default logic."""
    return pipeline.preprocess_image(image)

def preprocess_images(images: List[Tuple[Image.Image, str]]) -> List[Image.Image]:
    """Preprocess multiple images; each is a tuple of (Image, name)."""
    img_list = [img_tuple[0] for img_tuple in images]
    return [pipeline.preprocess_image(img) for img in img_list]


# ======================================
# Gaussian & Mesh Serialization
# ======================================
def pack_state(gs: Gaussian, mesh: MeshExtractResult) -> dict:
    """Serialize Gaussian + Mesh objects into a dict for internal usage."""
    return {
        'gaussian': {
            **gs.init_params,
            '_xyz': gs._xyz.cpu().numpy(),
            '_features_dc': gs._features_dc.cpu().numpy(),
            '_scaling': gs._scaling.cpu().numpy(),
            '_rotation': gs._rotation.cpu().numpy(),
            '_opacity': gs._opacity.cpu().numpy(),
        },
        'mesh': {
            'vertices': mesh.vertices.cpu().numpy(),
            'faces': mesh.faces.cpu().numpy(),
        },
    }

def unpack_state(state: dict) -> Tuple[Gaussian, edict]:
    """Restore Gaussian + Mesh objects from a serialized dict state."""
    gs = Gaussian(
        aabb=state['gaussian']['aabb'],
        sh_degree=state['gaussian']['sh_degree'],
        mininum_kernel_size=state['gaussian']['mininum_kernel_size'],
        scaling_bias=state['gaussian']['scaling_bias'],
        opacity_bias=state['gaussian']['opacity_bias'],
        scaling_activation=state['gaussian']['scaling_activation'],
    )
    gs._xyz = torch.tensor(state['gaussian']['_xyz'], device='cuda')
    gs._features_dc = torch.tensor(state['gaussian']['_features_dc'], device='cuda')
    gs._scaling = torch.tensor(state['gaussian']['_scaling'], device='cuda')
    gs._rotation = torch.tensor(state['gaussian']['_rotation'], device='cuda')
    gs._opacity = torch.tensor(state['gaussian']['_opacity'], device='cuda')

    mesh = edict(
        vertices=torch.tensor(state['mesh']['vertices'], device='cuda'),
        faces=torch.tensor(state['mesh']['faces'], device='cuda'),
    )
    return gs, mesh


# ======================================
# Utility: Seed Logic
# ======================================
def get_seed(randomize_seed: bool, seed: int) -> int:
    """Return a random seed if 'randomize_seed' is True; otherwise, user-provided seed."""
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed


# ======================================
# Core Function: Image -> 3D
# ======================================
def image_to_3d(
    image: Image.Image,
    multiimages: List[Tuple[Image.Image, str]],
    is_multiimage: bool,
    seed: int,
    ss_guidance_strength: float,
    ss_sampling_steps: int,
    slat_guidance_strength: float,
    slat_sampling_steps: int,
    multiimage_algo: Literal["multidiffusion", "stochastic"],
    req: gr.Request
) -> Tuple[dict, str]:
    """
    Convert input image(s) into a 3D Gaussian + Mesh representation, then create a preview video.
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))

    if not is_multiimage:
        # Single image pipeline
        outputs = pipeline.run(
            image,
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=False,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
        )
    else:
        # Multi-image pipeline
        outputs = pipeline.run_multi_image(
            [img_tuple[0] for img_tuple in multiimages],
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=False,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
            mode=multiimage_algo,
        )

    # Render color video from Gaussian and geometry (normal) from Mesh
    color_frames = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
    geo_frames = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']

    # Combine side-by-side
    video_frames = [np.concatenate([c, g], axis=1) for c, g in zip(color_frames, geo_frames)]
    video_path = os.path.join(user_dir, 'sample.mp4')
    imageio.mimsave(video_path, video_frames, fps=15)

    # Pack state for future extraction
    state = pack_state(outputs['gaussian'][0], outputs['mesh'][0])
    torch.cuda.empty_cache()
    return state, video_path


# ======================================
# Extraction Functions
# ======================================
def extract_glb(state: dict, mesh_simplify: float, texture_size: int, req: gr.Request) -> Tuple[str, str]:
    """Convert Gaussian+Mesh to a simplified GLB with specified texture resolution."""
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    gs, mesh = unpack_state(state)

    glb = postprocessing_utils.to_glb(
        gs, 
        mesh, 
        simplify=mesh_simplify, 
        texture_size=texture_size, 
        verbose=False
    )
    glb_path = os.path.join(user_dir, 'sample.glb')
    glb.export(glb_path)

    torch.cuda.empty_cache()
    return glb_path, glb_path  # (model_output, file_download_button)

def extract_gaussian(state: dict, req: gr.Request) -> Tuple[str, str]:
    """Export the Gaussian representation to a .ply file (can be large)."""
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    gs, _ = unpack_state(state)

    ply_path = os.path.join(user_dir, 'sample.ply')
    gs.save_ply(ply_path)

    torch.cuda.empty_cache()
    return ply_path, ply_path


# ======================================
# Example Functions (Multi-Image)
# ======================================
def prepare_multi_example() -> List[Image.Image]:
    """
    Load multiple sets of images from assets/example_multi_image.
    Concatenate each set horizontally as a single 'gallery' example.
    """
    if not os.path.exists("assets/example_multi_image"):
        return []

    cases = set(fn.split('_')[0] for fn in os.listdir("assets/example_multi_image"))
    examples = []
    for case in cases:
        rows = []
        for i in range(1, 4):
            fn = f"assets/example_multi_image/{case}_{i}.png"
            if not os.path.exists(fn):
                continue
            img = Image.open(fn).convert("RGBA")
            W, H = img.size
            new_w = int(W / H * 512)
            rows.append(np.array(img.resize((new_w, 512))))
        if rows:
            combined = np.concatenate(rows, axis=1)
            examples.append(Image.fromarray(combined))
    return examples

def split_image(image: Image.Image) -> List[Image.Image]:
    """
    Split one large RGBA image horizontally into sub-images 
    based on alpha transitions, for multi-view usage.
    """
    arr = np.array(image)
    alpha = arr[..., 3]
    alpha_mask = np.any(alpha > 0, axis=0)

    start_positions = np.where(~alpha_mask[:-1] & alpha_mask[1:])[0].tolist()
    end_positions = np.where(alpha_mask[:-1] & ~alpha_mask[1:])[0].tolist()

    subimages = []
    for s, e in zip(start_positions, end_positions):
        segment = Image.fromarray(arr[:, s:e+1])
        subimages.append(preprocess_image(segment))

    return subimages


# ======================================
# Build the Gradio UI
# ======================================
def build_interface():
    """
    Construct and return the Gradio Blocks interface (dark theme).
    Includes tabs for single vs. multi-image, generation settings,
    and extraction of GLB/Gaussian.
    """
    with gr.Blocks(
        title="TRELLIS: Image to 3D",
        theme=themes.Dracula(),
        css=None,
        analytics_enabled=False
    ) as demo:

        # Attribution to the creator of the Colab/IPYNB
        gr.Markdown("""
        **Note**: This Colab Notebook was set up by [David Kaiser](https://github.com/jackel27/Trellis-Colab).  
        Feel free to check out the repo for the installation script and environment details!

        ---
        """)

        gr.Markdown("""
        # Image to 3D with TRELLIS
        Convert one or multiple images into a 3D mesh + Gaussian representation (via Microsoft TRELLIS).  
        The pipeline uses diffusion-based sampling and can optionally combine multiple viewpoints in a single model.

        **Steps**:
        1. Choose **Single Image** or **Multiple Images**.
        2. Upload your image(s) and tune the generation settings.
        3. Click **Generate 3D Model**.
        4. Preview the side-by-side color+geometry video.
        5. Optionally **Export** as GLB or Gaussian for further usage.

        ---
        """)

        with gr.Row():
            with gr.Column():
                # Tabs for single or multiple images
                with gr.Tabs() as input_tabs:
                    with gr.Tab(label="Single Image", id=0) as single_tab:
                        image_prompt = gr.Image(
                            label="Image Prompt (RGBA)",
                            format="png",
                            image_mode="RGBA",
                            type="pil",
                            height=300
                        )

                    with gr.Tab(label="Multiple Images", id=1) as multi_tab:
                        multiimage_prompt = gr.Gallery(
                            label="Multi-Image Prompt",
                            format="png",
                            type="pil",
                            height=300,
                            columns=3
                        )
                        gr.Markdown("""
                        **Experimental**  
                        Provide multiple angles of the *same* object. 
                        The pipeline attempts to unify them into a single 3D asset.
                        """)

                # Generation Settings
                with gr.Accordion("Generation Settings", open=False):
                    seed = gr.Slider(0, MAX_SEED, value=0, step=1, label="Seed")
                    randomize_seed = gr.Checkbox(value=True, label="Randomize Seed")

                    gr.Markdown("**Stage 1: Sparse Structure**")
                    with gr.Row():
                        ss_guidance_strength = gr.Slider(
                            0.0, 10.0, value=7.5, step=0.1, label="Guidance Strength"
                        )
                        ss_sampling_steps = gr.Slider(
                            1, 50, value=12, step=1, label="Sampling Steps"
                        )

                    gr.Markdown("**Stage 2: Structured Latent**")
                    with gr.Row():
                        slat_guidance_strength = gr.Slider(
                            0.0, 10.0, value=3.0, step=0.1, label="Guidance Strength"
                        )
                        slat_sampling_steps = gr.Slider(
                            1, 50, value=12, step=1, label="Sampling Steps"
                        )

                    multiimage_algo = gr.Radio(
                        choices=["stochastic", "multidiffusion"],
                        value="stochastic",
                        label="Multi-Image Mode"
                    )

                generate_btn = gr.Button("Generate 3D Model", variant="primary")

                # GLB Extraction Settings
                with gr.Accordion("Export Settings (GLB)", open=False):
                    mesh_simplify = gr.Slider(
                        0.9, 0.98, value=0.95, step=0.01,
                        label="Mesh Simplification"
                    )
                    texture_size = gr.Slider(
                        512, 2048, value=1024, step=512,
                        label="Texture Resolution"
                    )

                with gr.Row():
                    extract_glb_btn = gr.Button("Export as GLB", interactive=False)
                    extract_gs_btn = gr.Button("Export as Gaussian", interactive=False)

                gr.Markdown("""
                *Note: Gaussian export can be **quite large** (~50MB+), so please allow time for saving.*
                """)

            # Right column: preview video + 3D viewer
            with gr.Column():
                video_output = gr.Video(
                    label="Preview (Color + Geometry)", 
                    autoplay=True, 
                    loop=True, 
                    height=300
                )
                model_output = LitModel3D(
                    label="Extracted 3D Model",
                    exposure=10.0,
                    height=300
                )

                with gr.Row():
                    download_glb = gr.DownloadButton(
                        label="Download .GLB",
                        interactive=False
                    )
                    download_gs = gr.DownloadButton(
                        label="Download .PLY",
                        interactive=False
                    )

        # Hidden states
        is_multiimage = gr.State(False)
        output_buf = gr.State()

        # Examples at the bottom
        with gr.Row(visible=True) as single_example_row:
            if os.path.exists("assets/example_image"):
                examples_list = [
                    os.path.join("assets/example_image", fn)
                    for fn in os.listdir("assets/example_image")
                ]
            else:
                examples_list = []

            examples = gr.Examples(
                examples=examples_list,
                inputs=[image_prompt],
                fn=preprocess_image,
                outputs=[image_prompt],
                run_on_click=True,
                examples_per_page=64,
                label="Single Image Examples"
            )

        with gr.Row(visible=False) as multi_example_row:
            multi_examples_list = prepare_multi_example()
            examples_multi = gr.Examples(
                examples=multi_examples_list,
                inputs=[image_prompt],
                fn=split_image,
                outputs=[multiimage_prompt],
                run_on_click=True,
                examples_per_page=8,
                label="Multi-View Examples"
            )

        # -- Session Lifecycle --
        demo.load(start_session)
        demo.unload(end_session)

        # -- Tab Switching Logic --
        def single_tab_fn():
            return (
                False,  # is_multiimage = False
                gr.Row.update(visible=True),   # Show single_image examples
                gr.Row.update(visible=False)   # Hide multi_image examples
            )

        def multi_tab_fn():
            return (
                True,  # is_multiimage = True
                gr.Row.update(visible=False),
                gr.Row.update(visible=True)
            )

        single_tab.select(single_tab_fn, outputs=[is_multiimage, single_example_row, multi_example_row])
        multi_tab.select(multi_tab_fn, outputs=[is_multiimage, single_example_row, multi_example_row])

        # -- Preprocessing on Upload --
        image_prompt.upload(
            preprocess_image,
            inputs=[image_prompt],
            outputs=[image_prompt],
        )
        multiimage_prompt.upload(
            preprocess_images,
            inputs=[multiimage_prompt],
            outputs=[multiimage_prompt],
        )

        # -- Generate Callback --
        generate_btn.click(
            get_seed,
            inputs=[randomize_seed, seed],
            outputs=[seed]
        ).then(
            image_to_3d,
            inputs=[
                image_prompt,
                multiimage_prompt,
                is_multiimage,
                seed,
                ss_guidance_strength,
                ss_sampling_steps,
                slat_guidance_strength,
                slat_sampling_steps,
                multiimage_algo
            ],
            outputs=[output_buf, video_output]
        ).then(
            lambda: (gr.Button.update(interactive=True), gr.Button.update(interactive=True)),
            outputs=[extract_glb_btn, extract_gs_btn]
        )

        # Reset extract buttons when clearing the video
        video_output.clear(
            lambda: (gr.Button.update(interactive=False), gr.Button.update(interactive=False)),
            outputs=[extract_glb_btn, extract_gs_btn],
        )

        # -- GLB Extraction --
        extract_glb_btn.click(
            extract_glb,
            inputs=[output_buf, mesh_simplify, texture_size],
            outputs=[model_output, download_glb]
        ).then(
            lambda: gr.DownloadButton.update(interactive=True),
            outputs=[download_glb]
        )

        # -- Gaussian Extraction --
        extract_gs_btn.click(
            extract_gaussian,
            inputs=[output_buf],
            outputs=[model_output, download_gs]
        ).then(
            lambda: gr.DownloadButton.update(interactive=True),
            outputs=[download_gs]
        )

        # Reset download buttons when clearing the 3D model output
        model_output.clear(
            lambda: gr.DownloadButton.update(interactive=False),
            outputs=[download_glb],
        )

        return demo


# ======================================
# Main Entry
# ======================================
if __name__ == "__main__":
    demo = build_interface()
    demo.launch()
