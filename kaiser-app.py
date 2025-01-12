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

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils


# ======================================
# Constants and Global Config
# ======================================
MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)

# Initialize the pipeline globally
# (If you prefer lazy-loading or dynamic choice, you can do so inside __main__ or a function)
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()

# ======================================
# Session Management
# ======================================
def start_session(req: gr.Request):
    """Create a temporary user directory upon session start."""
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)

def end_session(req: gr.Request):
    """Remove the user directory on session end/close."""
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)

# ======================================
# Preprocessing Helpers
# ======================================
def preprocess_image(image: Image.Image) -> Image.Image:
    """Preprocess the input image."""
    processed_image = pipeline.preprocess_image(image)
    return processed_image

def preprocess_images(images: List[Tuple[Image.Image, str]]) -> List[Image.Image]:
    """Preprocess multiple input images at once."""
    images = [img_tuple[0] for img_tuple in images]
    processed_images = [pipeline.preprocess_image(image) for image in images]
    return processed_images

# ======================================
# Gaussian / Mesh Pack & Unpack
# ======================================
def pack_state(gs: Gaussian, mesh: MeshExtractResult) -> dict:
    """Serialize Gaussian + Mesh objects into a dict for internal storage."""
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
    """Restore Gaussian + Mesh from a serialized state dict."""
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
# Utility: Random Seed
# ======================================
def get_seed(randomize_seed: bool, seed: int) -> int:
    """Return a random seed if requested, otherwise the user-specified seed."""
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed

# ======================================
# Core Function: Image to 3D
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
    Convert input image(s) into a 3D Gaussian + Mesh representation,
    then generate a short video preview (combined color + geometry).
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))

    if not is_multiimage:
        # Single image mode
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
        # Multi-image mode
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

    # Create a preview video (120 frames)
    video_color = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
    video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']

    # Combine the color + geometry in a side-by-side video
    video = [np.concatenate([video_color[i], video_geo[i]], axis=1) for i in range(len(video_color))]
    video_path = os.path.join(user_dir, 'sample.mp4')
    imageio.mimsave(video_path, video, fps=15)

    # Store the state for further extraction (GLB, Gaussian, etc.)
    state = pack_state(outputs['gaussian'][0], outputs['mesh'][0])
    torch.cuda.empty_cache()
    return state, video_path

# ======================================
# Extraction: GLB
# ======================================
def extract_glb(
    state: dict,
    mesh_simplify: float,
    texture_size: int,
    req: gr.Request
) -> Tuple[str, str]:
    """
    Convert the stored Gaussian+Mesh into a simplified GLB with a texture.
    """
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
    return glb_path, glb_path  # (model_output, download_glb)

# ======================================
# Extraction: Gaussian
# ======================================
def extract_gaussian(state: dict, req: gr.Request) -> Tuple[str, str]:
    """
    Export the Gaussian representation as a .ply file (can be large).
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    gs, _mesh = unpack_state(state)

    ply_path = os.path.join(user_dir, 'sample.ply')
    gs.save_ply(ply_path)
    torch.cuda.empty_cache()
    return ply_path, ply_path

# ======================================
# Multi-Example Helpers
# ======================================
def prepare_multi_example() -> List[Image.Image]:
    """
    Load and combine multiple example images side-by-side for demonstration.
    """
    multi_case = list(set([fn.split('_')[0] for fn in os.listdir("assets/example_multi_image")]))
    images = []
    for case in multi_case:
        subimgs = []
        for i in range(1, 4):
            path = f'assets/example_multi_image/{case}_{i}.png'
            img = Image.open(path)
            W, H = img.size
            # Resize to a uniform height=512
            new_w = int(W / H * 512)
            img = img.resize((new_w, 512))
            subimgs.append(np.array(img))
        # Combine horizontally
        combined = np.concatenate(subimgs, axis=1)
        images.append(Image.fromarray(combined))
    return images

def split_image(image: Image.Image) -> List[Image.Image]:
    """
    Split an input image horizontally into multiple sub-images for multi-view usage.
    (Used by the multi-image tab examples.)
    """
    arr = np.array(image)
    alpha = arr[..., 3]
    alpha_bool = np.any(alpha > 0, axis=0)
    start_pos = np.where(~alpha_bool[:-1] & alpha_bool[1:])[0].tolist()
    end_pos = np.where(alpha_bool[:-1] & ~alpha_bool[1:])[0].tolist()

    subimages = []
    for s, e in zip(start_pos, end_pos):
        sub_image = Image.fromarray(arr[:, s:e+1])
        subimages.append(preprocess_image(sub_image))
    return subimages

# ======================================
# Build the Gradio Interface
# ======================================
def build_interface():
    """
    Construct and return the Gradio Blocks interface with a dark theme (Dracula).
    """
    with gr.Blocks(
        title="Image to 3D Asset with TRELLIS",
        theme=themes.Dracula(),  # Dark theme
        css=None,  # You can specify custom CSS if you like
        analytics_enabled=False
    ) as demo:

        # Explanation at the top
        gr.Markdown("""
        # Image to 3D with [TRELLIS](https://trellis3d.github.io/)
        Convert a single or multiple images into a 3D asset (Gaussian + Mesh)
        in just a few clicks!

        **Steps**:
        1. **Upload** an image (PNG, RGBA, or use background removal).
        2. Tweak generation settings if desired.
        3. Click **Generate 3D Model**.
        4. Export as **GLB** or **Gaussian** for further usage.

        ---
        """)

        with gr.Row():
            with gr.Column():
                with gr.Tabs() as input_tabs:
                    with gr.Tab(label="Single Image", id=0) as single_tab:
                        image_prompt = gr.Image(
                            label="Image Prompt",
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
                        **Experimental**: Provide multiple angles of the same object for a
                        more complete 3D reconstruction.  
                        Use the controls below to experiment with different multi-image
                        generation modes (stochastic vs. multi-diffusion).
                        """)

                # Generation Settings in an Accordion
                with gr.Accordion(label="Generation Settings", open=False):
                    seed = gr.Slider(0, MAX_SEED, value=0, step=1, label="Seed")
                    randomize_seed = gr.Checkbox(value=True, label="Randomize Seed")
                    gr.Markdown("**Stage 1: Sparse Structure Generation**")
                    with gr.Row():
                        ss_guidance_strength = gr.Slider(
                            0.0, 10.0, value=7.5, step=0.1,
                            label="Guidance Strength"
                        )
                        ss_sampling_steps = gr.Slider(
                            1, 50, value=12, step=1,
                            label="Sampling Steps"
                        )
                    gr.Markdown("**Stage 2: Structured Latent Generation**")
                    with gr.Row():
                        slat_guidance_strength = gr.Slider(
                            0.0, 10.0, value=3.0, step=0.1,
                            label="Guidance Strength"
                        )
                        slat_sampling_steps = gr.Slider(
                            1, 50, value=12, step=1,
                            label="Sampling Steps"
                        )
                    multiimage_algo = gr.Radio(
                        choices=["stochastic", "multidiffusion"],
                        value="stochastic",
                        label="Multi-image Algorithm"
                    )

                generate_btn = gr.Button("Generate 3D Model", variant="primary")

                with gr.Accordion(label="GLB Extraction Settings", open=False):
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
                *Gaussian export can be quite large (~50MB+). Please be patient
                when downloading or visualizing it.*
                """)

            with gr.Column():
                video_output = gr.Video(
                    label="Generated 3D Preview",
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
                        label="Download .PLY (Gaussian)",
                        interactive=False
                    )

        # Hidden states
        is_multiimage = gr.State(False)
        output_buf = gr.State()

        # Example images at the bottom
        with gr.Row(visible=True) as single_example_row:
            examples = gr.Examples(
                examples=[
                    os.path.join("assets/example_image", fn)
                    for fn in os.listdir("assets/example_image")
                ],
                inputs=[image_prompt],
                fn=preprocess_image,
                outputs=[image_prompt],
                run_on_click=True,
                examples_per_page=64,
                label="Single Image Examples"
            )

        with gr.Row(visible=False) as multi_example_row:
            examples_multi = gr.Examples(
                examples=prepare_multi_example(),
                inputs=[image_prompt],
                fn=split_image,
                outputs=[multiimage_prompt],
                run_on_click=True,
                examples_per_page=8,
                label="Multi-View Examples"
            )

        # Session Management
        demo.load(start_session)
        demo.unload(end_session)

        # Tab Switching Logic
        def single_tab_fn():
            return (False, gr.Row.update(visible=True), gr.Row.update(visible=False))

        def multi_tab_fn():
            return (True, gr.Row.update(visible=False), gr.Row.update(visible=True))

        single_tab.select(single_tab_fn, outputs=[is_multiimage, single_example_row, multi_example_row])
        multi_tab.select(multi_tab_fn, outputs=[is_multiimage, single_example_row, multi_example_row])

        # Preprocessing on upload
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

        # Generate 3D Callback
        generate_btn.click(
            get_seed,
            inputs=[randomize_seed, seed],
            outputs=[seed]
        ).then(
            image_to_3d,
            inputs=[image_prompt, multiimage_prompt, is_multiimage, seed,
                    ss_guidance_strength, ss_sampling_steps,
                    slat_guidance_strength, slat_sampling_steps, multiimage_algo],
            outputs=[output_buf, video_output],
        ).then(
            lambda: (gr.Button.update(interactive=True),
                     gr.Button.update(interactive=True)),
            outputs=[extract_glb_btn, extract_gs_btn],
        )

        # Reset Extract Buttons when cleared
        video_output.clear(
            lambda: (gr.Button.update(interactive=False),
                     gr.Button.update(interactive=False)),
            outputs=[extract_glb_btn, extract_gs_btn],
        )

        # GLB Extraction
        extract_glb_btn.click(
            extract_glb,
            inputs=[output_buf, mesh_simplify, texture_size],
            outputs=[model_output, download_glb],
        ).then(
            lambda: gr.DownloadButton.update(interactive=True),
            outputs=[download_glb],
        )

        # Gaussian Extraction
        extract_gs_btn.click(
            extract_gaussian,
            inputs=[output_buf],
            outputs=[model_output, download_gs],
        ).then(
            lambda: gr.DownloadButton.update(interactive=True),
            outputs=[download_gs],
        )

        # Reset Download buttons when clearing model_output
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
