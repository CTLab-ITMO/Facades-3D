from __future__ import annotations

import gc
import os
from contextlib import suppress
from pathlib import Path
from threading import Event
from typing import cast

from default_configs import (
    DEFAULT_GENERATION_CONFIG,
    DEFAULT_MODEL_CONFIG,
    ENVIRONMENT_CONFIG,
    GenerationParametersConfig,
    ModelParametersConfig,
)

WallModelConfig = ModelParametersConfig
WallGenerationConfig = GenerationParametersConfig

os.environ["ATTN_BACKEND"] = ENVIRONMENT_CONFIG.attention_backend
os.environ["SPCONV_ALGO"] = ENVIRONMENT_CONFIG.spconv_algorithm

import numpy as np
import torch
import trimesh
from PIL import Image
from trimesh.visual.texture import TextureVisuals

from facade_semantic_map_gen_v2 import render_facade_segmentation
from request_control import raise_if_cancelled


MODEL_CONFIG = DEFAULT_MODEL_CONFIG

if not os.getenv("SKIP_MODEL_LOAD"):
    from diffusers import ControlNetModel, DPMSolverMultistepScheduler, StableDiffusionControlNetPipeline

    controlnet = ControlNetModel.from_pretrained(
        MODEL_CONFIG.controlnet_id,
        torch_dtype=torch.float16,
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        MODEL_CONFIG.base_model,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(
        str(MODEL_CONFIG.lora_path.parent),
        weight_name=MODEL_CONFIG.lora_path.name,
        adapter_name=MODEL_CONFIG.lora_adapter_name,
    )

if not os.getenv("SKIP_MODEL_LOAD"):
    from trellis_image_to_facade import TrellisImageToFacadePipeline
    from trellis.utils import postprocessing_utils

    pipeline = TrellisImageToFacadePipeline.from_pretrained(MODEL_CONFIG.trellis_model)


def _cleanup_cuda_state() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def release_generation_memory() -> None:
    if not os.getenv("SKIP_MODEL_LOAD"):
        return

    try:
        if "pipe" in globals():
            pipe.to("cpu", silence_dtype_warnings=True)
    finally:
        try:
            if "pipeline" in globals():
                pipeline.cpu()
        finally:
            _cleanup_cuda_state()


def is_ip_adapter_loaded(pipe):
    return (
        getattr(pipe.unet, "encoder_hid_proj", None) is not None
        and getattr(pipe.unet.config, "encoder_hid_dim_type", None) == "ip_image_proj"
    )


def enable_ip_adapter(pipe):
    if is_ip_adapter_loaded(pipe):
        return

    pipe.load_ip_adapter(
        MODEL_CONFIG.ip_adapter_repository,
        subfolder=MODEL_CONFIG.ip_adapter_subfolder,
        weight_name=MODEL_CONFIG.ip_adapter_weight_name,
    )


def disable_ip_adapter(pipe):
    if not is_ip_adapter_loaded(pipe):
        return

    pipe.unload_ip_adapter()


def compute_normal(face_points: np.ndarray) -> np.ndarray:
    first_edge = face_points[1] - face_points[0]
    second_edge = face_points[2] - face_points[0]
    normal = np.cross(first_edge, second_edge)
    norm = np.linalg.norm(normal)
    if norm != 0:
        normal = normal / norm
    return normal


def get_wall_size(vertices: np.ndarray) -> tuple[float, float]:
    width = np.linalg.norm(vertices[1] - vertices[0])
    height = np.linalg.norm(vertices[2] - vertices[1])

    if np.abs(vertices[1][1] - vertices[0][1]) >= 0.01:
        width, height = height, width

    return float(width), float(height)


def recover_balconies(semantic_map: Image.Image) -> Image.Image:
    pixels = np.array(semantic_map)
    overlaps_windows = np.all(pixels == (170, 255, 170), axis=-1)
    overlaps_doors = np.all(pixels == (170, 255, 255), axis=-1)
    pixels[overlaps_windows | overlaps_doors] = (170, 255, 85)
    return Image.fromarray(pixels)


def create_plane_from_4_points(points: np.ndarray) -> trimesh.Trimesh:
    vertices = np.array(points, dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _create_torch_generator(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    return torch.Generator(device="cuda").manual_seed(seed)


def gen_wall_image(
    semantic_map: Image.Image,
    style_ref: Image.Image | None,
    style_ref_scale: float,
    width_px: int,
    height_px: int,
    seed: int | None = None,
    config: GenerationParametersConfig | None = None,
    cancel_event: Event | None = None,
) -> Image.Image:
    config = config or DEFAULT_GENERATION_CONFIG
    raise_if_cancelled(cancel_event)

    semantic_map = recover_balconies(semantic_map)
    use_style_reference = style_ref is not None and style_ref_scale > 0.0

    if use_style_reference:
        enable_ip_adapter(pipe)
        pipe.set_ip_adapter_scale(style_ref_scale)
    else:
        disable_ip_adapter(pipe)

    try:
        pipe.to("cuda")
        raise_if_cancelled(cancel_event)

        pipeline_kwargs: dict[str, object] = {
            "prompt": config.prompt,
            "negative_prompt": config.negative_prompt,
            "image": semantic_map,
            "width": width_px,
            "height": height_px,
            "num_inference_steps": config.diffusion_steps,
            "guidance_scale": config.guidance_scale,
            "controlnet_conditioning_scale": config.controlnet_conditioning_scale,
            "control_guidance_start": config.control_guidance_start,
            "control_guidance_end": config.control_guidance_end,
            "cross_attention_kwargs": {"scale": config.cross_attention_scale},
            "generator": _create_torch_generator(seed),
        }
        if use_style_reference:
            pipeline_kwargs["ip_adapter_image"] = style_ref

        image = pipe(**pipeline_kwargs).images[0]
        raise_if_cancelled(cancel_event)
        return image
    finally:
        with suppress(Exception):
            pipe.to("cpu", silence_dtype_warnings=True)


def add_borders_to_image(
    image: Image.Image,
    border_size: int = 16,
    color: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> Image.Image:
    image = image.convert("RGBA")
    bordered_image = Image.new(
        "RGBA",
        (image.width + border_size * 2, image.height + border_size * 2),
        color,
    )
    bordered_image.paste(image, (border_size, border_size))
    return bordered_image


def gen_wall_model(
    image: Image.Image,
    semantic_map: Image.Image,
    seed: int | None = None,
    config: GenerationParametersConfig | None = None,
    cancel_event: Event | None = None,
) -> trimesh.Trimesh:
    config = config or DEFAULT_GENERATION_CONFIG
    raise_if_cancelled(cancel_event)

    try:
        pipeline.cuda()
        raise_if_cancelled(cancel_event)

        run_kwargs: dict[str, object] = {
            "sparse_structure_sampler_params": {
                "steps": config.sparse_structure_steps,
                "cfg_strength": config.sparse_structure_cfg_strength,
            },
            "slat_sampler_params": {
                "steps": config.slat_steps,
                "cfg_strength": config.slat_cfg_strength,
            },
            "mode": config.trellis_mode,
        }
        if seed is not None:
            run_kwargs["seed"] = seed

        outputs = pipeline.run_multi_image([image], semantic_map, **run_kwargs)
        raise_if_cancelled(cancel_event)

        mesh = postprocessing_utils.to_glb(
            outputs["gaussian"][0],
            outputs["mesh"][0],
            simplify=config.mesh_simplify,
            texture_size=config.texture_size,
        )
        raise_if_cancelled(cancel_event)
        return mesh
    finally:
        with suppress(Exception):
            pipeline.cpu()


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _resize_texture_for_packing(image: Image.Image, shrink_px: int) -> Image.Image:
    width, height = image.size
    if shrink_px <= 0:
        return image
    if width <= shrink_px or height <= shrink_px:
        return image
    if not (_is_power_of_two(width) and _is_power_of_two(height)):
        return image

    return image.resize((width - shrink_px, height - shrink_px), Image.Resampling.LANCZOS)


def _postprocess_texture(mesh: trimesh.Trimesh, factor: float, shrink_px: int) -> None:
    uv = mesh.visual.uv
    texture = mesh.visual.material.baseColorTexture
    if not isinstance(texture, Image.Image):
        texture = Image.fromarray(np.asarray(texture).astype(np.uint8))

    texture = _resize_texture_for_packing(texture, shrink_px=shrink_px)
    texture_array = np.clip(np.asarray(texture, dtype=np.float32) * factor, 0.0, 255.0)

    mesh.visual = TextureVisuals(uv=uv, image=Image.fromarray(texture_array.astype(np.uint8)))


def _generate_wall_mesh(
    file_stem: str,
    width: float,
    height: float,
    visual_path: str | Path | None,
    style_ref: Image.Image | None,
    style_ref_scale: float,
    pixels_per_meter: float,
    seed: int | None = None,
    config: GenerationParametersConfig | None = None,
    cancel_event: Event | None = None,
) -> trimesh.Trimesh:
    config = config or DEFAULT_GENERATION_CONFIG
    raise_if_cancelled(cancel_event)
    resolved_visual_path = Path(visual_path) if visual_path is not None else None
    width_px = int(width * pixels_per_meter + 0.5)
    height_px = int(height * pixels_per_meter + 0.5)
    semantic_map = render_facade_segmentation(
        width_m=width,
        height_m=height,
        pixels_per_meter=pixels_per_meter,
        seed=seed,
    )
    if resolved_visual_path is not None:
        semantic_map.save(resolved_visual_path / f"{file_stem}-sm.png")

    image = gen_wall_image(
        semantic_map,
        style_ref,
        style_ref_scale,
        width_px,
        height_px,
        seed=seed,
        config=config,
        cancel_event=cancel_event,
    )
    raise_if_cancelled(cancel_event)
    if resolved_visual_path is not None:
        image.save(resolved_visual_path / f"{file_stem}.png")

    pipeline.coords_dump_name = (
        str(resolved_visual_path / f"{file_stem}-ss.npy")
        if resolved_visual_path is not None
        else None
    )
    mesh = gen_wall_model(
        add_borders_to_image(image, border_size=config.border_size),
        semantic_map,
        seed=seed,
        config=config,
        cancel_event=cancel_event,
    )
    raise_if_cancelled(cancel_event)
    _postprocess_texture(
        mesh,
        config.texture_brightness_factor,
        config.texture_postprocess_shrink_px,
    )
    raise_if_cancelled(cancel_event)
    return mesh


def gen_wall_new(
    wall_idx: int,
    w: float,
    h: float,
    visual_path: str | Path | None,
    style_ref: Image.Image | None,
    style_ref_scale: float,
    px_per_meter: float | None = None,
    seed: int | None = None,
    config: GenerationParametersConfig | None = None,
    cancel_event: Event | None = None,
) -> trimesh.Trimesh:
    return _generate_wall_mesh(
        str(wall_idx),
        w,
        h,
        visual_path,
        style_ref,
        style_ref_scale,
        cast(float, px_per_meter),
        seed=seed,
        config=config,
        cancel_event=cancel_event,
    )


def place_wall_mesh(
    wall_mesh: trimesh.Trimesh,
    vertices: np.ndarray,
    wall_idx: int,
    visual_path: str | Path | None,
    config: GenerationParametersConfig | None = None,
) -> trimesh.Trimesh:
    config = config or DEFAULT_GENERATION_CONFIG
    width, height = get_wall_size(vertices)
    x, _, z = compute_normal(vertices)
    angle = np.arctan2(-x, -z)

    size_x, size_y, _ = wall_mesh.bounding_box.extents
    scale_x = width / size_x
    scale_y = height / size_y
    scale_z = config.depth_scale_reference / max(size_x, size_y)

    transform = trimesh.transformations.compose_matrix(
        angles=[0, angle, 0],
        translate=vertices.mean(axis=0),
        scale=[scale_x, scale_y, scale_z],
    )
    wall_mesh.apply_transform(transform)
    return wall_mesh
