from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeEnvironmentConfig:
    attention_backend: str = "xformers"
    spconv_algorithm: str = "native"


@dataclass(frozen=True)
class ModelParametersConfig:
    base_model: str = "CompVis/stable-diffusion-v1-4"
    controlnet_id: str = "lllyasviel/sd-controlnet-seg"
    lora_path: Path = Path("./aux/buildingface.safetensors")
    lora_adapter_name: str = "buildingface"
    ip_adapter_repository: str = "h94/IP-Adapter"
    ip_adapter_subfolder: str = "models"
    ip_adapter_weight_name: str = "ip-adapter_sd15.bin"
    trellis_model: str = "microsoft/TRELLIS-image-large"


@dataclass(frozen=True)
class GenerationParametersConfig:
    output_filename: str = "building.glb"
    cluster_count: int = 64
    seed: int | None = None
    temp_root: Path | None = None

    prompt: str = "modern style architecture, urban house"
    negative_prompt: str = "old, dark, distorted"
    diffusion_steps: int = 30
    guidance_scale: float = 7.0
    controlnet_conditioning_scale: float = 1.1
    control_guidance_start: float = 0.0
    control_guidance_end: float = 1.0
    cross_attention_scale: float = 0.7
    style_ref_scale: float = 0.5

    sparse_structure_steps: int = 25
    sparse_structure_cfg_strength: float = 7.0
    slat_steps: int = 25
    slat_cfg_strength: float = 3.0
    trellis_mode: str = "multidiffusion"

    mesh_simplify: float = 0.95
    texture_size: int = 512
    border_size: int = 16
    texture_brightness_factor: float = 1.4
    texture_postprocess_shrink_px: int = 4
    depth_scale_reference: float = 32.0
    max_wall_aspect_ratio: float = 1.5

    def __post_init__(self) -> None:
        if self.cluster_count <= 0:
            raise ValueError("cluster_count must be positive")
        if self.style_ref_scale < 0:
            raise ValueError("style_ref_scale must be non-negative")
        if self.max_wall_aspect_ratio <= 0:
            raise ValueError("max_wall_aspect_ratio must be positive")
        if self.diffusion_steps <= 0:
            raise ValueError("diffusion_steps must be positive")
        if self.sparse_structure_steps <= 0:
            raise ValueError("sparse_structure_steps must be positive")
        if self.slat_steps <= 0:
            raise ValueError("slat_steps must be positive")
        if self.texture_size <= 0:
            raise ValueError("texture_size must be positive")
        if self.border_size < 0:
            raise ValueError("border_size must be non-negative")
        if self.texture_postprocess_shrink_px < 0:
            raise ValueError("texture_postprocess_shrink_px must be non-negative")
        if self.depth_scale_reference <= 0:
            raise ValueError("depth_scale_reference must be positive")


ENVIRONMENT_CONFIG = RuntimeEnvironmentConfig()
DEFAULT_MODEL_CONFIG = ModelParametersConfig()
DEFAULT_GENERATION_CONFIG = GenerationParametersConfig()
