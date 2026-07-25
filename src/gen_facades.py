from __future__ import annotations

import argparse
import math
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from threading import Event
from typing import Callable

import numpy as np
import trimesh
from PIL import Image
from shapely.geometry import Polygon
from tqdm import tqdm

from default_configs import DEFAULT_GENERATION_CONFIG, GenerationParametersConfig
from gen_wall import gen_wall_new, place_wall_mesh
from request_control import raise_if_cancelled
from walls_clustering import cluster_points

Vertex = tuple[float, float, float]
Face = list[int]
FacesByGroup = dict[str, list[Face]]
GroupItems = list[tuple[str, list[Face]]]
ProgressCallback = Callable[[int, int], None]


class DSU:
    def __init__(self, size: int) -> None:
        self.parents = list(range(size))
        self.ranks = [0] * size

    def get(self, vertex: int) -> int:
        if vertex == self.parents[vertex]:
            return vertex

        self.parents[vertex] = self.get(self.parents[vertex])
        return self.parents[vertex]

    def unite(self, first: int, second: int) -> None:
        first = self.get(first)
        second = self.get(second)
        if first == second:
            return

        if self.ranks[first] > self.ranks[second]:
            first, second = second, first

        self.parents[first] = second
        if self.ranks[first] == self.ranks[second]:
            self.ranks[second] += 1


def generate_groups(lines: list[str]) -> list[int]:
    vertex_count = sum(bool(line) and line.startswith("v ") for line in lines)
    dsu = DSU(vertex_count)

    for line in lines:
        line = line.strip()
        if not line or not line.startswith("f "):
            continue

        previous_vertex: int | None = None
        for part in line.split()[1:]:
            vertex = int(part.split("/")[0]) - 1
            if previous_vertex is not None:
                dsu.unite(previous_vertex, vertex)
            previous_vertex = vertex

    groups = [-1] * vertex_count
    group_count = 0
    for vertex in range(vertex_count):
        parent = dsu.get(vertex)
        if groups[parent] == -1:
            groups[parent] = group_count
            group_count += 1
        groups[vertex] = groups[parent]

    return groups


def parse_obj_groups(
    path: str | Path,
    gen_groups: bool = True,
    max_wall_aspect_ratio: float | None = DEFAULT_GENERATION_CONFIG.max_wall_aspect_ratio,
) -> tuple[list[Vertex], FacesByGroup]:
    vertices: list[Vertex] = []
    groups: FacesByGroup = {}
    current_group: str = "building"

    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    group_ids = generate_groups(lines) if gen_groups else []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("v "):
            _, x, y, z = line.split()
            vertices.append((float(x), float(y), float(z)))
            continue

        if line.startswith("o "):
            if gen_groups:
                continue

            current_group = line.split(maxsplit=1)[1]
            groups.setdefault(current_group, [])
            continue

        if not line.startswith("f "):
            continue

        indices = [int(part.split("/")[0]) - 1 for part in line.split()[1:]]
        if gen_groups:
            current_group = f"building_{group_ids[indices[-1]]}"

        groups.setdefault(current_group, []).append(indices)

    if max_wall_aspect_ratio is None:
        return vertices, groups

    return split_long_vertical_faces(
        vertices,
        groups,
        max_aspect_ratio=max_wall_aspect_ratio,
    )


def compute_normal(face_points: np.ndarray) -> np.ndarray:
    first_edge = face_points[1] - face_points[0]
    second_edge = face_points[2] - face_points[0]
    normal = np.cross(first_edge, second_edge)
    norm = np.linalg.norm(normal)
    if norm != 0:
        normal = normal / norm
    return normal


def _edge_vertical_fraction(first: np.ndarray, second: np.ndarray) -> float:
    edge = second - first
    length = float(np.linalg.norm(edge))
    if length == 0.0:
        return math.inf
    return abs(float(edge[1])) / length


def _append_vertex(vertices: list[Vertex], point: np.ndarray) -> int:
    vertices.append((float(point[0]), float(point[1]), float(point[2])))
    return len(vertices) - 1


def split_long_vertical_face(
    face: Face,
    vertices: list[Vertex],
    max_aspect_ratio: float = DEFAULT_GENERATION_CONFIG.max_wall_aspect_ratio,
    normal_tolerance: float = 0.01,
) -> list[Face]:
    if max_aspect_ratio <= 0:
        raise ValueError("max_aspect_ratio must be positive")
    if len(face) != 4:
        return [face]

    points = np.asarray([vertices[index] for index in face], dtype=float)
    normal = compute_normal(points)
    if np.linalg.norm(normal) == 0.0 or abs(float(normal[1])) >= normal_tolerance:
        return [face]

    pair_0_vertical_fraction = (
        _edge_vertical_fraction(points[0], points[1])
        + _edge_vertical_fraction(points[2], points[3])
    ) / 2.0
    pair_1_vertical_fraction = (
        _edge_vertical_fraction(points[1], points[2])
        + _edge_vertical_fraction(points[3], points[0])
    ) / 2.0

    width_edge_start = 0 if pair_0_vertical_fraction <= pair_1_vertical_fraction else 1
    a = width_edge_start
    b = (a + 1) % 4
    c = (a + 2) % 4
    d = (a + 3) % 4

    first_width_vector = points[b] - points[a]
    opposite_width_vector = points[c] - points[d]
    first_height_vector = points[c] - points[b]
    opposite_height_vector = points[d] - points[a]

    if not np.allclose(first_width_vector, opposite_width_vector, rtol=1e-3, atol=1e-5):
        return [face]
    if not np.allclose(first_height_vector, opposite_height_vector, rtol=1e-3, atol=1e-5):
        return [face]

    width = (
        float(np.linalg.norm(first_width_vector))
        + float(np.linalg.norm(opposite_width_vector))
    ) / 2.0
    height = (
        float(np.linalg.norm(first_height_vector))
        + float(np.linalg.norm(opposite_height_vector))
    ) / 2.0

    if width == 0.0 or height == 0.0:
        return [face]

    aspect_ratio = width / height
    if aspect_ratio + 1e-9 < max_aspect_ratio:
        return [face]

    segment_count = math.floor(aspect_ratio / max_aspect_ratio + 1e-9) + 1
    first_edge_indices = [face[a]]
    opposite_edge_indices = [face[d]]

    for segment_index in range(1, segment_count):
        fraction = segment_index / segment_count
        first_point = points[a] + first_width_vector * fraction
        opposite_point = points[d] + opposite_width_vector * fraction
        first_edge_indices.append(_append_vertex(vertices, first_point))
        opposite_edge_indices.append(_append_vertex(vertices, opposite_point))

    first_edge_indices.append(face[b])
    opposite_edge_indices.append(face[c])

    return [
        [
            first_edge_indices[index],
            first_edge_indices[index + 1],
            opposite_edge_indices[index + 1],
            opposite_edge_indices[index],
        ]
        for index in range(segment_count)
    ]


def split_long_vertical_faces(
    vertices: list[Vertex],
    groups: FacesByGroup,
    max_aspect_ratio: float = DEFAULT_GENERATION_CONFIG.max_wall_aspect_ratio,
) -> tuple[list[Vertex], FacesByGroup]:
    split_vertices = list(vertices)
    split_groups: FacesByGroup = {}

    for group_name, faces in groups.items():
        split_faces: list[Face] = []
        for face in faces:
            split_faces.extend(
                split_long_vertical_face(
                    face,
                    split_vertices,
                    max_aspect_ratio=max_aspect_ratio,
                )
            )
        split_groups[group_name] = split_faces

    return split_vertices, split_groups


def create_plane_from_n_points(points: np.ndarray) -> trimesh.Trimesh:
    vertices = np.asarray(points, dtype=float)
    source_normal = compute_normal(vertices)
    if np.linalg.norm(source_normal) == 0.0:
        raise ValueError("Cannot triangulate a degenerate horizontal face")

    polygon = Polygon(vertices[:, [0, 2]])
    triangulated_vertices, faces = trimesh.creation.triangulate_polygon(polygon)
    faces = np.asarray(faces, dtype=int)

    y_coordinate = vertices[0, 1]
    triangulated_vertices_3d = np.column_stack(
        [
            triangulated_vertices[:, 0],
            np.full(len(triangulated_vertices), y_coordinate),
            triangulated_vertices[:, 1],
        ]
    )

    for face_index, face in enumerate(faces):
        generated_normal = compute_normal(triangulated_vertices_3d[face])
        if np.dot(generated_normal, source_normal) < 0.0:
            faces[face_index] = face[[0, 2, 1]]

    return trimesh.Trimesh(
        vertices=triangulated_vertices_3d,
        faces=faces,
        process=False,
    )


def get_wall_size(vertices: np.ndarray) -> tuple[float, float]:
    width = np.linalg.norm(vertices[1] - vertices[0])
    height = np.linalg.norm(vertices[2] - vertices[1])

    if np.abs(vertices[1][1] - vertices[0][1]) >= 0.01:
        width, height = height, width

    return float(width), float(height)


def get_wall_sizes(groups_items: GroupItems, vertices: list[Vertex]) -> np.ndarray:
    wall_sizes: list[tuple[float, float]] = []

    for _, faces in groups_items:
        for face in faces:
            points = np.array([vertices[index] for index in face])
            normal = compute_normal(points)
            if abs(normal[1]) >= 0.01:
                continue
            wall_sizes.append(get_wall_size(points))

    return np.array(wall_sizes)


def derive_seed(base_seed: int | None, offset: int) -> int | None:
    if base_seed is None:
        return None
    return (base_seed + offset) % (2**31 - 1)


def build_facades(
    wall_sizes: np.ndarray,
    visual_path: Path | None,
    storage_path: Path,
    style_ref: Image.Image | None,
    style_ref_scale: float,
    pixels_per_meter: int,
    config: GenerationParametersConfig,
    cancel_event: Event | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    raise_if_cancelled(cancel_event)
    random_state = derive_seed(config.seed, 0)
    frame_sizes, labels = cluster_points(
        wall_sizes,
        config.cluster_count,
        random_state=0 if random_state is None else random_state,
    )

    total_models = len(frame_sizes)
    for frame_index, (width, height) in enumerate(frame_sizes):
        print()

        raise_if_cancelled(cancel_event)
        wall = gen_wall_new(
            frame_index,
            width,
            height,
            visual_path,
            style_ref,
            style_ref_scale,
            pixels_per_meter,
            seed=derive_seed(config.seed, frame_index),
            config=config,
            cancel_event=cancel_event,
        )
        raise_if_cancelled(cancel_event)
        wall.export(str(storage_path / f"wall_{frame_index}.glb"))

        completed_models = frame_index + 1
        print(
            f"\nGenerated wall model {completed_models}/{total_models}\n",
            file=sys.stderr,
            flush=True,
        )
        if progress_callback is not None:
            progress_callback(completed_models, total_models)

    raise_if_cancelled(cancel_event)
    return frame_sizes, labels


def build_scene(
    groups_items: GroupItems,
    vertices: list[Vertex],
    labels: np.ndarray,
    visual_path: Path | None,
    storage_path: Path,
    config: GenerationParametersConfig,
    cancel_event: Event | None = None,
) -> trimesh.Scene:
    raise_if_cancelled(cancel_event)
    scene = trimesh.Scene()
    wall_index = 0

    for name, faces in tqdm(groups_items):
        raise_if_cancelled(cancel_event)
        walls: list[trimesh.Trimesh] = []

        for face in faces:
            raise_if_cancelled(cancel_event)
            points = np.array([vertices[index] for index in face])
            normal = compute_normal(points)

            if abs(normal[1]) >= 0.01:
                plane = create_plane_from_n_points(points)
                plane.apply_translation(-normal * 0.2)
                walls.append(plane)
                continue

            wall_mesh = trimesh.load(
                str(storage_path / f"wall_{labels[wall_index]}.glb"),
                force="scene",
            ).to_mesh()
            if isinstance(wall_mesh.visual.material, trimesh.visual.material.PBRMaterial):
                wall_mesh.visual.material = wall_mesh.visual.material.to_simple()

            walls.append(
                place_wall_mesh(
                    wall_mesh,
                    points,
                    wall_index,
                    visual_path,
                    config=config,
                )
            )
            wall_index += 1

        raise_if_cancelled(cancel_event)
        scene.add_geometry(trimesh.util.concatenate(walls), node_name=name, geom_name=name)

    raise_if_cancelled(cancel_event)
    return scene


def generate_facade_scene(
    input_path: str | Path,
    visual_path: str | Path | None,
    pixels_per_meter: int,
    style_ref: Image.Image | None,
    config: GenerationParametersConfig | None = None,
    cancel_event: Event | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[trimesh.Scene, np.ndarray, np.ndarray]:
    raise_if_cancelled(cancel_event)
    if pixels_per_meter <= 0:
        raise ValueError("pixels_per_meter must be positive")

    config = config or DEFAULT_GENERATION_CONFIG
    resolved_visual_path = Path(visual_path) if visual_path is not None else None
    if resolved_visual_path is not None:
        resolved_visual_path.mkdir(parents=True, exist_ok=True)

    if config.temp_root is not None:
        config.temp_root.mkdir(parents=True, exist_ok=True)

    effective_style_ref_scale = config.style_ref_scale if style_ref is not None else 0.0
    vertices, groups = parse_obj_groups(
        input_path,
        gen_groups=False,
        max_wall_aspect_ratio=config.max_wall_aspect_ratio,
    )
    raise_if_cancelled(cancel_event)
    groups_items = list(groups.items())
    wall_sizes = get_wall_sizes(groups_items, vertices)
    if len(wall_sizes) == 0:
        raise ValueError("The OBJ model does not contain any vertical wall faces")

    with tempfile.TemporaryDirectory(
        prefix="facade-generation-",
        dir=config.temp_root,
    ) as temporary_directory:
        storage_path = Path(temporary_directory)
        frame_sizes, labels = build_facades(
            wall_sizes,
            resolved_visual_path,
            storage_path,
            style_ref,
            effective_style_ref_scale,
            pixels_per_meter,
            config,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
        )
        raise_if_cancelled(cancel_event)
        scene = build_scene(
            groups_items,
            vertices,
            labels,
            resolved_visual_path,
            storage_path,
            config,
            cancel_event=cancel_event,
        )

    raise_if_cancelled(cancel_event)
    return scene, frame_sizes, labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate detailed facade meshes for an OBJ scene.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Input OBJ scene path.",
    )
    parser.add_argument(
        "--output-filename",
        "--result-path",
        dest="output_filename",
        type=Path,
        required=True,
        help="Output GLB scene path. --result-path is accepted as a legacy alias.",
    )
    parser.add_argument(
        "--visual-path",
        type=Path,
        default=None,
        help="Optional directory for intermediate visual outputs. If omitted, intermediate outputs are not written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_GENERATION_CONFIG.seed,
        help="Seed for generation.",
    )
    parser.add_argument(
        "--temp-root",
        type=Path,
        default=DEFAULT_GENERATION_CONFIG.temp_root,
        help="Directory for storing temporary files (wall meshes).",
    )

    parser.add_argument(
        "--pixels-per-meter",
        type=int,
        required=True,
        help="Number of pixels per meter when generating facade image. Used to infer resolution of facade image.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_GENERATION_CONFIG.prompt,
        help="Prompt for facade image generation.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=DEFAULT_GENERATION_CONFIG.negative_prompt,
        help="Negative prompt for facade image generation.",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=DEFAULT_GENERATION_CONFIG.diffusion_steps,
        help="Diffusion steps for facade image generation.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=DEFAULT_GENERATION_CONFIG.guidance_scale,
        help="Diffusion guidance scale for facade image generation.",
    )
    parser.add_argument(
        "--cross-attention-scale",
        type=float,
        default=DEFAULT_GENERATION_CONFIG.cross_attention_scale,
        help="Cross-attention scale (LoRA) for facade image generation.",
    )
    parser.add_argument(
        "--controlnet-conditioning-scale",
        type=float,
        default=DEFAULT_GENERATION_CONFIG.controlnet_conditioning_scale,
        help="ControlNet conditioning scale for facade image generation.",
    )
    parser.add_argument(
        "--control-guidance-start",
        type=float,
        default=DEFAULT_GENERATION_CONFIG.control_guidance_start,
        help="ControlNet guidance start for facade image generation.",
    )
    parser.add_argument(
        "--control-guidance-end",
        type=float,
        default=DEFAULT_GENERATION_CONFIG.control_guidance_end,
        help="ControlNet guidance end for facade image generation.",
    )
    style_group = parser.add_mutually_exclusive_group(required=True)
    style_group.add_argument(
        "--style-ref-path",
        type=Path,
        help="Style-reference image path.",
    )
    style_group.add_argument(
        "--no-style-ref",
        action="store_true",
        help="Disable style-reference conditioning.",
    )
    parser.add_argument(
        "--style-ref-scale",
        type=float,
        default=DEFAULT_GENERATION_CONFIG.style_ref_scale,
        help="IP-Adapter scale when style reference is enabled.",
    )

    parser.add_argument(
        "--cluster-count",
        type=int,
        required=True,
        help="Number of different wall meshes to generate.",
    )
    parser.add_argument(
        "--max-wall-aspect-ratio",
        type=float,
        default=DEFAULT_GENERATION_CONFIG.max_wall_aspect_ratio,
        help="Split vertical quad faces so every resulting wall is narrower than this width/height ratio.",
    )
    parser.add_argument(
        "--slat-steps",
        type=int,
        default=DEFAULT_GENERATION_CONFIG.slat_steps,
        help="SLAT generation steps for TRELLIS.",
    )
    parser.add_argument(
        "--slat-cfg-strength",
        type=float,
        default=DEFAULT_GENERATION_CONFIG.slat_cfg_strength,
        help="SLAT generation CFG strength for TRELLIS.",
    )
    parser.add_argument(
        "--border-size",
        type=int,
        default=DEFAULT_GENERATION_CONFIG.border_size,
        help="Border size for input image for TRELLIS.",
    )
    parser.add_argument(
        "--trellis-mode",
        default=DEFAULT_GENERATION_CONFIG.trellis_mode,
        help="Generation mode for TRELLIS.",
    )
    parser.add_argument(
        "--mesh-simplify",
        type=float,
        default=DEFAULT_GENERATION_CONFIG.mesh_simplify,
        help="Mesh postprocessing simplification (ratio of vertices to remove).",
    )
    parser.add_argument(
        "--texture-size",
        type=int,
        default=DEFAULT_GENERATION_CONFIG.texture_size,
        help="Mesh texture size. Must be a power of two.",
    )
    parser.add_argument(
        "--texture-brightness-factor",
        type=float,
        default=DEFAULT_GENERATION_CONFIG.texture_brightness_factor,
        help="After wall mesh is generated, its texture channel values are multiplied by this number.",
    )
    parser.add_argument(
        "--texture-postprocess-shrink-px",
        type=int,
        default=DEFAULT_GENERATION_CONFIG.texture_postprocess_shrink_px,
        help="Shrink each generated texture by this many pixels per dimension before atlas packing.",
    )
    parser.add_argument(
        "--depth-scale-reference",
        type=float,
        default=DEFAULT_GENERATION_CONFIG.depth_scale_reference,
        help="Used to control balconies' depth.",
    )

    return parser.parse_args()


def create_generation_config(args: argparse.Namespace) -> GenerationParametersConfig:
    return replace(
        DEFAULT_GENERATION_CONFIG,
        output_filename=str(args.output_filename),
        cluster_count=args.cluster_count,
        seed=args.seed,
        temp_root=args.temp_root,
        style_ref_scale=args.style_ref_scale,
        max_wall_aspect_ratio=args.max_wall_aspect_ratio,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        diffusion_steps=args.diffusion_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        control_guidance_start=args.control_guidance_start,
        control_guidance_end=args.control_guidance_end,
        cross_attention_scale=args.cross_attention_scale,
        slat_steps=args.slat_steps,
        slat_cfg_strength=args.slat_cfg_strength,
        trellis_mode=args.trellis_mode,
        mesh_simplify=args.mesh_simplify,
        texture_size=args.texture_size,
        border_size=args.border_size,
        texture_brightness_factor=args.texture_brightness_factor,
        texture_postprocess_shrink_px=args.texture_postprocess_shrink_px,
        depth_scale_reference=args.depth_scale_reference,
    )


def load_style_reference(path: Path | None) -> Image.Image | None:
    if path is None:
        return None
    with Image.open(path) as image:
        return image.copy()


def main() -> None:
    args = parse_args()
    config = create_generation_config(args)
    style_ref = load_style_reference(args.style_ref_path)

    scene, frame_sizes, labels = generate_facade_scene(
        input_path=args.input_path,
        visual_path=args.visual_path,
        pixels_per_meter=args.pixels_per_meter,
        style_ref=style_ref,
        config=config,
    )

    output_path = Path(config.output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(output_path))


if __name__ == "__main__":
    main()
