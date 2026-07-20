from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from PIL import Image, ImageDraw


ClassName = Literal[
    "facade_wall",
    "window",
    "door",
    "cornice",
    "sill",
    "balcony",
    "molding",
]
TopDecoration = Literal["none", "molding", "cornice"]


COLORS: dict[ClassName, tuple[int, int, int]] = {
    "facade_wall": (0, 0, 255),
    "window": (0, 85, 255),
    "door": (0, 170, 255),
    "cornice": (0, 255, 255),
    "sill": (85, 255, 170),
    "balcony": (170, 255, 85),
    "molding": (255, 85, 0),
}


BALCONY_OVERLAP_WINDOW_COLOR: tuple[int, int, int] = (170, 255, 170)
BALCONY_OVERLAP_DOOR_COLOR: tuple[int, int, int] = (170, 255, 255)


@dataclass(frozen=True)
class GeneratorConfig:

    target_upper_floor_height_m: float = 3.75
    target_ground_floor_height_m: float = 4.5
    min_upper_floors: int = 1
    max_ground_floor_fraction: float = 0.32


    roof_offset_m: float = 1.0


    target_bay_width_m: float = 2.7
    min_bays: int = 1
    max_bays: int = 2 ** 20


    upper_window_width_min_m: float = 0.7
    upper_window_width_max_m: float = 1.35
    upper_window_width_bay_fraction: float = 0.50
    upper_window_height_min_m: float = 1.0
    upper_window_height_max_m: float = 1.75
    upper_window_bottom_min_m: float = 0.9
    upper_window_bottom_max_m: float = 1.25


    ground_window_width_bay_fraction: float = 0.72
    ground_window_height_min_m: float = 1.75
    ground_window_height_max_m: float = 2.35
    ground_window_bottom_m: float = 0.75

    ground_door_width_bay_fraction: float = 0.66
    ground_door_width_min_m: float = 0.95
    ground_door_width_max_m: float = 1.45
    ground_door_height_min_m: float = 2.25
    ground_door_height_max_m: float = 2.85
    ground_door_bottom_m: float = 0.0

    ground_entrance_probability: float = 0.18
    ground_min_doors: int = 1
    ground_max_doors: int = 2


    ground_empty_bay_probability: float = 0.1
    upper_empty_bay_probability: float = 0.2


    balcony_probability: float = 0.8


    balcony_max_count_per_floor: int = 2 ** 20


    balcony_target_bays_per_balcony: float = 6.0
    balcony_count_jitter: int = 1
    balcony_min_width_bays: int = 2
    balcony_max_width_bays: int = 4


    balcony_width_2b_probability: float = 0.4
    balcony_width_3b_probability: float = 0.4
    balcony_width_4b_probability: float = 0.2

    balcony_bottom_offset_m: float = 0.10
    balcony_height_m: float = 1.30
    balcony_side_inset_m: float = 0.08

    balcony_has_door_probability: float = 0.0
    balcony_door_width_bay_fraction: float = 0.52
    balcony_door_width_min_m: float = 0.85
    balcony_door_width_max_m: float = 1.25
    balcony_door_bottom_m: float = 0.18


    sill_width_extra_m: float = 0.22
    sill_height_m: float = 0.10
    sill_gap_under_window_m: float = 0.08


    window_cornice_probability: float = 0.42
    window_cornice_width_extra_m: float = 0.22
    window_cornice_height_m: float = 0.10
    window_cornice_gap_above_window_m: float = 0.07

    window_side_molding_probability: float = 0.22
    window_side_molding_width_m: float = 0.08
    window_side_molding_gap_m: float = 0.06


    floor_top_molding_probability: float = 0.38
    floor_molding_height_m: float = 0.12
    floor_molding_y_inset_m: float = 0.03


    ground_top_molding_probability: float = 0.70
    ground_top_cornice_probability: float = 0.15


    top_cornice_probability: float = 0.65
    top_cornice_height_m: float = 0.18
    top_cornice_y_inset_m: float = 0.25


    one_upper_concept_probability: float = 0.70
    two_upper_concepts_probability: float = 0.30
    three_upper_concepts_probability: float = 0.00


    min_pixel_rect_size: int = 1
    grid_border_px: int = 8
    grid_border_color: tuple[int, int, int] = (255, 255, 255)


    debug_line_width_px: int = 2
    debug_bay_boundary_color: tuple[int, int, int] = (255, 0, 255)
    debug_floor_boundary_color: tuple[int, int, int] = (255, 0, 0)
    debug_facade_border_color: tuple[int, int, int] = (255, 255, 0)


@dataclass(frozen=True)
class BuildingGrid:
    width_m: float
    height_m: float
    floor_zone_height_m: float
    roof_offset_m: float
    ground_floor_height_m: float
    upper_floor_height_m: float
    upper_floor_count: int
    bay_count: int
    bay_width_m: float


@dataclass(frozen=True)
class FacadeGeometry:

    ground_window_width_m: float
    ground_window_bottom_m: float
    ground_window_top_m: float
    ground_door_width_m: float
    ground_door_bottom_m: float
    ground_door_top_m: float


    upper_window_width_m: float
    upper_window_bottom_m: float
    upper_window_top_m: float
    balcony_door_width_m: float
    balcony_door_bottom_m: float
    balcony_door_top_m: float


    sill_width_extra_m: float
    sill_height_m: float
    sill_gap_under_window_m: float

    window_cornice_width_extra_m: float
    window_cornice_height_m: float
    window_cornice_gap_above_window_m: float

    window_side_molding_width_m: float
    window_side_molding_gap_m: float

    floor_molding_height_m: float
    floor_molding_y_inset_m: float

    top_cornice_height_m: float
    top_cornice_y_inset_m: float


@dataclass(frozen=True)
class Element:
    class_name: ClassName
    x0_m: float
    y0_m: float
    x1_m: float
    y1_m: float

    def shifted(self, dx_m: float = 0.0, dy_m: float = 0.0) -> "Element":
        return Element(
            class_name=self.class_name,
            x0_m=self.x0_m + dx_m,
            y0_m=self.y0_m + dy_m,
            x1_m=self.x1_m + dx_m,
            y1_m=self.y1_m + dy_m,
        )


@dataclass(frozen=True)
class FloorConcept:
    elements: tuple[Element, ...]
    top_decoration: TopDecoration


@dataclass(frozen=True)
class UpperFloorStyle:
    add_window_cornices: bool
    add_side_moldings: bool
    top_decoration: TopDecoration


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def choose_upper_concept_count(rng: random.Random, cfg: GeneratorConfig) -> int:
    return rng.choices(
        [1, 2, 3],
        weights=[
            cfg.one_upper_concept_probability,
            cfg.two_upper_concepts_probability,
            cfg.three_upper_concepts_probability,
        ],
        k=1,
    )[0]


def determine_building_grid(
    width_m: float,
    height_m: float,
    cfg: GeneratorConfig,
    include_ground_floor: bool = True,
) -> BuildingGrid:
    if width_m <= 0:
        raise ValueError("width_m must be positive")
    if height_m <= 0:
        raise ValueError("height_m must be positive")
    if cfg.roof_offset_m < 0:
        raise ValueError("roof_offset_m must be non-negative")
    if cfg.roof_offset_m >= height_m:
        raise ValueError("roof_offset_m must be smaller than height_m")


    roof_offset_m = cfg.roof_offset_m
    floor_zone_height_m = height_m - roof_offset_m

    if include_ground_floor:
        ground_floor_height_m = min(
            cfg.target_ground_floor_height_m,
            floor_zone_height_m * cfg.max_ground_floor_fraction,
        )
        remaining_height_m = max(0.0, floor_zone_height_m - ground_floor_height_m)
    else:
        ground_floor_height_m = 0.0
        remaining_height_m = floor_zone_height_m

    estimated_upper_floors = round(remaining_height_m / cfg.target_upper_floor_height_m)
    upper_floor_count = max(cfg.min_upper_floors, estimated_upper_floors)
    upper_floor_height_m = remaining_height_m / upper_floor_count

    estimated_bays = round(width_m / cfg.target_bay_width_m)
    bay_count = int(clamp(estimated_bays, cfg.min_bays, cfg.max_bays))
    bay_width_m = width_m / bay_count

    return BuildingGrid(
        width_m=width_m,
        height_m=height_m,
        floor_zone_height_m=floor_zone_height_m,
        roof_offset_m=roof_offset_m,
        ground_floor_height_m=ground_floor_height_m,
        upper_floor_height_m=upper_floor_height_m,
        upper_floor_count=upper_floor_count,
        bay_count=bay_count,
        bay_width_m=bay_width_m,
    )


def sample_facade_geometry(grid: BuildingGrid, rng: random.Random, cfg: GeneratorConfig) -> FacadeGeometry:
    """Sample all spatial sizes once for the entire facade."""
    bay_w = grid.bay_width_m

    ground_window_width = min(bay_w * cfg.ground_window_width_bay_fraction, bay_w * 0.92)
    ground_window_height = rng.uniform(cfg.ground_window_height_min_m, cfg.ground_window_height_max_m)
    ground_window_bottom = cfg.ground_window_bottom_m
    ground_window_top = min(
        ground_window_bottom + ground_window_height,
        max(grid.ground_floor_height_m, 0.0) - cfg.floor_molding_height_m * 2.0
        if grid.ground_floor_height_m > 0
        else ground_window_bottom + ground_window_height,
    )

    ground_door_width = clamp(
        bay_w * cfg.ground_door_width_bay_fraction,
        cfg.ground_door_width_min_m,
        min(cfg.ground_door_width_max_m, bay_w * 0.90),
    )
    ground_door_height = rng.uniform(cfg.ground_door_height_min_m, cfg.ground_door_height_max_m)
    ground_door_bottom = cfg.ground_door_bottom_m
    ground_door_top = min(
        ground_door_bottom + ground_door_height,
        max(grid.ground_floor_height_m, 0.0) - cfg.floor_molding_height_m
        if grid.ground_floor_height_m > 0
        else ground_door_bottom + ground_door_height,
    )

    upper_window_width = clamp(
        bay_w * cfg.upper_window_width_bay_fraction,
        cfg.upper_window_width_min_m,
        min(cfg.upper_window_width_max_m, bay_w * 0.86),
    )
    upper_window_height = rng.uniform(cfg.upper_window_height_min_m, cfg.upper_window_height_max_m)
    upper_window_bottom = rng.uniform(cfg.upper_window_bottom_min_m, cfg.upper_window_bottom_max_m)
    upper_window_top = min(
        upper_window_bottom + upper_window_height,
        grid.upper_floor_height_m - cfg.floor_molding_height_m,
    )

    balcony_door_width = clamp(
        bay_w * cfg.balcony_door_width_bay_fraction,
        cfg.balcony_door_width_min_m,
        min(cfg.balcony_door_width_max_m, bay_w * 0.88),
    )
    balcony_door_bottom = cfg.balcony_door_bottom_m
    balcony_door_top = upper_window_top

    return FacadeGeometry(
        ground_window_width_m=ground_window_width,
        ground_window_bottom_m=ground_window_bottom,
        ground_window_top_m=ground_window_top,
        ground_door_width_m=ground_door_width,
        ground_door_bottom_m=ground_door_bottom,
        ground_door_top_m=ground_door_top,
        upper_window_width_m=upper_window_width,
        upper_window_bottom_m=upper_window_bottom,
        upper_window_top_m=upper_window_top,
        balcony_door_width_m=balcony_door_width,
        balcony_door_bottom_m=balcony_door_bottom,
        balcony_door_top_m=balcony_door_top,
        sill_width_extra_m=cfg.sill_width_extra_m,
        sill_height_m=cfg.sill_height_m,
        sill_gap_under_window_m=cfg.sill_gap_under_window_m,
        window_cornice_width_extra_m=cfg.window_cornice_width_extra_m,
        window_cornice_height_m=cfg.window_cornice_height_m,
        window_cornice_gap_above_window_m=cfg.window_cornice_gap_above_window_m,
        window_side_molding_width_m=cfg.window_side_molding_width_m,
        window_side_molding_gap_m=cfg.window_side_molding_gap_m,
        floor_molding_height_m=cfg.floor_molding_height_m,
        floor_molding_y_inset_m=cfg.floor_molding_y_inset_m,
        top_cornice_height_m=cfg.top_cornice_height_m,
        top_cornice_y_inset_m=cfg.top_cornice_y_inset_m,
    )


def bay_bounds(bay_index: int, bay_width_m: float) -> tuple[float, float]:
    x0_m = bay_index * bay_width_m
    x1_m = x0_m + bay_width_m
    return x0_m, x1_m


def rect_around(
    element: Element,
    class_name: ClassName,
    extra_x_m: float,
    y0_m: float,
    height_m: float,
) -> Element:
    return Element(
        class_name=class_name,
        x0_m=element.x0_m - extra_x_m,
        y0_m=y0_m,
        x1_m=element.x1_m + extra_x_m,
        y1_m=y0_m + height_m,
    )


def intersect_elements(a: Element, b: Element) -> Element | None:
    x0 = max(a.x0_m, b.x0_m)
    y0 = max(a.y0_m, b.y0_m)
    x1 = min(a.x1_m, b.x1_m)
    y1 = min(a.y1_m, b.y1_m)

    if x1 <= x0 or y1 <= y0:
        return None

    return Element("balcony", x0, y0, x1, y1)


def sample_balcony_width_bays(grid: BuildingGrid, rng: random.Random, cfg: GeneratorConfig) -> int:
    candidates: list[int] = []
    weights: list[float] = []

    if cfg.balcony_min_width_bays <= 2 <= cfg.balcony_max_width_bays and grid.bay_count >= 2:
        candidates.append(2)
        weights.append(cfg.balcony_width_2b_probability)

    if cfg.balcony_min_width_bays <= 3 <= cfg.balcony_max_width_bays and grid.bay_count >= 3:
        candidates.append(3)
        weights.append(cfg.balcony_width_3b_probability)

    if cfg.balcony_min_width_bays <= 4 <= cfg.balcony_max_width_bays and grid.bay_count >= 4:
        candidates.append(4)
        weights.append(cfg.balcony_width_4b_probability)

    if not candidates:
        return rng.randint(
            cfg.balcony_min_width_bays,
            min(cfg.balcony_max_width_bays, grid.bay_count),
        )

    return rng.choices(candidates, weights=weights, k=1)[0]


def sample_upper_floor_style(rng: random.Random, cfg: GeneratorConfig) -> UpperFloorStyle:
    top_decoration: TopDecoration = "none"
    if rng.random() < cfg.floor_top_molding_probability:
        top_decoration = "molding"

    return UpperFloorStyle(
        add_window_cornices=rng.random() < cfg.window_cornice_probability,
        add_side_moldings=rng.random() < cfg.window_side_molding_probability,
        top_decoration=top_decoration,
    )


def choose_ground_top_decoration(rng: random.Random, cfg: GeneratorConfig) -> TopDecoration:
    none_weight = max(0.0, 1.0 - cfg.ground_top_molding_probability - cfg.ground_top_cornice_probability)
    return rng.choices(
        ["molding", "cornice", "none"],
        weights=[cfg.ground_top_molding_probability, cfg.ground_top_cornice_probability, none_weight],
        k=1,
    )[0]


def generate_roof_zone_elements(
    grid: BuildingGrid,
    rng: random.Random,
    cfg: GeneratorConfig,
    geom: FacadeGeometry,
) -> tuple[Element, ...]:
    """Generate elements located between the highest floor and the roof line."""
    if grid.roof_offset_m <= 0.0:
        return ()
    if rng.random() >= cfg.top_cornice_probability:
        return ()

    roof_zone_bottom_m = grid.floor_zone_height_m
    roof_zone_top_m = grid.height_m
    y1 = roof_zone_top_m - geom.top_cornice_y_inset_m
    y0 = y1 - geom.top_cornice_height_m


    y0 = max(roof_zone_bottom_m, y0)
    y1 = min(roof_zone_top_m, y1)
    if y1 <= y0:
        return ()

    return (Element("cornice", 0.0, y0, grid.width_m, y1),)


def generate_ground_floor(
    grid: BuildingGrid,
    rng: random.Random,
    cfg: GeneratorConfig,
    geom: FacadeGeometry,
) -> FloorConcept:
    elements: list[Element] = []
    floor_h = grid.ground_floor_height_m
    bay_w = grid.bay_width_m

    max_doors = min(cfg.ground_max_doors, grid.bay_count)
    min_doors = min(cfg.ground_min_doors, max_doors)
    random_door_count = sum(rng.random() < cfg.ground_entrance_probability for _ in range(grid.bay_count))
    door_count = int(clamp(random_door_count, min_doors, max_doors))

    central_bays = sorted(
        range(grid.bay_count),
        key=lambda i: abs((i + 0.5) - grid.bay_count / 2.0),
    )
    candidate_doors = central_bays[: max(door_count + 2, door_count)]
    door_bays = set(rng.sample(candidate_doors, k=door_count))

    for bay_idx in range(grid.bay_count):
        bx0, bx1 = bay_bounds(bay_idx, bay_w)
        cx = (bx0 + bx1) / 2.0

        if bay_idx in door_bays:
            elements.append(
                Element(
                    "door",
                    cx - geom.ground_door_width_m / 2.0,
                    geom.ground_door_bottom_m,
                    cx + geom.ground_door_width_m / 2.0,
                    geom.ground_door_top_m,
                )
            )
            continue

        if rng.random() < cfg.ground_empty_bay_probability:
            continue

        elements.append(
            Element(
                "window",
                cx - geom.ground_window_width_m / 2.0,
                geom.ground_window_bottom_m,
                cx + geom.ground_window_width_m / 2.0,
                geom.ground_window_top_m,
            )
        )

    top_decoration = choose_ground_top_decoration(rng, cfg)

    if top_decoration == "molding":
        elements.append(
            Element(
                "molding",
                0.0,
                floor_h - geom.floor_molding_y_inset_m - geom.floor_molding_height_m,
                grid.width_m,
                floor_h - geom.floor_molding_y_inset_m,
            )
        )
    elif top_decoration == "cornice":
        cornice_y0 = max(0.0, floor_h - geom.top_cornice_y_inset_m - geom.top_cornice_height_m)
        elements.append(
            Element(
                "cornice",
                0.0,
                cornice_y0,
                grid.width_m,
                cornice_y0 + geom.top_cornice_height_m,
            )
        )

    return FloorConcept(tuple(elements), top_decoration=top_decoration)


def generate_balcony_spans(grid: BuildingGrid, rng: random.Random, cfg: GeneratorConfig) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    occupied = [False] * grid.bay_count

    if grid.bay_count < cfg.balcony_min_width_bays:
        return spans
    if rng.random() >= cfg.balcony_probability:
        return spans


    max_count_by_spacing = max(1, (grid.bay_count + 1) // (cfg.balcony_min_width_bays + 1))
    max_count = min(cfg.balcony_max_count_per_floor, max_count_by_spacing)

    desired_count = max(1, math.ceil(grid.bay_count / cfg.balcony_target_bays_per_balcony))
    min_count = max(1, desired_count - cfg.balcony_count_jitter)
    max_sampled_count = min(max_count, desired_count + cfg.balcony_count_jitter)

    if min_count > max_sampled_count:
        min_count = max_sampled_count

    target_count = rng.randint(min_count, max_sampled_count)

    for _ in range(target_count):
        span_width = sample_balcony_width_bays(grid, rng, cfg)
        starts = list(range(0, grid.bay_count - span_width + 1))
        rng.shuffle(starts)

        chosen_start: int | None = None
        for start in starts:
            end = start + span_width

            check_left = max(0, start - 1)
            check_right = min(grid.bay_count, end + 1)
            if any(occupied[check_left:check_right]):
                continue

            chosen_start = start
            break

        if chosen_start is None:
            continue

        chosen_end = chosen_start + span_width
        for bay_idx in range(chosen_start, chosen_end):
            occupied[bay_idx] = True
        spans.append((chosen_start, chosen_end))

    return sorted(spans)


def generate_upper_floor_concept(
    grid: BuildingGrid,
    rng: random.Random,
    cfg: GeneratorConfig,
    geom: FacadeGeometry,
    style: UpperFloorStyle,
) -> FloorConcept:
    elements: list[Element] = []
    floor_h = grid.upper_floor_height_m
    bay_w = grid.bay_width_m

    if style.top_decoration == "molding":
        elements.append(
            Element(
                "molding",
                0.0,
                max(0.0, floor_h - geom.floor_molding_y_inset_m - geom.floor_molding_height_m),
                grid.width_m,
                floor_h - geom.floor_molding_y_inset_m,
            )
        )
    elif style.top_decoration == "cornice":
        elements.append(
            Element(
                "cornice",
                0.0,
                max(0.0, floor_h - geom.top_cornice_y_inset_m - geom.top_cornice_height_m),
                grid.width_m,
                floor_h - geom.top_cornice_y_inset_m,
            )
        )

    balcony_spans = generate_balcony_spans(grid, rng, cfg)

    def span_for_bay(bay_idx: int) -> tuple[int, int] | None:
        for span in balcony_spans:
            if span[0] <= bay_idx < span[1]:
                return span
        return None

    balcony_door_bay_by_span: dict[tuple[int, int], int | None] = {}
    for span in balcony_spans:
        if rng.random() < cfg.balcony_has_door_probability:
            balcony_door_bay_by_span[span] = rng.randint(span[0], span[1] - 1)
        else:
            balcony_door_bay_by_span[span] = None

    empty_bays: set[int] = set()
    for bay_idx in range(grid.bay_count):
        if span_for_bay(bay_idx) is not None:
            continue
        if rng.random() < cfg.upper_empty_bay_probability:
            empty_bays.add(bay_idx)

    for bay_idx in range(grid.bay_count):
        bx0, bx1 = bay_bounds(bay_idx, bay_w)
        cx = (bx0 + bx1) / 2.0

        current_span = span_for_bay(bay_idx)
        balcony_door_bay = None if current_span is None else balcony_door_bay_by_span[current_span]

        if current_span is not None and balcony_door_bay == bay_idx:
            elements.append(
                Element(
                    "door",
                    cx - geom.balcony_door_width_m / 2.0,
                    geom.balcony_door_bottom_m,
                    cx + geom.balcony_door_width_m / 2.0,
                    geom.balcony_door_top_m,
                )
            )
            continue

        if bay_idx in empty_bays:
            continue

        window = Element(
            "window",
            cx - geom.upper_window_width_m / 2.0,
            geom.upper_window_bottom_m,
            cx + geom.upper_window_width_m / 2.0,
            geom.upper_window_top_m,
        )

        sill_y1 = window.y0_m - geom.sill_gap_under_window_m
        sill_y0 = sill_y1 - geom.sill_height_m
        if sill_y0 > 0.0:
            elements.append(
                rect_around(
                    window,
                    "sill",
                    geom.sill_width_extra_m,
                    sill_y0,
                    geom.sill_height_m,
                )
            )

        if current_span is None:
            if style.add_side_moldings:
                left_x0 = window.x0_m - geom.window_side_molding_gap_m - geom.window_side_molding_width_m
                left_x1 = window.x0_m - geom.window_side_molding_gap_m
                right_x0 = window.x1_m + geom.window_side_molding_gap_m
                right_x1 = window.x1_m + geom.window_side_molding_gap_m + geom.window_side_molding_width_m

                elements.append(Element("molding", left_x0, window.y0_m, left_x1, window.y1_m))
                elements.append(Element("molding", right_x0, window.y0_m, right_x1, window.y1_m))

            if style.add_window_cornices:
                cornice_y0 = window.y1_m + geom.window_cornice_gap_above_window_m
                if cornice_y0 + geom.window_cornice_height_m < floor_h:
                    elements.append(
                        rect_around(
                            window,
                            "cornice",
                            geom.window_cornice_width_extra_m,
                            cornice_y0,
                            geom.window_cornice_height_m,
                        )
                    )

        elements.append(window)

    for start_bay, end_bay in balcony_spans:
        x0 = start_bay * bay_w + cfg.balcony_side_inset_m
        x1 = end_bay * bay_w - cfg.balcony_side_inset_m
        y0 = cfg.balcony_bottom_offset_m
        y1 = y0 + cfg.balcony_height_m
        elements.append(Element("balcony", x0, y0, x1, y1))

    return FloorConcept(tuple(elements), top_decoration=style.top_decoration)


def generate_facade_structure(
    width_m: float,
    height_m: float,
    seed: int | None = None,
    cfg: GeneratorConfig | None = None,
    include_ground_floor: bool = True,
) -> tuple[BuildingGrid, FloorConcept, list[FloorConcept], tuple[Element, ...]]:
    cfg = cfg or GeneratorConfig()
    rng = random.Random(seed)

    grid = determine_building_grid(
        width_m=width_m,
        height_m=height_m,
        cfg=cfg,
        include_ground_floor=include_ground_floor,
    )
    geom = sample_facade_geometry(grid, rng, cfg)

    if include_ground_floor:
        ground_floor = generate_ground_floor(grid, rng, cfg, geom)
    else:
        ground_floor = FloorConcept(elements=(), top_decoration="none")

    concept_count = choose_upper_concept_count(rng, cfg)

    upper_concepts: list[FloorConcept] = []
    for _ in range(concept_count):
        style = sample_upper_floor_style(rng, cfg)
        upper_concepts.append(generate_upper_floor_concept(grid, rng, cfg, geom, style))

    upper_floors: list[FloorConcept] = []
    for upper_floor_idx in range(grid.upper_floor_count):
        concept = upper_concepts[upper_floor_idx % concept_count]


        upper_floors.append(concept)

    roof_elements = generate_roof_zone_elements(grid, rng, cfg, geom)

    return grid, ground_floor, upper_floors, roof_elements


def generate_facade_elements(
    width_m: float,
    height_m: float,
    seed: int | None = None,
    cfg: GeneratorConfig | None = None,
    include_ground_floor: bool = True,
) -> tuple[BuildingGrid, list[Element]]:
    grid, ground_floor, upper_floors, roof_elements = generate_facade_structure(
        width_m=width_m,
        height_m=height_m,
        seed=seed,
        cfg=cfg,
        include_ground_floor=include_ground_floor,
    )

    elements: list[Element] = []
    elements.extend(ground_floor.elements)

    for upper_floor_idx, floor in enumerate(upper_floors):
        y_offset = grid.ground_floor_height_m + upper_floor_idx * grid.upper_floor_height_m
        elements.extend(element.shifted(dy_m=y_offset) for element in floor.elements)

    elements.extend(roof_elements)

    return grid, elements


def element_to_floor_pixel_box(
    element: Element,
    image_width_px: int,
    floor_height_px: int,
    pixels_per_meter: float,
    cfg: GeneratorConfig,
) -> tuple[int, int, int, int] | None:
    x0 = int(round(element.x0_m * pixels_per_meter))
    x1 = int(round(element.x1_m * pixels_per_meter))
    y0 = floor_height_px - int(round(element.y1_m * pixels_per_meter))
    y1 = floor_height_px - int(round(element.y0_m * pixels_per_meter))

    x0 = max(0, min(image_width_px, x0))
    x1 = max(0, min(image_width_px, x1))
    y0 = max(0, min(floor_height_px, y0))
    y1 = max(0, min(floor_height_px, y1))

    if x1 - x0 < cfg.min_pixel_rect_size or y1 - y0 < cfg.min_pixel_rect_size:
        return None

    return x0, y0, x1 - 1, y1 - 1


def element_to_global_pixel_box(
    element: Element,
    image_width_px: int,
    image_height_px: int,
    pixels_per_meter: float,
    cfg: GeneratorConfig,
) -> tuple[int, int, int, int] | None:
    x0 = int(round(element.x0_m * pixels_per_meter))
    x1 = int(round(element.x1_m * pixels_per_meter))
    y0 = image_height_px - int(round(element.y1_m * pixels_per_meter))
    y1 = image_height_px - int(round(element.y0_m * pixels_per_meter))

    x0 = max(0, min(image_width_px, x0))
    x1 = max(0, min(image_width_px, x1))
    y0 = max(0, min(image_height_px, y0))
    y1 = max(0, min(image_height_px, y1))

    if x1 - x0 < cfg.min_pixel_rect_size or y1 - y0 < cfg.min_pixel_rect_size:
        return None

    return x0, y0, x1 - 1, y1 - 1


def render_global_elements(
    image: Image.Image,
    elements: Iterable[Element],
    pixels_per_meter: float,
    cfg: GeneratorConfig,
) -> None:
    draw = ImageDraw.Draw(image)
    width_px, height_px = image.size

    for element in elements:
        pixel_box = element_to_global_pixel_box(element, width_px, height_px, pixels_per_meter, cfg)
        if pixel_box is None:
            continue
        draw.rectangle(pixel_box, fill=COLORS[element.class_name])


def render_floor_image(
    floor_elements: Iterable[Element],
    width_px: int,
    floor_height_px: int,
    pixels_per_meter: float,
    cfg: GeneratorConfig,
    repaint_balcony_overlaps: bool = True,
) -> Image.Image:
    floor_elements = list(floor_elements)

    floor_image = Image.new("RGBA", (width_px, floor_height_px), (0, 0, 0, 0))
    draw = ImageDraw.Draw(floor_image)

    non_balconies = [element for element in floor_elements if element.class_name != "balcony"]
    balconies = [element for element in floor_elements if element.class_name == "balcony"]

    for element in non_balconies:
        pixel_box = element_to_floor_pixel_box(
            element,
            width_px,
            floor_height_px,
            pixels_per_meter,
            cfg,
        )
        if pixel_box is None:
            continue
        draw.rectangle(pixel_box, fill=COLORS[element.class_name] + (255,))

    for balcony in balconies:
        balcony_box = element_to_floor_pixel_box(
            balcony,
            width_px,
            floor_height_px,
            pixels_per_meter,
            cfg,
        )
        if balcony_box is None:
            continue

        draw.rectangle(balcony_box, fill=COLORS["balcony"] + (255,))

        if not repaint_balcony_overlaps:
            continue

        for other in non_balconies:
            if other.class_name not in {"window", "door"}:
                continue

            overlap = intersect_elements(balcony, other)
            if overlap is None:
                continue

            overlap_box = element_to_floor_pixel_box(
                overlap,
                width_px,
                floor_height_px,
                pixels_per_meter,
                cfg,
            )
            if overlap_box is None:
                continue

            color = BALCONY_OVERLAP_WINDOW_COLOR if other.class_name == "window" else BALCONY_OVERLAP_DOOR_COLOR
            draw.rectangle(overlap_box, fill=color + (255,))

    return floor_image


def draw_debug_overlay(
    image: Image.Image,
    grid: BuildingGrid,
    ground_h_px: int,
    upper_h_px: int,
    roof_h_px: int,
    cfg: GeneratorConfig,
) -> None:
    """Draw bay, floor and facade boundaries on top of the segmentation map.

    This is intended only for visual debugging. The debug colors are not part of
    the facade-element semantic label palette.
    """
    width_px, height_px = image.size
    draw = ImageDraw.Draw(image)
    line_width = max(1, cfg.debug_line_width_px)


    for bay_idx in range(1, grid.bay_count):
        x = int(round(bay_idx * width_px / grid.bay_count))
        x = max(0, min(width_px - 1, x))
        draw.line(
            [(x, 0), (x, height_px - 1)],
            fill=cfg.debug_bay_boundary_color,
            width=line_width,
        )


    if grid.ground_floor_height_m > 0.0:
        ground_top_y = height_px - ground_h_px
        if 0 <= ground_top_y < height_px:
            draw.line(
                [(0, ground_top_y), (width_px - 1, ground_top_y)],
                fill=cfg.debug_floor_boundary_color,
                width=line_width,
            )

        for upper_boundary_idx in range(1, grid.upper_floor_count):
            y = height_px - ground_h_px - upper_boundary_idx * upper_h_px
            if 0 <= y < height_px:
                draw.line(
                    [(0, y), (width_px - 1, y)],
                    fill=cfg.debug_floor_boundary_color,
                    width=line_width,
                )
    else:
        for upper_boundary_idx in range(1, grid.upper_floor_count):
            y = height_px - upper_boundary_idx * upper_h_px
            if 0 <= y < height_px:
                draw.line(
                    [(0, y), (width_px - 1, y)],
                    fill=cfg.debug_floor_boundary_color,
                    width=line_width,
                )


    if roof_h_px > 0:
        roof_boundary_y = roof_h_px
        if 0 <= roof_boundary_y < height_px:
            draw.line(
                [(0, roof_boundary_y), (width_px - 1, roof_boundary_y)],
                fill=cfg.debug_floor_boundary_color,
                width=line_width,
            )


    for offset in range(line_width):
        draw.rectangle(
            [offset, offset, width_px - 1 - offset, height_px - 1 - offset],
            outline=cfg.debug_facade_border_color,
        )


def render_facade_segmentation(
    width_m: float,
    height_m: float,
    pixels_per_meter: float,
    seed: int | None = None,
    cfg: GeneratorConfig | None = None,
    include_ground_floor: bool = True,
    repaint_balcony_overlaps: bool = True,
    debug: bool = False,
) -> Image.Image:
    if pixels_per_meter <= 0:
        raise ValueError("pixels_per_meter must be positive")

    cfg = cfg or GeneratorConfig()
    image_width_px = int(round(width_m * pixels_per_meter))
    image_height_px = int(round(height_m * pixels_per_meter))

    grid, ground_floor, upper_floors, roof_elements = generate_facade_structure(
        width_m=width_m,
        height_m=height_m,
        seed=seed,
        cfg=cfg,
        include_ground_floor=include_ground_floor,
    )

    image = Image.new("RGB", (image_width_px, image_height_px), COLORS["facade_wall"])

    roof_h_px = int(round(grid.roof_offset_m * pixels_per_meter))
    roof_h_px = max(0, min(image_height_px - 1, roof_h_px))
    floor_zone_h_px = image_height_px - roof_h_px

    if grid.ground_floor_height_m > 0.0:
        upper_h_px = int(round(grid.upper_floor_height_m * pixels_per_meter))
        ground_h_px = floor_zone_h_px - upper_h_px * grid.upper_floor_count

        if ground_h_px <= 0:
            ground_h_px = int(round(grid.ground_floor_height_m * pixels_per_meter))
            upper_h_px = max(1, (floor_zone_h_px - ground_h_px) // max(1, grid.upper_floor_count))

        ground_img = render_floor_image(
            floor_elements=ground_floor.elements,
            width_px=image_width_px,
            floor_height_px=ground_h_px,
            pixels_per_meter=pixels_per_meter,
            cfg=cfg,
            repaint_balcony_overlaps=repaint_balcony_overlaps,
        )
        image.paste(ground_img, (0, image_height_px - ground_h_px), ground_img)
    else:
        ground_h_px = 0
        upper_h_px = max(1, math.ceil(floor_zone_h_px / max(1, grid.upper_floor_count)))

    rendered_floor_cache: dict[int, Image.Image] = {}
    for upper_floor_idx, floor in enumerate(upper_floors):
        cache_key = id(floor)
        if cache_key not in rendered_floor_cache:
            rendered_floor_cache[cache_key] = render_floor_image(
                floor_elements=floor.elements,
                width_px=image_width_px,
                floor_height_px=upper_h_px,
                pixels_per_meter=pixels_per_meter,
                cfg=cfg,
                repaint_balcony_overlaps=repaint_balcony_overlaps,
            )

        floor_img = rendered_floor_cache[cache_key]
        paste_y = image_height_px - ground_h_px - (upper_floor_idx + 1) * upper_h_px

        if paste_y < 0:
            crop_top = -paste_y
            if crop_top >= floor_img.height:
                continue
            cropped = floor_img.crop((0, crop_top, floor_img.width, floor_img.height))
            image.paste(cropped, (0, 0), cropped)
        else:
            image.paste(floor_img, (0, paste_y), floor_img)

    render_global_elements(image, roof_elements, pixels_per_meter, cfg)

    if debug:
        draw_debug_overlay(
            image=image,
            grid=grid,
            ground_h_px=ground_h_px,
            upper_h_px=upper_h_px,
            roof_h_px=roof_h_px,
            cfg=cfg,
        )

    return image


def save_facade_segmentation(
    output_path: str | Path,
    width_m: float,
    height_m: float,
    pixels_per_meter: float,
    seed: int | None = None,
    cfg: GeneratorConfig | None = None,
    include_ground_floor: bool = True,
    repaint_balcony_overlaps: bool = True,
    debug: bool = False,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = render_facade_segmentation(
        width_m=width_m,
        height_m=height_m,
        pixels_per_meter=pixels_per_meter,
        seed=seed,
        cfg=cfg,
        include_ground_floor=include_ground_floor,
        repaint_balcony_overlaps=repaint_balcony_overlaps,
        debug=debug,
    )
    image.save(output)
    return output


def render_samples_grid(
    width_m: float,
    height_m: float,
    pixels_per_meter: float,
    sample_count: int,
    columns: int,
    seed: int | None = None,
    cfg: GeneratorConfig | None = None,
    include_ground_floor: bool = True,
    repaint_balcony_overlaps: bool = True,
    debug: bool = False,
) -> Image.Image:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if columns <= 0:
        raise ValueError("columns must be positive")

    cfg = cfg or GeneratorConfig()
    rng = random.Random(seed)
    sample_seeds = [rng.randrange(0, 2**31 - 1) for _ in range(sample_count)]

    samples = [
        render_facade_segmentation(
            width_m=width_m,
            height_m=height_m,
            pixels_per_meter=pixels_per_meter,
            seed=sample_seed,
            cfg=cfg,
            include_ground_floor=include_ground_floor,
            repaint_balcony_overlaps=repaint_balcony_overlaps,
            debug=debug,
        )
        for sample_seed in sample_seeds
    ]

    sample_w, sample_h = samples[0].size
    rows = math.ceil(sample_count / columns)
    border = cfg.grid_border_px

    grid_w = columns * sample_w + (columns + 1) * border
    grid_h = rows * sample_h + (rows + 1) * border

    canvas = Image.new("RGB", (grid_w, grid_h), cfg.grid_border_color)
    for idx, sample in enumerate(samples):
        row = idx // columns
        col = idx % columns
        x = border + col * (sample_w + border)
        y = border + row * (sample_h + border)
        canvas.paste(sample, (x, y))

    return canvas


def save_samples_grid(
    output_path: str | Path,
    width_m: float,
    height_m: float,
    pixels_per_meter: float,
    sample_count: int = 12,
    columns: int = 4,
    seed: int | None = None,
    cfg: GeneratorConfig | None = None,
    include_ground_floor: bool = True,
    repaint_balcony_overlaps: bool = True,
    debug: bool = False,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = render_samples_grid(
        width_m=width_m,
        height_m=height_m,
        pixels_per_meter=pixels_per_meter,
        sample_count=sample_count,
        columns=columns,
        seed=seed,
        cfg=cfg,
        include_ground_floor=include_ground_floor,
        repaint_balcony_overlaps=repaint_balcony_overlaps,
        debug=debug,
    )
    image.save(output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate rectangular facade-element segmentation maps.")

    parser.add_argument("--width-m", type=float, required=True, help="Facade width in meters.")
    parser.add_argument("--height-m", type=float, required=True, help="Facade height in meters.")
    parser.add_argument("--ppm", type=float, required=True, help="Pixels per meter.")
    parser.add_argument("--output", type=Path, default=Path("facade_segmentation.png"), help="Output PNG path.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")

    parser.add_argument(
        "--grid-output",
        type=Path,
        default=None,
        help="If specified, also save many samples as one grid PNG.",
    )
    parser.add_argument("--samples", type=int, default=12, help="Number of samples in the grid image.")
    parser.add_argument("--columns", type=int, default=4, help="Number of columns in the grid image.")

    parser.add_argument(
        "--no-ground-floor",
        action="store_true",
        help="Generate all floors as non-ground floors; do not create a special ground floor.",
    )
    parser.add_argument(
        "--no-repaint-balcony-overlaps",
        action="store_true",
        help="Keep old behavior: balconies simply overwrite windows/doors instead of repainting overlap zones.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Draw bay boundaries, floor boundaries and the facade border on top of the output image.",
    )
    parser.add_argument(
        "--roof-offset-m",
        type=float,
        default=None,
        help="Height in meters between the highest floor and the roof. Overrides GeneratorConfig.roof_offset_m.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = GeneratorConfig() if args.roof_offset_m is None else GeneratorConfig(roof_offset_m=args.roof_offset_m)

    save_facade_segmentation(
        output_path=args.output,
        width_m=args.width_m,
        height_m=args.height_m,
        pixels_per_meter=args.ppm,
        seed=args.seed,
        cfg=cfg,
        include_ground_floor=not args.no_ground_floor,
        repaint_balcony_overlaps=not args.no_repaint_balcony_overlaps,
        debug=args.debug,
    )
    print(f"Saved {args.output}")

    if args.grid_output is not None:
        save_samples_grid(
            output_path=args.grid_output,
            width_m=args.width_m,
            height_m=args.height_m,
            pixels_per_meter=args.ppm,
            sample_count=args.samples,
            columns=args.columns,
            seed=args.seed,
            cfg=cfg,
            include_ground_floor=not args.no_ground_floor,
            repaint_balcony_overlaps=not args.no_repaint_balcony_overlaps,
            debug=args.debug,
        )
        print(f"Saved {args.grid_output}")


if __name__ == "__main__":
    main()
