# Facades-3D

This is an app that generates 3D models of residential buildings with detailed exteriors that fit the provided mass model. The below image shows transition from input mass model to generated result.

<img src="https://github.com/CTLab-ITMO/Facades-3D/releases/download/webp-files/itmo1-transition.webp" alt="itmo1-transition" width="75%">

This app also accepts text prompt and style reference image for generation. The above image was generated with [this](./example_data/style1.png) reference image. Example generated with [this](./example_data/style2.png) reference image is listed below.

<img src="https://github.com/CTLab-ITMO/Facades-3D/releases/download/webp-files/itmo2-transition.webp" alt="itmo2-transition" width="75%">

More visual results can be seen in [visuals.md](./visuals.md).

## Requirements

### Hardware

* 8-core CPU
* 28 Gb of RAM
* NVIDIA GPU with 16 Gb VRAM and compute capability `>=7.0` (at least V100)
* 64 Gb of disk space

### Software

* Ubuntu 24 (or 22)
* NVIDIA driver `>=525.60.13`
* Python (to run without docker you need exactly `3.10`)
* git lfs
* CUDA `==12.6` (to run without docker)
* Docker Engine / Docker Compose (for docker runs)

## Clone

```bash
# Clone this repo
git clone --recurse-submodules https://github.com/CTLab-ITMO/Facades-3D.git

# Download sources from TRELLIS repo
./install/install-trellis-sources.sh

# If `aux/buildingface.safetensors` wasn't downloaded automatically
git lfs pull
```

## Run

### Without Docker

Dependency installation to run without docker.

```bash
# Assuming python and CUDA already are installed,
# installs to your current python environment (venv or conda).
./install/install-local-dependencies.sh
```

#### Standalone CLI App

File `gen_facades.py` provides interface for standalone CLI app.

```
$ SKIP_MODEL_LOAD=1 python src/gen_facades.py -h
usage: gen_facades.py [-h]
                      --input-path INPUT_PATH
                      --output-filename OUTPUT_FILENAME
                      [--visual-path VISUAL_PATH]
                      [--seed SEED]
                      [--temp-root TEMP_ROOT]
                      --pixels-per-meter PIXELS_PER_METER
                      [--prompt PROMPT]
                      [--negative-prompt NEGATIVE_PROMPT]
                      [--diffusion-steps DIFFUSION_STEPS]
                      [--guidance-scale GUIDANCE_SCALE]
                      [--cross-attention-scale CROSS_ATTENTION_SCALE]
                      [--controlnet-conditioning-scale CONTROLNET_CONDITIONING_SCALE]
                      [--control-guidance-start CONTROL_GUIDANCE_START]
                      [--control-guidance-end CONTROL_GUIDANCE_END]
                      (--style-ref-path STYLE_REF_PATH | --no-style-ref)
                      [--style-ref-scale STYLE_REF_SCALE]
                      --cluster-count CLUSTER_COUNT
                      [--max-wall-aspect-ratio MAX_WALL_ASPECT_RATIO]
                      [--slat-steps SLAT_STEPS]
                      [--slat-cfg-strength SLAT_CFG_STRENGTH]
                      [--border-size BORDER_SIZE]
                      [--trellis-mode TRELLIS_MODE]
                      [--mesh-simplify MESH_SIMPLIFY]
                      [--texture-size TEXTURE_SIZE]
                      [--texture-brightness-factor TEXTURE_BRIGHTNESS_FACTOR]
                      [--texture-postprocess-shrink-px TEXTURE_POSTPROCESS_SHRINK_PX]
                      [--depth-scale-reference DEPTH_SCALE_REFERENCE]

Generate detailed facade meshes for an OBJ scene.

options:
  -h, --help            show this help message and exit
  --input-path INPUT_PATH
                        Input OBJ scene path. (default: None)
  --output-filename OUTPUT_FILENAME, --result-path OUTPUT_FILENAME
                        Output GLB scene path. --result-path is accepted as a legacy alias. (default: None)
  --visual-path VISUAL_PATH
                        Optional directory for intermediate visual outputs. If omitted, intermediate outputs are not written. (default: None)
  --seed SEED           Seed for generation. (default: None)
  --temp-root TEMP_ROOT
                        Directory for storing temporary files (wall meshes). (default: None)
  --pixels-per-meter PIXELS_PER_METER
                        Number of pixels per meter when generating facade image. Used to infer resolution of facade image. (default: None)
  --prompt PROMPT       Prompt for facade image generation. (default: modern style architecture, urban house)
  --negative-prompt NEGATIVE_PROMPT
                        Negative prompt for facade image generation. (default: old, dark, distorted)
  --diffusion-steps DIFFUSION_STEPS
                        Diffusion steps for facade image generation. (default: 30)
  --guidance-scale GUIDANCE_SCALE
                        Diffusion guidance scale for facade image generation. (default: 7.0)
  --cross-attention-scale CROSS_ATTENTION_SCALE
                        Cross-attention scale (LoRA) for facade image generation. (default: 0.7)
  --controlnet-conditioning-scale CONTROLNET_CONDITIONING_SCALE
                        ControlNet conditioning scale for facade image generation. (default: 1.1)
  --control-guidance-start CONTROL_GUIDANCE_START
                        ControlNet guidance start for facade image generation. (default: 0.0)
  --control-guidance-end CONTROL_GUIDANCE_END
                        ControlNet guidance end for facade image generation. (default: 1.0)
  --style-ref-path STYLE_REF_PATH
                        Style-reference image path. (default: None)
  --no-style-ref        Disable style-reference conditioning. (default: False)
  --style-ref-scale STYLE_REF_SCALE
                        IP-Adapter scale when style reference is enabled. (default: 0.5)
  --cluster-count CLUSTER_COUNT
                        Number of different wall meshes to generate. (default: None)
  --max-wall-aspect-ratio MAX_WALL_ASPECT_RATIO
                        Split vertical quad faces so every resulting wall is narrower than this width/height ratio. (default: 1.5)
  --slat-steps SLAT_STEPS
                        SLAT generation steps for TRELLIS. (default: 25)
  --slat-cfg-strength SLAT_CFG_STRENGTH
                        SLAT generation CFG strength for TRELLIS. (default: 3.0)
  --border-size BORDER_SIZE
                        Border size for input image for TRELLIS. (default: 16)
  --trellis-mode TRELLIS_MODE
                        Generation mode for TRELLIS. (default: multidiffusion)
  --mesh-simplify MESH_SIMPLIFY
                        Mesh postprocessing simplification (ratio of vertices to remove). (default: 0.95)
  --texture-size TEXTURE_SIZE
                        Mesh texture size. Must be a power of two. (default: 512)
  --texture-brightness-factor TEXTURE_BRIGHTNESS_FACTOR
                        After wall mesh is generated, its texture channel values are multiplied by this number. (default: 1.4)
  --texture-postprocess-shrink-px TEXTURE_POSTPROCESS_SHRINK_PX
                        Shrink each generated texture by this many pixels per dimension before atlas packing. (default: 4)
  --depth-scale-reference DEPTH_SCALE_REFERENCE
                        Used to control balconies' depth. (default: 32.0)
```

Usage example:

```bash
python gen_facades.py --input-path mass_model.obj --output-filename result.glb --visual-path ./visuals --pixels-per-meter 32 --style-ref-path style.png --style-ref-scale 0.6 --cluster-count 8 --mesh-simplify 0.95 --texture-size 512 --seed 123
```

All default parameters are stored in [`default_configs.py`](./src/default_configs.py) file.

#### Web Server

You can run a server that accepts generation requests via HTTP REST API or WebSocket messages. To run:

```bash
python src/server.py --host 0.0.0.0 --port 8000
```

All generation parameters are passed via HTTP request form or WebSocket message. Files (input mass model and style reference) are also included into request. For more details about API and server behavior refer [API.md](./API.md).

### With Docker

#### Docker Engine

You can run server inside [Docker container](./Dockerfile). Build it with

```bash
DOCKER_BUILDKIT=1 docker build --progress=plain --build-arg MAX_JOBS=8 -t facades-3d-generation-server .
```

Build process takes approximately 10 minutes because of heavy runtime environment and specific kernels compilation. It also mounts `pip` and `torch_extensions` cache to your host machine cache during build, so building process can affect your filesystem.

> [!IMPORTANT]
> The built container carries the environment built for specific GPU architecture. To run it on different GPUs you have to adjust `TORCH_CUDA_ARCH_LIST` and `CUDAARCHS` building arguments to specify the required GPU architecture. Default arguments specify build for NVIDIA V100. See [Dockerfile](./Dockerfile) for details.

The default attention backend is `xformers`, but you can enforce usage of `flash-attn` if your GPU allows it.

After the container is built, run it with

```bash
docker run --rm -it --gpus all --ipc=host -p 8000:8000 -v "$HOME/.cache/torch:/root/.cache/torch" -v "$HOME/.cache/huggingface:/root/.cache/huggingface" facades-3d-generation-server
```

This command mounts `torch` and `huggingface` caches to your host machine cache to avoid downloading checkpoints on every run.

`-p 8000:8000` forwards a port from your host machine to container. If you serve only internal requests, restrict port usage with `-p 127.0.0.1:8000:8000` (this restricts connections from outside of your host machine).

#### Docker Compose

You can also configure run specification with Docker Compose (see [docker-compose.yaml](./docker-compose.yaml)). To start

```bash
docker compose up
```

To stop

```bash
docker compose down
```

## File Format

Input mass model must be written in [`.obj`](https://paulbourke.net/dataformats/obj/) format. The resulting model is returned in [`.glb`](https://www.khronos.org/gltf/) format (binary `glTF`). This model contains one or several meshes according to object grouping in the input file.

To make each building a separate mesh in the resulting model, you have to assemble their faces into objects in the input file. Example of input file structure:

```
# Vertices and their positions
v 1 2 3
v ...
...

# Faces of the first building
o B_1  # Name of the first building
f 1 2 3 4
f ...
...

# Faces of the second building
o B_2  # Name of the second building
f 5 6 7 8
f ...
...

# Faces of the third building
o B_3  # Name of the third building
f 9 10 11 12
f ...
...
```

In this case the resulting model will contain three independent meshes with names `B_1`, `B_2` and `B_3`. Names can be arbitrary and they will be preserved in the output file. If buildings are not assembled into separate objects in the input file, they will be represented as one mesh in the resulting model. For more details see [`small_scene.obj`](./example_data/small_scene.obj) and other files from [`example_data`](./example_data/).

Vertex coordinates will also be preserved by 3D model generator. It means that position of each building in the resulting scene will be almost identical to position of its mass model in the input scene.

Number of objects (buildings) in one file and number of faces in one object (building) is not limited by implementation.

## API

Server accepts requests via HTTP REST API or WebSocket messages. For more details about API and server behavior refer [API.md](./API.md). You can also run API tests with [`api_test_client.py`](./test/api_test_client.py).

## Limitations

This work has got several limitations:
* heavy runtime that requires modern GPU, while generation of one wall mesh can still take several minutes, so the whole scene can take several hours to be generated;
* text prompt is used only for facade image generation; this is a limitation of 3D mesh generator model, since it does not accept text prompts at the moment;
* all 3D meshes are made from a template which is generated in a procedural manner and does not fit for provided input mass model, text prompt or style reference; fixing this limitation is a place for future research;

## Acknowledgement and Licenses

This project was done while working in CT Lab at ITMO University and is published under [MIT License](./LICENSE).

This project relies on
* Stable Diffusion v1.4 / [License](https://huggingface.co/spaces/CompVis/stable-diffusion-license);
* Building Facades LoRA checkpoint presented by [this paper](https://doi.org/10.1007/978-981-99-8405-3_3) downloaded from [here](https://civitai.com/models/11661/buildingfacade);
* TRELLIS &mdash; a model for 3D mesh generation / [License](https://github.com/microsoft/TRELLIS/blob/main/LICENSE);
* Submodules for TRELLIS:
  - diffoctreerast / [License](https://github.com/JeffreyXiang/diffoctreerast/blob/master/LICENSE);
  - FlexiCubes / [License](https://github.com/nv-tlabs/FlexiCubes/blob/main/LICENSE.txt);
  - mip-splatting / [License](https://github.com/autonomousvision/mip-splatting/blob/main/LICENSE.md);
  - nvdiffrast / [License](https://github.com/NVlabs/nvdiffrast/blob/main/LICENSE.txt);

<img src="https://github.com/CTLab-ITMO/Facades-3D/releases/download/webp-files/low-city.webp" alt="low-city" width="100%">
