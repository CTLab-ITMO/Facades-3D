from typing import *
import torch
import numpy as np
from PIL import Image
from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline
from trellis.pipelines import samplers


class TrellisImageToFacadePipeline(TrellisImageTo3DPipeline):
    """
    Pipeline for generating facade 3D models with TRELLIS from a list of images.
    """

    @staticmethod
    def from_pretrained(path: str) -> "TrellisImageToFacadePipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(TrellisImageToFacadePipeline, TrellisImageToFacadePipeline).from_pretrained(path)
        new_pipeline = TrellisImageToFacadePipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_image_cond_model(args['image_cond_model'])

        return new_pipeline

    def sample_sparse_structure(
        self,
        cond: dict,
        semantic_map: Image.Image,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution * 4
        w, h = semantic_map.size
        max_size = max(w, h)
        new_w = int(w / max_size * reso + 0.5)
        new_h = int(h / max_size * reso + 0.5)
        semantic_map = semantic_map.resize((new_w, new_h), Image.NEAREST)

        ss = torch.zeros((reso, reso, reso), dtype=torch.uint8)

        start_i = (reso - new_h) // 2
        start_j = (reso - new_w) // 2

        def is_balcony(pix):
            return pix == (170, 255, 85) or pix == (170, 255, 170) or pix == (170, 255, 255)

        for pix_i, i in enumerate(range(start_i, start_i + new_h)):
            for pix_j, j in enumerate(range(start_j, start_j + new_w)):
                ss[i][j][31] = 1
                ss[i][j][32] = 1

                pix = semantic_map.getpixel((pix_j, pix_i))

                left_pix, right_pix = (0, 0, 0), (0, 0, 0)
                up_pix, down_pix = (0, 0, 0), (0, 0, 0)

                if pix_j - 1 >= 0:
                    left_pix = semantic_map.getpixel((pix_j - 1, pix_i))

                if pix_j + 1 < new_w:
                    right_pix = semantic_map.getpixel((pix_j + 1, pix_i))

                if pix_i - 1 >= 0:
                    up_pix = semantic_map.getpixel((pix_j, pix_i - 1))

                if pix_i + 1 < new_h:
                    down_pix = semantic_map.getpixel((pix_j, pix_i + 1))

                if pix == (0, 85, 255) or pix == (170, 255, 170):  # window
                    ss[i][j][32] = 0

                if pix ==  (0, 170, 255) or pix == (170, 255, 255):  # door
                    ss[i][j][32] = 0

                if pix == (85, 255, 170):  # sill
                    ss[i][j][33] = 1

                if pix == (0, 255, 255):  # cornice
                    ss[i][j][33] = 1

                if pix == (255, 85, 0):  # molding
                    ss[i][j][33] = 1

                if is_balcony(pix):
                    if not is_balcony(left_pix) or not is_balcony(right_pix) or not is_balcony(down_pix):
                        ss[i][j][33] = 1
                        ss[i][j][34] = 1
                        ss[i][j][35] = 1

                    ss[i][j][36] = 1

        to_del = torch.zeros_like(ss)
        for i in range(reso):
            for j in range(reso):
                for k in range(reso):
                    if ss[i][j][k] == 0:
                        continue

                    to_del[i][j][k] = 1
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            for dk in [-1, 0, 1]:
                                ni, nj, nk = i + di, j + dj, k + dk
                                if ni < 0 or reso <= ni or nj < 0 or reso <= nj or nk < 0 or reso <= nk:
                                    to_del[i][j][k] = 0
                                    continue

                                if ss[ni][nj][nk] == 0:
                                    to_del[i][j][k] = 0

        for i in range(reso):
            for j in range(reso):
                for k in range(reso):
                    if to_del[i][j][k] == 1:
                        ss[i][j][k] = 0

        ss = torch.flip(ss, [0])
        ss = torch.permute(ss, (1, 2, 0)).unsqueeze(0).unsqueeze(0)
        coords = torch.argwhere(ss>0)[:, [0, 2, 3, 4]].int().cuda()

        if self.coords_dump_name is not None:
            np.save(self.coords_dump_name, coords.cpu().numpy())

        return coords

    @torch.no_grad()
    def run_multi_image(
        self,
        images: List[Image.Image],
        semantic_map: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian'],
        preprocess_image: bool = True,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ) -> dict:
        if preprocess_image:
            images = [self.preprocess_image(image) for image in images]
        cond = self.get_cond(images)
        cond['neg_cond'] = cond['neg_cond'][:1]
        torch.manual_seed(seed)
        ss_steps = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('sparse_structure_sampler', len(images), ss_steps, mode=mode):
            coords = self.sample_sparse_structure(cond, semantic_map, num_samples, sparse_structure_sampler_params)
        slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('slat_sampler', len(images), slat_steps, mode=mode):
            slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)
