#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))
glm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../third_party/glm/")

setup(
    name="diff_gaussian_rasterization_metric",
    version='0.0.1',
    description='export two metrics from diff_gaussian_rasterization_ashawkey',
    packages=['diff_gaussian_rasterization_metric'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization_metric._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": ["-I" + glm_path]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
