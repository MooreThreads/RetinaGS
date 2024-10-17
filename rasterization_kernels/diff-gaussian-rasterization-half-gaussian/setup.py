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

# this package is modified from https://github.com/ashawkey/diff-gaussian-rasterization.git

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))
glm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")

setup(
    name="diff_gaussian_rasterization_half_gaussian",
    version='0.0.2',
    description='add space boundary basing on diff_gaussian_rasterization_ashawkey',
    packages=['diff_gaussian_rasterization_half_gaussian'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization_half_gaussian._C",
            sources=[
            "cuda_rasterizer_hgs/rasterizer_impl.cu",
            "cuda_rasterizer_hgs/forward.cu",
            "cuda_rasterizer_hgs/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": ["-I" + glm_path]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
