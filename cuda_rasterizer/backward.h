/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <ATen/ATen.h>

namespace BACKWARD
{
	template <typename floatp, typename floatp2, typename floatp3, typename floatp4>
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const floatp* bg_color,
		const floatp2* means2D,
		const floatp4* conic_opacity,
		const floatp* colors,
		const floatp* final_Ts,
		const uint32_t* n_contrib,
		const floatp* dL_dpixels,
		floatp3* dL_dmean2D,
		floatp4* dL_dconic2D,
		floatp* dL_dopacity,
		floatp* dL_dcolors);

	template <typename floatp, typename floatp3, typename floatp4, typename vec3p, typename vec4p>
	void preprocess(
		int P, int D, int M,
		const floatp3* means,
		const int* radii,
		const floatp* shs,
		const bool* clamped,
		const vec3p* scales,
		const vec4p* rotations,
		const floatp scale_modifier,
		const floatp* cov3Ds,
		const floatp* view,
		const floatp* proj,
		const floatp focal_x, floatp focal_y,
		const floatp tan_fovx, floatp tan_fovy,
		const vec3p* campos,
		const floatp3* dL_dmean2D,
		const floatp* dL_dconics,
		vec3p* dL_dmeans,
		floatp* dL_dcolor,
		floatp* dL_dcov3D,
		floatp* dL_dsh,
		vec3p* dL_dscale,
		vec4p* dL_drot);

	

}

#endif
