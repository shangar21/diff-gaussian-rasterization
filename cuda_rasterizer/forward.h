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

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <ATen/ATen.h>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	template <typename floatp, typename floatp2, typename floatp3, typename floatp4, typename vec3p, typename vec4p>
	void preprocess(int P, int D, int M,
		const floatp* orig_points,
		const vec3p* scales,
		const floatp scale_modifier,
		const vec4p* rotations,
		const floatp* opacities,
		const floatp* shs,
		bool* clamped,
		const floatp* cov3D_precomp,
		const floatp* colors_precomp,
		const floatp* viewmatrix,
		const floatp* projmatrix,
		const vec3p* cam_pos,
		const int W, int H,
		const floatp focal_x, floatp focal_y,
		const floatp tan_fovx, floatp tan_fovy,
		int* radii,
		floatp2* points_xy_image,
		floatp* depths,
		floatp* cov3Ds,
		floatp* colors,
		floatp4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered);

	// Main rasterization method.

	template <typename floatp, typename floatp2, typename floatp4>
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const floatp2* points_xy_image,
		const floatp* features,
		const floatp4* conic_opacity,
		floatp* final_T,
		uint32_t* n_contrib,
		const floatp* bg_color,
		floatp* out_color);

}

#endif
