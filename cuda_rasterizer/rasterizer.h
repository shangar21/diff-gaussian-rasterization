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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>
#include "types.h"

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			floatp* means3D,
			floatp* viewmatrix,
			floatp* projmatrix,
			bool* present);

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const floatp* background,
			const int width, int height,
			const floatp* means3D,
			const floatp* shs,
			const floatp* colors_precomp,
			const floatp* opacities,
			const floatp* scales,
			const floatp scale_modifier,
			const floatp* rotations,
			const floatp* cov3D_precomp,
			const floatp* viewmatrix,
			const floatp* projmatrix,
			const floatp* cam_pos,
			const floatp tan_fovx, floatp tan_fovy,
			const bool prefiltered,
			floatp* out_color,
			int* radii = nullptr,
			bool debug = false);

		static void backward(
			const int P, int D, int M, int R,
			const floatp* background,
			const int width, int height,
			const floatp* means3D,
			const floatp* shs,
			const floatp* colors_precomp,
			const floatp* scales,
			const floatp scale_modifier,
			const floatp* rotations,
			const floatp* cov3D_precomp,
			const floatp* viewmatrix,
			const floatp* projmatrix,
			const floatp* campos,
			const floatp tan_fovx, floatp tan_fovy,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const floatp* dL_dpix,
			floatp* dL_dmean2D,
			floatp* dL_dconic,
			floatp* dL_dopacity,
			floatp* dL_dcolor,
			floatp* dL_dmean3D,
			floatp* dL_dcov3D,
			floatp* dL_dsh,
			floatp* dL_dscale,
			floatp* dL_drot,
			bool debug);
	};
};

#endif
