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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
template <typename floatp, typename vecp3>
__device__ vecp3 computeColorFromSH(int idx, int deg, int max_coeffs, const vecp3* means, vecp3 campos, const floatp* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	vecp3 pos = means[idx];
	vecp3 dir = pos - campos;
	dir = dir / glm::length(dir);

	vecp3* sh = ((vecp3*)shs) + idx * max_coeffs;
	vecp3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		floatp x = dir.x;
		floatp y = dir.y;
		floatp z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			floatp xx = x * x, yy = y * y, zz = z * z;
			floatp xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
template <typename floatp, typename floatp3>
__device__ floatp3 computeCov2D(const floatp3& mean, floatp focal_x, floatp focal_y, floatp tan_fovx, floatp tan_fovy, const floatp* cov3D, const floatp* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	floatp3 t = transformPoint4x3(mean, viewmatrix);

	const floatp limx = 1.3f * tan_fovx;
	const floatp limy = 1.3f * tan_fovy;
	const floatp txtz = t.x / t.z;
	const floatp tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::tmat3x3<floatp> J = glm::tmat3x3<floatp>(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::tmat3x3<floatp> W = glm::tmat3x3<floatp>(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::tmat3x3<floatp> T = W * J;

	glm::tmat3x3<floatp> Vrk = glm::tmat3x3<floatp>(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::tmat3x3<floatp> cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { floatp(cov[0][0]), floatp(cov[0][1]), floatp(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
template <typename floatp, typename vec3p, typename vec4p>
__device__ void computeCov3D(const vec3p scale, floatp mod, const vec4p rot, floatp* cov3D)
{
	// Create scaling matrix
	glm::tmat3x3<floatp> S = glm::tmat3x3<floatp>(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	vec4p q = rot;// / glm::length(rot);
	floatp r = q.x;
	floatp x = q.y;
	floatp y = q.z;
	floatp z = q.w;

	// Compute rotation matrix from quaternion
	glm::tmat3x3<floatp> R = glm::tmat3x3<floatp>(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::tmat3x3<floatp> M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::tmat3x3<floatp> Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template <int C, typename floatp, typename floatp2, typename floatp3, typename floatp4, typename vec3p, typename vec4p>
__global__ void preprocessCUDA(int P, int D, int M,
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
	const floatp tan_fovx, floatp tan_fovy,
	const floatp focal_x, floatp focal_y,
	int* radii,
	floatp2* points_xy_image,
	floatp* depths,
	floatp* cov3Ds,
	floatp* rgb,
	floatp4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	floatp3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	floatp3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	floatp4 p_hom = transformPoint4x4(p_orig, projmatrix);
	floatp p_w = 1.0f / (p_hom.w + 0.0000001f);
	floatp3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const floatp* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D<floatp, vec3p, vec4p>(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	floatp3 cov = computeCov2D<floatp, floatp3>(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	floatp det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	floatp det_inv = 1.f / det;
	floatp3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	floatp mid = 0.5f * (cov.x + cov.z);
	floatp lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	floatp lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	floatp my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	floatp2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		vec3p result = computeColorFromSH<floatp, vec3p>(idx, D, M, (vec3p*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one floatp4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS, typename floatp, typename floatp2, typename floatp4>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const floatp2* __restrict__ points_xy_image,
	const floatp* __restrict__ features,
	const floatp4* __restrict__ conic_opacity,
	floatp* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const floatp* __restrict__ bg_color,
	floatp* __restrict__ out_color)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	floatp2 pixf = { (floatp)pix.x, (floatp)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ floatp2 collected_xy[BLOCK_SIZE];
	__shared__ floatp4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	floatp T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	floatp C[CHANNELS] = { 0 };

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			floatp2 xy = collected_xy[j];
			floatp2 d = { xy.x - pixf.x, xy.y - pixf.y };
			floatp4 con_o = collected_conic_opacity[j];
			floatp power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			floatp alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			floatp test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

template <typename floatp, typename floatp2, typename floatp4>
void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const floatp2* means2D,
	const floatp* colors,
	const floatp4* conic_opacity,
	floatp* final_T,
	uint32_t* n_contrib,
	const floatp* bg_color,
	floatp* out_color)
{
	renderCUDA<NUM_CHANNELS, floatp, floatp2, floatp4> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color);
}

template <typename floatp, typename floatp2, typename floatp3, typename floatp4, typename vec3p, typename vec4p>
void FORWARD::preprocess(int P, int D, int M,
	const floatp* means3D,
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
	floatp2* means2D,
	floatp* depths,
	floatp* cov3Ds,
	floatp* rgb,
	floatp4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS, floatp, floatp2, floatp3, floatp4, vec3p, vec4p> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}

template void FORWARD::render<float, float2, float4>(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* features,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color
	);

template void FORWARD::preprocess<float, float2, float3, float4, glm::vec3, glm::vec4>(
		int P, int D, int M,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		bool* clamped,
		const float* cov3D_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float* colors,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered
	);

