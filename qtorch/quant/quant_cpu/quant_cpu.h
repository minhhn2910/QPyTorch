#pragma once

#include <torch/extension.h>
#include <torch/torch.h>

// Compatibility macros for PyTorch 2.4
#ifndef AT_ASSERT
#define AT_ASSERT TORCH_CHECK
#endif

// Ensure we have the right namespace
using namespace at;
using namespace torch;

// Function declarations
at::Tensor fixed_point_quantize_nearest(at::Tensor a, int wl, int fl, bool use_clamp, bool symmetric);
std::tuple<at::Tensor, at::Tensor> fixed_point_quantize_nearest_mask(at::Tensor a, int wl, int fl, bool symmetric);
at::Tensor block_quantize_nearest(at::Tensor a, int wl, int dim);
at::Tensor float_quantize_nearest(at::Tensor a, int man_bits, int exp_bits);
at::Tensor fixed_point_quantize_stochastic(at::Tensor a, int wl, int fl, bool use_clamp, bool symmetric);
std::tuple<at::Tensor, at::Tensor> fixed_point_quantize_stochastic_mask(at::Tensor a, int wl, int fl, bool symmetric);
at::Tensor block_quantize_stochastic(at::Tensor a, int wl, int dim);
at::Tensor float_quantize_stochastic(at::Tensor a, int man_bits, int exp_bits);
at::Tensor posit_quantize_nearest(at::Tensor a, int nsize, int es, float scale);
uint16_t convert_to_posit(at::Tensor a, int nsize, int es, float scale);
at::Tensor posit_sigmoid(at::Tensor a, int nsize, int es, float scale);
at::Tensor posit_tanh(at::Tensor a, int nsize, int es, float scale);
at::Tensor posit_tanh_enhanced(at::Tensor a, int nsize, int es, float scale);
at::Tensor new_format_quantize(at::Tensor a, float scale);
at::Tensor act_format_quantize(at::Tensor a, float scale);
at::Tensor configurable_table_quantize(at::Tensor a, at::Tensor lookup_table, float scale);
at::Tensor configurable_table_quantize_rounding_hint(at::Tensor a, at::Tensor lookup_table, at::Tensor rounding_hint, float scale);

#include <stdint.h>

unsigned int clip_exponent(int exp_bits, int man_bits, unsigned int old_num,
                           unsigned int quantized_num);

unsigned int clip_max_exponent(int man_bits,
                               unsigned int max_exponent,
                               unsigned int quantized_num);

template <typename T>
T clamp_helper(T a, T min, T max);

template <typename T>
T clamp_mask_helper(T a, T min, T max, uint8_t *mask);

void fixed_min_max(int wl, int fl, bool symmetric, float *t_min, float *t_max);

float round(float a, float r, int sigma);
