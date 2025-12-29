/**
 * @file vae_gemm.h
 * @brief Ternary VAE GEMM Inference - Dense243 Weight Loading and Inference
 *
 * Provides C++ infrastructure for loading Dense243-packed VAE weights and
 * performing inference using ternary GEMM operations.
 *
 * Architecture:
 *   Encoder (A/B): 9 -> 256 -> 128 -> 64 -> 16 (mu/logvar)
 *   Decoder: 16 -> 32 -> 64 -> 27
 *
 * Memory: ~46KB total for all weights (Dense243 packed + float32 biases)
 *
 * @date 2025-12-29
 * @version 1.0
 */

#pragma once

#include <stdint.h>
#include <stddef.h>
#include <string>
#include <vector>
#include <array>
#include <memory>
#include <cmath>
#include <cstring>
#include <fstream>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration
struct VAEModel;

/**
 * @brief Load VAE model from binary files
 *
 * @param model_dir Directory containing vae_encoder_A.bin, etc.
 * @return Pointer to loaded model (NULL on failure)
 */
VAEModel* vae_load_model(const char* model_dir);

/**
 * @brief Free VAE model memory
 */
void vae_free_model(VAEModel* model);

/**
 * @brief Encode input to latent space (using encoder A)
 *
 * @param model    Loaded VAE model
 * @param input    Input vector [batch_size, 9]
 * @param mu       Output mu vector [batch_size, 16]
 * @param batch_size Number of samples
 */
void vae_encode_A(
    VAEModel* model,
    const float* input,
    float* mu,
    int batch_size
);

/**
 * @brief Encode input to latent space (using encoder B)
 */
void vae_encode_B(
    VAEModel* model,
    const float* input,
    float* mu,
    int batch_size
);

/**
 * @brief Decode latent to output
 *
 * @param model    Loaded VAE model
 * @param z        Latent vector [batch_size, 16]
 * @param output   Output logits [batch_size, 27]
 * @param batch_size Number of samples
 */
void vae_decode(
    VAEModel* model,
    const float* z,
    float* output,
    int batch_size
);

/**
 * @brief Full forward pass (encode A + decode)
 *
 * @param model    Loaded VAE model
 * @param input    Input [batch_size, 9]
 * @param output   Output logits [batch_size, 27]
 * @param batch_size Number of samples
 */
void vae_forward(
    VAEModel* model,
    const float* input,
    float* output,
    int batch_size
);

#ifdef __cplusplus
}
#endif

// =============================================================================
// C++ Implementation (header-only for simplicity)
// =============================================================================

#ifdef VAE_GEMM_IMPLEMENTATION

#include "tritnet_gemm.h"

// Layer configuration
struct LayerConfig {
    char name[32];
    int in_features;
    int out_features;
    int weight_offset;
    int weight_size;
    int bias_offset;
    int bias_size;
    bool has_activation;
};

// VAE Model structure
struct VAEModel {
    // Encoder A weights (Dense243 packed)
    std::vector<uint8_t> encoder_A_weights;
    std::vector<float> encoder_A_biases;
    std::vector<LayerConfig> encoder_A_layers;

    // Encoder B weights
    std::vector<uint8_t> encoder_B_weights;
    std::vector<float> encoder_B_biases;
    std::vector<LayerConfig> encoder_B_layers;

    // Decoder weights
    std::vector<uint8_t> decoder_weights;
    std::vector<float> decoder_biases;
    std::vector<LayerConfig> decoder_layers;

    // Intermediate buffers (preallocated for batch processing)
    std::vector<float> buffer1;  // Max size: batch * 256
    std::vector<float> buffer2;  // Max size: batch * 256

    // Model dimensions
    int input_dim = 9;
    int latent_dim = 16;
    int output_dim = 27;
    int max_batch_size = 64;
};

// Unpack Dense243 byte to 5 trits {-1, 0, +1}
static inline void unpack_dense243(uint8_t packed, int8_t* trits) {
    int val = packed;
    for (int i = 0; i < 5; ++i) {
        trits[i] = (val % 3) - 1;  // {0,1,2} -> {-1,0,+1}
        val /= 3;
    }
}

// Simple ternary GEMM (C = A @ B + bias, with optional ReLU)
// A: [M, K] float, B: [K_packed, N] Dense243, C: [M, N] float
static void ternary_linear(
    const float* A,        // [M, K]
    const uint8_t* B,      // [K/5, N] Dense243
    const float* bias,     // [N]
    float* C,              // [M, N]
    int M, int N, int K,
    bool apply_relu
) {
    const int K_packed = (K + 4) / 5;

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = bias ? bias[n] : 0.0f;

            for (int k_pack = 0; k_pack < K_packed; ++k_pack) {
                // Unpack 5 weight trits
                int8_t trits[5];
                unpack_dense243(B[n * K_packed + k_pack], trits);

                // Multiply-accumulate
                int k_base = k_pack * 5;
                for (int t = 0; t < 5 && (k_base + t) < K; ++t) {
                    sum += A[m * K + k_base + t] * trits[t];
                }
            }

            // Apply ReLU if needed
            if (apply_relu && sum < 0.0f) {
                sum = 0.0f;
            }

            C[m * N + n] = sum;
        }
    }
}

// Load binary file
static bool load_binary_file(const std::string& path, std::vector<uint8_t>& data) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return false;

    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    data.resize(size);
    file.read(reinterpret_cast<char*>(data.data()), size);
    return true;
}

// Parse encoder/decoder binary file
static bool parse_layer_file(
    const std::vector<uint8_t>& data,
    std::vector<uint8_t>& weights,
    std::vector<float>& biases,
    std::vector<LayerConfig>& layers
) {
    if (data.size() < 20) return false;

    const uint8_t* ptr = data.data();

    // Header: magic(4) + version(4) + num_layers(4) + weight_bytes(4) + bias_bytes(4)
    char magic[5] = {0};
    memcpy(magic, ptr, 4); ptr += 4;

    uint32_t version, num_layers, weight_bytes, bias_bytes;
    memcpy(&version, ptr, 4); ptr += 4;
    memcpy(&num_layers, ptr, 4); ptr += 4;
    memcpy(&weight_bytes, ptr, 4); ptr += 4;
    memcpy(&bias_bytes, ptr, 4); ptr += 4;

    // Parse layer configs
    layers.resize(num_layers);
    for (uint32_t i = 0; i < num_layers; ++i) {
        memcpy(layers[i].name, ptr, 32); ptr += 32;
        memcpy(&layers[i].in_features, ptr, 4); ptr += 4;
        memcpy(&layers[i].out_features, ptr, 4); ptr += 4;
        memcpy(&layers[i].weight_offset, ptr, 4); ptr += 4;
        memcpy(&layers[i].weight_size, ptr, 4); ptr += 4;
        memcpy(&layers[i].bias_offset, ptr, 4); ptr += 4;
        memcpy(&layers[i].bias_size, ptr, 4); ptr += 4;
        memcpy(&layers[i].has_activation, ptr, 1); ptr += 1;
    }

    // Copy weights
    weights.resize(weight_bytes);
    memcpy(weights.data(), ptr, weight_bytes);
    ptr += weight_bytes;

    // Copy biases (convert bytes to floats)
    size_t num_floats = bias_bytes / sizeof(float);
    biases.resize(num_floats);
    memcpy(biases.data(), ptr, bias_bytes);

    return true;
}

// Run encoder forward pass
static void encoder_forward(
    const float* input,    // [batch, 9]
    float* mu_out,         // [batch, 16]
    const std::vector<uint8_t>& weights,
    const std::vector<float>& biases,
    const std::vector<LayerConfig>& layers,
    float* buffer1,
    float* buffer2,
    int batch_size
) {
    const float* current_input = input;
    float* current_output = buffer1;
    int bias_float_offset = 0;

    for (size_t i = 0; i < layers.size(); ++i) {
        const LayerConfig& layer = layers[i];

        // Get pointers
        const uint8_t* W = weights.data() + layer.weight_offset;
        const float* b = biases.data() + (bias_float_offset / sizeof(float));
        bias_float_offset += layer.bias_size;

        // Determine output destination
        bool is_last = (i == layers.size() - 1);
        bool is_fc_mu = (strncmp(layer.name, "fc_mu", 5) == 0);

        if (is_fc_mu) {
            // fc_mu outputs to mu_out
            ternary_linear(
                current_input, W, b, mu_out,
                batch_size, layer.out_features, layer.in_features,
                layer.has_activation
            );
            // Skip fc_logvar for inference (we only need mu)
            break;
        } else {
            // Regular layer
            ternary_linear(
                current_input, W, b, current_output,
                batch_size, layer.out_features, layer.in_features,
                layer.has_activation
            );
        }

        // Swap buffers
        current_input = current_output;
        current_output = (current_output == buffer1) ? buffer2 : buffer1;
    }
}

// Run decoder forward pass
static void decoder_forward(
    const float* z,        // [batch, 16]
    float* output,         // [batch, 27]
    const std::vector<uint8_t>& weights,
    const std::vector<float>& biases,
    const std::vector<LayerConfig>& layers,
    float* buffer1,
    float* buffer2,
    int batch_size
) {
    const float* current_input = z;
    float* current_output = buffer1;
    int bias_float_offset = 0;

    for (size_t i = 0; i < layers.size(); ++i) {
        const LayerConfig& layer = layers[i];

        const uint8_t* W = weights.data() + layer.weight_offset;
        const float* b = biases.data() + (bias_float_offset / sizeof(float));
        bias_float_offset += layer.bias_size;

        bool is_last = (i == layers.size() - 1);
        float* dest = is_last ? output : current_output;

        ternary_linear(
            current_input, W, b, dest,
            batch_size, layer.out_features, layer.in_features,
            layer.has_activation
        );

        if (!is_last) {
            current_input = current_output;
            current_output = (current_output == buffer1) ? buffer2 : buffer1;
        }
    }
}

// =============================================================================
// Public API Implementation
// =============================================================================

VAEModel* vae_load_model(const char* model_dir) {
    std::string dir(model_dir);
    if (dir.back() != '/' && dir.back() != '\\') {
        dir += "/";
    }

    VAEModel* model = new VAEModel();

    // Load encoder A
    std::vector<uint8_t> enc_a_data;
    if (!load_binary_file(dir + "vae_encoder_A.bin", enc_a_data)) {
        delete model;
        return nullptr;
    }
    if (!parse_layer_file(enc_a_data, model->encoder_A_weights,
                          model->encoder_A_biases, model->encoder_A_layers)) {
        delete model;
        return nullptr;
    }

    // Load encoder B
    std::vector<uint8_t> enc_b_data;
    if (!load_binary_file(dir + "vae_encoder_B.bin", enc_b_data)) {
        delete model;
        return nullptr;
    }
    if (!parse_layer_file(enc_b_data, model->encoder_B_weights,
                          model->encoder_B_biases, model->encoder_B_layers)) {
        delete model;
        return nullptr;
    }

    // Load decoder
    std::vector<uint8_t> dec_data;
    if (!load_binary_file(dir + "vae_decoder_A.bin", dec_data)) {
        delete model;
        return nullptr;
    }
    if (!parse_layer_file(dec_data, model->decoder_weights,
                          model->decoder_biases, model->decoder_layers)) {
        delete model;
        return nullptr;
    }

    // Allocate buffers
    model->buffer1.resize(model->max_batch_size * 256);
    model->buffer2.resize(model->max_batch_size * 256);

    return model;
}

void vae_free_model(VAEModel* model) {
    delete model;
}

void vae_encode_A(
    VAEModel* model,
    const float* input,
    float* mu,
    int batch_size
) {
    encoder_forward(
        input, mu,
        model->encoder_A_weights,
        model->encoder_A_biases,
        model->encoder_A_layers,
        model->buffer1.data(),
        model->buffer2.data(),
        batch_size
    );
}

void vae_encode_B(
    VAEModel* model,
    const float* input,
    float* mu,
    int batch_size
) {
    encoder_forward(
        input, mu,
        model->encoder_B_weights,
        model->encoder_B_biases,
        model->encoder_B_layers,
        model->buffer1.data(),
        model->buffer2.data(),
        batch_size
    );
}

void vae_decode(
    VAEModel* model,
    const float* z,
    float* output,
    int batch_size
) {
    decoder_forward(
        z, output,
        model->decoder_weights,
        model->decoder_biases,
        model->decoder_layers,
        model->buffer1.data(),
        model->buffer2.data(),
        batch_size
    );
}

void vae_forward(
    VAEModel* model,
    const float* input,
    float* output,
    int batch_size
) {
    // Temporary latent buffer
    std::vector<float> mu(batch_size * model->latent_dim);

    // Encode
    vae_encode_A(model, input, mu.data(), batch_size);

    // Decode
    vae_decode(model, mu.data(), output, batch_size);
}

#endif // VAE_GEMM_IMPLEMENTATION
