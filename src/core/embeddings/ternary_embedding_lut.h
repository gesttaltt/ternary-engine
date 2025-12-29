/**
 * @file ternary_embedding_lut.h
 * @brief Pre-computed Ternary Embedding Lookup Table
 *
 * Provides O(1) access to learned hyperbolic embeddings for all 19,683
 * balanced ternary values (9 trits). Embeddings preserve p-adic hierarchy:
 *   - v=0 values: radius ~0.99 (outer shell)
 *   - v=9 values: radius ~0.01 (near origin)
 *
 * Use cases:
 *   1. Geometric nearest-neighbor search
 *   2. Hierarchical clustering respecting p-adic valuation
 *   3. Approximate ternary operations via embedding arithmetic
 *   4. Similarity search in ternary space
 *
 * Memory: 1.2 MB for [19683, 16] float32 embeddings
 *
 * @date 2025-12-29
 * @version 1.0
 */

#pragma once

#include <stdint.h>
#include <stddef.h>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>

#ifdef __cplusplus
extern "C" {
#endif

// Constants
#define TERNARY_LUT_SIZE 19683   // 3^9 = 19683 possible values
#define TERNARY_EMBED_DIM 16     // Embedding dimension

/**
 * @brief Ternary Embedding LUT structure
 */
typedef struct {
    float* embeddings;      // [19683, 16] float32
    int num_entries;        // 19683
    int embed_dim;          // 16
    int loaded;             // 1 if loaded, 0 otherwise
} TernaryEmbeddingLUT;

/**
 * @brief Load embedding LUT from binary file
 *
 * @param path Path to ternary_embeddings.bin
 * @return Pointer to loaded LUT (NULL on failure)
 */
TernaryEmbeddingLUT* ternary_lut_load(const char* path);

/**
 * @brief Free LUT memory
 */
void ternary_lut_free(TernaryEmbeddingLUT* lut);

/**
 * @brief Get embedding for a ternary value
 *
 * @param lut     Loaded LUT
 * @param index   Ternary index [0, 19682]
 * @param output  Output buffer [16 floats]
 */
void ternary_lut_get(const TernaryEmbeddingLUT* lut, int index, float* output);

/**
 * @brief Find K nearest neighbors to a query embedding
 *
 * @param lut       Loaded LUT
 * @param query     Query embedding [16 floats]
 * @param k         Number of neighbors
 * @param indices   Output indices [k]
 * @param distances Output distances [k]
 */
void ternary_lut_knn(
    const TernaryEmbeddingLUT* lut,
    const float* query,
    int k,
    int* indices,
    float* distances
);

/**
 * @brief Find nearest neighbor to a query embedding
 *
 * @param lut   Loaded LUT
 * @param query Query embedding [16 floats]
 * @return Index of nearest neighbor
 */
int ternary_lut_nearest(const TernaryEmbeddingLUT* lut, const float* query);

/**
 * @brief Compute midpoint embedding and find nearest ternary value
 *
 * @param lut Loaded LUT
 * @param a   First ternary index
 * @param b   Second ternary index
 * @return Nearest ternary index to midpoint(a, b)
 */
int ternary_lut_midpoint(const TernaryEmbeddingLUT* lut, int a, int b);

/**
 * @brief Get radius (distance from origin) of ternary value
 *
 * @param lut   Loaded LUT
 * @param index Ternary index
 * @return Radius in [0, 1]
 */
float ternary_lut_radius(const TernaryEmbeddingLUT* lut, int index);

/**
 * @brief Convert 9 balanced trits to ternary index
 *
 * @param trits Array of 9 trits in {-1, 0, +1}
 * @return Ternary index [0, 19682]
 */
int trits_to_index(const int8_t* trits);

/**
 * @brief Convert ternary index to 9 balanced trits
 *
 * @param index  Ternary index [0, 19682]
 * @param trits  Output array of 9 trits
 */
void index_to_trits(int index, int8_t* trits);

/**
 * @brief Compute 3-adic valuation of index
 *
 * @param index Ternary index
 * @return Valuation [0, 9]
 */
int ternary_valuation(int index);

#ifdef __cplusplus
}
#endif

// =============================================================================
// C++ Implementation (header-only)
// =============================================================================

#ifdef TERNARY_EMBEDDING_LUT_IMPLEMENTATION

static float euclidean_distance(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

TernaryEmbeddingLUT* ternary_lut_load(const char* path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return nullptr;
    }

    // Read header
    char magic[4];
    file.read(magic, 4);
    if (memcmp(magic, "TLUT", 4) != 0) {
        return nullptr;
    }

    uint32_t version, num_entries, embed_dim;
    file.read(reinterpret_cast<char*>(&version), 4);
    file.read(reinterpret_cast<char*>(&num_entries), 4);
    file.read(reinterpret_cast<char*>(&embed_dim), 4);

    if (num_entries != TERNARY_LUT_SIZE || embed_dim != TERNARY_EMBED_DIM) {
        return nullptr;
    }

    // Allocate LUT
    TernaryEmbeddingLUT* lut = new TernaryEmbeddingLUT();
    lut->num_entries = num_entries;
    lut->embed_dim = embed_dim;
    lut->embeddings = new float[num_entries * embed_dim];
    lut->loaded = 1;

    // Read embeddings
    file.read(reinterpret_cast<char*>(lut->embeddings),
              num_entries * embed_dim * sizeof(float));

    return lut;
}

void ternary_lut_free(TernaryEmbeddingLUT* lut) {
    if (lut) {
        delete[] lut->embeddings;
        delete lut;
    }
}

void ternary_lut_get(const TernaryEmbeddingLUT* lut, int index, float* output) {
    if (index < 0 || index >= lut->num_entries) {
        memset(output, 0, lut->embed_dim * sizeof(float));
        return;
    }
    memcpy(output, &lut->embeddings[index * lut->embed_dim],
           lut->embed_dim * sizeof(float));
}

void ternary_lut_knn(
    const TernaryEmbeddingLUT* lut,
    const float* query,
    int k,
    int* indices,
    float* distances
) {
    // Compute all distances
    std::vector<std::pair<float, int>> dist_idx(lut->num_entries);
    for (int i = 0; i < lut->num_entries; ++i) {
        dist_idx[i] = {
            euclidean_distance(query, &lut->embeddings[i * lut->embed_dim], lut->embed_dim),
            i
        };
    }

    // Partial sort for top-k
    std::partial_sort(dist_idx.begin(), dist_idx.begin() + k, dist_idx.end());

    // Output
    for (int i = 0; i < k; ++i) {
        indices[i] = dist_idx[i].second;
        distances[i] = dist_idx[i].first;
    }
}

int ternary_lut_nearest(const TernaryEmbeddingLUT* lut, const float* query) {
    int best_idx = 0;
    float best_dist = euclidean_distance(query, lut->embeddings, lut->embed_dim);

    for (int i = 1; i < lut->num_entries; ++i) {
        float dist = euclidean_distance(query, &lut->embeddings[i * lut->embed_dim], lut->embed_dim);
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = i;
        }
    }
    return best_idx;
}

int ternary_lut_midpoint(const TernaryEmbeddingLUT* lut, int a, int b) {
    float midpoint[TERNARY_EMBED_DIM];
    const float* emb_a = &lut->embeddings[a * lut->embed_dim];
    const float* emb_b = &lut->embeddings[b * lut->embed_dim];

    for (int i = 0; i < lut->embed_dim; ++i) {
        midpoint[i] = (emb_a[i] + emb_b[i]) * 0.5f;
    }

    return ternary_lut_nearest(lut, midpoint);
}

float ternary_lut_radius(const TernaryEmbeddingLUT* lut, int index) {
    const float* emb = &lut->embeddings[index * lut->embed_dim];
    float sum = 0.0f;
    for (int i = 0; i < lut->embed_dim; ++i) {
        sum += emb[i] * emb[i];
    }
    return sqrtf(sum);
}

int trits_to_index(const int8_t* trits) {
    int index = 0;
    int power = 1;
    for (int i = 0; i < 9; ++i) {
        index += (trits[i] + 1) * power;  // {-1,0,+1} -> {0,1,2}
        power *= 3;
    }
    return index;
}

void index_to_trits(int index, int8_t* trits) {
    for (int i = 0; i < 9; ++i) {
        trits[i] = (index % 3) - 1;  // {0,1,2} -> {-1,0,+1}
        index /= 3;
    }
}

int ternary_valuation(int index) {
    if (index == 0) return 9;
    int v = 0;
    while (index % 3 == 0 && v < 9) {
        index /= 3;
        v++;
    }
    return v;
}

#endif // TERNARY_EMBEDDING_LUT_IMPLEMENTATION
