#pragma once
#include "utils.h"

#include <vector>
#include <hip/hip_cooperative_groups.h>

#include <thrust/sort.h>    
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>


#include <hipcub/hipcub.hpp>

// #include <hipcub/device/device_scan.cuh>
// #include <hipcub/device/device_histogram.cuh>

__global__ void compute_bucket_off(const uint32_t *keys, uint32_t *bucket_off, size_t total_items, uint32_t buckets_count, size_t scale) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > total_items) return;

    uint32_t total_bins = (total_items / scale) * buckets_count;
    uint32_t prev_key = (idx == 0) ? 0 : keys[idx - 1];
    uint32_t curr_key = (idx == total_items) ? total_bins : keys[idx];

    if (prev_key != curr_key) {
        for (uint32_t b = prev_key; b < curr_key; b++) {
            bucket_off[b] = idx - (b / buckets_count) * scale;
        }
    }
}

template<typename FieldT, typename HostFT>
class BucketContext {
    TypedGpuArena arena;
    void *temp_storage;
    size_t temp_storage_bytes;

public:
    FieldT *scalars;

    uint32_t *window_scalars_as_keys_in;
    uint32_t *window_scalars_as_keys;
    uint32_t *indices_as_vals_in;
    uint32_t *indices_as_vals;

    uint32_t *buckets_off;

    size_t scale;
    size_t window_bits;
    size_t windows_count, buckets_count;

    BucketContext(size_t scale_, size_t window_bits_) 
    : scale(scale_), window_bits(window_bits_), windows_count(ceil_div(FieldT::nbits, window_bits)), buckets_count(1 << window_bits)
    {
        arena.register_alloc(scalars, scale);
        arena.register_alloc(window_scalars_as_keys_in, windows_count * scale);
        arena.register_alloc(window_scalars_as_keys, windows_count * scale);
        arena.register_alloc(indices_as_vals_in, windows_count * scale);
        arena.register_alloc(indices_as_vals, windows_count * scale);
        arena.register_alloc(buckets_off, windows_count * buckets_count);
        arena.commit("BucketContext");

        temp_storage = 0, temp_storage_bytes = 0;

        size_t num_items_total = windows_count * scale;
        hipcub::DeviceRadixSort::SortPairs(
            temp_storage, temp_storage_bytes,
            window_scalars_as_keys_in, window_scalars_as_keys,
            indices_as_vals_in, indices_as_vals,
            num_items_total);
        hipDeviceSynchronize();
        CUDA_CHECK(hipGetLastError());

        hipError_t err = hipMalloc(&temp_storage, temp_storage_bytes);
        if (err != hipSuccess) throw std::runtime_error("hipMalloc failed");

        printf("[BucketContext] Successfully alloc %f GB memory.\n", double(temp_storage_bytes) / double(1 << 30));
    }

    ~BucketContext() { 
        hipFree(temp_storage); 
    }

    void load_scalars(const HostFT *host_scalars) { hipMemcpy(scalars, host_scalars, scale * sizeof(FieldT), hipMemcpyHostToDevice); }
    void load_scalars(FieldT *device_scalars) { hipMemcpy(scalars, device_scalars, scale * sizeof(FieldT), hipMemcpyDeviceToDevice); }

    void process(bool from = true, bool extraction = true, bool sort = true, bool scatter = true)
    {   
        if (from) thrust::for_each(thrust::hip::par.on(0), scalars, scalars + scale, [] __device__ (FieldT &x) { x.from(); });

        if (extraction) {
            size_t scale = this->scale;
            size_t window_bits = this->window_bits;
            size_t windows_count = this->windows_count;
            size_t buckets_count = this->buckets_count;
            const size_t block_size = 256;
            kernel<<<ceil_div(scale, block_size), block_size, 0, 0>>>([=] __device__ (FieldT *scalars, uint32_t *keys, uint32_t *vals) {
                namespace cg = cooperative_groups;
                cg::grid_group g = cg::this_grid();
                int idx = g.thread_rank();

                if (idx < scale) {
                    for (int j = 0; j < windows_count; j++) {
                        uint32_t key = get_window_by_ptr(scalars + idx, j * window_bits, window_bits);
                        keys[j * scale + idx] = j * buckets_count + key;
                        vals[j * scale + idx] = idx;
                    }
                }
            }, scalars, window_scalars_as_keys_in, indices_as_vals_in);
        }

        hipDeviceSynchronize();
        CUDA_CHECK(hipGetLastError());

        if (sort) {
            size_t num_items_total = windows_count * scale;
            hipcub::DeviceRadixSort::SortPairs(
                temp_storage, temp_storage_bytes,
                window_scalars_as_keys_in, window_scalars_as_keys,
                indices_as_vals_in, indices_as_vals,
                num_items_total);
            hipDeviceSynchronize();
            CUDA_CHECK(hipGetLastError());
        }

        if (scatter) {
            size_t num_items_total = windows_count * scale;
            size_t threads = 256;
            size_t blocks = ceil_div(num_items_total + 1, threads);
            compute_bucket_off<<<blocks, threads>>>(window_scalars_as_keys, buckets_off, num_items_total, buckets_count, scale);
            hipDeviceSynchronize();
            CUDA_CHECK(hipGetLastError());
        }
    }
};
