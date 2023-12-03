#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <random>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <cuda_runtime.h>
#include <cuda.h>

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/count.h>
#include <thrust/transform.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/iterator/counting_iterator.h>
#include <cmath>

// Global variables
int THREADS;
int BLOCKS;
size_t NUM_VALS;

// Function to generate sorted data
std::vector<int> generate_sorted_data()
{
    std::vector<int> data(NUM_VALS);
    for (size_t i = 0; i < NUM_VALS; ++i)
    {
        data[i] = static_cast<int>(i);
    }
    return data;
}

// Function to generate reverse sorted data
std::vector<int> generate_reverse_sorted_data()
{
    std::vector<int> data(NUM_VALS);
    for (size_t i = 0; i < NUM_VALS; ++i)
    {
        data[i] = static_cast<int>(NUM_VALS - i - 1);
    }
    return data;
}

// Function to generate random data
std::vector<int> generate_random_data()
{
    std::vector<int> data(NUM_VALS);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<std::int64_t> dis(0, static_cast<std::int64_t>(NUM_VALS) * 10);

    for (size_t i = 0; i < NUM_VALS; ++i)
    {
        data[i] = static_cast<int>(dis(gen));
    }
    return data;
}

// Function to generate 1% perturbed data
std::vector<int> generate_perturbed_data()
{
    std::vector<int> data = generate_sorted_data();
    size_t perturb_count = std::max(1UL, NUM_VALS / 100);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<std::int64_t> dis(0, static_cast<std::int64_t>(NUM_VALS) * 10);
    std::uniform_int_distribution<size_t> index_dis(0, NUM_VALS - 1);

    for (size_t i = 0; i < perturb_count; ++i)
    {
        data[index_dis(gen)] = static_cast<int>(dis(gen));
    }
    return data;
}

// Check if the data is correctly sorted
bool is_correct(const std::vector<int> &data)
{
    for (size_t i = 1; i < data.size(); i++)
    {
        if (data[i - 1] > data[i])
        {
            return false;
        }
    }
    return true;
}

/**
 * GPU Sample Sort
 * -----------------------
 * Copyright (c) 2009-2019 Nikolaj Leischner and Vitaly Osipov
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 **/

#include <algorithm>
#include <stack>
#include <vector>
#include <queue>
#include <random>
#include <limits>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "detail/constants.h"
#include "detail/bucket.h"
#include "detail/create_bst.h"
#include "detail/find_buckets.h"
#include "detail/scatter.h"
#include "detail/quicksort.h"
#include "detail/copy_buckets.h"
#include "detail/temporary_device_memory.h"

namespace SampleSort
{

    int clamp(int value, int lo, int hi)
    {
        return std::max(lo, std::min(value, hi));
    }

    template <int COPY_THREADS, int MAX_BLOCK_COUNT, bool KEYS_ONLY, typename KeyType, typename ValueType>
    void move_to_output(std::priority_queue<Bucket> &swapped_buckets, KeyType *keys,
                        const TemporaryDeviceMemory<KeyType> &keys_buffer, ValueType *values,
                        const TemporaryDeviceMemory<ValueType> &values_buffer)
    {
        int batch_size = static_cast<int>(std::min(swapped_buckets.size(), static_cast<size_t>(MAX_BLOCK_COUNT)));
        TemporaryDeviceMemory<Bucket> dev_swapped_bucket_data((size_t)batch_size);
        std::vector<Bucket> swapped_bucket_data;

        while (!swapped_buckets.empty())
        {
            swapped_bucket_data.clear();
            batch_size = static_cast<int>(std::min(swapped_buckets.size(), static_cast<size_t>(MAX_BLOCK_COUNT)));

            for (int i = 0; i < batch_size; ++i)
            {
                swapped_bucket_data.push_back(swapped_buckets.top());
                swapped_buckets.pop();
            }

            dev_swapped_bucket_data.copy_to_device(swapped_bucket_data.data());

            if (KEYS_ONLY)
                copy_buckets<COPY_THREADS><<<batch_size, COPY_THREADS>>>(keys, keys_buffer.data, dev_swapped_bucket_data.data);
            else
                copy_buckets<COPY_THREADS><<<batch_size, COPY_THREADS>>>(keys, keys_buffer.data, values, values_buffer.data, dev_swapped_bucket_data.data);
        }
    }

    template <bool KEYS_ONLY, typename KeyType, typename ValueType, typename CompType>
    void sort_buckets(std::priority_queue<Bucket> &small_buckets, KeyType *keys,
                      const TemporaryDeviceMemory<KeyType> &keys_buffer, ValueType *values,
                      const TemporaryDeviceMemory<ValueType> &values_buffer, CompType comp,
                      int sort_threads, int max_block_count)
    {
        // Below this size odd-even-merge-sort is used in the CTA sort.
        const unsigned int LOCAL_SORT_SIZE = 2048;
        // Might want to choose a different size for key-value sorting, since the
        // shared memory requirements are higher.
        const unsigned int LOCAL_SORT_SIZE_KV = 2048;
        int batch_size = static_cast<int>(std::min(small_buckets.size(), static_cast<size_t>(MAX_BLOCK_COUNT)));
        TemporaryDeviceMemory<Bucket> dev_small_bucket_data((size_t)batch_size);
        std::vector<Bucket> small_bucket_data;

        while (!small_buckets.empty())
        {
            small_bucket_data.clear();
            batch_size = static_cast<int>(std::min(small_buckets.size(), static_cast<size_t>(MAX_BLOCK_COUNT)));

            for (int i = 0; i < batch_size; ++i)
            {
                small_bucket_data.push_back(small_buckets.top());
                small_buckets.pop();
            }

            dev_small_bucket_data.copy_to_device(small_bucket_data.data());

            if (KEYS_ONLY)
                quicksort<LOCAL_SORT_SIZE, sort_threads><<<batch_size, sort_threads>>>(keys, keys_buffer.data, dev_small_bucket_data.data, comp);
            else
                quicksort<LOCAL_SORT_SIZE_KV, sort_threads><<<batch_size, sort_threads>>>(keys, keys_buffer.data, values, values_buffer.data, dev_small_bucket_data.data, comp);
        }
    }

    template <typename KeyPtrType, typename ValuePtrType, typename StrictWeakOrdering, bool KEYS_ONLY>
    void sort(KeyPtrType begin, KeyPtrType end, ValuePtrType values_begin, StrictWeakOrdering comp)
    {
        int sort_threads = THREADS;
        int max_block_count = BLOCKS;

        const int A = 32;
        // Smaller oversampling factor, used when all buckets are smaller than some size.
        const int SMALL_A = A / 2;
        // How large should the largest bucket be to allow using the smaller oversampling factor?
        const int REDUCED_OVERSAMPLING_LIMIT = 1 << 25;
        // Number of replicated bucket counters per thread block in the bucket finding / scattering kernels.
        const int COUNTERS = 1;
        // Factor for additional counter replication in the bucket finding kernel.
        const int COUNTER_COPIES = 1;

        const int LOCAL_SORT_SIZE = 2048;
        const int BST_THREADS = 128;
        const int FIND_THREADS = 128;
        const int SCATTER_THREADS = 128;
        // Must be a power of 2.
        const int LOCAL_THREADS = 256;
        const int COPY_THREADS = 128;

        // The number of elements/thread is chosen so that at least this many CTAs are used, if possible.
        const int DESIRED_CTA_COUNT = 1024;

        const int MAX_BLOCK_COUNT = (1 << 29) - 1;

        typedef typename thrust::iterator_traits<KeyPtrType>::value_type KeyType;
        typedef typename thrust::iterator_traits<ValuePtrType>::value_type ValueType;
        typedef StrictWeakOrdering CompType;

        KeyType *keys = thrust::raw_pointer_cast(&*begin);
        ValueType *values = thrust::raw_pointer_cast(&*values_begin);

        const int size = static_cast<int>(end - begin);
        if (size == 0)
            return;

        const int block_sort_limit = clamp(static_cast<int>(size / (2 * std::sqrt(static_cast<float>(K)))), 1 << 14,
                                           1 << 18);

        std::stack<Bucket> large_buckets;
        // Buckets are ordered by size, which improves the performance of the
        // CTA level sorting. Helps the gpu's scheduler?
        std::priority_queue<Bucket> small_buckets;
        std::priority_queue<Bucket> swapped_buckets;

        // Push the whole input on a stack.
        Bucket init(0, size);

        if (size < block_sort_limit)
            small_buckets.push(init);
        else
            large_buckets.push(init);

        TemporaryDeviceMemory<KeyType> keys_buffer(size);

        std::random_device rd;
        std::mt19937 gen(rd());
        // Seeded with a constant value for reproducible benchmark results.
        gen.seed(17);
        std::uniform_int_distribution<int> distribution;
        auto *rng = new Lrand48();

        TemporaryDeviceMemory<ValueType> values_buffer(KEYS_ONLY ? size : 0);

        // Cooperatively k-way split large buckets. Search tree creation is done for several large buckets in parallel.
        while (!large_buckets.empty())
        {
            // Grab as many large buckets as possible, within the CTA count limitation for a kernel call.
            std::vector<Bucket> buckets;
            int max_blocks_per_bucket = 0;
            while (!large_buckets.empty() && buckets.size() < MAX_BLOCK_COUNT)
            {
                Bucket b = large_buckets.top();
                // Adjust the number of elements/thread according to the bucket size.
                int keys_per_thread =
                    static_cast<int>(std::max(1, static_cast<int>(ceil(
                                                     static_cast<double>(b.size) / (DESIRED_CTA_COUNT * FIND_THREADS)))));
                int block_count =
                    static_cast<int>(ceil((static_cast<double>(b.size) / (keys_per_thread * FIND_THREADS))));

                b.keys_per_thread = keys_per_thread;
                max_blocks_per_bucket = std::max(max_blocks_per_bucket, block_count);
                buckets.push_back(b);
                large_buckets.pop();
            }

            // Copy bucket parameters to the GPU.
            TemporaryDeviceMemory<Bucket> dev_bucketParams(buckets.size());
            dev_bucketParams.copy_to_device(buckets.data());

            // Create the binary search trees.
            TemporaryDeviceMemory<KeyType> bst(K * buckets.size());

            rng->init(static_cast<int>((buckets.size() * BST_THREADS)), distribution(gen));

            const int bst_blocks = static_cast<int>(buckets.size());

            // One CTA creates the search tree for one bucket. In the first step only
            // one multiprocessor will be occupied. If no bucket is larger than a certain size,
            // use less oversampling.
            if (block_sort_limit < REDUCED_OVERSAMPLING_LIMIT)
            {
                TemporaryDeviceMemory<KeyType> sample(SMALL_A * K * buckets.size());
                TemporaryDeviceMemory<KeyType> sample_buffer(SMALL_A * K * buckets.size());
                create_bst<K, SMALL_A, BST_THREADS, LOCAL_SORT_SIZE><<<bst_blocks, BST_THREADS>>>(keys, keys_buffer.data, dev_bucketParams.data, bst.data, sample.data, sample_buffer.data, *rng,
                                                                                                  comp);
            }
            else
            {
                TemporaryDeviceMemory<KeyType> sample(A * K * buckets.size());
                TemporaryDeviceMemory<KeyType> sample_buffer(A * K * buckets.size());
                create_bst<K, A, BST_THREADS, LOCAL_SORT_SIZE><<<bst_blocks, BST_THREADS>>>(keys, keys_buffer.data, dev_bucketParams.data, bst.data, sample.data, sample_buffer.data, *rng,
                                                                                            comp);
            }

            rng->destroy();

            // Fetch the bucket parameters again which now contain information about which buckets
            // have only equal splitters. Would be sufficient to just fetch an array of bool flags instead
            // of all parameters. But from profiling it looks as if that would be over-optimization.
            dev_bucketParams.copy_to_host(buckets.data());

            TemporaryDeviceMemory<int> dev_bucket_counters(static_cast<size_t>(K * COUNTERS * max_blocks_per_bucket));

            std::vector<int> new_bucket_bounds(K * buckets.size());
            TemporaryDeviceMemory<int> dev_new_bucket_bounds(K * buckets.size());

            // Loop over the large buckets. The limit for considering a bucket to be large should ensure
            // that the bucket-finding and scattering kernels are launched with a sufficient number of CTAs
            // to make use of all available multiprocessors.
            for (int i = 0; i < buckets.size(); ++i)
            {
                Bucket b = buckets[i];

                int block_count = static_cast<int>(ceil(
                    static_cast<double>(b.size) / (FIND_THREADS * b.keys_per_thread)));

                int from = b.start;
                int to = b.start + b.size;

                KeyType *input = b.flipped ? keys_buffer.data : keys;
                KeyType *output = b.flipped ? keys : keys_buffer.data;
                ValueType *values_input = b.flipped ? values_buffer.data : values;
                ValueType *values_output = b.flipped ? values : values_buffer.data;

                cudaMemcpyToSymbol(bst_cache, bst.data + K * i, K * sizeof(KeyType), 0, cudaMemcpyDeviceToDevice);

                // If all keys in the sample are equal, check if the whole bucket contains only one key.
                if (b.degenerated)
                {
                    thrust::device_ptr<KeyType> dev_input(input + from);
                    KeyType min_key, max_key;
                    cudaMemcpy(&min_key, thrust::min_element(dev_input, dev_input + b.size).get(), sizeof(KeyType),
                               cudaMemcpyDeviceToHost);
                    cudaMemcpy(&max_key, thrust::max_element(dev_input, dev_input + b.size).get(), sizeof(KeyType),
                               cudaMemcpyDeviceToHost);

                    if (!comp(min_key, max_key) && !comp(max_key, min_key))
                    {
                        buckets[i].constant = true;
                        // Skip the rest, the bucket is already sorted.
                        continue;
                    }
                }

                // Find buckets.
                if (!b.degenerated)
                    find_buckets<K, LOG_K, FIND_THREADS, COUNTERS, COUNTER_COPIES, false>
                        <<<block_count, FIND_THREADS>>>(input, from, to, dev_bucket_counters.data, b.keys_per_thread, comp);
                else
                    find_buckets<K, LOG_K, FIND_THREADS, COUNTERS, COUNTER_COPIES, true>
                        <<<block_count, FIND_THREADS>>>(input, from, to, dev_bucket_counters.data, b.keys_per_thread, comp);

                // Scan over the bucket counters, yielding the array positions the blocks of the scattering kernel need to write to.
                thrust::device_ptr<int> dev_counters(dev_bucket_counters.data);
                thrust::inclusive_scan(dev_counters, dev_counters + K * COUNTERS * block_count, dev_counters);

                if (KEYS_ONLY)
                {
                    if (!b.degenerated)
                        scatter<K, LOG_K, FIND_THREADS, SCATTER_THREADS, COUNTERS, false>
                            <<<block_count, SCATTER_THREADS>>>(input, from, to, output, dev_bucket_counters.data,
                                                               dev_new_bucket_bounds.data + K * i, b.keys_per_thread, comp);
                    else
                        scatter<K, LOG_K, FIND_THREADS, SCATTER_THREADS, COUNTERS, true>
                            <<<block_count, SCATTER_THREADS>>>(input, from, to, output, dev_bucket_counters.data,
                                                               dev_new_bucket_bounds.data + K * i, b.keys_per_thread,
                                                               comp);
                }
                else
                {
                    if (!b.degenerated)
                        scatter<K, LOG_K, FIND_THREADS, SCATTER_THREADS, COUNTERS, false>
                            <<<block_count, SCATTER_THREADS>>>(input, values_input, from, to, output, values_output,
                                                               dev_bucket_counters.data, dev_new_bucket_bounds.data + K * i,
                                                               b.keys_per_thread, comp);
                    else
                        scatter<K, LOG_K, FIND_THREADS, SCATTER_THREADS, COUNTERS, true>
                            <<<block_count, SCATTER_THREADS>>>(input, values_input, from, to, output, values_output,
                                                               dev_bucket_counters.data, dev_new_bucket_bounds.data + K * i,
                                                               b.keys_per_thread, comp);
                }
            }

            dev_new_bucket_bounds.copy_to_host(new_bucket_bounds.data());

            for (int i = 0; i < buckets.size(); i++)
            {
                if (!buckets[i].degenerated)
                {
                    for (int j = 0; j < K; j++)
                    {
                        int start = (j > 0) ? new_bucket_bounds[K * i + j - 1] : buckets[i].start;
                        int bucket_size = new_bucket_bounds[K * i + j] - start;
                        Bucket new_bucket(start, bucket_size, !buckets[i].flipped);

                        // Depending on it's size push the bucket on a different stack.
                        if (new_bucket.size > block_sort_limit)
                            large_buckets.push(new_bucket);
                        else if (new_bucket.size > 1)
                            small_buckets.push(new_bucket);
                        else if (new_bucket.size == 1 && new_bucket.flipped)
                            swapped_buckets.push(new_bucket);
                    }
                }
                else if (!buckets[i].constant)
                {
                    // There are only 3 buckets if all splitters were equal.
                    for (int j = 0; j < 3; j++)
                    {
                        int start = (j > 0) ? new_bucket_bounds[K * i + j - 1] : buckets[i].start;
                        int bucket_size = new_bucket_bounds[K * i + j] - start;
                        Bucket new_bucket(start, bucket_size, !buckets[i].flipped);

                        // Bucket with id 1 contains only equal keys, there is no need to sort it.
                        if (j == 1)
                        {
                            if (new_bucket.flipped)
                                swapped_buckets.push(new_bucket);
                        }
                        else if (new_bucket.size > block_sort_limit)
                            large_buckets.push(new_bucket);
                        else if (new_bucket.size > 1)
                            small_buckets.push(new_bucket);
                        else if (new_bucket.size == 1 && new_bucket.flipped)
                            swapped_buckets.push(new_bucket);
                    }
                }
                else
                {
                    // The bucket only contains equal keys. No need for sorting.
                    if (buckets[i].flipped)
                        swapped_buckets.push(buckets[i]);
                }
            }
        }
        delete rng;

        CALI_MARK_BEGIN("comp_small");
        move_to_output<COPY_THREADS, MAX_BLOCK_COUNT, KEYS_ONLY>(swapped_buckets, keys, keys_buffer, values, values_buffer);
        CALI_MARK_END("comp_small");

        CALI_MARK_BEGIN("comp_large");
        sort_buckets<KEYS_ONLY>(small_buckets, keys, keys_buffer, values, values_buffer, comp, sort_threads, max_block_count);
        CALI_MARK_END("comp_large");
    }

    void sort_by_key(std::uint16_t *keys, std::uint16_t *keys_end, std::uint64_t *values)
    {
        SampleSort::sort<std::uint16_t *, std::uint64_t *, thrust::less<std::uint16_t>, false>(keys, keys_end, values, thrust::less<std::uint16_t>());
    }

    void sort_by_key(std::uint32_t *keys, std::uint32_t *keys_end, std::uint64_t *values)
    {
        SampleSort::sort<std::uint32_t *, std::uint64_t *, thrust::less<std::uint32_t>, false>(keys, keys_end, values, thrust::less<std::uint32_t>());
    }

    void sort_by_key(std::uint64_t *keys, std::uint64_t *keys_end, std::uint64_t *values)
    {
        SampleSort::sort<std::uint64_t *, std::uint64_t *, thrust::less<std::uint64_t>, false>(keys, keys_end, values, thrust::less<std::uint64_t>());
    }

    void sort(std::uint16_t *keys, std::uint16_t *keys_end)
    {
        SampleSort::sort<std::uint16_t *, std::uint16_t *, thrust::less<std::uint16_t>, true>(keys, keys_end, 0, thrust::less<std::uint16_t>());
    }

    void sort(std::uint32_t *keys, std::uint32_t *keys_end)
    {
        SampleSort::sort<std::uint32_t *, std::uint32_t *, thrust::less<std::uint32_t>, true>(keys, keys_end, 0, thrust::less<std::uint32_t>());
    }

    void sort(std::uint64_t *keys, std::uint64_t *keys_end)
    {
        SampleSort::sort<std::uint64_t *, std::uint64_t *, thrust::less<std::uint64_t>, true>(keys, keys_end, 0, thrust::less<std::uint64_t>());
    }
}

int main(int argc, char **argv)
{
    CALI_MARK_BEGIN("main");

    // Check for correct number of arguments
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <size> <numThreads> <inputType>\n";
        return 1;
    }

    // Parse the command line arguments
    NUM_VALS = std::stoul(argv[1]);
    THREADS = std::stoi(argv[2]);
    BLOCKS = (NUM_VALS + THREADS - 1) / THREADS;
    std::string inputTypeShort = argv[3];

    std::string inputType;
    if (inputTypeShort == "r")
    {
        inputType = "Random";
    }
    else if (inputTypeShort == "s")
    {
        inputType = "Sorted";
    }
    else if (inputTypeShort == "rs")
    {
        inputType = "Reverse Sorted";
    }
    else if (inputTypeShort == "p")
    {
        inputType = "1% Perturbed";
    }
    else
    {
        std::cerr << "Invalid input type. Use 'r', 's', 'rs', or 'p'.\n";
        return 1;
    }

    adiak::init(NULL);
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();

    std::string algorithm = "SampleSort";
    std::string programmingModel = "CUDA";
    std::string datatype = "int";
    size_t sizeOfDatatype = sizeof(int);
    int group_number = 13;
    std::string implementation_source = "Online";

    adiak::value("Algorithm", algorithm);
    adiak::value("ProgrammingModel", programmingModel);
    adiak::value("Datatype", datatype);
    adiak::value("SizeOfDatatype", sizeOfDatatype);
    adiak::value("InputSize", NUM_VALS);
    adiak::value("InputType", inputType);
    adiak::value("num_threads", THREADS);
    adiak::value("num_blocks", BLOCKS);
    adiak::value("group_num", group_number);
    adiak::value("implementation_source", implementation_source);

    CALI_MARK_BEGIN("data_init");
    std::vector<int> data;
    if (inputType == "Sorted")
    {
        data = generate_sorted_data();
    }
    else if (inputType == "Reverse Sorted")
    {
        data = generate_reverse_sorted_data();
    }
    else if (inputType == "Random")
    {
        data = generate_random_data();
    }
    else if (inputType == "1% Perturbed")
    {
        data = generate_perturbed_data();
    }
    else
    {
        std::cerr << "Invalid input type. Use 'Sorted', 'Reverse Sorted', 'Random', or '1% Perturbed'.\n";
        return 1;
    }
    CALI_MARK_END("data_init");

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_small");
    CALI_MARK_END("comm_small");
    CALI_MARK_END("comm");

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("cudaMemcpy");
    thrust::device_vector<std::uint32_t> d_data(data.begin(), data.end());
    CALI_MARK_END("cudaMemcpy");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    SampleSort::sort(d_data.data().get(), d_data.data().get() + d_data.size(), thrust::less<std::uint32_t>());

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("cudaMemcpy");
    thrust::copy(d_data.begin(), d_data.end(), data.begin());
    CALI_MARK_END("cudaMemcpy");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    CALI_MARK_BEGIN("correctness_check");
    bool correct = is_correct(data);
    if (!correct)
    {
        std::cerr << "Error: The algorithm did not sort the data correctly." << std::endl;
    }
    CALI_MARK_END("correctness_check");

    CALI_MARK_END("main");
    return 0;
}
