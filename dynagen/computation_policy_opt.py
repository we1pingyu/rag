from .computation_policy_interface import *
from tqdm import tqdm
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor


class MultiStreamBase:
    def __init__(self, size):
        self.size = size
        self.streams = [torch.cuda.Stream() for _ in range(size)]
        self.executors = ThreadPoolExecutor(max_workers=size)
        self.execute_idx = 0

    def run(self, need_sync, func, *args):
        use_stream = self.streams[self.execute_idx]
        self.execute_idx = (self.execute_idx + 1) % self.size

        def _run_func():
            with torch.cuda.stream(use_stream):
                func(*args)
            return use_stream if need_sync else None

        return self.executors.submit(_run_func)


def wait_stream_finish(f):
    stream = f.result()
    if stream is not None:
        stream.synchronize()


class CacheLoaderManager(MultiStreamBase):
    def __init__(self, size):
        super().__init__(size)

    def load_cache(self, need_sync, func, *args):
        return self.run(need_sync, func, *args)


class ComputationStreamAlterManager(MultiStreamBase):
    def __init__(self, size):
        super().__init__(size)

    def compute(self, need_sync, func, *args):
        return self.run(need_sync, func, *args)


class ComputationPolicyOptimize(ComputationPolicyInterface):
    def generation_loop_normal(self, this, evaluate):
        raise NotImplementedError()

    def generation_loop_overlap_single_batch(self, this, evaluate):
        # Helper functions for asynchronous operations
        def load_weight_async(i, j):
            return this.cache_loader.load_cache(True, lambda: this.load_weight(i, j, 0, overlap=False))

        def load_cache_async(i, j):
            return this.cache_loader.load_cache(True, lambda: this.load_cache_dyn(i, j, 0, load_to_cpu=False))

        # Create prefetching buffers
        weight_futures = [None for _ in range(this.num_layers)]
        cache_futures = [None for _ in range(this.num_layers)]

        # Initialize with first layer weight
        weight_futures[0] = load_weight_async(0, 0)
        this.sync()

        # Main generation loop
        for i in tqdm(range(this.execute_gen_len), desc="Generating"):
            this.update_attention_mask(i, 0)

            for j in range(this.num_layers):
                # Prefetch weights and caches for upcoming layers
                prefetch_distance = 2 if i == 0 else 2

                for offset in range(1, prefetch_distance + 1):
                    next_j = j + offset
                    next_i = i

                    # Handle wrap-around for layer index
                    if next_j >= this.num_layers:
                        next_j = next_j - this.num_layers
                        next_i = i + 1

                    # Skip if beyond generation length
                    if next_i >= this.execute_gen_len:
                        continue

                    # Limit number of concurrent prefetches
                    loading_weights = sum(x is not None for x in weight_futures)
                    loading_caches = sum(x is not None for x in cache_futures)

                    # Prefetch weight if slot is empty and not too many already loading
                    if weight_futures[next_j] is None and loading_weights <= prefetch_distance:
                        weight_futures[next_j] = load_weight_async(next_i, next_j)

                    # Prefetch cache if slot is empty and not too many already loading
                    if cache_futures[next_j] is None and loading_caches <= prefetch_distance:
                        cache_futures[next_j] = load_cache_async(next_i, next_j)

                # Wait for current weight and cache to be ready
                if weight_futures[j] is not None:
                    this.env.disk.synchronize()
                    wait_stream_finish(weight_futures[j])
                    weight_futures[j] = None

                if this.layers[j].need_cache and cache_futures[j] is not None:
                    this.env.disk.synchronize()
                    wait_stream_finish(cache_futures[j])
                cache_futures[j] = None

                # Compute current layer
                this.load_hidden(i, j, 0)
                this.compute_layer(i, j, 0, cpu_delegation=None)

                if j == this.num_layers - 1:
                    this.sync()

                this.store_cache(i, j - 1, 0)
                this.store_hidden(i, j, 0)
                this.sync()

                if i == 0:
                    this.sync()

    def generation_loop_overlap_multi_batch(self, this):
        # Simplified implementation without DynagenOpt
        # Prologue - initialize weights and hidden state
        for k in range(this.num_gpu_batches):
            this.load_weight(0, 0, k, overlap=False)
        this.load_hidden(0, 0, 0)
        this.sync()

        # Create prefetching buffers
        weight_futures = [[None for _ in range(this.num_layers)] for _ in range(this.num_gpu_batches)]
        cache_futures = [[None for _ in range(this.num_layers)] for _ in range(this.num_gpu_batches)]

        # Helper functions for asynchronous operations
        def load_weight_async(i, j, k):
            return this.cache_loader.load_cache(True, lambda: this.load_weight(i, j, k, overlap=False))

        def load_cache_async(i, j, k):
            return this.cache_loader.load_cache(True, lambda: this.load_cache(i, j, k, overlap=False))

        # Generate tokens
        for i in tqdm(range(this.execute_gen_len), desc="Generating"):

            # Update attention masks for all batches
            for k in range(this.num_gpu_batches):
                this.update_attention_mask(i, k)

            # Process each layer
            for j in range(this.num_layers):
                # Process each batch
                for k in range(this.num_gpu_batches):
                    # Prefetch weights and cache for next operations with a simple sliding window
                    if i == 0:
                        prefetch_distance = 1  # How many steps ahead to prefetch
                    else:
                        prefetch_distance = 1  # How many steps ahead to prefetch

                    # Prefetch weights for upcoming operations
                    for offset in range(1, prefetch_distance + 1):
                        next_j = j + offset
                        next_i = i
                        next_k = k

                        # Handle wrap-around for layer index
                        if next_j >= this.num_layers:
                            next_j = next_j - this.num_layers
                            next_i = i + 1

                        # Skip if beyond generation length
                        if next_i >= this.execute_gen_len:
                            continue

                        # Prefetch weight if slot is empty
                        if weight_futures[next_k][next_j] is None:
                            weight_futures[next_k][next_j] = load_weight_async(next_i, next_j, next_k)

                        # Prefetch cache if slot is empty
                        if cache_futures[next_k][next_j] is None:
                            cache_futures[next_k][next_j] = load_cache_async(next_i, next_j, next_k)

                    # Wait for current weight and cache to be ready
                    if weight_futures[k][j] is not None:
                        this.env.disk.synchronize()
                        wait_stream_finish(weight_futures[k][j])
                        weight_futures[k][j] = None

                    if cache_futures[k][j] is not None:
                        this.env.disk.synchronize()
                        wait_stream_finish(cache_futures[k][j])
                        cache_futures[k][j] = None

                    # Compute current layer for current batch
                    this.store_hidden(i, j, k - 1)
                    this.load_hidden(i, j, k + 1)
                    this.compute_layer(i, j, k)
                    this.store_cache(i, j, k - 1, overlap=False)
                    # this.sync()

            # Check for early stopping condition
            if this.task.stop and np.all(this.stopped):
                break

        # Epilogue - store final hidden state
        this.store_hidden(this.execute_gen_len - 1, this.num_layers - 1, this.num_gpu_batches - 1)
