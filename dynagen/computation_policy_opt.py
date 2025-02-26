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
        # Implementation remains unchanged
        def load_layer_weight(i, j):
            this.load_weight(i, j, 0, overlap=False)

        def load_layer_cache(i, j, k, load_to_cpu=False):
            this.load_cache_dyn(i, j, k, load_to_cpu=load_to_cpu)

        def compute_layer(i, j, layers_weights_sync, layers_cache_sync):
            wait_stream_finish(layers_weights_sync[j])
            layers_weights_sync[j] = None
            if this.layers[j].need_cache:
                wait_stream_finish(layers_cache_sync[j])
            layers_cache_sync[j] = None
            cpu_del = j % 4 == 1
            this.load_hidden(i, j, 0)
            this.compute_layer(i, j, 0, cpu_delegation=None)
            if j == this.num_layers - 1:
                this.sync()
            this.store_cache(i, j - 1, 0)
            this.store_hidden(i, j, 0)
            this.sync()

        layers_weights_sync = [None for _ in range(this.num_layers)]
        layers_cache_sync = [None for _ in range(this.num_layers)]
        f = this.cache_loader.load_cache(True, load_layer_weight, 0, 0)
        layers_weights_sync[0] = f
        this.sync()
        for i in tqdm(range(this.execute_gen_len), desc="Generating"):
            this.update_attention_mask(i, 0)
            for j in range(this.num_layers):
                loading_weights = sum(x is not None for x in layers_weights_sync)
                loading_caches = sum(x is not None for x in layers_cache_sync)
                step = j + 2 if i == 0 else j + 10
                for l in range(j + 1, step):
                    layer = l
                    token = i
                    if layer >= this.num_layers:
                        layer = layer - this.num_layers
                        token = i + 1
                    if token >= this.execute_gen_len:
                        continue
                    if layers_weights_sync[layer] is None and loading_weights <= 3:
                        f = this.cache_loader.load_cache(True, load_layer_weight, token, layer)
                        layers_weights_sync[layer] = f
                    if layers_cache_sync[layer] is None and loading_caches <= 3:
                        f = this.cache_loader.load_cache(True, load_layer_cache, token, layer, 0, 0)
                        layers_cache_sync[layer] = f

                compute_layer(i, j, layers_weights_sync, layers_cache_sync)
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
                    if i==0:
                        prefetch_distance = 1  # How many steps ahead to prefetch
                    else:
                        prefetch_distance = 4  # How many steps ahead to prefetch

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
                        wait_stream_finish(weight_futures[k][j])
                        weight_futures[k][j] = None

                    if cache_futures[k][j] is not None:
                        wait_stream_finish(cache_futures[k][j])
                        cache_futures[k][j] = None

                    # Compute current layer for current batch
                    this.store_hidden(i, j, k - 1)
                    this.load_hidden(i, j, k + 1)
                    this.compute_layer(i, j, k)
                    this.store_cache(i, j, k - 1, overlap=False)
                    this.sync()

            # Check for early stopping condition
            if this.task.stop and np.all(this.stopped):
                break

        # Epilogue - store final hidden state
        this.store_hidden(this.execute_gen_len - 1, this.num_layers - 1, this.num_gpu_batches - 1)
