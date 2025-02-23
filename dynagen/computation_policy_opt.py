from .computation_policy_interface import *

from .timers import timers
from tqdm import tqdm
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from math import ceil


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
        # print("Synchronizing stream")
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
            # this.load_weight(i, j + 1, 0)
            # this.load_cache(i,j+1,0)
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
            # timers("generate").start()
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

            # timers("generate").stop()

    def generation_loop_overlap_multi_batch(self, this):
        def load_layer_weight(i, j, k):
            this.load_weight(i, j, k, overlap=False)

        def load_layer_cache(i, j, k, load_to_cpu=False):
            this.load_cache_dyn(i, j, k, load_to_cpu=load_to_cpu)

        def compute_layer(i, j, k, layers_weights_sync, layers_cache_sync, cpu_del):
            wait_stream_finish(layers_weights_sync[k][j])
            layers_weights_sync[k][j] = None
            if this.layers[j].need_cache:
                wait_stream_finish(layers_cache_sync[k][j])
            layers_cache_sync[k][j] = None
            this.store_hidden(i, j, k - 1)
            this.load_hidden(i, j, k + 1)
            this.compute_layer(i, j, k, cpu_delegation=cpu_del[(i, j, k)])
            this.store_cache(i, j, k - 1, overlap=False)

        optimizer = DynagenOpt(
            this.num_layers,
            this.policy.gpu_batch_size,
            this.num_gpu_batches,
            this.execute_gen_len,
        )
        optimizer.optimize()
        cache_prefetch, weight_prefetch, cpu_delegation = optimizer.get_policy()

        layers_weights_sync = [[None for _ in range(this.num_layers)] for _ in range(this.num_gpu_batches)]
        layers_cache_sync = [[None for _ in range(this.num_layers)] for _ in range(this.num_gpu_batches)]
        w = this.cache_loader.load_cache(True, load_layer_weight, 0, 0, 0)
        layers_weights_sync[0][0] = w
        this.load_hidden(0, 0, 0)
        this.sync()
        for i in tqdm(range(this.execute_gen_len)):
            timers("generate").start()

            for k in range(this.num_gpu_batches):
                this.update_attention_mask(i, k)

            for j in range(this.num_layers):
                for k in range(this.num_gpu_batches):
                    cache_prefetches = []
                    if (i, j, k) in cache_prefetch:
                        cache_prefetches = cache_prefetch[(i, j, k)]
                    weight_prefetches = []
                    if (i, j, k) in weight_prefetch:
                        weight_prefetches = weight_prefetch[(i, j, k)]
                    for token, layer, batch in cache_prefetches:
                        f = this.cache_loader.load_cache(
                            True, load_layer_cache, token, layer, batch, cpu_delegation[(token, layer, batch)]
                        )
                        layers_cache_sync[batch][layer] = f
                    for token, layer, batch in weight_prefetches:
                        f = this.cache_loader.load_cache(True, load_layer_weight, token, layer, batch)
                        layers_weights_sync[batch][layer] = f

                    compute_layer(i, j, k, layers_weights_sync, layers_cache_sync, cpu_delegation)

                    if i == 0:
                        this.sync()

            timers("generate").stop()


class DynagenOpt:
    def __init__(
        self,
        num_layers,
        batch_size,
        num_gpu_batches,
        gen_len,
    ):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_gpu_batches = num_gpu_batches
        self.gen_len = gen_len

        self.cache_prefetch = np.array([None] * gen_len * num_layers * num_gpu_batches)
        self.weight_prefetch = np.array([None] * gen_len * num_layers * num_gpu_batches)
        self.cpu_delegation = np.array([0] * gen_len * num_layers * num_gpu_batches)

    def get_policy(self):
        cache_prefetch = {}
        weight_prefetch = {}
        cpu_delegation = {}
        for i in range(self.gen_len):
            for j in range(self.num_layers):
                for k in range(self.num_gpu_batches):
                    cache_prefetch_idx = self.cache_prefetch[self._idx(i, j, k)]
                    weight_prefetch_idx = self.weight_prefetch[self._idx(i, j, k)]

                    if cache_prefetch_idx is not None:
                        if self._decode(cache_prefetch_idx) not in cache_prefetch:
                            cache_prefetch[self._decode(cache_prefetch_idx)] = []
                        cache_prefetch[self._decode(cache_prefetch_idx)].append((i, j, k))

                    if weight_prefetch_idx is not None:
                        if self._decode(weight_prefetch_idx) not in weight_prefetch:
                            weight_prefetch[self._decode(weight_prefetch_idx)] = []
                        weight_prefetch[self._decode(weight_prefetch_idx)].append((i, j, k))

                    cpu_delegation[(i, j, k)] = self.cpu_delegation[self._idx(i, j, 0)]

        return cache_prefetch, weight_prefetch, cpu_delegation

    def _idx(self, token, layer, batch):
        return token * self.num_layers * self.num_gpu_batches + layer * self.num_gpu_batches + batch

    def _decode(self, idx):
        token = idx // (self.num_layers * self.num_gpu_batches)
        layer = (idx % (self.num_layers * self.num_gpu_batches)) // self.num_gpu_batches
        batch = idx % self.num_gpu_batches
        return (token, layer, batch)

    def optimize(self):
        layers_weights_sync = [[None for _ in range(self.num_layers)] for _ in range(self.num_gpu_batches)]
        layers_cache_sync = [[None for _ in range(self.num_layers)] for _ in range(self.num_gpu_batches)]
        layers_weights_sync[0][0] = 1
        for i in range(self.gen_len):
            for j in range(self.num_layers):
                for k in range(self.num_gpu_batches):
                    loading_weights = sum(x is not None for sublist in layers_weights_sync for x in sublist)
                    loading_caches = sum(x is not None for sublist in layers_cache_sync for x in sublist)
                    step = k + 2 if i == 0 else k + self.num_gpu_batches * 10
                    for l in range(k + 1, step):
                        batch = l % self.num_gpu_batches
                        layer = j + l // self.num_gpu_batches
                        token = i
                        if layer >= self.num_layers:
                            layer = layer - self.num_layers
                            token = i + 1
                        if token >= self.gen_len:
                            continue
                        if layers_weights_sync[batch][layer] is None and loading_weights <= 4:
                            self.weight_prefetch[self._idx(token, layer, batch)] = self._idx(i, j, k)
                            layers_weights_sync[batch][layer] = 1
                            loading_weights += 1
                        if layers_cache_sync[batch][layer] is None and loading_caches <= 4:
                            self.cache_prefetch[self._idx(token, layer, batch)] = self._idx(i, j, k)
                            self.cpu_delegation[self._idx(token, layer, batch)] = 0
                            layers_cache_sync[batch][layer] = 1
                            loading_caches += 1
                    # compute
                    layers_weights_sync[k][j] = None
                    layers_cache_sync[k][j] = None
