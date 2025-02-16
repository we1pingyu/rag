from .computation_policy_interface import *

# from optimize.network_config import Llama13BConfig
# from timer import timers
from tqdm import tqdm
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor

# from dynagen.optimize.dynagen_optimize import DynagenOpt, DynagenOptWorksetHeuristic


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

    def generation_loop_overlap_single_batch(self, this, evaluate, profile_dir):
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

    # def generation_loop_overlap_multi_batch(self, this, profile_dir):
    #     def load_layer_weight(i, j, k):
    #         this.load_weight(i, j, k, overlap=False)

    #     def load_layer_cache(i, j, k, load_to_cpu=False):
    #         this.load_cache_dyn(i, j, k, load_to_cpu=load_to_cpu)

    #     def compute_layer(i, j, k, layers_weights_sync, layers_cache_sync, cpu_del):
    #         wait_stream_finish(layers_weights_sync[k][j])
    #         layers_weights_sync[k][j] = None
    #         if this.layers[j].need_cache:
    #             wait_stream_finish(layers_cache_sync[k][j])
    #         layers_cache_sync[k][j] = None
    #         this.store_hidden(i, j, k - 1)
    #         this.load_hidden(i, j, k + 1)
    #         this.compute_layer(i, j, k, cpu_delegation=cpu_del[(i, j, k)])
    #         this.store_cache(i, j, k - 1, overlap=False)

    #     # optimizer = DynagenOptWorksetHeuristic(this.num_layers, this.policy.gpu_batch_size, this.num_gpu_batches, 1024, this.execute_gen_len, 23, Llama13BConfig())
    #     # optimizer.optimize()
    #     optimizer = DynagenOpt(
    #         this.num_layers,
    #         this.policy.gpu_batch_size,
    #         this.num_gpu_batches,
    #         1024,
    #         this.execute_gen_len,
    #         Llama13BConfig(),
    #     )
    #     optimizer.optimize_alter_v2()
    #     cache_prefetch, weight_prefetch, cpu_delegation = optimizer.get_policy()

    #     layers_weights_sync = [[None for _ in range(this.num_layers)] for _ in range(this.num_gpu_batches)]
    #     layers_cache_sync = [[None for _ in range(this.num_layers)] for _ in range(this.num_gpu_batches)]
    #     w = this.cache_loader.load_cache(True, load_layer_weight, 0, 0, 0)
    #     layers_weights_sync[0][0] = w
    #     this.load_hidden(0, 0, 0)
    #     this.sync()
    #     for i in tqdm(range(this.execute_gen_len)):
    #         timers("generate").start()

    #         for k in range(this.num_gpu_batches):
    #             this.update_attention_mask(i, k)

    #         for j in range(this.num_layers):
    #             for k in range(this.num_gpu_batches):
    #                 cache_prefetches = []
    #                 if (i, j, k) in cache_prefetch:
    #                     cache_prefetches = cache_prefetch[(i, j, k)]
    #                 weight_prefetches = []
    #                 if (i, j, k) in weight_prefetch:
    #                     weight_prefetches = weight_prefetch[(i, j, k)]
    #                 for token, layer, batch in cache_prefetches:
    #                     f = this.cache_loader.load_cache(
    #                         True, load_layer_cache, token, layer, batch, cpu_delegation[(token, layer, batch)]
    #                     )
    #                     layers_cache_sync[batch][layer] = f
    #                 for token, layer, batch in weight_prefetches:
    #                     f = this.cache_loader.load_cache(True, load_layer_weight, token, layer, batch)
    #                     layers_weights_sync[batch][layer] = f

    #                 compute_layer(i, j, k, layers_weights_sync, layers_cache_sync, cpu_delegation)

    #                 if i == 0:
    #                     this.sync()

    #         timers("generate").stop()
