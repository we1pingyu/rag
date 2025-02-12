from .computation_policy_interface import *
from .timers import timers
from tqdm import tqdm
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
import time

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
      
      if need_sync:
        use_stream.synchronize()
  
    return self.executors.submit(_run_func)

def wait_stream_finish(f):
  f.result()

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

class ComputationPolicyAlterStream(ComputationPolicyInterface):
  def generation_loop_normal(self, this, evaluate):
    raise NotImplementedError()
  
  def generation_loop_overlap_single_batch(self, this, evaluate, profile_dir):
    def load_layer_weight(i, j, load_to_cpu=False):
      this.load_weight(i, j, 0, overlap=False)
    
    def load_layer_cache(i, j, k, load_to_cpu=False):
      this.load_cache_dyn(i, j, k, load_to_cpu=load_to_cpu)
      
    def compute_layer(i, j, weight_handle, cache_handle):
      # if this.weight_read_buf[j].val is None:
      wait_stream_finish(weight_handle)
      # this.load_weight_stream.synchronize()
      if this.layers[j].need_cache:
        wait_stream_finish(cache_handle)
        
      cpu_del = this.cpu_del[j]
      this.load_hidden(i, j, 0)
      this.compute_layer(i, j, 0, cpu_delegation=cpu_del)
      this.store_cache(i, j - 1, 0)
      this.store_hidden(i, j, 0)
      # this.pop_weight(i, j, 0)
    
    layers_weights_sync = [None for _ in range(this.num_layers * 2)]
    layers_cache_sync = [None for _ in range(this.num_layers * 2)]
    f = this.cache_loader.load_cache(True, load_layer_weight, 0, 0, this.cpu_del[0])
    layers_weights_sync[0] = f
    for i in tqdm(range(this.execute_gen_len)):
      timers("generate").start()                    
      this.update_attention_mask(i, 0)
      for j in range(this.num_layers):
          # load weight and cache
          for k in range(j + 1, j + 6):
            if layers_weights_sync[k] is None:
              f = this.cache_loader.load_cache(True, load_layer_weight, i, k, this.cpu_del[k % this.num_layers])
              layers_weights_sync[k] = f
          for k in range(j + 1, min(j + 4, this.num_layers + 5)):
            if layers_cache_sync[k] is None:
              f = this.cache_loader.load_cache(True, load_layer_cache, i, k, 0, this.cpu_del[k % this.num_layers])
              layers_cache_sync[k] = f
          compute_layer(i, j, layers_weights_sync[j], layers_cache_sync[j])
          if i==0:
            this.sync()
          if j == this.num_layers - 1:
            layers_weights_sync = layers_weights_sync[this.num_layers:] + [None for _ in range(this.num_layers)]
            layers_cache_sync = layers_cache_sync[this.num_layers:] + [None for _ in range(this.num_layers)]

      timers("generate").stop()

  
  def generation_loop_overlap_multi_batch(self, this, profile_dir):
    def load_layer_weight(i, j):
      this.load_weight(i, j, 0, overlap=False)
      
    def load_layer_cache(i, j, k, load_to_cpu=False):
      this.load_cache_dyn(i, j, k, load_to_cpu=load_to_cpu)
      
    def compute_layer(i, j, k, weight_handle, cache_handle):
      wait_stream_finish(weight_handle)
      if this.layers[j].need_cache:
        wait_stream_finish(cache_handle)
        
      cpu_del = this.cpu_del[j]
      this.load_hidden(i, j, k)
      this.compute_layer(i, j, k, cpu_delegation=cpu_del)
      this.store_cache(i, j - 1, k, overlap=False)
      this.store_hidden(i, j, k)

    for i in tqdm(range(this.execute_gen_len)):
      timers("generate").start()
      
      for k in range(this.num_gpu_batches):
        this.update_attention_mask(i, k)
      
      layers_weights_sync = [None for _ in range(this.num_layers)]
      layer_cache_sync = [[None for _ in range(this.num_gpu_batches)] for _ in range(this.num_layers)]
      # load the first weight
      w = this.cache_loader.load_cache(True, load_layer_weight, i, 0)
      layers_weights_sync[0] = w
      for j in range(this.num_layers):
        futures = []
        for k in range(this.num_gpu_batches):
          preload_layer = j
          preload_batch = k + 1
          if k == this.num_gpu_batches - 1:
            preload_layer = j + 1
            preload_batch = 0
          
          if preload_layer < this.num_layers and this.layers[preload_layer].need_cache:
            # load next batch's cache
            c = this.cache_loader.load_cache(True, load_layer_cache, i, preload_layer, preload_batch)
            layer_cache_sync[preload_layer][preload_batch] = c
          
          if k == 0 and preload_layer + 1 < this.num_layers:
            # The first batch: load next layer's weight
            w = this.cache_loader.load_cache(True, load_layer_weight, i, preload_layer + 1)
            layers_weights_sync[preload_layer + 1] = w
          
          f = this.stream_manager.compute(True, compute_layer, i, j, k, layers_weights_sync[j], layer_cache_sync[j][k])
          futures.append(f)

        # sync
        for f in futures:
          f.result()
        if i==0:
          this.sync()
        
        this.pop_weight(i, j, 0)
      
      timers("generate").stop()
