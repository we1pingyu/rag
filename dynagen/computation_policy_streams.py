from .computation_policy_interface import *
from .timers import timers
from tqdm import tqdm
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor

class ComputationStreams:
  def __init__(self, size):
    self.size = size
    self.streams = [torch.cuda.Stream() for _ in range(size)]
    self.executors = ThreadPoolExecutor(max_workers=size)
  
  def compute(self, i, func, *args):
    def _run_func(func, *args):
      with torch.cuda.stream(self.streams[i]):
        func(*args)
    return self.executors.submit(_run_func, func, *args)


class ComputationPolicyStream(ComputationPolicyInterface):
  def generation_loop_normal(self, this, evaluate):
    for i in range(this.execute_gen_len):
      timers("generate").start()
      for k in range(this.num_gpu_batches):
        this.update_attention_mask(i, k)
      for j in range(this.num_layers):
        for k in range(this.num_gpu_batches):
          this.load_weight(i, j, k, overlap=False)
        for k in range(this.num_gpu_batches):
          this.load_cache(i, j, k, overlap=False)
          this.load_hidden(i, j, k)
          this.compute_layer(i, j, k)
          if evaluate and j == this.num_layers - 1:
            this.sync()
            break
          this.sync()
          this.store_hidden(i, j, k)
          this.store_cache(i, j, k, overlap=False)
      timers("generate").stop()

  def generation_loop_debug_normal(self, this):
    raise NotImplementedError()
  
  def generation_loop_overlap_single_batch(self, this, evaluate, profile_dir):
    raise NotImplementedError()
  
  def generation_loop_overlap_multi_batch(self, this, profile_dir):
    print('start generation loop')
    def compute_layer(i, j, k):
      this.load_cache(i, j, k, overlap=False)
      this.load_hidden(i, j, k)
      this.compute_layer(i, j, k)
      this.store_hidden(i, j, k)
      this.store_cache(i, j, k, overlap=False)
    
    for i in tqdm(range(this.execute_gen_len)):
      timers("generate").start()
      
      for k in range(this.num_gpu_batches):
        this.update_attention_mask(i, k)
      
      for j in range(this.num_layers):
        this.load_weight(i, j, 0, overlap=False)
        futures = []
        
        for k in range(this.num_gpu_batches):
          f = this.stream_manager.compute(k % this.stream_manager.size, compute_layer, i, j, k)
          futures.append(f)
        
        for f in futures:
          f.result()
        this.sync()

        this.pop_weight(i, j, 0)
        
      timers("generate").stop()

  def generation_loop_debug_single_batch(self, this):
    raise NotImplementedError()


  def generation_loop_debug_multi_batch(self, this):
    raise NotImplementedError()
