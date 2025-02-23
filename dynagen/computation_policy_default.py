from .computation_policy_interface import *
from .timers import timers
from tqdm import tqdm
import numpy as np
import torch
import time


class ComputationPolicyImpl(ComputationPolicyInterface):
    def generation_loop_normal(self, this, evaluate):
        sum_attn = 0.0
        sum_mlp = 0.0
        count_attn = 0
        count_mlp = 0
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
                    start_time = time.time()
                    this.compute_layer(i, j, k)
                    duration = time.time() - start_time
                    if evaluate and j == this.num_layers - 1:
                        this.sync()
                        break
                    if i > 0:
                        if j % 2 == 1:
                            sum_attn += duration
                            count_attn += 1
                        else:
                            sum_mlp += duration
                    count_mlp += 1
                    this.sync()
                    this.store_hidden(i, j, k)
                    this.store_cache(i, j, k, overlap=False)
            timers("generate").stop()
        if count_attn > 0:
            avg_attn = sum_attn / count_attn
            avg_mlp = sum_mlp / count_mlp
            print("Average compute_layer time for attn:", avg_attn, "seconds")
            print("Average compute_layer time for mlp:", avg_mlp, "seconds")

    def generation_loop_debug_normal(self, this):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill_total").reset()
        timers("decoding_gpu_batch").reset()

        timers("load_weight").reset()
        timers("load_cache_prefill").reset()
        timers("load_cache_decoding").reset()
        timers("store_cache_prefill").reset()
        timers("store_cache_decoding").reset()
        timers("compute_layer_prefill").reset()
        timers("compute_layer_decoding").reset()
        load_weight_timer = timers("load_weight")

        for i in range(this.execute_gen_len):
            if i == 0:
                timers("prefill_total").start()
                load_cache_timer = timers("load_cache_prefill")
                store_cache_timer = timers("store_cache_prefill")
                compute_layer_timer = timers("compute_layer_prefill")
            else:
                load_cache_timer = timers("load_cache_decoding")
                store_cache_timer = timers("store_cache_decoding")
                compute_layer_timer = timers("compute_layer_decoding")

            for k in range(this.num_gpu_batches):
                this.update_attention_mask(i, k)

            for j in range(this.num_layers):
                if i > 0:
                    timers("decoding_gpu_batch").start()

                load_weight_timer.start(this.sync)
                for k in range(this.num_gpu_batches):
                    this.load_weight(i, j, k)
                load_weight_timer.stop(this.sync)

                for k in range(this.num_gpu_batches):
                    load_cache_timer.start(this.sync)
                    this.load_cache(i, j, k)
                    load_cache_timer.stop(this.sync)
                    this.load_hidden(i, j, k)
                    compute_layer_timer.start(this.sync)
                    this.compute_layer(i, j, k)
                    compute_layer_timer.stop(this.sync)
                    this.store_hidden(i, j, k)
                    store_cache_timer.start(this.sync)
                    this.store_cache(i, j, k)
                    store_cache_timer.stop(this.sync)

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches:
                    break
            if batch_ct >= execute_num_batches:
                break
            if i == 0:
                timers("prefill_total").stop(this.sync)

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(this.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill_total").costs[0])
            else:
                timers("generate").costs.append(this.num_layers * batch_cost)

        # Debug the costs of individual functions
        print(f"#layers: {this.num_layers}")

        print(f"#batches prefill:  " f"{this.num_layers * this.num_gpu_batches}")
        print(f"#batches decoding: " f"{(this.task.gen_len - 1) * this.num_layers * this.num_gpu_batches}")
        print(f"load_weight            (per-layer)" f": {np.mean(timers('load_weight').costs):.6f} s")
        for stage in ["prefill", "decoding"]:
            for func in ["load_cache", "store_cache", "compute_layer"]:
                name = func + "_" + stage
                costs = timers(name).costs
                print(f"{name:22s} (per-batch): {np.mean(costs):.6f} s")

    def generation_loop_overlap_single_batch(self, this, evaluate):
        # Prologue
        this.load_weight(0, 0, 0)
        this.sync()

        # Generate
        for i in tqdm(range(this.execute_gen_len)):
            timers("generate").start()
            this.update_attention_mask(i, 0)
            for j in range(this.num_layers):
                this.load_weight(i, j + 1, 0)
                this.load_cache(i, j + 1, 0)
                this.load_hidden(i, j, 0)
                this.compute_layer(i, j, 0)
                if evaluate and j == this.num_layers - 1:
                    this.sync()
                    break
                this.store_cache(i, j - 1, 0, overlap=False)
                this.store_hidden(i, j, 0)

                this.sync()
            timers("generate").stop()

            if this.task.stop and np.all(this.stopped):
                break

    def generation_loop_overlap_multi_batch(self, this):
        # Prologue
        for k in range(this.num_gpu_batches):
            this.load_weight(0, 0, k)
        this.load_hidden(0, 0, 0)
        this.sync()
        # Generate

        for i in tqdm(range(this.execute_gen_len)):
            timers("generate").start()
            for k in range(this.num_gpu_batches):
                this.update_attention_mask(i, k)
            for j in range(this.num_layers):
                for k in range(this.num_gpu_batches):
                    this.load_weight(i, j + 1, k)
                    this.load_cache(i, j, k + 1)
                    this.store_hidden(i, j, k - 1)
                    this.load_hidden(i, j, k + 1)
                    this.compute_layer(i, j, k)
                    this.store_cache(i, j, k - 1, overlap=False)
                    this.sync()
            timers("generate").stop()

        # Epilogue
        this.store_hidden(this.execute_gen_len - 1, this.num_layers - 1, this.num_gpu_batches - 1)

    def generation_loop_debug_single_batch(self, this):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill").reset()
        timers("decoding_gpu_batch").reset()

        # Prologue
        for k in range(this.num_gpu_batches):
            this.load_weight(0, 0, k)
        this.sync()

        # Generate
        for i in range(this.execute_gen_len):
            if i == 0:
                timers("prefill").start()
            this.update_attention_mask(i, 0)
            for j in range(this.num_layers):
                if i > 0:
                    timers("decoding_gpu_batch").start()
                this.load_weight(i, j + 1, 0)
                this.load_cache(i, j + 1, 0)
                this.load_hidden(i, j, 0)
                this.compute_layer(i, j, 0)
                this.store_cache(i, j - 1, 0)
                this.store_hidden(i, j, 0)
                this.sync()

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches:
                    break
            if batch_ct >= execute_num_batches:
                break
            if i == 0:
                timers("prefill").stop()

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(this.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill").costs[0])
            else:
                timers("generate").costs.append(this.num_layers * batch_cost)

    def generation_loop_debug_multi_batch(self, this):
        execute_num_batches = 20
        batch_ct = 0
        pbar = tqdm(total=execute_num_batches)
        timers("prefill").reset()
        timers("decoding_gpu_batch").reset()

        # Prologue
        for k in range(this.num_gpu_batches):
            this.load_weight(0, 0, k)
        this.load_hidden(0, 0, 0)
        this.sync()

        # Generate
        for i in range(this.execute_gen_len):
            if i == 0:
                timers("prefill").start()
            for k in range(this.num_gpu_batches):
                this.update_attention_mask(i, k)
            for j in range(this.num_layers):
                if i > 0:
                    timers("decoding_gpu_batch").start()
                for k in range(this.num_gpu_batches):
                    this.load_weight(i, j + 1, k)
                    this.load_cache(i, j, k + 1)
                    this.store_hidden(i, j, k - 1)
                    this.load_hidden(i, j, k + 1)
                    this.compute_layer(i, j, k)
                    this.store_cache(i, j, k - 1)
                    this.sync()

                if i > 0:
                    timers("decoding_gpu_batch").stop()
                    pbar.update(1)
                    batch_ct += 1
                if batch_ct >= execute_num_batches:
                    break
            if batch_ct >= execute_num_batches:
                break
            if i == 0:
                timers("prefill").stop()

        # Convert "decoding_gpu_batch" timer to "generate" timer
        batch_cost = np.mean(timers("decoding_gpu_batch").costs[10:])
        for i in range(this.execute_gen_len):
            if i == 0:
                timers("generate").costs.append(timers("prefill").costs[0])
            else:
                timers("generate").costs.append(this.num_layers * batch_cost)
