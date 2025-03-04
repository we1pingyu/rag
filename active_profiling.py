import time
import threading
import numpy as np
import psutil
import torch
import sys
import xgboost as xgb
import numpy as np
import gc
import os
import shutil
import random
import json
import glob
import math
from scipy.optimize import linprog
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from utils import batch_query, batch_generate_responses, split_batch, init_dynagen_model
from pymilvus import Partition
from sklearn.model_selection import train_test_split
from itertools import product

learning_samples = 10


class ActiveProfilingProcessor:
    def __init__(
        self,
        questions: List[Dict],
        embedding_model,
        model_name,
        tokenizer,
        collection,
        partition_size_gb: float,
        partition_names: List[str],
        total_cpu_gb: int = 128,
        gpu_memory_gb: int = 24,
        safety_margin: float = 0.9,
    ):
        self.questions = questions
        self.embedding_model = embedding_model
        self.model_name = model_name
        self.model, self.model_config, self.env = init_dynagen_model(model_name, tokenizer, [0, 50, 0, 50])
        self.tokenizer = tokenizer
        self.collection = collection
        self.partition_names = partition_names
        self.gpu_memory_gb = gpu_memory_gb
        self.total_cpu_gb = total_cpu_gb * safety_margin  # Leave some swap space
        self.partition_size_gb = partition_size_gb
        self.loaded_partitions = set()
        self.prev_gen_time = 3000

        # Get model requirements
        self.total_weight_gb = self.model_config.model_bytes() / (1024**3)
        self.compute_weight_gb = self.total_weight_gb / self.model_config.num_hidden_layers
        print(f"Model weight size: {self.total_weight_gb:.2f} GB")
        print(f"Estimated cache size per batch: {self.estimate_cache_size(1):.2f} GB")
        print(f"Estimated hidden size per batch: {self.estimate_hidden_size(1):.2f} GB")

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in GB"""
        process = psutil.Process()
        memory_info = process.memory_info()
        gpu_memory = torch.cuda.memory_allocated() / (1024**3)
        return {"cpu_used": memory_info.rss / (1024**3), "gpu_used": gpu_memory}

    def estimate_hidden_size(self, batch_size: int) -> float:
        return self.model_config.hidden_bytes(batch_size, 512 + 16) / (1024**3)

    def estimate_cache_size(self, batch_size: int, layers=1) -> float:
        """Estimate cache size in GB for given batch size"""
        return self.model_config.cache_bytes(batch_size, 512 + 16, layers) / (1024**3)

    def estimate_attention_working_memory(self, batch_size: int) -> float:
        """Estimate attention mechanism working memory in GB for given batch size

        This includes:
        - QKV projections
        - Attention matrices
        - Output projections
        """
        input_dim = self.model_config.input_dim  # h1 in cost_model
        n_head = self.model_config.n_head  # nh in cost_model
        head_dim = input_dim // n_head
        seq_len = 512 + 16  # Current fixed sequence length in the processor

        # Memory in bytes (following cost_model calculations)
        # QKV projections: 3 projections each of size batch_size * seq_len * input_dim * 2 bytes (assuming float16)
        qkv_memory = 3 * batch_size * seq_len * input_dim * 2

        # Attention calculation:
        # - Q, K, V split into heads: 3 * batch_size * n_head * seq_len * head_dim * 2 bytes
        # - Attention scores: batch_size * n_head * seq_len * seq_len * 2 bytes
        # - Attention output: batch_size * seq_len * input_dim * 2 bytes
        attention_memory = (
            (3 * batch_size * n_head * seq_len * head_dim * 2)
            + (batch_size * n_head * seq_len * seq_len * 2)
            + (batch_size * seq_len * input_dim * 2)
        )

        # Output projection: batch_size * seq_len * input_dim * 2 bytes
        output_memory = batch_size * seq_len * input_dim * 2

        total_memory_bytes = qkv_memory + attention_memory + output_memory
        return total_memory_bytes / (1024**3)  # Convert to GB

    def estimate_mlp_working_memory(self, batch_size: int) -> float:
        """Estimate MLP layer working memory in GB for given batch size

        This includes intermediate activations in feed-forward networks
        """
        input_dim = self.model_config.input_dim  # h1 in cost_model
        intermediate_size = self.model_config.intermediate_size  # h2 in cost_model
        seq_len = 512 + 16  # Current fixed sequence length

        # Following cost_model calculations:
        # - First projection (input_dim to intermediate_size): batch_size * seq_len * intermediate_size * 2 bytes
        mlp1_memory = batch_size * seq_len * intermediate_size * 2

        # - Second projection (intermediate_size to input_dim): batch_size * seq_len * input_dim * 2 bytes
        mlp2_memory = batch_size * seq_len * input_dim * 2

        # - Intermediate activations: batch_size * seq_len * intermediate_size * 2 bytes
        activation_memory = batch_size * seq_len * intermediate_size * 2

        total_memory_bytes = mlp1_memory + mlp2_memory + activation_memory
        return total_memory_bytes / (1024**3)  # Convert to GB

    def estimate_total_working_memory(self, batch_size: int) -> float:
        """Estimate total working memory needed for computation, including attention and MLP states"""
        attention_memory = self.estimate_attention_working_memory(batch_size)
        mlp_memory = self.estimate_mlp_working_memory(batch_size)

        # Add some overhead for other operations
        overhead_factor = 1.1
        return (attention_memory + mlp_memory) * overhead_factor

    # Enhanced gpu_memory_available method
    def gpu_memory_available(self, batch_size) -> float:
        """Calculate available GPU memory after accounting for model and computation memory needs

        Returns the amount of available GPU memory in GB after allocating memory for:
        - Model weights on GPU
        - KV Cache on GPU
        - Hidden states
        - Working memory for attention mechanism
        - Working memory for MLP layers
        """
        batch_size_split, _ = split_batch(batch_size)
        # Original computations
        weights_memory = self.compute_weight_gb * 2  # Compute weight allocation
        cache_memory = 2 * self.estimate_cache_size(batch_size_split)  # KV cache
        hidden_memory = self.estimate_hidden_size(batch_size)  # Hidden states

        # New working memory computations
        working_memory = self.estimate_total_working_memory(batch_size_split)

        # Total GPU memory needed with proper safety margins
        total_needed = weights_memory + cache_memory + hidden_memory + working_memory

        # Return available GPU memory after allocating for computation
        return self.gpu_memory_gb - 1.1 * total_needed

    def update_resident_partitions(self, new_resident_partitions: int):
        """Update number of resident partitions"""
        total_partitions = len(self.partition_names)
        new_resident_partitions = min(new_resident_partitions, total_partitions)

        new_loaded = set(range(new_resident_partitions))
        to_release = self.loaded_partitions - new_loaded
        to_load = new_loaded - self.loaded_partitions

        for partition_idx in to_release:
            if partition_idx < total_partitions:
                partition = Partition(self.collection, f"partition_{partition_idx}")
                partition.release()

        valid_partitions = [f"partition_{i}" for i in to_load if i < total_partitions]
        if valid_partitions:
            self.collection.load(partition_names=valid_partitions)

        self.loaded_partitions = new_loaded
        return new_resident_partitions

    def try_configuration(
        self,
        batch_size: int,
        cache_gpu_percent: int,
        cache_cpu_percent: int,
        w_gpu_percent: int,
        w_cpu_percent: int,
        resident_partitions: int,
        num_test_batches: int = 1,
    ) -> Optional[Dict]:
        """Try a specific configuration"""
        thread_exceptions = []

        def exception_handler(args):
            if isinstance(args.exc_value, (OSError, RuntimeError)) and (
                "Cannot allocate memory" in str(args.exc_value) or "[Errno 12]" in str(args.exc_value)
            ):
                thread_exceptions.append(args)

        threading.excepthook = exception_handler

        try:
            # if os.path.exists("./dynagen_offload_dir"):
            #     shutil.rmtree("./dynagen_offload_dir")
            #     os.makedirs("./dynagen_offload_dir")
            batch_size_split, num_batch = split_batch(batch_size)
            print(f"Available GPU memory: {self.gpu_memory_available(batch_size):.2f} GB")
            print("Splitting batch into batches:", batch_size_split, ", num_batch: ", num_batch)
            self.model.update_policy(cache_gpu_percent, cache_cpu_percent, batch_size_split, num_batch)
            self.model.update_weight(w_gpu_percent, w_cpu_percent)
            actual_resident_partitions = self.update_resident_partitions(resident_partitions)

            timing_stats = {"query_times": [], "generation_times": [], "total_times": []}

            for i in range(num_test_batches):
                if thread_exceptions:
                    print(f"Child thread OOM detected: {thread_exceptions[0].exc_value}")
                    return {"error": "cpu_oom"}

                start_idx = i * batch_size
                if start_idx >= len(self.questions):
                    break

                batch = self.questions[start_idx : start_idx + batch_size]
                query_texts = [q["question"] for q in batch]
                query_embeddings = self.embedding_model.embed_documents(query_texts)

                query_start = time.time()
                batch_results, _ = batch_query(
                    collection=self.collection,
                    questions=batch,
                    query_texts=query_texts,
                    query_embeddings=query_embeddings,
                    partition_names=self.partition_names,
                    resident_partitions=actual_resident_partitions,
                )
                query_time = time.time() - query_start

                if thread_exceptions:
                    print(f"Child thread OOM detected: {thread_exceptions[0].exc_value}")
                    return {"error": "cpu_oom"}

                # Set timeout to 2x the previous generation time
                timeout = self.prev_gen_time * 2  # Min 10s, max 300s
                gen_start = time.time()
                generation_result = batch_generate_responses(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    batch_results=batch_results,
                    max_new_tokens=16,
                    batch_size=batch_size,
                    env=self.env,
                )
                gen_time = time.time() - gen_start
                cache_files = glob.glob("./dynagen_offload_dir/t_*")
                for file_path in cache_files:
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        continue
                print(f"delete {len(cache_files)} files on disk")
                if generation_result[1] == "timeout":
                    print(f"Generation timeout detected (>{timeout:.2f}s)")
                    return {"error": "cpu_oom"}
                elif "out of memory" in generation_result[1]:
                    return {"error": "gpu_oom"}

                responses, _ = generation_result

                # Update previous generation time for next batch
                self.prev_gen_time = gen_time

                timing_stats["query_times"].append(query_time)
                timing_stats["generation_times"].append(gen_time)
                timing_stats["total_times"].append(query_time + gen_time)

            if thread_exceptions:
                print(f"Child thread OOM detected: {thread_exceptions[0].exc_value}")
                return {"error": "cpu_oom"}

            avg_query_time = np.mean(timing_stats["query_times"])
            avg_gen_time = np.mean(timing_stats["generation_times"])
            avg_total_time = np.mean(timing_stats["total_times"])

            memory_usage = self.get_memory_usage()

            return {
                "batch_size": batch_size,
                "cache_gpu_percent": cache_gpu_percent,
                "cache_cpu_percent": cache_cpu_percent,
                "w_gpu_percent": w_gpu_percent,
                "w_cpu_percent": w_cpu_percent,
                "resident_partitions": actual_resident_partitions,
                "avg_query_time": avg_query_time,
                "avg_gen_time": avg_gen_time,
                "avg_total_time": avg_total_time,
                "memory_usage": memory_usage,
                "success": True,
            }

        except (RuntimeError, torch.cuda.OutOfMemoryError, OSError, Exception) as e:
            error_msg = str(e)

            if isinstance(e, torch.cuda.OutOfMemoryError) or "CUDA out of memory" in error_msg:
                print(f"GPU OOM detected: {error_msg}")
                return {"error": "gpu_oom"}
            elif (
                "Cannot allocate memory" in error_msg
                or "[Errno 12]" in error_msg
                or "Resource temporarily unavailable" in error_msg
                or "libgomp" in error_msg
            ):
                print(f"CPU OOM detected: {error_msg}")
                return {"error": "cpu_oom"}
            elif "Exception in thread" in error_msg and "Cannot allocate memory" in error_msg:
                print(f"Thread-related CPU OOM detected: {error_msg}")
                return {"error": "cpu_oom"}
            else:
                print(f"Other error detected: {error_msg}")
                return {"error": "other", "message": error_msg}

        finally:
            threading.excepthook = sys.__excepthook__

    def calculate_memory_distribution(
        self, batch_size: int, available_gpu_memory: float, available_cpu_memory: float
    ) -> Dict[str, int]:
        """Calculate memory distribution across tiers"""
        cache_size = self.estimate_cache_size(batch_size, self.model_config.num_hidden_layers)
        # Initialize distribution
        distribution = {
            "w_gpu_percent": 0,
            "w_cpu_percent": 0,
            "cache_gpu_percent": 0,
            "cache_cpu_percent": 0,
            "resident_partitions": 0,
        }

        # First priority: Model weights on GPU
        if available_gpu_memory >= self.total_weight_gb:
            distribution["w_gpu_percent"] = 100
            available_gpu_memory -= self.total_weight_gb
        else:
            # Split weights between GPU, CPU and disk
            gpu_weight_percent = int((available_gpu_memory / self.total_weight_gb) * 100)
            distribution["w_gpu_percent"] = gpu_weight_percent

            # Calculate how much of remaining weights can fit in CPU
            remaining_weight_gb = self.total_weight_gb * (100 - gpu_weight_percent) / 100
            if available_cpu_memory >= remaining_weight_gb:
                # Can fit all remaining weights in CPU
                distribution["w_cpu_percent"] = 100 - gpu_weight_percent
                available_cpu_memory -= remaining_weight_gb
            else:
                # Can only fit part of remaining weights in CPU
                cpu_weight_percent = int((available_cpu_memory / self.total_weight_gb) * 100)
                distribution["w_cpu_percent"] = cpu_weight_percent
                # The rest will implicitly go to disk (100 - gpu - cpu)
                available_cpu_memory = 0

            available_gpu_memory = 0

        # Second priority: Cache on GPU if space available
        if available_gpu_memory > 0:
            possible_gpu_cache_percent = int((available_gpu_memory / cache_size) * 100)
            distribution["cache_gpu_percent"] = min(100, possible_gpu_cache_percent)
            if distribution["cache_gpu_percent"] < 100:
                distribution["cache_cpu_percent"] = min(
                    100 - distribution["cache_gpu_percent"], int((available_cpu_memory / cache_size) * 100)
                )
                available_cpu_memory -= cache_size * distribution["cache_cpu_percent"] / 100
        elif available_cpu_memory > 0:
            possible_cpu_cache_percent = int((available_cpu_memory / cache_size) * 100)
            distribution["cache_cpu_percent"] = min(100, possible_cpu_cache_percent)
            available_cpu_memory -= cache_size * distribution["cache_cpu_percent"] / 100
        else:
            distribution["cache_cpu_percent"] = 0

        # Finally, use remaining CPU memory for resident partitions
        distribution["resident_partitions"] = max(0, int(available_cpu_memory / self.partition_size_gb))
        
        # New constraint: Check if disk storage exceeds 50% of total
        weight_on_disk_percent = 100 - distribution["w_gpu_percent"] - distribution["w_cpu_percent"]
        cache_on_disk_percent = 100 - distribution["cache_gpu_percent"] - distribution["cache_cpu_percent"]
        weight_on_disk = (weight_on_disk_percent / 100) * self.total_weight_gb
        cache_on_disk = (cache_on_disk_percent / 100) * cache_size
        total_on_disk = weight_on_disk + cache_on_disk
        max_allowed_on_disk = 0.3 * (self.total_weight_gb + cache_size)
        
        if total_on_disk > max_allowed_on_disk:
            print(f"Warning: Disk storage constraint violated. Adjusting configuration.")
            print(f"Current disk usage: {total_on_disk:.2f} GB, Max allowed: {max_allowed_on_disk:.2f} GB")
            
            # We need to reduce disk usage by moving more to GPU/CPU
            excess_disk_gb = total_on_disk - max_allowed_on_disk
            
            # Try to move weight from disk to CPU/GPU first
            if weight_on_disk_percent > 0:
                # How much weight percentage needs to be moved from disk (as percentage of total weight)
                weight_to_move_percent = min(weight_on_disk_percent, 
                                        int(excess_disk_gb / self.total_weight_gb * 100))
                
                # Attempt to move to CPU first
                cpu_capacity_percent = max(0, 100 - distribution["w_cpu_percent"])
                weight_to_cpu_percent = min(weight_to_move_percent, cpu_capacity_percent)
                distribution["w_cpu_percent"] += weight_to_cpu_percent
                weight_to_move_percent -= weight_to_cpu_percent
                
                # If needed, try to move remaining weight to GPU
                if weight_to_move_percent > 0:
                    gpu_capacity_percent = max(0, 100 - distribution["w_gpu_percent"])
                    weight_to_gpu_percent = min(weight_to_move_percent, gpu_capacity_percent)
                    distribution["w_gpu_percent"] += weight_to_gpu_percent
            
            # Recalculate disk usage after weight adjustments
            weight_on_disk_percent = 100 - distribution["w_gpu_percent"] - distribution["w_cpu_percent"]
            weight_on_disk = (weight_on_disk_percent / 100) * self.total_weight_gb
            total_on_disk = weight_on_disk + cache_on_disk
            
            # If still exceeding limit, try to move cache from disk to CPU/GPU
            if total_on_disk > max_allowed_on_disk and cache_on_disk_percent > 0:
                excess_disk_gb = total_on_disk - max_allowed_on_disk
                cache_to_move_percent = min(cache_on_disk_percent, 
                                        int(excess_disk_gb / cache_size * 100))
                
                # Attempt to move to CPU first
                cpu_capacity_percent = max(0, 100 - distribution["cache_cpu_percent"])
                cache_to_cpu_percent = min(cache_to_move_percent, cpu_capacity_percent)
                distribution["cache_cpu_percent"] += cache_to_cpu_percent
                cache_to_move_percent -= cache_to_cpu_percent
                
                # If needed, try to move remaining cache to GPU
                if cache_to_move_percent > 0:
                    gpu_capacity_percent = max(0, 100 - distribution["cache_gpu_percent"])
                    cache_to_gpu_percent = min(cache_to_move_percent, gpu_capacity_percent)
                    distribution["cache_gpu_percent"] += cache_to_gpu_percent
            
            # Final check to verify we've met the constraint
            weight_on_disk_percent = 100 - distribution["w_gpu_percent"] - distribution["w_cpu_percent"]
            cache_on_disk_percent = 100 - distribution["cache_gpu_percent"] - distribution["cache_cpu_percent"]
            weight_on_disk = (weight_on_disk_percent / 100) * self.total_weight_gb
            cache_on_disk = (cache_on_disk_percent / 100) * cache_size
            total_on_disk = weight_on_disk + cache_on_disk
            
            print(f"After adjustment: Disk usage: {total_on_disk:.2f} GB, Max allowed: {max_allowed_on_disk:.2f} GB")
            if total_on_disk > max_allowed_on_disk:
                print("Warning: Could not fully satisfy disk storage constraint.")
        
        return distribution
    def find_optimal_config_with_model(
        self,
        batch_size: int,
        inf_model,
        query_model,
        search_space_size: int = 500,  # Kept for backward compatibility but not used
    ):
        """Find optimal configuration for a given batch size using linear programming approach"""
        if inf_model is None or query_model is None:
            return None

        # Check GPU memory capacity (after accounting for compute space)
        batch_size_split, _ = split_batch(batch_size)
        if self.gpu_memory_available(batch_size) <= 0:
            print(f"Batch size {batch_size} exceeds GPU compute space")
            return None

        # Get the linear regression coefficients
        inf_coef = inf_model.coef_
        inf_intercept = inf_model.intercept_
        query_coef = query_model.coef_
        query_intercept = query_model.intercept_

        # Define variables for linear programming
        # x[0] = w_gpu_percent
        # x[1] = w_cpu_percent
        # x[2] = cache_gpu_percent
        # x[3] = cache_cpu_percent
        # x[4] = resident_partitions
        # x[5] = maximum latency (we want to minimize this)

        # Our objective is to minimize the maximum latency
        c = [0, 0, 0, 0, 0, 1]

        # Define constraints for the optimization problem

        # Cache size in GB for the given batch size
        cache_size = self.estimate_cache_size(batch_size, self.model_config.num_hidden_layers)

        # Hidden state size in GB for the given batch size
        hidden_size = self.estimate_hidden_size(batch_size)

        # 1. Inference latency <= max_latency
        # inf_latency = inf_coef[0]*w_gpu + inf_coef[1]*w_cpu + inf_coef[2]*cache_gpu*batch_size +
        #               inf_coef[3]*cache_cpu*batch_size + inf_coef[4]*batch_size +
        #               inf_coef[5]*log(batch_size) + inf_intercept <= max_latency
        #
        # Rearranged: inf_coef[0]*w_gpu + inf_coef[1]*w_cpu + inf_coef[2]*cache_gpu*batch_size +
        #             inf_coef[3]*cache_cpu*batch_size - max_latency <= -inf_coef[4]*batch_size -
        #             inf_coef[5]*log(batch_size) - inf_intercept
        A_inf_latency = [
            inf_coef[0],
            inf_coef[1],
            inf_coef[2] * batch_size,
            inf_coef[3] * batch_size,
            0,  # resident_partitions not used in inference model
            -1,  # -max_latency
        ]
        b_inf_latency = -inf_coef[4] * batch_size - inf_coef[5] * math.log(batch_size) - inf_intercept

        # 2. Query latency <= max_latency
        # query_latency = query_coef[0]*resident_partitions + query_coef[1]*batch_size + query_intercept <= max_latency
        # Rearranged: query_coef[0]*resident_partitions - max_latency <= -query_coef[1]*batch_size - query_intercept
        A_query_latency = [0, 0, 0, 0, query_coef[0], -1]  # Only resident_partitions and max_latency
        b_query_latency = -query_coef[1] * batch_size - query_intercept

        # 3. Weight distribution constraint: w_gpu + w_cpu <= 100
        A_weight_dist = [1, 1, 0, 0, 0, 0]
        b_weight_dist = 100

        # 4. Cache distribution constraint: cache_gpu + cache_cpu <= 100
        A_cache_dist = [0, 0, 1, 1, 0, 0]
        b_cache_dist = 100

        # 5. GPU memory constraint: (w_gpu*self.total_weight_gb/100) + (cache_gpu*cache_size/100) + hidden_size <= gpu_available
        gpu_available = self.gpu_memory_available(batch_size)
        A_gpu_mem = [self.total_weight_gb / 100, 0, cache_size / 100, 0, 0, 0]
        b_gpu_mem = gpu_available - hidden_size

        # 6. CPU memory constraint: (w_cpu*self.total_weight_gb/100) + (cache_cpu*cache_size/100) +
        #                          (resident_partitions*self.partition_size_gb) <= self.total_cpu_gb
        A_cpu_mem = [0, self.total_weight_gb / 100, 0, cache_size / 100, self.partition_size_gb, 0]
        b_cpu_mem = self.total_cpu_gb

        # 7. NEW CONSTRAINT: cache_gpu_percent + cache_cpu_percent >= 20
        # Rearranged: -cache_gpu_percent - cache_cpu_percent <= -20
        A_cache_min = [0, 0, -1, -1, 0, 0]
        b_cache_min = -40

        # 8. NEW CONSTRAINT: w_gpu_percent + w_cpu_percent >= 20
        # Rearranged: -w_gpu_percent - w_cpu_percent <= -20
        A_weight_min = [-1, -1, 0, 0, 0, 0]
        b_weight_min = -40
        
        # 9. NEW CONSTRAINT: Weight and cache on disk <= 50% of total weight and cache
        # Disk weight = (100 - w_gpu - w_cpu) * total_weight_gb / 100
        # Disk cache = (100 - cache_gpu - cache_cpu) * cache_size / 100
        # Constraint: Disk weight + Disk cache <= 0.5 * (total_weight_gb + cache_size)
        # Expanded: (100 - w_gpu - w_cpu) * total_weight_gb / 100 + (100 - cache_gpu - cache_cpu) * cache_size / 100 <= 0.5 * (total_weight_gb + cache_size)
        # Simplified: -w_gpu * total_weight_gb/100 - w_cpu * total_weight_gb/100 - cache_gpu * cache_size/100 - cache_cpu * cache_size/100 <= 
        #             0.5 * (total_weight_gb + cache_size) - total_weight_gb - cache_size
        # Further simplified: -w_gpu * total_weight_gb/100 - w_cpu * total_weight_gb/100 - cache_gpu * cache_size/100 - cache_cpu * cache_size/100 <= 
        #                     -0.5 * (total_weight_gb + cache_size)
        A_disk_limit = [
            -self.total_weight_gb / 100,  # w_gpu_percent coefficient
            -self.total_weight_gb / 100,  # w_cpu_percent coefficient
            -cache_size / 100,            # cache_gpu_percent coefficient
            -cache_size / 100,            # cache_cpu_percent coefficient
            0,                            # resident_partitions coefficient
            0,                            # max_latency coefficient
        ]
        b_disk_limit = -0.3 * (self.total_weight_gb + cache_size)

        # Combine all constraints
        A_ub = np.array(
            [
                A_inf_latency,
                A_query_latency,
                A_weight_dist,
                A_cache_dist,
                A_gpu_mem,
                A_cpu_mem,
                A_cache_min,
                A_weight_min,
                A_disk_limit,  # Added the new disk storage constraint
            ]
        )
        b_ub = np.array(
            [
                b_inf_latency,
                b_query_latency,
                b_weight_dist,
                b_cache_dist,
                b_gpu_mem,
                b_cpu_mem,
                b_cache_min,
                b_weight_min,
                b_disk_limit,  # Added the new disk storage constraint limit
            ]
        )

        # Variable bounds
        # w_gpu_percent, w_cpu_percent, cache_gpu_percent, cache_cpu_percent >= 0
        # resident_partitions >= 0 and <= len(self.partition_names)
        # max_latency >= 0
        bounds = [
            (0, 100),  # w_gpu_percent
            (0, 100),  # w_cpu_percent
            (0, 100),  # cache_gpu_percent
            (0, 100),  # cache_cpu_percent
            (0, len(self.partition_names)),  # resident_partitions
            (0, None),  # max_latency (no upper bound)
        ]

        # Solve the linear programming problem
        try:
            result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

            if result.success:
                # Round solution values to integers
                w_gpu_percent = round(result.x[0])
                w_cpu_percent = round(result.x[1])
                cache_gpu_percent = round(result.x[2])
                cache_cpu_percent = round(result.x[3])
                resident_partitions = round(result.x[4])
                max_latency = result.x[5]

                # Double-check the disk storage constraint is satisfied
                weight_on_disk_percent = 100 - w_gpu_percent - w_cpu_percent
                cache_on_disk_percent = 100 - cache_gpu_percent - cache_cpu_percent
                weight_on_disk = (weight_on_disk_percent / 100) * self.total_weight_gb
                cache_on_disk = (cache_on_disk_percent / 100) * cache_size
                total_on_disk = weight_on_disk + cache_on_disk
                max_allowed_on_disk = 0.5 * (self.total_weight_gb + cache_size)
                
                if total_on_disk > max_allowed_on_disk + 0.1:  # Add small tolerance for floating point errors
                    print(f"Warning: Disk storage constraint violated. Total on disk: {total_on_disk:.2f} GB, Max allowed: {max_allowed_on_disk:.2f} GB")
                    # Adjust the solution to meet the constraint
                    # This is a simplistic approach - we could use a more sophisticated method if needed
                    if weight_on_disk > 0:
                        additional_weight_needed = min(weight_on_disk_percent, 
                                                round((total_on_disk - max_allowed_on_disk) / self.total_weight_gb * 100))
                        # First try to put more weight on CPU
                        if w_cpu_percent + additional_weight_needed <= 100:
                            w_cpu_percent += additional_weight_needed
                        # If that's not enough, try GPU
                        else:
                            remaining = additional_weight_needed - (100 - w_cpu_percent)
                            w_cpu_percent = 100
                            w_gpu_percent += remaining
                    
                    # If we still haven't met the constraint, adjust cache percentages
                    weight_on_disk_percent = 100 - w_gpu_percent - w_cpu_percent
                    weight_on_disk = (weight_on_disk_percent / 100) * self.total_weight_gb
                    total_on_disk = weight_on_disk + cache_on_disk
                    
                    if total_on_disk > max_allowed_on_disk + 0.1 and cache_on_disk > 0:
                        additional_cache_needed = min(cache_on_disk_percent,
                                                round((total_on_disk - max_allowed_on_disk) / cache_size * 100))
                        # First try to put more cache on CPU
                        if cache_cpu_percent + additional_cache_needed <= 100:
                            cache_cpu_percent += additional_cache_needed
                        # If that's not enough, try GPU
                        else:
                            remaining = additional_cache_needed - (100 - cache_cpu_percent)
                            cache_cpu_percent = 100
                            cache_gpu_percent += remaining

                # Predict latencies using the optimized configuration
                X_inf = np.array(
                    [
                        [
                            w_gpu_percent,
                            w_cpu_percent,
                            cache_gpu_percent * batch_size,
                            cache_cpu_percent * batch_size,
                            batch_size,
                            math.log(batch_size),
                        ]
                    ]
                )

                X_query = np.array([[resident_partitions, batch_size]])

                predicted_inf_latency = inf_model.predict(X_inf)[0]
                predicted_query_latency = query_model.predict(X_query)[0]

                best_config = {
                    "batch_size": int(batch_size),
                    "cache_gpu_percent": int(cache_gpu_percent),
                    "cache_cpu_percent": int(cache_cpu_percent),
                    "w_gpu_percent": int(w_gpu_percent),
                    "w_cpu_percent": int(w_cpu_percent),
                    "resident_partitions": int(resident_partitions),
                    "predicted_inf_latency": float(predicted_inf_latency),
                    "predicted_query_latency": float(predicted_query_latency),
                    "max_latency": float(max_latency),
                }

                os.makedirs(f"{self.model_name}_data", exist_ok=True)

                # Save the optimal configuration for this batch size
                config_file = f"{self.model_name}_data/optimal_config_batch{batch_size}.json"
                all_configs_file = f"{self.model_name}_data/all_optimal_configs.json"

                with open(config_file, "w") as f:
                    json.dump(best_config, f, indent=4)

                # Update the record of all optimal configurations
                all_configs = []
                if os.path.exists(all_configs_file):
                    with open(all_configs_file, "r") as f:
                        try:
                            all_configs = json.load(f)
                        except:
                            all_configs = []

                # Check if a configuration for this batch size already exists
                exists = False
                for i, config in enumerate(all_configs):
                    if config.get("batch_size") == batch_size:
                        all_configs[i] = best_config
                        exists = True
                        break

                if not exists:
                    all_configs.append(best_config)

                # Sort configurations by batch size
                all_configs.sort(key=lambda x: x.get("batch_size", 0))

                with open(all_configs_file, "w") as f:
                    json.dump(all_configs, f, indent=4)

                print(f"Saved optimal configuration for batch size {batch_size} to {config_file}")

                return best_config
            else:
                print(f"Optimization failed: {result.message}")
                return None
        except Exception as e:
            print(f"Error in optimization: {e}")

            # Fallback to a simple feasible solution if optimization fails
            print("Falling back to default configuration...")

            # Calculate a basic memory distribution
            distribution = self.calculate_memory_distribution(
                batch_size, self.gpu_memory_available(batch_size), self.total_cpu_gb
            )

            # Create a fallback configuration
            X_inf = np.array(
                [
                    [
                        distribution["w_gpu_percent"],
                        distribution["w_cpu_percent"],
                        distribution["cache_gpu_percent"] * batch_size,
                        distribution["cache_cpu_percent"] * batch_size,
                        batch_size,
                        math.log(batch_size),
                    ]
                ]
            )

            X_query = np.array([[distribution["resident_partitions"], batch_size]])

            predicted_inf_latency = inf_model.predict(X_inf)[0]
            predicted_query_latency = query_model.predict(X_query)[0]

            return {
                "batch_size": int(batch_size),
                "cache_gpu_percent": int(distribution["cache_gpu_percent"]),
                "cache_cpu_percent": int(distribution["cache_cpu_percent"]),
                "w_gpu_percent": int(distribution["w_gpu_percent"]),
                "w_cpu_percent": int(distribution["w_cpu_percent"]),
                "resident_partitions": int(distribution["resident_partitions"]),
                "predicted_inf_latency": float(predicted_inf_latency),
                "predicted_query_latency": float(predicted_query_latency),
                "max_latency": max(predicted_inf_latency, predicted_query_latency),
            }

    def find_optimal_config(self) -> List[Dict]:
        """Find optimal configuration through profiling with learning-based approach"""
        batch_sizes = [2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256]
        optimal_configs = []
        prev_best_max_time = float("inf")

        # Store training data for the learning model
        training_data = {
            "batch_size": [],
            "cache_gpu_percent": [],
            "cache_cpu_percent": [],
            "w_gpu_percent": [],
            "w_cpu_percent": [],
            "resident_partitions": [],
            "inference_latency": [],
            "query_latency": [],
        }

        # Cost models for latency prediction
        inf_model, query_model = None, None

        for batch_size in batch_sizes:
            print(f"\nProfiling batch_size={batch_size}")

            # Calculate available memory after compute space
            batch_size_split, _ = split_batch(batch_size)
            if self.gpu_memory_available(batch_size) <= 0:
                print(f"Batch size {batch_size} exceeds GPU compute space")
                break

            # Initial distribution - start with a reasonable default
            distribution = self.calculate_memory_distribution(
                batch_size, self.gpu_memory_available(batch_size), self.total_cpu_gb
            )

            best_config = None
            best_max_time = float("inf")
            retries = 0
            max_retries = 10

            # Try getting an optimized config from the model if we have enough data
            if len(training_data["batch_size"]) >= learning_samples:
                # Train the cost models
                inf_model, query_model = self.learning_cost_model(
                    training_data["batch_size"],
                    training_data["cache_gpu_percent"],
                    training_data["cache_cpu_percent"],
                    training_data["w_gpu_percent"],
                    training_data["w_cpu_percent"],
                    training_data["resident_partitions"],
                    training_data["inference_latency"],
                    training_data["query_latency"],
                )

                # Get model-based prediction for this batch size
                predicted_config = self.find_optimal_config_with_model(batch_size, inf_model, query_model)

                # If we have a valid prediction, try it first
                if predicted_config:
                    print("\nTrying model-predicted configuration:")
                    print(f"- Batch size: {predicted_config['batch_size']}")
                    print(f"- Cache GPU%: {predicted_config['cache_gpu_percent']}")
                    print(f"- Cache CPU%: {predicted_config['cache_cpu_percent']}")
                    print(f"- Weight GPU%: {predicted_config['w_gpu_percent']}")
                    print(f"- Weight CPU%: {predicted_config['w_cpu_percent']}")
                    print(f"- Resident partitions: {predicted_config['resident_partitions']}")
                    print(f"- Predicted inference time: {predicted_config['predicted_inf_latency']:.3f}s")
                    print(f"- Predicted query time: {predicted_config['predicted_query_latency']:.3f}s")

                    result = self.try_configuration(
                        batch_size=batch_size,
                        cache_gpu_percent=predicted_config["cache_gpu_percent"],
                        cache_cpu_percent=predicted_config["cache_cpu_percent"],
                        w_gpu_percent=predicted_config["w_gpu_percent"],
                        w_cpu_percent=predicted_config["w_cpu_percent"],
                        resident_partitions=predicted_config["resident_partitions"],
                    )

                    if result.get("success", False):
                        # Store the model's prediction result for comparison
                        max_time = max(result["avg_gen_time"], result["avg_query_time"])
                        if max_time < best_max_time:
                            best_max_time = max_time
                            best_config = result

                        # Add to training data
                        training_data["batch_size"].append(batch_size)
                        training_data["cache_gpu_percent"].append(result["cache_gpu_percent"])
                        training_data["cache_cpu_percent"].append(result["cache_cpu_percent"])
                        training_data["w_gpu_percent"].append(result["w_gpu_percent"])
                        training_data["w_cpu_percent"].append(result["w_cpu_percent"])
                        training_data["resident_partitions"].append(result["resident_partitions"])
                        training_data["inference_latency"].append(result["avg_gen_time"])
                        training_data["query_latency"].append(result["avg_query_time"])

                        # Initialize configurations_tried for tracking already tested configs
                        configurations_tried = set()
                        config_key = (
                            batch_size,
                            result["cache_gpu_percent"],
                            result["cache_cpu_percent"],
                            result["w_gpu_percent"],
                            result["w_cpu_percent"],
                            result["resident_partitions"],
                        )
                        configurations_tried.add(config_key)

                        # Note: We continue with the optimization process to verify if this is truly the best config
                    else:
                        # Initialize configurations_tried for tracking already tested configs
                        configurations_tried = set()

            # If we don't have a model yet or the model prediction failed, use the iterative approach
            else:
                # Initialize configurations_tried for tracking already tested configs
                configurations_tried = set()

            while retries < max_retries:
                # Check if we've already tried this configuration
                config_key = (
                    batch_size,
                    distribution["cache_gpu_percent"],
                    distribution["cache_cpu_percent"],
                    distribution["w_gpu_percent"],
                    distribution["w_cpu_percent"],
                    distribution["resident_partitions"],
                )

                if config_key in configurations_tried:
                    # Skip configurations we've already tried
                    print("Skipping already tried configuration")
                    retries += 1
                    continue

                configurations_tried.add(config_key)

                print("\nTrying configuration:")
                print(f"- Batch size: {batch_size}")
                print(f"- Cache GPU%: {distribution['cache_gpu_percent']}")
                print(f"- Cache CPU%: {distribution['cache_cpu_percent']}")
                print(f"- Weight GPU%: {distribution['w_gpu_percent']}")
                print(f"- Weight CPU%: {distribution['w_cpu_percent']}")
                print(f"- Resident partitions: {distribution['resident_partitions']}")

                result = self.try_configuration(batch_size=batch_size, **distribution)

                if result.get("success", False):
                    # Add to training data
                    training_data["batch_size"].append(batch_size)
                    training_data["cache_gpu_percent"].append(result["cache_gpu_percent"])
                    training_data["cache_cpu_percent"].append(result["cache_cpu_percent"])
                    training_data["w_gpu_percent"].append(result["w_gpu_percent"])
                    training_data["w_cpu_percent"].append(result["w_cpu_percent"])
                    training_data["resident_partitions"].append(result["resident_partitions"])
                    training_data["inference_latency"].append(result["avg_gen_time"])
                    training_data["query_latency"].append(result["avg_query_time"])

                    max_time = max(result["avg_gen_time"], result["avg_query_time"])
                    time_diff_percent = (
                        abs(result["avg_gen_time"] - result["avg_query_time"])
                        / min(result["avg_gen_time"], result["avg_query_time"])
                        * 100
                    )

                    print(f"\nConfiguration successful:")
                    print(f"Generation time: {result['avg_gen_time']:.3f}s")
                    print(f"Query time: {result['avg_query_time']:.3f}s")

                    if max_time < best_max_time:
                        best_max_time = max_time
                        best_config = result
                        cache_size = self.estimate_cache_size(batch_size, self.model_config.num_hidden_layers)

                        # Check if times are balanced
                        if time_diff_percent <= 10:
                            print("Times are balanced, ending batch profiling")
                            break

                        # Adjust distribution based on bottleneck using heuristics
                        if result["avg_gen_time"] > result["avg_query_time"]:
                            print("Generation time is bottleneck, adjusting distribution")
                            # Try to improve generation time
                            if distribution["resident_partitions"] > 0:
                                freed_partitions = min(2, distribution["resident_partitions"])
                                distribution["resident_partitions"] -= freed_partitions
                                freed_memory = freed_partitions * self.partition_size_gb

                                # Use freed memory for weights or cache
                                if distribution["w_gpu_percent"] + distribution["w_cpu_percent"] < 100:
                                    additional_weight = int((freed_memory / self.total_weight_gb) * 100)
                                    distribution["w_cpu_percent"] = min(
                                        100 - distribution["w_gpu_percent"],
                                        distribution["w_cpu_percent"] + additional_weight,
                                    )
                                elif distribution["cache_gpu_percent"] + distribution["cache_cpu_percent"] < 100:
                                    additional_cache = int(freed_memory / cache_size * 100)
                                    distribution["cache_cpu_percent"] = min(
                                        100 - distribution["cache_cpu_percent"],
                                        distribution["cache_cpu_percent"] + additional_cache,
                                    )
                                else:
                                    break
                            else:
                                break
                        else:
                            print("Query time is bottleneck, adjusting distribution")
                            # Try to improve query time
                            cache_size = self.estimate_cache_size(batch_size, self.model_config.num_hidden_layers)
                            if distribution["cache_cpu_percent"] * cache_size / 100 > self.partition_size_gb:
                                reduction = int(self.partition_size_gb / cache_size * 100)
                                distribution["cache_cpu_percent"] -= reduction
                                distribution["resident_partitions"] += 1
                            elif distribution["w_cpu_percent"] * self.total_weight_gb / 100 > self.partition_size_gb:
                                reduction = int(self.partition_size_gb / self.total_weight_gb * 100)
                                distribution["w_cpu_percent"] -= reduction
                                distribution["resident_partitions"] += 1
                    else:
                        break
                else:
                    error_type = result.get("error")
                    cache_size = self.estimate_cache_size(batch_size, self.model_config.num_hidden_layers)
                    available_gpu_memory = (
                        self.gpu_memory_available(batch_size)
                        - distribution["cache_gpu_percent"] * cache_size / 10
                        - distribution["w_gpu_percent"] * self.total_weight_gb / 100
                    )
                    if error_type == "gpu_oom" and available_gpu_memory < 0.3 * self.gpu_memory_gb:
                        print("GPU memory exhausted, retrying with reduced tensors")
                        # Calculate available CPU memory
                        used_cpu_memory = (
                            (self.total_weight_gb * distribution["w_cpu_percent"] / 100)
                            + (cache_size * distribution["cache_cpu_percent"] / 100)
                            + (distribution["resident_partitions"] * self.partition_size_gb)
                        )
                        available_cpu_memory = self.total_cpu_gb - used_cpu_memory

                        # Then try to move cache from GPU first
                        remove_gpu_tensor_gb = 2
                        remove_cpu_tensor_gb = 5
                        if distribution["cache_gpu_percent"] * cache_size / 100 > remove_gpu_tensor_gb:
                            reduction = int(remove_gpu_tensor_gb / cache_size * 100)
                            distribution["cache_gpu_percent"] -= reduction

                            # Calculate how much can fit in CPU
                            if available_cpu_memory >= remove_gpu_tensor_gb:
                                # Can fit in CPU
                                distribution["cache_cpu_percent"] += reduction
                                available_cpu_memory -= remove_gpu_tensor_gb
                            else:
                                # Can only fit part in CPU, rest goes to disk
                                cpu_possible_percent = int((available_cpu_memory / cache_size) * 100)
                                distribution["cache_cpu_percent"] += cpu_possible_percent
                                # Rest implicitly goes to disk
                                available_cpu_memory = 0

                        # Try to move weights if needed
                        elif distribution["w_gpu_percent"] * self.total_weight_gb / 100 > remove_cpu_tensor_gb:
                            reduction = int(remove_cpu_tensor_gb / self.total_weight_gb * 100)
                            distribution["w_gpu_percent"] -= reduction

                            # Calculate how much can fit in CPU
                            if available_cpu_memory >= remove_cpu_tensor_gb:
                                # Can fit in CPU
                                distribution["w_cpu_percent"] += reduction
                                available_cpu_memory -= remove_cpu_tensor_gb
                            else:
                                # Can only fit part in CPU, rest goes to disk
                                cpu_possible_percent = int((available_cpu_memory / self.total_weight_gb) * 100)
                                distribution["w_cpu_percent"] += cpu_possible_percent
                                # Rest implicitly goes to disk
                                available_cpu_memory = 0
                        else:
                            break
                    else:
                        print("CPU memory exhausted, retrying with reduced tensors and partitions")
                        # Reduce CPU usage
                        if best_config and best_config["avg_query_time"] < best_config["avg_gen_time"]:
                            if distribution["resident_partitions"] > 0:
                                distribution["resident_partitions"] -= 1
                            elif distribution["cache_cpu_percent"] * cache_size / 100 > 10:
                                reduction = int(10 / cache_size * 100)
                                distribution["cache_cpu_percent"] -= reduction
                            elif distribution["w_cpu_percent"] * self.total_weight_gb / 100 > 10:
                                reduction = int(10 / self.total_weight_gb * 100)
                                distribution["w_cpu_percent"] -= reduction
                            else:
                                break
                        else:
                            if distribution["cache_cpu_percent"] * cache_size / 100 > 10:
                                reduction = int(10 / cache_size * 100)
                                distribution["cache_cpu_percent"] -= reduction
                            elif distribution["w_cpu_percent"] * self.total_weight_gb / 100 > 10:
                                reduction = int(10 / self.total_weight_gb * 100)
                                distribution["w_cpu_percent"] -= reduction
                            elif distribution["resident_partitions"] > 0:
                                distribution["resident_partitions"] -= 1
                            else:
                                break

                retries += 1

                # Clear CUDA cache before retry
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if best_config:
                optimal_configs.append(best_config)

                # Check if performance degraded too much compared to previous batch
                if optimal_configs and best_max_time > 2 * prev_best_max_time:
                    print(f"\nPerformance degraded too much at batch_size={batch_size}")
                    print(f"Current max time: {best_max_time:.3f}s")
                    print(f"Previous max time: {prev_best_max_time:.3f}s")
                    break

                prev_best_max_time = best_max_time
            else:
                # If no valid configuration found for this batch size
                print(f"\nNo valid configuration found for batch_size={batch_size}")
                break

        return optimal_configs

    def learning_cost_model(
        self,
        batch_size: List[int],
        cache_gpu_percent: List[int],
        cache_cpu_percent: List[int],
        w_gpu_percent: List[int],
        w_cpu_percent: List[int],
        resident_partitions: List[int],
        inference_latency: List[float],
        query_latency: List[float],
    ):
        """Build a cost model to predict inference and query latencies based on configuration parameters"""
        import numpy as np
        from sklearn.linear_model import LinearRegression
        import json
        import os
        import math

        # 创建特征矩阵
        # 推理延迟模型的特征：根据注释中的公式
        # a * w_gpu_percent + b * w_cpu_percent + c * cache_gpu_percent * batch_size +
        # d * cache_cpu_percent * batch_size + e * batch_size + f * log(batch_size) + g
        X_inf = np.column_stack(
            [
                w_gpu_percent,
                w_cpu_percent,
                [cache_gpu_percent[i] * batch_size[i] for i in range(len(batch_size))],
                [cache_cpu_percent[i] * batch_size[i] for i in range(len(batch_size))],
                batch_size,
                [math.log(bs) if bs > 0 else 0 for bs in batch_size],
            ]
        )

        # 查询延迟模型的特征：根据注释中的公式
        # a * resident_partitions + b * batch_size + c
        X_query = np.column_stack([resident_partitions, batch_size])

        # 创建目标向量
        y_inf = np.array(inference_latency)
        y_query = np.array(query_latency)

        # 训练推理延迟模型
        inf_model = LinearRegression()
        inf_model.fit(X_inf, y_inf)

        # 训练查询延迟模型
        query_model = LinearRegression()
        query_model.fit(X_query, y_query)

        # 计算两个模型的MSE
        inf_pred = inf_model.predict(X_inf)
        inf_mse = np.mean((inf_pred - y_inf) ** 2)

        query_pred = query_model.predict(X_query)
        query_mse = np.mean((query_pred - y_query) ** 2)

        print(f"推理模型MSE: {inf_mse:.4f}")
        print(f"查询模型MSE: {query_mse:.4f}")

        # 保存模型参数和数据到人类可读的文件
        os.makedirs(f"{self.model_name}_data", exist_ok=True)

        # 保存推理模型参数
        inf_model_params = {
            "coefficients": inf_model.coef_.tolist(),
            "intercept": float(inf_model.intercept_),
            "features": [
                "w_gpu_percent",
                "w_cpu_percent",
                "cache_gpu_percent * batch_size",
                "cache_cpu_percent * batch_size",
                "batch_size",
                "log(batch_size)",
            ],
            "mse": float(inf_mse),
            "equation": "inference_latency = "
            + f"{inf_model.coef_[0]:.4f} * w_gpu_percent + "
            + f"{inf_model.coef_[1]:.4f} * w_cpu_percent + "
            + f"{inf_model.coef_[2]:.4f} * cache_gpu_percent * batch_size + "
            + f"{inf_model.coef_[3]:.4f} * cache_cpu_percent * batch_size + "
            + f"{inf_model.coef_[4]:.4f} * batch_size + "
            + f"{inf_model.coef_[5]:.4f} * log(batch_size) + "
            + f"{inf_model.intercept_:.4f}",
        }

        with open(f"{self.model_name}_data/inference_model_params.json", "w") as f:
            json.dump(inf_model_params, f, indent=4)

        # 保存查询模型参数
        query_model_params = {
            "coefficients": query_model.coef_.tolist(),
            "intercept": float(query_model.intercept_),
            "features": ["resident_partitions", "batch_size"],
            "mse": float(query_mse),
            "equation": "query_latency = "
            + f"{query_model.coef_[0]:.4f} * resident_partitions + "
            + f"{query_model.coef_[1]:.4f} * batch_size + "
            + f"{query_model.intercept_:.4f}",
        }

        with open(f"{self.model_name}_data/query_model_params.json", "w") as f:
            json.dump(query_model_params, f, indent=4)

        # 保存训练数据
        training_data = {
            "batch_size": batch_size,
            "cache_gpu_percent": cache_gpu_percent,
            "cache_cpu_percent": cache_cpu_percent,
            "w_gpu_percent": w_gpu_percent,
            "w_cpu_percent": w_cpu_percent,
            "resident_partitions": resident_partitions,
            "inference_latency": inference_latency,
            "query_latency": query_latency,
        }

        # 转换列表为JSON可序列化格式
        for key in training_data:
            training_data[key] = [
                float(x) if isinstance(x, (np.integer, np.floating)) else x for x in training_data[key]
            ]

        with open(f"{self.model_name}_data/training_data.json", "w") as f:
            json.dump(training_data, f, indent=4)

        # 保存完整的训练样本供可视化和分析
        samples = []
        for i in range(len(batch_size)):
            samples.append(
                {
                    "batch_size": batch_size[i],
                    "cache_gpu_percent": cache_gpu_percent[i],
                    "cache_cpu_percent": cache_cpu_percent[i],
                    "w_gpu_percent": w_gpu_percent[i],
                    "w_cpu_percent": w_cpu_percent[i],
                    "resident_partitions": resident_partitions[i],
                    "inference_latency": float(inference_latency[i]),
                    "query_latency": float(query_latency[i]),
                    "inference_predicted": float(inf_pred[i]),
                    "query_predicted": float(query_pred[i]),
                    "inference_error": float(inf_pred[i] - inference_latency[i]),
                    "query_error": float(query_pred[i] - query_latency[i]),
                }
            )

        with open(f"{self.model_name}_data/training_samples.json", "w") as f:
            json.dump(samples, f, indent=4)

        return inf_model, query_model

    def run(self):
        """Execute active profiling"""
        print("\nStarting active profiling:")
        optimal_configs = self.find_optimal_config()

        if not optimal_configs:
            print("\nNo valid configurations found during profiling")
            return []

        print("\nOptimal configurations found:")
        for config in optimal_configs:
            print("\nConfiguration:")
            print(f"- Batch size: {config['batch_size']}")
            print(f"- Cache GPU%: {config['cache_gpu_percent']}")
            print(f"- Cache CPU%: {config['cache_cpu_percent']}")
            print(f"- Weight GPU%: {config['w_gpu_percent']}")
            print(f"- Weight CPU%: {config['w_cpu_percent']}")
            print(f"- Resident partitions: {config['resident_partitions']}")
            print(f"- Average query time: {config['avg_query_time']:.3f}s")
            print(f"- Average generation time: {config['avg_gen_time']:.3f}s")
            print(f"- Average total time: {config['avg_total_time']:.3f}s")
            print(f"- Memory usage:")
            print(f"  - CPU: {config['memory_usage']['cpu_used']:.2f} GB")
            print(f"  - GPU: {config['memory_usage']['gpu_used']:.2f} GB")

        self.env.close_copy_threads()
        return optimal_configs
