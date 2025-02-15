import time
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import numpy as np
from utils import batch_query, batch_generate_responses, batch_query_qdrant, Question
from pymilvus import Partition
import psutil
import torch
import resource


class ActiveProfilingProcessor:
    def __init__(
        self,
        questions: List[Dict],
        embedding_model,
        model,
        tokenizer,
        collection,
        partition_names: List[str],
        model_config,
        use_qdrant: bool = False,
        dynagen: bool = False,
        env: Optional[object] = None,
        total_cpu_gb: int = 128,
        gpu_memory_gb: int = 24,
        safety_margin: float = 0.9,
    ):
        # Existing initialization code remains the same
        self.questions = questions
        self.embedding_model = embedding_model
        self.model = model
        self.tokenizer = tokenizer
        self.collection = collection
        self.partition_names = partition_names
        self.model_config = model_config
        self.use_qdrant = use_qdrant
        self.dynagen = dynagen
        self.env = env
        self.total_cpu_gb = total_cpu_gb
        self.gpu_memory_gb = gpu_memory_gb
        self.safety_margin = safety_margin
        self.partition_size_gb = 8.5
        self.results = []
        self.base_time = time.time()
        self.profiling_results = []

        # Add weight size estimation
        self.total_weight_gb = self.model_config.model_bytes() / (1024**3)
        print(f"Estimated model weight size: {self.total_weight_gb:.2f} GB")

    def estimate_memory_requirements(
        self, batch_size: int, cache_gpu_percent: int, w_gpu_percent: int
    ) -> Dict[str, float]:
        """Estimate memory requirements for a given configuration"""
        cache_size_gb = self.estimate_cache_size(batch_size)
        gpu_cache_gb = cache_size_gb * cache_gpu_percent / 100
        gpu_weight_gb = self.total_weight_gb * w_gpu_percent / 100

        total_gpu_gb = gpu_cache_gb + gpu_weight_gb
        return {"total_gpu": total_gpu_gb, "cache_gpu": gpu_cache_gb, "weight_gpu": gpu_weight_gb}

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in GB"""
        process = psutil.Process()
        memory_info = process.memory_info()
        gpu_memory = torch.cuda.memory_allocated() / (1024**3)

        return {"cpu_used": memory_info.rss / (1024**3), "gpu_used": gpu_memory}

    def estimate_cache_size(self, batch_size: int) -> float:
        """Estimate cache size in GB for given batch size"""
        return (self.model_config.cache_bytes(batch_size, 2048)) / (1024**3)

    def update_resident_partitions(self, new_resident_partitions: int):
        """Update number of resident partitions"""
        if self.use_qdrant:
            return

        if not hasattr(self, "loaded_partitions"):
            self.loaded_partitions = set()

        # Ensure we don't exceed available partitions
        total_partitions = len(self.partition_names)
        new_resident_partitions = min(new_resident_partitions, total_partitions)

        new_loaded = set(range(new_resident_partitions))
        to_release = self.loaded_partitions - new_loaded
        to_load = new_loaded - self.loaded_partitions

        if to_release:
            for partition_idx in to_release:
                if partition_idx < total_partitions:  # Safety check
                    partition = Partition(self.collection, f"partition_{partition_idx}")
                    partition.release()

        if to_load:
            valid_partitions = [f"partition_{i}" for i in to_load if i < total_partitions]
            if valid_partitions:
                self.collection.load(partition_names=valid_partitions)

        self.loaded_partitions = new_loaded
        self.resident_partitions = new_resident_partitions

    def try_configuration(
        self,
        batch_size: int,
        cache_gpu_percent: int,
        cache_cpu_percent: int,
        w_gpu_percent: int,
        w_cpu_percent: int,
        resident_partitions: int,
        num_test_batches: int = 1,
        max_retries: int = 1,
    ) -> Optional[Dict]:
        """Try a specific configuration with weight distribution parameters"""
        initial_w_gpu = w_gpu_percent
        initial_cache_cpu = cache_cpu_percent
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Estimate memory requirements
                mem_req = self.estimate_memory_requirements(batch_size, cache_gpu_percent, w_gpu_percent)

                # Check if configuration exceeds GPU memory
                if mem_req["total_gpu"] > self.gpu_memory_gb * self.safety_margin:
                    print(
                        f"Warning: Configuration would exceed GPU memory ({mem_req['total_gpu']:.2f} GB > {self.gpu_memory_gb * self.safety_margin:.2f} GB)"
                    )
                    raise RuntimeError("Predicted GPU memory overflow")

                # Check CPU memory requirements
                estimated_cache_size = self.estimate_cache_size(batch_size)
                cpu_memory_needed = (
                    (estimated_cache_size * cache_cpu_percent / 100)  # Cache on CPU
                    + (self.total_weight_gb * w_cpu_percent / 100)  # Weights on CPU
                    + (resident_partitions * self.partition_size_gb)  # Assuming 2GB per partition
                )

                if cpu_memory_needed > self.total_cpu_gb * self.safety_margin:
                    print(
                        f"Warning: Configuration would exceed CPU memory ({cpu_memory_needed:.2f} GB > {self.total_cpu_gb * self.safety_margin:.2f} GB)"
                    )
                    raise RuntimeError("Predicted CPU memory overflow")

                self.update_resident_partitions(resident_partitions)
                print(f"\nTrying configuration:")
                print(f"- Batch size: {batch_size}")
                print(f"- Cache GPU%: {cache_gpu_percent}")
                print(f"- Cache CPU%: {cache_cpu_percent}")
                print(f"- Cache Disk%: {100 - cache_gpu_percent - cache_cpu_percent}")
                print(f"- Weight GPU%: {w_gpu_percent}")
                print(f"- Weight CPU%: {w_cpu_percent}")
                print(f"- Resident partitions: {resident_partitions}")

                if self.dynagen:
                    self.model.update_policy(cache_gpu_percent, cache_cpu_percent, batch_size)
                    self.model.update_weight(w_gpu_percent, w_cpu_percent)

                # available_cpu_mem = self.total_cpu_gb - self.partition_size_gb * resident_partitions
                # print(f"Available CPU memory: {available_cpu_mem:.2f} GB")
                # resource.setrlimit(
                #     resource.RLIMIT_AS, (int(available_cpu_mem * 1024**3), int(available_cpu_mem * 1024**3))
                # )
                timing_stats = {"query_times": [], "generation_times": [], "total_times": []}

                # Process test batches
                for i in range(num_test_batches):
                    start_idx = i * batch_size
                    if start_idx + batch_size > len(self.questions):
                        break

                    batch = self.questions[start_idx : start_idx + batch_size]
                    query_texts = [q["question"] for q in batch]

                    # Embedding generation
                    query_embeddings = self.embedding_model.embed_documents(query_texts)

                    # Vector search
                    query_start = time.time()
                    if self.use_qdrant:
                        batch_results, _ = batch_query_qdrant(
                            client=self.collection,
                            questions=batch,
                            query_texts=query_texts,
                            query_embeddings=query_embeddings,
                            partition_names=self.partition_names,
                            resident_partitions=resident_partitions,
                        )
                    else:
                        batch_results, _ = batch_query(
                            collection=self.collection,
                            questions=batch,
                            query_texts=query_texts,
                            query_embeddings=query_embeddings,
                            partition_names=self.partition_names,
                            resident_partitions=resident_partitions,
                        )
                    query_time = time.time() - query_start

                    # Response generation
                    gen_start = time.time()
                    responses, _ = batch_generate_responses(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        batch_results=batch_results,
                        max_new_tokens=128,
                        batch_size=batch_size,
                        dynagen=self.dynagen,
                        env=self.env,
                    )
                    gen_time = time.time() - gen_start

                    timing_stats["query_times"].append(query_time)
                    timing_stats["generation_times"].append(gen_time)
                    timing_stats["total_times"].append(query_time + gen_time)

                # Calculate averages
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
                    "resident_partitions": resident_partitions,
                    "avg_query_time": avg_query_time,
                    "avg_gen_time": avg_gen_time,
                    "avg_total_time": avg_total_time,
                    "memory_usage": memory_usage,
                    "success": True,
                }

            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                retry_count += 1
                error_msg = str(e)

                # Handle CPU OOM
                if "Cannot allocate memory" in error_msg or "CPU memory overflow" in error_msg:
                    print("\nCPU OOM detected, adjusting configuration...")

                    # First try reducing resident partitions
                    if resident_partitions > 1:
                        resident_partitions = max(1, resident_partitions - 2)
                        print(f"Reduced resident partitions to {resident_partitions}")
                        continue

                    # Then try offloading more cache to disk
                    if cache_cpu_percent > 0:
                        reduction = min(20, cache_cpu_percent)
                        cache_cpu_percent = max(0, cache_cpu_percent - reduction)
                        print(f"Reduced cache_cpu_percent to {cache_cpu_percent}")
                        continue

                    # Finally try offloading more weights to disk
                    if w_cpu_percent > 0:
                        reduction = min(20, w_cpu_percent)
                        w_cpu_percent = max(0, w_cpu_percent - reduction)
                        print(f"Reduced w_cpu_percent to {w_cpu_percent}")
                        continue

                # Handle GPU OOM
                else:
                    print(f"\nGPU OOM error: {error_msg}")
                    old_w_gpu = w_gpu_percent
                    w_gpu_percent = max(0, w_gpu_percent - 20)
                    w_cpu_percent = min(100, w_cpu_percent + 20)
                    print(f"Reducing w_gpu_percent from {old_w_gpu} to {w_gpu_percent}")

                # If we've exhausted all options or hit max retries
                if (
                    w_gpu_percent == 0 and w_cpu_percent == 0 and cache_cpu_percent == 0 and resident_partitions == 1
                ) or retry_count >= max_retries:
                    return {
                        "batch_size": batch_size,
                        "cache_gpu_percent": cache_gpu_percent,
                        "cache_cpu_percent": initial_cache_cpu,
                        "w_gpu_percent": initial_w_gpu,
                        "w_cpu_percent": 100 - initial_w_gpu,
                        "resident_partitions": resident_partitions,
                        "success": False,
                        "error": error_msg,
                    }

                # Clear CUDA cache before retry
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

    def find_optimal_config(self) -> List[Dict]:
        """Find optimal configuration through active profiling with prioritized memory tier policy

        Key optimizations:
        1. If all tensors can fit in GPU memory, use full GPU configuration and skip lower tiers
        2. When reducing GPU usage, adjust cache_gpu first. Only if cache_gpu=0 still causes OOM,
        then start reducing w_gpu
        3. Skip trying configurations with lower GPU usage if higher GPU usage works and doesn't improve
           max(generation_time, query_time)
        4. Memory tier priority: GPU > CPU > disk, skip lower tiers if higher tier works
        5. Balance generation and query time:
           - If query_time > generation_time: reduce cache_gpu to improve query performance
           - If generation_time > query_time:
             a) Reduce resident_partitions to free up CPU memory
             b) Increase cache_gpu with freed memory to improve generation time
             This works because fewer resident partitions means less CPU memory used for partitions,
             allowing more GPU cache which speeds up generation
        6. Early stopping when optimal balance is found or cannot be improved further
        """
        optimal_configs = []
        batch_sizes = [2, 4, 8, 16, 32]
        MAX_PARTITIONS = 10

        for batch_size in batch_sizes:
            print(f"\nProfiling batch_size={batch_size}")

            # Estimate cache size for current batch size
            estimated_cache_size = self.estimate_cache_size(batch_size)
            max_gpu_memory = self.gpu_memory_gb * self.safety_margin

            # Calculate total GPU memory needed for full GPU configuration
            total_gpu_needed = estimated_cache_size + self.total_weight_gb

            # Try maximum GPU configuration first
            if total_gpu_needed <= max_gpu_memory:
                print("\nAttempting full GPU configuration...")
                result = self.try_configuration(
                    batch_size=batch_size,
                    cache_gpu_percent=100,
                    cache_cpu_percent=0,
                    w_gpu_percent=100,
                    w_cpu_percent=0,
                    resident_partitions=0,
                    max_retries=1,
                )

                if result and result["success"]:
                    print(f"\nFull GPU configuration successful:")
                    print(f"- Query time: {result['avg_query_time']:.3f}s")
                    print(f"- Generation time: {result['avg_gen_time']:.3f}s")
                    optimal_configs.append(result)
                    continue  # Move to next batch size

            # Calculate maximum possible GPU percentages
            max_cache_gpu = min(100, int((max_gpu_memory * 0.5) / estimated_cache_size * 100))
            remaining_gpu = max_gpu_memory - (estimated_cache_size * max_cache_gpu / 100)
            max_w_gpu = min(100, int(remaining_gpu / self.total_weight_gb * 100))

            # Start with maximum possible cache_gpu, keeping w_gpu at max
            cache_gpu = max_cache_gpu
            w_gpu = max_w_gpu
            tried_configs = set()
            best_max_time = float("inf")  # Track the best max(query_time, gen_time)
            best_config = None

            while True:
                cache_cpu = min(
                    100 - cache_gpu, int((self.total_cpu_gb * self.safety_margin) / estimated_cache_size * 100)
                )

                # Calculate maximum resident partitions
                cpu_memory_for_partitions = (
                    self.total_cpu_gb * self.safety_margin
                    - estimated_cache_size * cache_cpu / 100
                    - self.total_weight_gb * (100 - w_gpu) / 100
                )
                max_resident_partitions = min(MAX_PARTITIONS, max(1, int(cpu_memory_for_partitions / 2)))

                config_key = f"{cache_gpu}_{cache_cpu}_{w_gpu}"
                if config_key in tried_configs:
                    if cache_gpu > 0:
                        cache_gpu = max(0, cache_gpu - 20)
                    else:
                        # Stop trying lower w_gpu values since they will only increase generation time
                        if best_config:
                            optimal_configs.append(best_config)
                        break
                    continue

                tried_configs.add(config_key)
                print(f"\nTrying configuration: cache_gpu={cache_gpu}, cache_cpu={cache_cpu}, w_gpu={w_gpu}")

                result = self.try_configuration(
                    batch_size=batch_size,
                    cache_gpu_percent=cache_gpu,
                    cache_cpu_percent=cache_cpu,
                    w_gpu_percent=w_gpu,
                    w_cpu_percent=100 - w_gpu,
                    resident_partitions=max_resident_partitions,
                    max_retries=1,
                )

                if result and result["success"]:
                    current_max_time = max(result["avg_query_time"], result["avg_gen_time"])
                    print(f"\nConfiguration successful:")
                    print(f"- Query time: {result['avg_query_time']:.3f}s")
                    print(f"- Generation time: {result['avg_gen_time']:.3f}s")
                    print(f"- Max time: {current_max_time:.3f}s")

                    if current_max_time < best_max_time:
                        best_max_time = current_max_time
                        best_config = result
                        print("\nNew best configuration found")

                        # If query time dominates, try reducing cache_gpu to improve it
                        if result["avg_query_time"] > result["avg_gen_time"]:
                            cache_gpu = max(0, cache_gpu - 20)
                        # If generation time dominates, try to balance by adjusting resident_partitions and cache_gpu
                        elif result["avg_gen_time"] > result["avg_query_time"]:
                            # Reduce resident partitions to free up CPU memory
                            new_resident_partitions = max(1, max_resident_partitions - 2)

                            # Calculate how much CPU memory we freed up
                            freed_memory = (
                                max_resident_partitions - new_resident_partitions
                            ) * 2  # Assuming 2GB per partition

                            # Convert freed CPU memory to potential cache_gpu increase
                            # Assuming 1:1 conversion ratio for simplicity
                            potential_cache_increase = min(
                                20, int((freed_memory / estimated_cache_size) * 100)  # Standard increment
                            )

                            if cache_gpu + potential_cache_increase <= 100:
                                max_resident_partitions = new_resident_partitions
                                cache_gpu = min(100, cache_gpu + potential_cache_increase)
                                print(f"\nAdjusting for generation time dominance:")
                                print(f"- Reduced resident partitions to: {new_resident_partitions}")
                                print(f"- Increased cache_gpu to: {cache_gpu}")
                            else:
                                # Can't improve further, save best config and stop
                                optimal_configs.append(best_config)
                                break
                        else:
                            # Times are balanced
                            optimal_configs.append(best_config)
                            break
                    else:
                        # Current configuration didn't improve max time
                        # No point trying lower GPU configurations
                        if best_config:
                            optimal_configs.append(best_config)
                        break
                else:
                    # Adjust strategy on failure
                    if cache_gpu > 0:
                        cache_gpu = max(0, cache_gpu - 20)
                    else:
                        w_gpu = max(0, w_gpu - 20)
                        if w_gpu == 0:
                            if best_config:
                                optimal_configs.append(best_config)
                            break

        return optimal_configs

    def run(self):
        """Execute active profiling"""
        print("\nStarting active profiling:")
        optimal_configs = self.find_optimal_config()

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

        return optimal_configs
