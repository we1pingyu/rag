import time
from typing import List, Dict, Optional, Tuple
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import numpy as np
import psutil
import torch
from tqdm import tqdm
from utils import batch_query, batch_generate_responses
from pymilvus import Partition
import sys


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
        total_cpu_gb: int = 128,
        gpu_memory_gb: int = 24,
        safety_margin: float = 0.9,
    ):
        self.questions = questions
        self.embedding_model = embedding_model
        self.model = model
        self.tokenizer = tokenizer
        self.collection = collection
        self.partition_names = partition_names
        self.model_config = model_config
        self.total_cpu_gb = total_cpu_gb * safety_margin
        self.gpu_memory_gb = gpu_memory_gb * safety_margin
        self.partition_size_gb = 9
        self.compute_space_per_batch = self.estimate_cache_size(1)  # GB per batch
        self.loaded_partitions = set()
        self.prev_gen_time = 3000

        # Get model requirements
        self.total_weight_gb = self.model_config.model_bytes() / (1024**3)
        print(f"Model weight size: {self.total_weight_gb:.2f} GB")
        print(f"Estimated cache size per batch: {self.estimate_cache_size(1):.2f} GB")

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in GB"""
        process = psutil.Process()
        memory_info = process.memory_info()
        gpu_memory = torch.cuda.memory_allocated() / (1024**3)
        return {"cpu_used": memory_info.rss / (1024**3), "gpu_used": gpu_memory}

    def generate_with_timeout(self, model, tokenizer, batch_results, max_new_tokens, batch_size, timeout):
        """Generate responses with timeout and proper thread cleanup"""

        def generate():
            return batch_generate_responses(
                model=model,
                tokenizer=tokenizer,
                batch_results=batch_results,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
            )

        # Use a custom Event to signal thread termination
        termination_event = threading.Event()
        result = [None]

        def wrapped_generate():
            try:
                result[0] = generate()
            except Exception as e:
                result[0] = None, f"error: {str(e)}"
            finally:
                termination_event.set()

        # Start generation in a separate thread
        thread = threading.Thread(target=wrapped_generate)
        thread.daemon = True  # Mark as daemon so it won't prevent program exit
        thread.start()

        # Wait for either completion or timeout
        timeout_occurred = not termination_event.wait(timeout=timeout)

        if timeout_occurred:
            print(f"Generation timed out after {timeout:.2f} seconds")
            # Force thread cleanup by clearing model cache
            if hasattr(model, "clear_cache"):
                model.clear_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None, "timeout"

        return result[0]

    def estimate_cache_size(self, batch_size: int) -> float:
        """Estimate cache size in GB for given batch size"""
        return self.model_config.cache_bytes(batch_size, 512) / (1024**3)

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
            self.model.update_policy(cache_gpu_percent, cache_cpu_percent, batch_size)
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
                generation_result = self.generate_with_timeout(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    batch_results=batch_results,
                    max_new_tokens=32,
                    batch_size=batch_size,
                    timeout=timeout,
                )
                gen_time = time.time() - gen_start

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
        cache_size = self.estimate_cache_size(batch_size)

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
        # distribution["resident_partitions"] = max(0, int(available_cpu_memory / self.partition_size_gb))
        distribution["resident_partitions"] = 0
        return distribution

    def find_optimal_config(self) -> List[Dict]:
        """Find optimal configuration through profiling"""
        batch_sizes = [2, 4, 8, 16, 32, 64]
        optimal_configs = []
        prev_best_max_time = float("inf")

        for batch_size in batch_sizes:
            print(f"\nProfiling batch_size={batch_size}")

            # Calculate available memory after compute space
            gpu_memory_available = self.gpu_memory_gb - self.compute_space_per_batch * batch_size
            if gpu_memory_available <= 0:
                print(f"Batch size {batch_size} exceeds GPU compute space")
                break

            # Initial distribution
            distribution = self.calculate_memory_distribution(batch_size, gpu_memory_available, self.total_cpu_gb)

            best_config = None
            best_max_time = float("inf")
            retries = 0
            max_retries = 10

            while retries < max_retries:
                print("\nTrying configuration:")
                print(f"- Batch size: {batch_size}")
                print(f"- Cache GPU%: {distribution['cache_gpu_percent']}")
                print(f"- Cache CPU%: {distribution['cache_cpu_percent']}")
                print(f"- Weight GPU%: {distribution['w_gpu_percent']}")
                print(f"- Weight CPU%: {distribution['w_cpu_percent']}")
                print(f"- Resident partitions: {distribution['resident_partitions']}")

                result = self.try_configuration(batch_size=batch_size, **distribution)

                if result.get("success", False):
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

                        # Check if times are balanced
                        if time_diff_percent <= 10:
                            print("Times are balanced, ending batch profiling")
                            break

                        # Adjust distribution based on bottleneck
                        if result["avg_gen_time"] > result["avg_query_time"]:
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
                                    additional_cache = int((freed_memory / self.estimate_cache_size(batch_size)) * 100)
                                    distribution["cache_cpu_percent"] = min(
                                        100 - distribution["cache_cpu_percent"],
                                        distribution["cache_cpu_percent"] + additional_cache,
                                    )
                                else:
                                    break
                            else:
                                break
                        else:
                            # Try to improve query time
                            if distribution["cache_cpu_percent"] > 0:
                                reduction = min(25, distribution["cache_cpu_percent"])
                                distribution["cache_cpu_percent"] -= reduction
                                freed_memory = self.estimate_cache_size(batch_size) * reduction / 100
                                additional_partitions = int(freed_memory / self.partition_size_gb)
                                distribution["resident_partitions"] += additional_partitions
                            else:
                                break
                    else:
                        break

                else:
                    error_type = result.get("error")
                    available_gpu_memory = (
                        self.gpu_memory_gb
                        - self.compute_space_per_batch * batch_size
                        - distribution["cache_gpu_percent"] * self.estimate_cache_size(batch_size) / 100
                        - distribution["w_gpu_percent"] * self.total_weight_gb / 100
                    )
                    if error_type == "gpu_oom" and available_gpu_memory < 0.2 * self.gpu_memory_gb:
                        # Calculate available CPU memory
                        used_cpu_memory = (
                            (self.total_weight_gb * distribution["w_cpu_percent"] / 100)
                            + (self.estimate_cache_size(batch_size) * distribution["cache_cpu_percent"] / 100)
                            + (distribution["resident_partitions"] * self.partition_size_gb)
                        )
                        available_cpu_memory = self.total_cpu_gb - used_cpu_memory

                        # Try to move weights from GPU first
                        if distribution["w_gpu_percent"] > 0:
                            reduction = min(25, distribution["w_gpu_percent"])
                            distribution["w_gpu_percent"] -= reduction

                            # Calculate how much can fit in CPU
                            weight_to_move_gb = self.total_weight_gb * reduction / 100
                            if available_cpu_memory >= weight_to_move_gb:
                                # Can fit in CPU
                                distribution["w_cpu_percent"] += reduction
                                available_cpu_memory -= weight_to_move_gb
                            else:
                                # Can only fit part in CPU, rest goes to disk
                                cpu_possible_percent = int((available_cpu_memory / self.total_weight_gb) * 100)
                                distribution["w_cpu_percent"] += cpu_possible_percent
                                # Rest implicitly goes to disk
                                available_cpu_memory = 0

                        # Then try to move cache if needed
                        elif distribution["cache_gpu_percent"] > 0:
                            reduction = min(25, distribution["cache_gpu_percent"])
                            distribution["cache_gpu_percent"] -= reduction

                            # Calculate how much can fit in CPU
                            cache_to_move_gb = self.estimate_cache_size(batch_size) * reduction / 100
                            if available_cpu_memory >= cache_to_move_gb:
                                # Can fit in CPU
                                distribution["cache_cpu_percent"] += reduction
                                available_cpu_memory -= cache_to_move_gb
                            else:
                                # Can only fit part in CPU, rest goes to disk
                                cpu_possible_percent = int(
                                    (available_cpu_memory / self.estimate_cache_size(batch_size)) * 100
                                )
                                distribution["cache_cpu_percent"] += cpu_possible_percent
                                # Rest implicitly goes to disk
                                available_cpu_memory = 0
                        else:
                            break
                    else:
                        # Reduce CPU usage
                        if distribution["resident_partitions"] > 0:
                            distribution["resident_partitions"] = max(0, distribution["resident_partitions"] - 2)
                        elif distribution["cache_cpu_percent"] > 0:
                            reduction = min(25, distribution["cache_cpu_percent"])
                            distribution["cache_cpu_percent"] -= reduction
                        elif distribution["w_cpu_percent"] > 0:
                            reduction = min(25, distribution["w_cpu_percent"])
                            distribution["w_cpu_percent"] -= reduction
                        else:
                            break
                    # else:
                    #     break

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

        return optimal_configs
