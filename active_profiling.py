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
import xgboost as xgb
import numpy as np
import gc
from sklearn.model_selection import train_test_split
from itertools import product


class ActiveProfilingProcessor:
    def __init__(
        self,
        questions: List[Dict],
        embedding_model,
        model,
        tokenizer,
        collection,
        partition_names: List[str],
        partition_size_gb: float,
        model_config,
        total_cpu_gb: int = 128,
        gpu_memory_gb: int = 24,
        safety_margin: float = 0.8,
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
        self.partition_size_gb = partition_size_gb * 1.2
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
        error_info = [None]

        def wrapped_generate():
            try:
                result[0] = generate()
            except Exception as e:
                import traceback
                error_traceback = traceback.format_exc()
                error_message = f"Error in generation thread:\n{str(e)}\n\nTraceback:\n{error_traceback}"
                print(error_message)
                result[0] = None, f"error: {error_message}"
                error_info[0] = error_message
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
            gc.collect()
            return None, "timeout"

        if error_info[0]:
            print(f"Generation completed with error: {error_info[0]}")
        else:
            print("Generation completed successfully")

        return result[0]

    def estimate_cache_size(self, batch_size: int) -> float:
        """Estimate cache size in GB for given batch size"""
        return self.model_config.cache_bytes(batch_size, 256 + 16) / (1024**3)

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
                generation_result = batch_generate_responses(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    batch_results=batch_results,
                    max_new_tokens=16,
                    batch_size=batch_size,
                    # timeout=timeout,
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

    def find_optimal_config_with_model(
        self,
        batch_size: int,
        inf_model,
        query_model,
        search_space_size: int = 500,
    ):
        """Find optimal configuration for a given batch size using the trained models"""
        if inf_model is None or query_model is None:
            return None

        import numpy as np

        # Define search space
        cache_gpu_range = np.linspace(0, 100, 10, dtype=int)
        cache_cpu_range = np.linspace(0, 100, 10, dtype=int)
        w_gpu_range = np.linspace(0, 100, 10, dtype=int)
        w_cpu_range = np.linspace(0, 100, 10, dtype=int)
        resident_partitions_range = np.arange(0, min(10, len(self.partition_names) + 1), 1, dtype=int)

        # Generate all valid configurations
        valid_configs = []

        # GPU memory capacity in GB (after accounting for compute space)
        gpu_memory_available = self.gpu_memory_gb - self.compute_space_per_batch * batch_size
        if gpu_memory_available <= 0:
            print(f"Batch size {batch_size} exceeds GPU compute space")
            return None

        # Sample from the search space
        import random

        candidates = []
        for _ in range(search_space_size):
            # Sample random configuration
            cache_gpu = random.choice(cache_gpu_range)
            cache_cpu = random.choice(cache_cpu_range)
            w_gpu = random.choice(w_gpu_range)
            w_cpu = random.choice(w_cpu_range)
            resident_part = random.choice(resident_partitions_range)

            # Check constraints
            # 1. Weight distribution constraint
            if w_gpu + w_cpu > 100:
                continue

            # 2. Cache distribution constraint
            if cache_gpu + cache_cpu > 100:
                continue

            # 3. GPU memory constraint
            gpu_usage = (w_gpu * self.total_weight_gb / 100) + (cache_gpu * self.estimate_cache_size(batch_size) / 100)
            if gpu_usage > gpu_memory_available:
                continue

            # 4. CPU memory constraint
            cpu_usage = (
                (w_cpu * self.total_weight_gb / 100)
                + (cache_cpu * self.estimate_cache_size(batch_size) / 100)
                + (resident_part * self.partition_size_gb)
            )
            if cpu_usage > self.total_cpu_gb:
                continue

            candidates.append([batch_size, cache_gpu, cache_cpu, w_gpu, w_cpu, resident_part])

        if not candidates:
            print(f"No valid configurations found for batch_size={batch_size}")
            return None

        # Convert to numpy for batch prediction
        candidates = np.array(candidates)

        # Predict latencies for all valid configurations
        inf_latencies = inf_model.predict(candidates)
        query_latencies = query_model.predict(candidates)

        # Compute max latency for each configuration
        max_latencies = np.maximum(inf_latencies, query_latencies)

        # Find the configuration with minimum max latency
        best_idx = np.argmin(max_latencies)
        best_config = {
            "batch_size": int(candidates[best_idx][0]),
            "cache_gpu_percent": int(candidates[best_idx][1]),
            "cache_cpu_percent": int(candidates[best_idx][2]),
            "w_gpu_percent": int(candidates[best_idx][3]),
            "w_cpu_percent": int(candidates[best_idx][4]),
            "resident_partitions": int(candidates[best_idx][5]),
            "predicted_inf_latency": float(inf_latencies[best_idx]),
            "predicted_query_latency": float(query_latencies[best_idx]),
        }

        return best_config

    def find_optimal_config(self) -> List[Dict]:
        """Find optimal configuration through profiling with learning-based approach"""
        batch_sizes = [2, 4, 8, 16, 32, 64]
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
            gpu_memory_available = self.gpu_memory_gb - self.compute_space_per_batch * batch_size
            if gpu_memory_available <= 0:
                print(f"Batch size {batch_size} exceeds GPU compute space")
                break

            # Initial distribution - start with a reasonable default
            distribution = self.calculate_memory_distribution(batch_size, gpu_memory_available, self.total_cpu_gb)

            best_config = None
            best_max_time = float("inf")
            retries = 0
            max_retries = 5

            # Try getting an optimized config from the model if we have enough data
            if len(training_data["batch_size"]) >= 5:
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
                        # If the prediction was successful, use it as our best config
                        best_config = result
                        best_max_time = max(result["avg_gen_time"], result["avg_query_time"])

                        # Add to training data
                        training_data["batch_size"].append(batch_size)
                        training_data["cache_gpu_percent"].append(result["cache_gpu_percent"])
                        training_data["cache_cpu_percent"].append(result["cache_cpu_percent"])
                        training_data["w_gpu_percent"].append(result["w_gpu_percent"])
                        training_data["w_cpu_percent"].append(result["w_cpu_percent"])
                        training_data["resident_partitions"].append(result["resident_partitions"])
                        training_data["inference_latency"].append(result["avg_gen_time"])
                        training_data["query_latency"].append(result["avg_query_time"])

                        # Successful prediction, add to optimal configs and continue to next batch size
                        optimal_configs.append(best_config)
                        prev_best_max_time = best_max_time

                        # Skip to next batch size
                        continue

            # If we don't have a model yet or the model prediction failed, use the iterative approach
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

                        # Try using the model if we have enough data now
                        if len(training_data["batch_size"]) >= 5:
                            # Update the cost models with the latest data
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

                            predicted_config = self.find_optimal_config_with_model(batch_size, inf_model, query_model)

                            if predicted_config:
                                # Check if we've already tried this configuration
                                model_config_key = (
                                    batch_size,
                                    predicted_config["cache_gpu_percent"],
                                    predicted_config["cache_cpu_percent"],
                                    predicted_config["w_gpu_percent"],
                                    predicted_config["w_cpu_percent"],
                                    predicted_config["resident_partitions"],
                                )

                                if model_config_key not in configurations_tried:
                                    # Try the model's prediction next
                                    distribution["cache_gpu_percent"] = predicted_config["cache_gpu_percent"]
                                    distribution["cache_cpu_percent"] = predicted_config["cache_cpu_percent"]
                                    distribution["w_gpu_percent"] = predicted_config["w_gpu_percent"]
                                    distribution["w_cpu_percent"] = predicted_config["w_cpu_percent"]
                                    distribution["resident_partitions"] = predicted_config["resident_partitions"]
                                    continue

                        # Check if times are balanced
                        if time_diff_percent <= 10:
                            print("Times are balanced, ending batch profiling")
                            break

                        # Adjust distribution based on bottleneck using heuristics
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
                            if (
                                distribution["cache_cpu_percent"] * self.estimate_cache_size(batch_size) / 100
                                > self.partition_size_gb
                            ):
                                reduction = int(self.partition_size_gb / self.estimate_cache_size(batch_size) * 100)
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

                        # Then try to move cache from GPU first
                        if distribution["cache_gpu_percent"] > 0:
                            reduction = min(10, distribution["cache_gpu_percent"])
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

                        # Try to move weights if needed
                        elif distribution["w_gpu_percent"] > 0:
                            reduction = min(10, distribution["w_gpu_percent"])
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
                        else:
                            break
                    else:
                        # Reduce CPU usage
                        if best_config and best_config["avg_query_time"] < best_config["avg_gen_time"]:
                            if distribution["resident_partitions"] > 0:
                                distribution["resident_partitions"] -= 1
                            elif distribution["cache_cpu_percent"] * self.estimate_cache_size(batch_size) / 100 > 10:
                                reduction = int(10 / self.estimate_cache_size(batch_size) * 100)
                                distribution["cache_cpu_percent"] -= reduction
                            elif distribution["w_cpu_percent"] * self.total_weight_gb / 100 > 10:
                                reduction = int(10 / self.total_weight_gb * 100)
                                distribution["w_cpu_percent"] -= reduction
                            else:
                                break
                        else:
                            if distribution["cache_cpu_percent"] * self.estimate_cache_size(batch_size) / 100 > 10:
                                reduction = int(10 / self.estimate_cache_size(batch_size) * 100)
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

        # Create feature matrix
        X = np.array(
            [batch_size, cache_gpu_percent, cache_cpu_percent, w_gpu_percent, w_cpu_percent, resident_partitions]
        ).T

        # Train two models: one for inference latency and one for query latency
        y_inference = np.array(inference_latency)
        y_query = np.array(query_latency)

        # If we have very little data, use a simple model
        if len(X) < 5:
            print("Not enough data for training XGBoost model, using simple heuristics")
            return None, None

        # Split data for training and validation
        X_train, X_val, y_inf_train, y_inf_val = train_test_split(X, y_inference, test_size=0.2, random_state=42)
        _, _, y_q_train, y_q_val = train_test_split(X, y_query, test_size=0.2, random_state=42)

        # Train inference latency model
        inf_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        inf_model.fit(X_train, y_inf_train)

        # Train query latency model
        query_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        query_model.fit(X_train, y_q_train)

        # Evaluate models
        print(f"Inference model validation score: {inf_model.score(X_val, y_inf_val):.4f}")
        print(f"Query model validation score: {query_model.score(X_val, y_q_val):.4f}")

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

        return optimal_configs
