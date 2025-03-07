import numpy as np
import time
import math
import json
import os
import glob
import csv
from queue import Queue, Empty
from threading import Event, Thread, Lock
from typing import List, Dict
from utils import (
    Question,
    BatchResult,
    log_timing,
    batch_query,
    batch_generate_responses,
    print_memory_usage,
    split_batch,
    init_dynagen_model,
)
from scipy.optimize import linprog
from pymilvus import Partition

min_batch = 32


class DynPipelineProcessor:

    def __init__(
        self,
        questions: List[Dict],
        batch_size: int,
        arrival_rates: List[float],
        embedding_model,
        model_name,
        tokenizer=None,
        collection=None,
        partition_size_gb: float = 11.0,
        partition_names: List[str] = None,
        total_cpu_gb: int = 166,
        gpu_memory_gb: int = 12,
        timing_stats: Dict[str, List[float]] = None,
        resident_partitions: int = 0,
        base_time: float = None,
        rate_change_interval: int = 600,  # 10 minutes in seconds
        seed: int = 42,
        max_disk_percent: float = 0.7,
    ):
        self.questions = [
            Question(question_text=q["question"], doc_id=q["doc_id"], arrival_time=0.0) for q in questions
        ]
        self.batch_size = batch_size
        self.arrival_rates = arrival_rates if arrival_rates else [16, 32, 64]
        self.rate_change_interval = rate_change_interval
        self.embedding_model = embedding_model
        self.model_name = model_name
        self.model, self.model_config, self.env = init_dynagen_model(model_name, tokenizer, [0, 50, 0, 50])
        self.tokenizer = tokenizer
        self.collection = collection
        self.partition_names = partition_names
        self.timing_stats = timing_stats if timing_stats else {}
        self.resident_partitions = resident_partitions
        self.base_time = base_time if base_time else time.time()

        # Memory parameters (will be set in initialize_memory_params)
        self.total_cpu_gb = total_cpu_gb
        self.gpu_memory_gb = gpu_memory_gb
        self.partition_size_gb = partition_size_gb
        self.total_weight_gb = self.model_config.model_bytes() / (1024**3)
        self.compute_weight_gb = self.total_weight_gb / self.model_config.num_hidden_layers
        self.max_disk_percent = max_disk_percent

        # Queues and state management
        self.question_queue = Queue()
        self.query_queue = Queue()
        self.stop_event = Event()
        self.query_lock = Lock()
        self.generation_lock = Lock()
        self.config_update_lock = Lock()

        # Batch management
        self.current_batch = []
        self.total_batches = 0  # Will be calculated after arrival times are generated
        self.completed_batches = 0

        # Configuration management
        self.current_arrival_rate_index = 0
        self.current_arrival_rate = self.arrival_rates[0]
        self.next_config_update_time = self.base_time + self.rate_change_interval
        self.current_config = {
            "batch_size": self.batch_size,
            "cache_gpu_percent": 50,
            "cache_cpu_percent": 50,
            "w_gpu_percent": 50,
            "w_cpu_percent": 50,
            "resident_partitions": resident_partitions,
        }

        # Get or load model parameters
        self.inf_model, self.query_model = None, None

        all_training_data = self.load_all_batch_samples()
        if len(all_training_data["batch_size"]) >= 5:  # Minimum number of samples needed
            self.inf_model, self.query_model = self.learning_cost_model(all_training_data)

            # Get coefficients and intercepts from the trained models
            self.inf_model_params = {
                "coefficients": self.inf_model.coef_.tolist(),
                "intercept": float(self.inf_model.intercept_),
            }
            self.query_model_params = {
                "coefficients": self.query_model.coef_.tolist(),
                "intercept": float(self.query_model.intercept_),
            }
        else:
            print("Not enough training data to learn cost models. Exiting.")
            return

        np.random.seed(seed)
        self.all_results = []
        self.results_lock = Lock()
        self.loaded_partitions = set([0])  # Start with partition_0 loaded

        # Track questions by interval
        self.interval_question_count = []
        self.interval_completion_count = []

        # Track queries and generations processed per interval
        self.interval_query_count = []
        self.interval_gen_count = []

        # Track questions that have been queued per interval
        self.interval_queued_count = []

        # Flags to signal when each worker should stop
        self.query_complete = Event()
        self.gen_complete = Event()

    def learning_cost_model(self, training_data):
        """Build a cost model to predict inference and query latencies based on loaded samples"""
        import numpy as np
        from sklearn.linear_model import LinearRegression
        import json
        import os
        import math

        # Create feature matrices
        # Inference latency model features: based on the formula
        # a * w_gpu_percent + b * w_cpu_percent + c * cache_gpu_percent * batch_size +
        # d * cache_cpu_percent * batch_size + e * batch_size + f * log(batch_size) + g
        X_inf = np.column_stack(
            [
                training_data["w_gpu_percent"],
                training_data["w_cpu_percent"],
                [
                    training_data["cache_gpu_percent"][i] * training_data["batch_size"][i]
                    for i in range(len(training_data["batch_size"]))
                ],
                [
                    training_data["cache_cpu_percent"][i] * training_data["batch_size"][i]
                    for i in range(len(training_data["batch_size"]))
                ],
                training_data["batch_size"],
                [math.log(bs) if bs > 0 else 0 for bs in training_data["batch_size"]],
            ]
        )

        # Query latency model features: based on the formula
        # a * resident_partitions + b * batch_size + c
        X_query = np.column_stack([training_data["resident_partitions"], training_data["batch_size"]])

        # Create target vectors
        y_inf = np.array(training_data["inference_latency"])
        y_query = np.array(training_data["query_latency"])

        # Train inference latency model
        inf_model = LinearRegression()
        inf_model.fit(X_inf, y_inf)

        # Train query latency model
        query_model = LinearRegression()
        query_model.fit(X_query, y_query)

        # Calculate MSE for both models
        inf_pred = inf_model.predict(X_inf)
        inf_mse = np.mean((inf_pred - y_inf) ** 2)

        query_pred = query_model.predict(X_query)
        query_mse = np.mean((query_pred - y_query) ** 2)

        print(f"Inference model MSE: {inf_mse:.4f}")
        print(f"Query model MSE: {query_mse:.4f}")

        # Save model parameters to human-readable files
        os.makedirs(f"{self.model_name}_data", exist_ok=True)

        # Save inference model parameters
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

        # Save query model parameters
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

        return inf_model, query_model

    def load_all_batch_samples(self) -> Dict[str, List]:
        """Load all batch samples from previously saved files"""
        batch_files = glob.glob(f"{self.model_name}_data/batch_*_samples.json")

        # Initialize data structure
        all_data = {
            "batch_size": [],
            "cache_gpu_percent": [],
            "cache_cpu_percent": [],
            "w_gpu_percent": [],
            "w_cpu_percent": [],
            "resident_partitions": [],
            "inference_latency": [],
            "query_latency": [],
        }

        # Load data from each batch file
        for batch_file in batch_files:
            try:
                with open(batch_file, "r") as f:
                    samples = json.load(f)

                for sample in samples:
                    if sample.get("success", False):
                        all_data["batch_size"].append(sample["batch_size"])
                        all_data["cache_gpu_percent"].append(sample["cache_gpu_percent"])
                        all_data["cache_cpu_percent"].append(sample["cache_cpu_percent"])
                        all_data["w_gpu_percent"].append(sample["w_gpu_percent"])
                        all_data["w_cpu_percent"].append(sample["w_cpu_percent"])
                        all_data["resident_partitions"].append(sample["resident_partitions"])
                        all_data["inference_latency"].append(sample["avg_gen_time"])
                        all_data["query_latency"].append(sample["avg_query_time"])
            except Exception as e:
                print(f"Error loading samples from {batch_file}: {e}")

        print(f"Loaded {len(all_data['batch_size'])} samples from {len(batch_files)} batch files")
        return all_data

    def _load_model_params(self, filename):
        """Load model parameters from JSON file"""
        with open(filename, "r") as f:
            return json.load(f)

    def estimate_hidden_size(self, batch_size: int) -> float:
        """Estimate hidden state size in GB for given batch size"""
        return self.model_config.hidden_bytes(batch_size, 512 + 16) / (1024**3)

    def estimate_cache_size(self, batch_size: int, layers=1) -> float:
        """Estimate cache size in GB for given batch size"""
        return self.model_config.cache_bytes(batch_size, 512 + 16, layers) / (1024**3)

    def estimate_total_working_memory(self, batch_size: int) -> float:
        """Estimate total working memory needed for computation, including attention and MLP states"""
        # Simplified approach - this could be expanded with the detailed calculations from ActiveProfilingProcessor
        batch_size_split, _ = split_batch(batch_size)

        # Basic estimation for attention mechanism memory usage
        input_dim = self.model_config.input_dim
        n_head = self.model_config.n_head
        seq_len = 512 + 16  # Typical sequence length

        # Simplified attention working memory estimate
        attention_memory = 4 * batch_size_split * seq_len * input_dim * 2 / (1024**3)  # in GB

        # Simplified MLP working memory estimate
        intermediate_size = input_dim * 4  # Typical intermediate size multiplier
        mlp_memory = 3 * batch_size_split * seq_len * intermediate_size * 2 / (1024**3)  # in GB

        # Add overhead factor
        return (attention_memory + mlp_memory) * 1.1

    def gpu_memory_available(self, batch_size) -> float:
        """Calculate available GPU memory after accounting for model and computation memory needs"""
        batch_size_split, _ = split_batch(batch_size)

        # Memory allocated for model weights on GPU
        weights_memory = self.compute_weight_gb * 2  # Compute weight allocation

        # Memory for KV cache
        cache_memory = self.estimate_cache_size(batch_size_split)

        # Memory for hidden states
        hidden_memory = self.estimate_hidden_size(batch_size)

        # Working memory for computation
        working_memory = self.estimate_total_working_memory(batch_size_split)

        # Total needed memory with safety margin
        total_needed = weights_memory + cache_memory + hidden_memory + working_memory

        # Return available memory after computations
        return self.gpu_memory_gb - 1.1 * total_needed

    def update_resident_partitions(self, new_resident_partitions: int):
        """Update number of resident partitions"""
        with self.config_update_lock:
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

    def _generate_arrival_times(self):
        """Generate arrival times based on time-varying arrival rates"""
        # 确保base_time已正确设置
        if self.base_time is None:
            self.base_time = time.time()

        print(f"Generating arrival times with base_time: {self.base_time}")
        print(f"Arrival rates: {self.arrival_rates} questions/minute")

        # Calculate expected duration for each rate interval
        interval_durations = [self.rate_change_interval] * len(self.arrival_rates)
        total_expected_duration = sum(interval_durations)

        # Calculate expected number of questions for each interval
        expected_questions_per_interval = [
            rate * (interval_durations[i] / 60) for i, rate in enumerate(self.arrival_rates)
        ]
        total_expected_questions = sum(expected_questions_per_interval)

        # 打印预期问题数量
        print(f"Expected questions per interval based on rates: {[int(q) for q in expected_questions_per_interval]}")
        print(f"Total expected questions: {int(total_expected_questions)}")

        # Use the expected number of questions for each interval without scaling
        questions_per_interval = [int(round(count)) for count in expected_questions_per_interval]

        # 确保问题数量不超过可用问题
        if len(self.questions) < total_expected_questions:
            print(
                f"Warning: Not enough questions available. Need {total_expected_questions} but only have {len(self.questions)}"
            )
            # 按比例缩减问题数量
            scale_factor = len(self.questions) / total_expected_questions
            questions_per_interval = [int(round(count * scale_factor)) for count in questions_per_interval]
            # 确保每个区间至少有一个问题
            questions_per_interval = [max(1, count) for count in questions_per_interval]
            # 调整总数不超过可用问题
            while sum(questions_per_interval) > len(self.questions):
                # 找到最大的区间并减少一个问题
                max_idx = questions_per_interval.index(max(questions_per_interval))
                questions_per_interval[max_idx] -= 1

        # 打印最终的问题分配
        print(f"Final questions per interval: {questions_per_interval}")
        print(f"Total questions to be processed: {sum(questions_per_interval)}")

        # Initialize tracking arrays for questions by interval
        self.interval_question_count = questions_per_interval
        self.interval_completion_count = [0] * len(self.arrival_rates)

        # Initialize new tracking arrays for query and generation completion
        self.interval_query_count = [0] * len(self.arrival_rates)
        self.interval_gen_count = [0] * len(self.arrival_rates)
        self.interval_queued_count = [0] * len(self.arrival_rates)

        # 只使用所需数量的问题，而不是全部问题
        used_questions = self.questions[: sum(questions_per_interval)]

        # 为每个问题分配区间ID
        question_index = 0
        for i, question_count in enumerate(questions_per_interval):
            for j in range(question_count):
                if question_index < len(used_questions):
                    # 直接设置区间ID
                    used_questions[question_index].interval_id = i
                    # 设置到达时间
                    interval_start = self.base_time + sum(interval_durations[:i])
                    # 平均分布到达时间
                    used_questions[question_index].arrival_time = (
                        interval_start + (j / question_count) * interval_durations[i]
                        if question_count > 0
                        else interval_start
                    )
                    question_index += 1

        # 只处理需要的问题
        self.questions = used_questions

        # Sort questions by arrival time
        self.questions.sort(key=lambda x: x.arrival_time)

        # 验证生成的时间戳是否合理
        if self.questions:
            min_arrival = min(q.arrival_time for q in self.questions)
            max_arrival = max(q.arrival_time for q in self.questions)
            print(f"Arrival times range: {min_arrival - self.base_time:.2f}s to {max_arrival - self.base_time:.2f}s")

        # Calculate total batches based on the expected questions
        self.total_batches = (sum(questions_per_interval) + self.batch_size - 1) // self.batch_size
        print(f"Total batches: {self.total_batches}")

    def find_optimal_config_for_arrival_rate(self, arrival_rate, current_interval=None):
        """
        Find optimal configuration based on queue size rather than arrival rate
        Returns only batch size, deferring other policy updates to generation worker
        """
        print(f"\nFinding optimal batch size for current queue")

        # Get current queue size
        current_queue_size = self.question_queue.qsize()
        print(f"Current queue size: approximately {current_queue_size} questions")

        # Available batch sizes
        batch_sizes = [32, 64, 96, 128, 160, 192, 224]

        # Choose appropriate batch size based on queue size
        optimal_batch_size = self.batch_size  # default to current

        # Choose largest batch size that doesn't exceed queue size
        # and doesn't exceed GPU memory
        for batch_size in sorted(batch_sizes, reverse=True):
            # Skip this batch size if it exceeds GPU memory
            if self.gpu_memory_available(batch_size) <= 0:
                print(f"Batch size {batch_size} exceeds GPU compute space")
                continue

            if batch_size <= current_queue_size:
                optimal_batch_size = batch_size
                break

        print(f"Selected optimal batch size: {optimal_batch_size}")

        # Return configuration with only batch size set
        # Other policy updates will be deferred to generation worker
        return {
            "batch_size": optimal_batch_size,
            "cache_gpu_percent": self.current_config["cache_gpu_percent"],
            "cache_cpu_percent": self.current_config["cache_cpu_percent"],
            "w_gpu_percent": self.current_config["w_gpu_percent"],
            "w_cpu_percent": self.current_config["w_cpu_percent"],
            "resident_partitions": self.current_config["resident_partitions"],
        }

    def find_optimal_config_with_linprog(self, batch_size):
        """Find optimal configuration for a given batch size using linear programming with disk constraints"""
        print(f"Finding optimal configuration for batch size {batch_size} using linear programming")

        # Extract model parameters
        inf_coef = self.inf_model_params["coefficients"]
        inf_intercept = self.inf_model_params["intercept"]
        query_coef = self.query_model_params["coefficients"]
        query_intercept = self.query_model_params["intercept"]

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
        A_query_latency = [0, 0, 0, 0, query_coef[0], -1]  # Only resident_partitions and max_latency
        b_query_latency = -query_coef[1] * batch_size - query_intercept

        # 3. Weight distribution constraint: w_gpu + w_cpu <= 100
        A_weight_dist = [1, 1, 0, 0, 0, 0]
        b_weight_dist = 100

        # 4. Cache distribution constraint: cache_gpu + cache_cpu <= 100
        A_cache_dist = [0, 0, 1, 1, 0, 0]
        b_cache_dist = 100

        # 5. GPU memory constraint
        gpu_available = self.gpu_memory_available(batch_size)
        A_gpu_mem = [self.total_weight_gb / 100, 0, cache_size / 100, 0, 0, 0]
        b_gpu_mem = gpu_available - hidden_size

        # 6. CPU memory constraint
        A_cpu_mem = [0, self.total_weight_gb / 100, 0, cache_size / 100, self.partition_size_gb, 0]
        b_cpu_mem = self.total_cpu_gb

        # 7. Minimum cache requirement (at least some percentage must be in memory)
        min_memory_percent = 100 - self.max_disk_percent * 100
        A_cache_min = [0, 0, -1, -1, 0, 0]
        b_cache_min = -min_memory_percent  # At least min_memory_percent of cache must be used

        # 8. Minimum weight requirement (at least some percentage must be in memory)
        A_weight_min = [-1, -1, 0, 0, 0, 0]
        b_weight_min = -min_memory_percent  # At least min_memory_percent of weights must be in memory

        # 9. NEW CONSTRAINT: Weight and cache on disk <= max_disk_percent of total weight and cache
        # Disk weight = (100 - w_gpu - w_cpu) * total_weight_gb / 100
        # Disk cache = (100 - cache_gpu - cache_cpu) * cache_size / 100
        # Constraint: Disk weight + Disk cache <= max_disk_percent * (total_weight_gb + cache_size)
        A_disk_limit = [
            -self.total_weight_gb / 100,  # w_gpu_percent coefficient
            -self.total_weight_gb / 100,  # w_cpu_percent coefficient
            -cache_size / 100,  # cache_gpu_percent coefficient
            -cache_size / 100,  # cache_cpu_percent coefficient
            0,  # resident_partitions coefficient
            0,  # max_latency coefficient
        ]
        b_disk_limit = -self.max_disk_percent * (self.total_weight_gb + cache_size)

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
        bounds = [
            (0, 100),  # w_gpu_percent
            (0, 100),  # w_cpu_percent
            (0, 100),  # cache_gpu_percent
            (0, 100),  # cache_cpu_percent
            (0, min(10, len(self.partition_names))),  # resident_partitions
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

                # Verify disk constraint is satisfied
                weight_on_disk_percent = 100 - w_gpu_percent - w_cpu_percent
                cache_on_disk_percent = 100 - cache_gpu_percent - cache_cpu_percent
                weight_on_disk = (weight_on_disk_percent / 100) * self.total_weight_gb
                cache_on_disk = (cache_on_disk_percent / 100) * cache_size
                total_on_disk = weight_on_disk + cache_on_disk
                max_allowed_on_disk = self.max_disk_percent * (self.total_weight_gb + cache_size)

                if total_on_disk > max_allowed_on_disk + 0.1:  # Small tolerance for floating-point errors
                    print(f"Warning: Disk storage constraint violated. Adjusting configuration.")
                    print(f"Current disk usage: {total_on_disk:.2f} GB, Max allowed: {max_allowed_on_disk:.2f} GB")

                    # Adjust weights and cache to satisfy the constraint
                    # This is simplified compared to the original but achieves the same goal
                    excess_disk_gb = total_on_disk - max_allowed_on_disk

                    # Try to move more weights to CPU/GPU first
                    if weight_on_disk_percent > 0:
                        weight_to_move_percent = min(
                            weight_on_disk_percent, int(excess_disk_gb / self.total_weight_gb * 100)
                        )
                        # Try CPU first, then GPU
                        weight_to_cpu = min(weight_to_move_percent, 100 - w_cpu_percent)
                        w_cpu_percent += weight_to_cpu
                        weight_to_move_percent -= weight_to_cpu

                        if weight_to_move_percent > 0:
                            w_gpu_percent += min(weight_to_move_percent, 100 - w_gpu_percent)

                    # Recalculate disk usage
                    weight_on_disk_percent = 100 - w_gpu_percent - w_cpu_percent
                    weight_on_disk = (weight_on_disk_percent / 100) * self.total_weight_gb
                    total_on_disk = weight_on_disk + cache_on_disk

                    # If still exceeding, adjust cache
                    if total_on_disk > max_allowed_on_disk and cache_on_disk_percent > 0:
                        excess_disk_gb = total_on_disk - max_allowed_on_disk
                        cache_to_move_percent = min(cache_on_disk_percent, int(excess_disk_gb / cache_size * 100))

                        # Try CPU first, then GPU
                        cache_to_cpu = min(cache_to_move_percent, 100 - cache_cpu_percent)
                        cache_cpu_percent += cache_to_cpu
                        cache_to_move_percent -= cache_to_cpu

                        if cache_to_move_percent > 0:
                            cache_gpu_percent += min(cache_to_move_percent, 100 - cache_gpu_percent)

                # Predict latencies using the optimized configuration
                inf_latency, query_latency = self.predict_latencies(
                    {
                        "w_gpu_percent": w_gpu_percent,
                        "w_cpu_percent": w_cpu_percent,
                        "cache_gpu_percent": cache_gpu_percent,
                        "cache_cpu_percent": cache_cpu_percent,
                        "resident_partitions": resident_partitions,
                    },
                    batch_size,
                )

                return {
                    "batch_size": batch_size,
                    "cache_gpu_percent": int(cache_gpu_percent),
                    "cache_cpu_percent": int(cache_cpu_percent),
                    "w_gpu_percent": int(w_gpu_percent),
                    "w_cpu_percent": int(w_cpu_percent),
                    "resident_partitions": int(resident_partitions),
                    "predicted_inf_latency": float(inf_latency),
                    "predicted_query_latency": float(query_latency),
                    "max_latency": float(max_latency),
                }
            else:
                print(f"Optimization failed: {result.message}")
                return None

        except Exception as e:
            print(f"Error in optimization: {e}")
            return None

    def predict_latencies(self, config, batch_size):
        """Predict inference and query latencies using the cost models"""
        # Extract model parameters
        inf_coef = self.inf_model_params["coefficients"]
        inf_intercept = self.inf_model_params["intercept"]
        query_coef = self.query_model_params["coefficients"]
        query_intercept = self.query_model_params["intercept"]

        # Predict inference latency
        inference_latency = (
            inf_coef[0] * config["w_gpu_percent"]
            + inf_coef[1] * config["w_cpu_percent"]
            + inf_coef[2] * config["cache_gpu_percent"] * batch_size
            + inf_coef[3] * config["cache_cpu_percent"] * batch_size
            + inf_coef[4] * batch_size
            + inf_coef[5] * math.log(batch_size)
            + inf_intercept
        )

        # Predict query latency
        query_latency = query_coef[0] * config["resident_partitions"] + query_coef[1] * batch_size + query_intercept

        return inference_latency, query_latency

    def update_configuration(self, current_time=None, force_update=False, next_interval_index=None):
        """Update only batch size configuration based on current queue size"""
        with self.config_update_lock:
            if current_time is None:
                current_time = time.time()

            # Check if it's time to update the configuration
            if force_update or current_time >= self.next_config_update_time:
                # Determine which interval to update to
                if next_interval_index is not None:
                    self.current_arrival_rate_index = next_interval_index
                else:
                    # Default behavior: move to next interval
                    self.current_arrival_rate_index = (self.current_arrival_rate_index + 1) % len(self.arrival_rates)

                self.current_arrival_rate = self.arrival_rates[self.current_arrival_rate_index]

                print(f"\n{'='*50}")
                print(f"Updating configuration at {current_time - self.base_time:.2f}s")
                print(f"New arrival rate: {self.current_arrival_rate} questions/minute")
                print(f"Current batch: {len(self.current_batch)} questions")
                print(f"Questions in queue: approximately {self.question_queue.qsize()} questions")
                print(f"Completed batches: {self.completed_batches}/{self.total_batches}")

                # Find optimal configuration for the current queue size
                new_config = self.find_optimal_config_for_arrival_rate(
                    self.current_arrival_rate, current_interval=self.current_arrival_rate_index
                )

                if new_config:
                    # Update batch size only
                    old_batch_size = self.batch_size
                    self.batch_size = new_config["batch_size"]

                    print(f"Updating batch size:")
                    print(f"- Old batch size: {old_batch_size}")
                    print(f"- New batch size: {self.batch_size}")

                    # Update current configuration (batch size only)
                    self.current_config["batch_size"] = self.batch_size
                    print("Batch size updated successfully")
                else:
                    print("Failed to find optimal batch size, keeping current settings")

                # Set next update time
                self.next_config_update_time = (
                    self.base_time + (self.current_arrival_rate_index + 1) * self.rate_change_interval
                )
                print(f"Next configuration update scheduled at: {self.next_config_update_time - self.base_time:.2f}s")
                print(f"{'='*50}\n")

                return True  # Configuration was updated

            return False  # No update was needed

    def _update_interval_counts(self, questions, counter_type):
        """集中式区间计数更新函数"""
        if not questions:
            return

        # 获取对应的计数器数组
        counter = None
        if counter_type == "query":
            counter = self.interval_query_count
        elif counter_type == "gen":
            counter = self.interval_gen_count
        elif counter_type == "completion":
            counter = self.interval_completion_count
        elif counter_type == "queued":
            counter = self.interval_queued_count
        else:
            # print(f"Warning: Unknown counter type '{counter_type}'")
            return

        # 确保计数器长度正确
        while len(counter) < len(self.arrival_rates):
            counter.append(0)

        # 遍历每个问题，更新相应的计数器
        for q in questions:
            try:
                # 检查arrival_time是否有效
                if not hasattr(q, "arrival_time") or q.arrival_time is None:
                    # print(f"Warning: Question has no arrival_time in {counter_type} update")
                    continue

                # 计算该问题所属的区间
                interval_idx = min(int((q.arrival_time - self.base_time) / self.rate_change_interval), len(counter) - 1)

                # 检查并修复负面区间索引
                if interval_idx < 0:
                    # print(f"Warning: Negative interval index detected ({interval_idx}), using index 0 instead")
                    interval_idx = 0

                # 增加对应区间的计数
                if 0 <= interval_idx < len(counter):
                    counter[interval_idx] += 1
                else:
                    print(f"Warning: Invalid interval index {interval_idx}, max is {len(counter)-1}")
            except Exception as e:
                print(f"Error in _update_interval_counts ({counter_type}): {str(e)}")

        # 打印当前计数器状态
        print(f"Updated {counter_type} counts: {counter}")
        print(f"Target counts: {self.interval_question_count}")

    def update_model_policy_for_batch(self, batch_size):
        """Update model policy specifically for the current batch size"""
        # For non-standard batch sizes, round up to the nearest standard batch size
        # This avoids assertion errors when processing small or odd-sized batches
        standard_batch_sizes = [32, 64, 96, 128, 160, 192, 224]

        # Find the next largest standard batch size
        policy_batch_size = batch_size
        for std_size in sorted(standard_batch_sizes):
            if std_size >= batch_size:
                policy_batch_size = std_size
                break

        print(f"Updating model policy for batch size {batch_size} (using policy size {policy_batch_size})")

        # Split batch for the policy batch size, not the actual batch size
        batch_size_split, num_batch = split_batch(policy_batch_size)

        print(f"- Split batch size: {batch_size_split}, num_batch: {num_batch}")

        try:
            # Use current cache settings but update batch settings
            self.model.update_policy(
                self.current_config["cache_gpu_percent"],
                self.current_config["cache_cpu_percent"],
                batch_size_split,
                num_batch,
            )
            print(f"Model policy updated for specific batch")
        except Exception as e:
            print(f"Error updating model policy for specific batch: {str(e)}")

    def _query_worker(self):
        """Execute query worker thread that processes as many questions as possible at once"""
        loop_counter = 0
        total_queries_processed = 0
        current_interval_idx = 0
        first_batch = True
        batch_sizes = [32, 64, 96, 128, 160, 192, 224]

        # 根据设置的到达率，选择合适的最大批处理大小
        max_questions = max(self.interval_question_count) if self.interval_question_count else 0
        max_batch_size = 224

        print(f"Query worker starting. Max questions per interval: {max_questions}")
        print(f"Selected max batch size: {max_batch_size}")
        print(f"Target questions per interval: {self.interval_question_count}")

        # 如果第一批次太大，调整等待策略
        first_batch_size = min(4, max(8, max_questions // 2))
        print(f"Will wait for batch size of {first_batch_size} before processing first batch")

        # Continue until all questions have been processed
        while not self.query_complete.is_set():
            loop_counter += 1
            current_time = time.time()

            # Determine current interval based on time
            elapsed_time = current_time - self.base_time
            interval_idx = min(int(elapsed_time / self.rate_change_interval), len(self.arrival_rates) - 1)

            # Update current interval if changed
            if interval_idx != current_interval_idx:
                print(f"Query worker moving to interval {interval_idx} from {current_interval_idx}")
                current_interval_idx = interval_idx

            # Log status periodically
            should_log = loop_counter % 20 == 0
            if should_log:
                print(
                    f"[Query Loop #{loop_counter}] Status: batch={len(self.current_batch)}, "
                    f"queue~{self.question_queue.qsize()}, completed={self.completed_batches}/{self.total_batches}"
                )
                print(f"Interval query counts: {self.interval_query_count}")
                print(f"Target interval counts: {self.interval_question_count}")

            # 获取当前队列大小
            current_queue_size = self.question_queue.qsize()

            # 对第一批次使用较小的等待阈值
            if first_batch and current_queue_size < first_batch_size:
                if should_log:
                    print(
                        f"[Query Loop #{loop_counter}] Waiting for {first_batch_size} questions before processing first batch. Current: {current_queue_size}"
                    )
                time.sleep(0.1)
                continue

            # For small remaining batches (less than 16), process them directly
            if 0 < current_queue_size < min_batch:
                optimal_batch_size = current_queue_size
                print(f"[Query Loop #{loop_counter}] Processing small batch of {optimal_batch_size} questions directly")
            else:
                # 选择最优批处理大小，但不超过max_batch_size
                optimal_batch_size = 2  # 默认最小值

                # 找到不超过队列大小且不超过max_batch_size的最大批处理大小
                for batch_size in sorted([bs for bs in batch_sizes if bs <= max_batch_size], reverse=True):
                    if batch_size <= current_queue_size and self.gpu_memory_available(batch_size) > 0:
                        optimal_batch_size = batch_size
                        break

            # 只在有问题且批处理大小有效时处理
            if current_queue_size > 0 and optimal_batch_size > 0:
                # 对首批使用之前计算的大小
                if first_batch:
                    optimal_batch_size = min(first_batch_size, optimal_batch_size)
                    first_batch = False

                print(
                    f"[Query Loop #{loop_counter}] Selected batch size {optimal_batch_size} for {current_queue_size} queued questions"
                )

                # 重置当前批次并收集问题
                self.current_batch = []

                # 从队列收集问题
                collected_count = 0
                for _ in range(optimal_batch_size):
                    if not self.question_queue.empty():
                        question = self.question_queue.get()
                        self.current_batch.append(question)
                        collected_count += 1
                    else:
                        break

                print(f"[Query Loop #{loop_counter}] Collected {collected_count} questions for processing")

                # 处理批次
                if len(self.current_batch) > 0:
                    # 使用集中式函数更新计数
                    self._update_interval_counts(self.current_batch, "queued")

                    with self.query_lock:
                        batch = self.current_batch
                        self.current_batch = []

                        # 执行查询
                        query_texts = [q.question_text for q in batch]
                        questions_dict = [{"question": q.question_text, "doc_id": q.doc_id} for q in batch]

                        # Embedding generation
                        embed_start = time.time()
                        query_embeddings = self.embedding_model.embed_documents(query_texts)
                        log_timing(self.timing_stats, "embedding_time", time.time() - embed_start)

                        # Query execution
                        batch_results, updated_timing_stats = batch_query(
                            collection=self.collection,
                            questions=questions_dict,
                            query_texts=query_texts,
                            query_embeddings=query_embeddings,
                            partition_names=self.partition_names,
                            timing_stats=self.timing_stats,
                            resident_partitions=self.resident_partitions,
                        )
                        self.timing_stats.update(updated_timing_stats)

                        # 将结果放入队列
                        batch_result = BatchResult(
                            questions=batch, query_embeddings=query_embeddings, query_results=batch_results
                        )
                        self.query_queue.put(batch_result)

                        # 更新查询计数
                        self._update_interval_counts(batch, "query")

                        total_queries_processed += len(batch)

                        current_time = time.time() - self.base_time
                        print(
                            f"[Query Loop #{loop_counter}] Query completed for batch of {len(batch)} at {current_time:.2f}s"
                        )
                        print(f"Total queries processed: {total_queries_processed}")
                        print_memory_usage("After query completion - ")
            else:
                # 如果没有问题或批处理大小无效，休眠
                time.sleep(0.1)

                # 检查是否所有问题都已处理并且队列为空
                if self.question_queue.empty() and total_queries_processed >= len(self.questions):
                    print(f"Query worker completed all available questions, setting query_complete signal")
                    self.query_complete.set()
                    break

        print(f"Query worker completed all required queries")

    def _generation_worker(self):
        """Execute generation worker thread with modified batching strategy that processes as many batch results as possible"""
        loop_counter = 0
        total_gens_processed = 0
        current_interval_idx = 0
        first_batch = True

        # Available batch sizes
        available_batch_sizes = [32, 64, 96, 128, 160, 192, 224]

        print(f"Generation worker starting. Target: {self.total_batches} batches")
        print(f"Target questions by interval: {self.interval_question_count}")

        # Continue until all generations are completed
        while not self.gen_complete.is_set():
            loop_counter += 1
            current_time = time.time()

            # Determine current interval based on time
            elapsed_time = current_time - self.base_time
            interval_idx = min(int(elapsed_time / self.rate_change_interval), len(self.arrival_rates) - 1)

            # Update current interval if changed
            if interval_idx != current_interval_idx:
                print(f"Generation worker moving to interval {interval_idx} from {current_interval_idx}")

                # Update policies when interval changes
                if not first_batch:
                    print("Updating policies for new interval")
                    self.update_policies_for_generation()

                current_interval_idx = interval_idx

            # Log less frequently
            if loop_counter % 10 == 0:
                print(f"[Generation Loop #{loop_counter}] Status: {total_gens_processed} questions processed")
                print(f"Interval generation counts: {self.interval_gen_count}")
                print(f"Target interval counts: {self.interval_question_count}")

            # Get a batch result from the queue
            try:
                batch_result = self.query_queue.get(timeout=0.5)

                # If this is the first batch, update policies
                if first_batch:
                    print("First batch received, updating policies for generation")
                    self.update_policies_for_generation()
                    first_batch = False

                num_questions = len(batch_result.questions)

                # For small remaining batches, process them directly without splitting
                if num_questions < min_batch:
                    chosen_batch_size = num_questions
                    print(
                        f"[Generation Loop #{loop_counter}] Processing small batch with {num_questions} questions directly"
                    )
                else:
                    # Choose largest batch size that doesn't exceed the number of questions
                    chosen_batch_size = max([bs for bs in available_batch_sizes if bs <= num_questions], default=2)
                    print(
                        f"[Generation Loop #{loop_counter}] Processing batch with {num_questions} questions using batch size {chosen_batch_size}"
                    )

                print_memory_usage("Before generation - ")

                with self.generation_lock:
                    # Process the questions and query results based on chosen batch size
                    if chosen_batch_size < num_questions and num_questions >= 16:
                        # We need to split the batch
                        process_questions = batch_result.questions[:chosen_batch_size]
                        process_query_results = batch_result.query_results[:chosen_batch_size]

                        # Put the remaining questions back in the queue for later
                        remaining_questions = batch_result.questions[chosen_batch_size:]
                        remaining_query_results = batch_result.query_results[chosen_batch_size:]

                        # Create a new batch result for remaining questions
                        remaining_batch = BatchResult(
                            questions=remaining_questions,
                            query_embeddings=batch_result.query_embeddings,  # Keep original embeddings
                            query_results=remaining_query_results,
                        )
                        self.query_queue.put(remaining_batch)
                        print(
                            f"[Generation Loop #{loop_counter}] Put {len(remaining_questions)} remaining questions back in the queue"
                        )
                    else:
                        # Process the entire batch
                        process_questions = batch_result.questions
                        process_query_results = batch_result.query_results

                    # Update model policy for this specific batch size
                    # The function now handles rounding to valid batch sizes
                    self.update_model_policy_for_batch(len(process_questions))

                    # Get current time for query completion
                    query_completion_time = time.time()

                    # Execute generation with exact batch size
                    if (
                        not len(process_questions)
                        == self.model.policy.gpu_batch_size * self.model.policy.num_gpu_batches
                    ):
                        batch_size_split, num_batch = split_batch(len(process_questions))
                        self.model.update_policy(
                            self.current_config["cache_gpu_percent"],
                            self.current_config["cache_cpu_percent"],
                            batch_size_split,
                            num_batch,
                        )
                    batch_responses, updated_timing_stats = batch_generate_responses(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        batch_results=process_query_results,
                        max_new_tokens=16,
                        batch_size=len(process_questions),  # Use actual number of questions
                        timing_stats=self.timing_stats,
                        dynagen=True,
                        env=self.env,
                    )
                    self.timing_stats.update(updated_timing_stats)

                    self.completed_batches += 1

                    # Get current time for generation completion
                    generation_completion_time = time.time()

                    # 使用集中式函数更新生成和完成计数
                    self._update_interval_counts(process_questions, "gen")
                    self._update_interval_counts(process_questions, "completion")

                    total_gens_processed += len(process_questions)

                    with self.results_lock:
                        for q, r in zip(process_questions, batch_responses):
                            # Calculate the interval for this question
                            q_interval = min(
                                int((q.arrival_time - self.base_time) / self.rate_change_interval),
                                len(self.arrival_rates) - 1,
                            )

                            # 添加当前的policy信息
                            self.all_results.append(
                                {
                                    "question": q,
                                    "result": r,
                                    "arrival_time": q.arrival_time,
                                    "query_completion_time": query_completion_time,
                                    "generation_completion_time": generation_completion_time,
                                    "interval_id": q_interval,
                                    "arrival_rate": self.arrival_rates[q_interval],
                                    # 添加policy信息
                                    "batch_size": self.current_config["batch_size"],
                                    "w_gpu_percent": self.current_config["w_gpu_percent"],
                                    "w_cpu_percent": self.current_config["w_cpu_percent"],
                                    "cache_gpu_percent": self.current_config["cache_gpu_percent"],
                                    "cache_cpu_percent": self.current_config["cache_cpu_percent"],
                                    "resident_partitions": self.current_config["resident_partitions"],
                                }
                            )

                    current_time = time.time() - self.base_time
                    print(
                        f"[Generation Loop #{loop_counter}] Generation completed at {current_time:.2f}s, "
                        f"{self.completed_batches}/{self.total_batches} batches done"
                    )
                    print(f"Total generations processed: {total_gens_processed}")
                    print(f"Interval generation counts: {self.interval_gen_count}")
                    print_memory_usage("After generation - ")

            except Empty:
                # Queue is empty, don't log this every time
                if loop_counter % 20 == 0:
                    print(f"[Generation Loop #{loop_counter}] Queue empty, waiting for more batches...")
                time.sleep(0.2)  # Small sleep to prevent CPU spinning

                # 检查是否所有查询都已完成且队列为空
                if self.query_complete.is_set() and self.query_queue.empty():
                    print(f"Generation worker: All queries completed and queue empty, setting gen_complete signal")
                    self.gen_complete.set()
                    break

                continue

        print(f"Generation worker completed all required generations")

    def _question_arrival_simulator(self):
        """Simulate question arrival based on time-varying arrival rates"""
        print(f"Question arrival simulator starting with base_time {self.base_time}")
        print(f"Total questions to be added to queue: {len(self.questions)}")

        # 记录当前已加入队列的问题数量
        added_count = 0

        for question in self.questions:
            if self.stop_event.is_set():
                break

            current_time = time.time()
            wait_time = question.arrival_time - current_time

            # 检查无效的等待时间
            if wait_time > 3600:  # 超过1小时的等待时间可能是错误的
                print(
                    f"Warning: Unusually long wait time detected ({wait_time:.2f}s). "
                    f"Current: {current_time}, Arrival: {question.arrival_time}"
                )
                # 调整为小的等待时间
                wait_time = 0.1

            if wait_time > 0:
                time.sleep(wait_time)

            self.question_queue.put(question)
            added_count += 1

            # 每10个问题记录一次
            if added_count % 10 == 0:
                current_time = time.time()
                elapsed = current_time - self.base_time
                rate_index = min(int(elapsed / self.rate_change_interval), len(self.arrival_rates) - 1)
                print(f"Added {added_count}/{len(self.questions)} questions to queue. Current interval: {rate_index}")

    def get_sorted_results(self):
        """Get all results sorted by question arrival time"""
        with self.results_lock:
            # Add 'completion_time' key to each result, using 'generation_completion_time'
            for result in self.all_results:
                result["completion_time"] = result["generation_completion_time"]
            sorted_results = sorted(self.all_results, key=lambda x: x["arrival_time"])
            return sorted_results

    def update_policies_for_generation(self):
        """Update model policy and weight distribution based on current configuration"""
        print(f"\nUpdating policies for generation worker")

        # Find optimal configuration for the current interval
        optimal_config = self.find_optimal_config_with_linprog(self.batch_size)

        if optimal_config:
            print(f"Found optimal configuration for batch size {self.batch_size}")

            # Update model policy
            batch_size_split, num_batch = split_batch(self.batch_size)

            print(f"Updating model policy:")
            print(f"- Cache GPU%: {optimal_config['cache_gpu_percent']}")
            print(f"- Cache CPU%: {optimal_config['cache_cpu_percent']}")
            print(f"- Split batch size: {batch_size_split}, num_batch: {num_batch}")

            try:
                self.model.update_policy(
                    optimal_config["cache_gpu_percent"],
                    optimal_config["cache_cpu_percent"],
                    batch_size_split,
                    num_batch,
                )
                print("Model policy updated successfully")
            except Exception as e:
                print(f"Error updating model policy: {str(e)}")

            # Update weight distribution
            print(f"Updating weight distribution:")
            print(f"- Weight GPU%: {optimal_config['w_gpu_percent']}")
            print(f"- Weight CPU%: {optimal_config['w_cpu_percent']}")

            try:
                self.model.update_weight(optimal_config["w_gpu_percent"], optimal_config["w_cpu_percent"])
                print("Weight distribution updated successfully")
            except Exception as e:
                print(f"Error updating weight distribution: {str(e)}")

            # Update resident partitions
            old_resident = self.resident_partitions

            try:
                self.resident_partitions = self.update_resident_partitions(optimal_config["resident_partitions"])
                print(f"Updated resident partitions: {old_resident} -> {self.resident_partitions}")
            except Exception as e:
                print(f"Error updating resident partitions: {str(e)}")

            # Update current configuration
            self.current_config = optimal_config
            print("All policies updated successfully")
        else:
            print("Failed to find optimal configuration, keeping current settings")

        print(f"{'='*50}\n")

    def run(self, model_config=None, total_cpu_gb=None, gpu_memory_gb=None, partition_size_gb=None):
        """Run the dynamic pipeline processing"""
        # Generate question arrival times
        self._generate_arrival_times()

        # 确保所有计数器数组有正确的长度
        count_length = len(self.arrival_rates)

        # 初始化为零数组，长度与arrival_rates一致
        self.interval_gen_count = [0] * count_length
        self.interval_query_count = [0] * count_length
        self.interval_completion_count = [0] * count_length
        self.interval_queued_count = [0] * count_length

        # 打印初始计数器状态进行验证
        print("\nInitial interval counts:")
        print(f"- Target question counts: {self.interval_question_count}")
        print(f"- Query counts: {self.interval_query_count}")
        print(f"- Generation counts: {self.interval_gen_count}")
        print(f"- Completion counts: {self.interval_completion_count}")

        # Set initial batch size to 32 for first run
        self.batch_size = 32

        # Initialize current configuration with default values
        self.current_config = {
            "batch_size": self.batch_size,
            "cache_gpu_percent": 50,
            "cache_cpu_percent": 50,
            "w_gpu_percent": 50,
            "w_cpu_percent": 50,
            "resident_partitions": self.resident_partitions,
        }

        # Create worker threads
        arrival_thread = Thread(target=self._question_arrival_simulator)
        query_thread = Thread(target=self._query_worker)
        generation_thread = Thread(target=self._generation_worker)

        try:
            # Start all threads
            arrival_thread.start()
            query_thread.start()
            generation_thread.start()

            # Wait for query and generation to complete
            query_thread.join()
            generation_thread.join()

            # Signal stop for arrival simulator
            self.stop_event.set()
            arrival_thread.join()

            print("Pipeline processing completed")

            # Print final stats
            print("\n===== Final Statistics =====")
            print(f"Total questions processed: {sum(self.interval_completion_count)}")
            print(f"Questions by interval: {self.interval_question_count}")
            print(f"Completions by interval: {self.interval_completion_count}")
            print(f"Query counts by interval: {self.interval_query_count}")
            print(f"Generation counts by interval: {self.interval_gen_count}")
            print("============================\n")

            # Export timing results to CSV
            self.export_timing_to_csv()

        except KeyboardInterrupt:
            print("\nStopping pipeline manager...")
            self.stop_event.set()
            self.query_complete.set()
            self.gen_complete.set()
            arrival_thread.join()
            query_thread.join()
            generation_thread.join()

    def export_timing_to_csv(self):
        """Export all timing information to a CSV file"""
        csv_filename = (
            f"{self.model_name}_{self.arrival_rates}_{self.rate_change_interval}_question_timings_pipeline.csv"
        )
        print(f"Exporting timing information to {csv_filename}")

        # Sort results by question arrival time
        sorted_results = sorted(self.all_results, key=lambda x: x["arrival_time"])

        with open(csv_filename, "w", newline="") as csvfile:
            # 添加policy信息到fieldnames
            fieldnames = [
                "question_id",
                "interval_id",
                "arrival_rate",
                "arrival_time",
                "query_completion_time",
                "generation_completion_time",
                "batch_size",
                "w_gpu_percent",
                "w_cpu_percent",
                "cache_gpu_percent",
                "cache_cpu_percent",
                "resident_partitions",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for i, result in enumerate(sorted_results):
                question = result["question"]

                # Calculate relative times from base_time
                arrival_time_rel = result["arrival_time"] - self.base_time
                query_completion_time_rel = result["query_completion_time"] - self.base_time
                generation_completion_time_rel = result["generation_completion_time"] - self.base_time

                # 添加包含policy信息的行
                writer.writerow(
                    {
                        "question_id": i,  # Use index as question ID
                        "interval_id": result["interval_id"],
                        "arrival_rate": result["arrival_rate"],
                        "arrival_time": f"{arrival_time_rel:.4f}",
                        "query_completion_time": f"{query_completion_time_rel:.4f}",
                        "generation_completion_time": f"{generation_completion_time_rel:.4f}",
                        "batch_size": result["batch_size"],
                        "w_gpu_percent": result["w_gpu_percent"],
                        "w_cpu_percent": result["w_cpu_percent"],
                        "cache_gpu_percent": result["cache_gpu_percent"],
                        "cache_cpu_percent": result["cache_cpu_percent"],
                        "resident_partitions": result["resident_partitions"],
                    }
                )

        print(f"Successfully exported {len(sorted_results)} question records to {csv_filename}")

    @property
    def total_questions(self):
        return len(self.questions)

    @property
    def expected_duration(self):
        """Return expected total runtime in seconds"""
        if self.questions:
            return self.questions[-1].arrival_time - self.base_time
        return 0.0
