from typing import List, Dict, Optional, Any
import time
import torch
import numpy as np
from tqdm import tqdm
from utils import Question, log_timing, batch_query
from vllm import LLM, SamplingParams


class VLLMProcessor:
    """
    Sequential Processor that uses VLLM for generation.
    Queries are processed in batches, with each batch fully completing (query + generation)
    before the next batch starts. Batch size is dynamically adjusted based on interval arrival rate.
    """

    def __init__(
        self,
        questions: List[Dict],
        batch_size: int,  # Now used as maximum batch size
        arrival_rates: List[float],  # questions per minute (multiple rates)
        rate_change_interval: int,  # interval time (seconds)
        embedding_model,
        model_name: str,  # Model name for VLLM initialization
        tokenizer,
        collection,
        partition_names: List[str],
        timing_stats: Dict[str, List[float]],
        resident_partitions: int = 0,
        base_time: float = None,
        max_new_tokens: int = 16,
        total_cpu_gb: int = 166,
        gpu_memory_gb: int = 12,
        gpu_memory_utilization: float = 0.5,
        partition_size_gb: int = 10,
        seed: int = 42,
        vllm_batch_size: Optional[int] = None,  # Max concurrent requests for VLLM'
    ):
        self.questions = [
            Question(question_text=q["question"], doc_id=q["doc_id"], arrival_time=0.0) for q in questions
        ]
        self.max_batch_size = 64  # Renamed to max_batch_size
        self.arrival_rates = arrival_rates if arrival_rates else [16]  # Default single arrival rate
        self.rate_change_interval = rate_change_interval
        self.embedding_model = embedding_model
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.collection = collection
        self.partition_names = partition_names
        self.timing_stats = timing_stats
        self.resident_partitions = resident_partitions
        self.base_time = base_time if base_time else time.time()
        self.results = []
        self.max_new_tokens = max_new_tokens
        self.gpu_memory_utilization = gpu_memory_utilization
        self.vllm_batch_size = vllm_batch_size if vllm_batch_size else batch_size
        self.total_cpu_gb = total_cpu_gb - (resident_partitions + 1) * partition_size_gb
        self.gpu_memory_gb = gpu_memory_gb

        # New interval tracking variables
        self.interval_question_count = []
        self.interval_completion_count = []

        # VLLM related attributes
        self.vllm_model = None
        self.sampling_params = SamplingParams(
            temperature=0.0, max_tokens=max_new_tokens, stop=None  # Deterministic generation
        )

        np.random.seed(seed)

        # Generate arrival times
        self._generate_arrival_times()

        # Initialize VLLM model
        self._init_vllm_model()

    def _init_vllm_model(self):
        """Initialize the VLLM model"""
        print(f"\nInitializing VLLM model: {self.model_name}")
        try:
            self.vllm_model = LLM(
                model=self.model_name,
                gpu_memory_utilization=self.gpu_memory_utilization,
                tensor_parallel_size=1,  # Adjust if using multiple GPUs
                trust_remote_code=True,
                cpu_offload_gb=self.total_cpu_gb,
                max_model_len=512 + 16,
            )
            print("VLLM model initialized successfully")
        except Exception as e:
            print(f"Error initializing VLLM model: {e}")
            raise

    def _generate_arrival_times(self):
        """Generate arrival times based on time-varying arrival rates using a Poisson process"""
        # Ensure base_time is correctly set
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

        # Print expected question counts
        print(f"Expected questions per interval based on rates: {[int(q) for q in expected_questions_per_interval]}")
        print(f"Total expected questions: {int(total_expected_questions)}")

        # Generate arrival times for each interval using Poisson process
        all_arrivals = []
        
        for i, rate in enumerate(self.arrival_rates):
            # Convert rate from questions/minute to questions/second
            rate_per_second = rate / 60.0
            
            # Start and end time for this interval
            interval_start = self.base_time + sum(interval_durations[:i])
            interval_end = interval_start + interval_durations[i]
            
            # Generate arrival times using homogeneous Poisson process
            current_time = interval_start
            interval_arrivals = []
            
            while current_time < interval_end:
                # Generate next inter-arrival time (exponential distribution)
                inter_arrival = np.random.exponential(1.0 / rate_per_second)
                current_time += inter_arrival
                
                if current_time < interval_end:
                    interval_arrivals.append((current_time, i))  # (arrival_time, interval_id)
            
            all_arrivals.extend(interval_arrivals)
            print(f"Interval {i}: Generated {len(interval_arrivals)} arrivals with rate {rate} q/min")
        
        # Sort all arrivals by time
        all_arrivals.sort()
        
        # Check if we have enough questions available
        if len(all_arrivals) > len(self.questions):
            print(f"Warning: Generated {len(all_arrivals)} arrivals but only have {len(self.questions)} questions")
            # Truncate arrivals to match available questions
            all_arrivals = all_arrivals[:len(self.questions)]
        
        # Count questions per interval in the final set
        questions_per_interval = [0] * len(self.arrival_rates)
        for _, interval_id in all_arrivals:
            questions_per_interval[interval_id] += 1
        
        # Initialize tracking arrays
        self.interval_question_count = questions_per_interval
        self.interval_completion_count = [0] * len(self.arrival_rates)
        
        # Assign arrival times to questions
        used_questions = []
        for i, (arrival_time, interval_id) in enumerate(all_arrivals):
            if i < len(self.questions):
                question = self.questions[i]
                question.arrival_time = arrival_time
                question.interval_id = interval_id
                used_questions.append(question)
        
        # Only process required questions
        self.questions = used_questions
        
        # Sort questions by arrival time
        self.questions.sort(key=lambda x: x.arrival_time)
        
        # Validate generated timestamps
        if self.questions:
            min_arrival = min(q.arrival_time for q in self.questions)
            max_arrival = max(q.arrival_time for q in self.questions)
            print(f"Final questions per interval: {questions_per_interval}")
            print(f"Total questions to be processed: {sum(questions_per_interval)}")
            print(f"Arrival times range: {min_arrival - self.base_time:.2f}s to {max_arrival - self.base_time:.2f}s")

    def get_current_interval(self, current_time):
        """Get the current interval based on timestamp"""
        if self.base_time is None:
            return 0

        elapsed_time = current_time - self.base_time
        interval_index = int(elapsed_time / self.rate_change_interval)

        # Cap at the last defined interval
        return min(interval_index, len(self.arrival_rates) - 1)

    def get_dynamic_batch_size(self, current_time, available_questions):
        """
        Calculate dynamic batch size based on current interval's arrival rate
        and available questions
        """
        interval = self.get_current_interval(current_time)

        # Get arrival rate for the current interval
        arrival_rate = self.arrival_rates[interval]

        # Calculate batch size as arrival rate * 4
        dynamic_batch_size = int(arrival_rate * 4)

        # Ensure it doesn't exceed the max batch size
        dynamic_batch_size = min(dynamic_batch_size, self.max_batch_size)

        # Ensure it doesn't exceed available questions
        dynamic_batch_size = min(dynamic_batch_size, available_questions)

        # Ensure batch size is at least 1
        return max(1, dynamic_batch_size)

    def format_prompt(self, context: str, query: str) -> str:
        """Format prompt for the LLM"""
        return f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"

    def process_batch_queries(self, batch: List[Question]):
        """Process a batch of queries to retrieve contexts"""
        query_texts = [q.question_text for q in batch]
        questions_dict = [{"question": q.question_text, "doc_id": q.doc_id} for q in batch]

        # Release all partitions before processing new batch if not using resident partitions
        if not self.resident_partitions:
            self.collection.release()

        # Embedding generation
        embed_start = time.time()  # This will be our wait_end_time
        query_embeddings = self.embedding_model.embed_documents(query_texts)
        log_timing(self.timing_stats, "embedding_time", time.time() - embed_start)

        # Query phase
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

        # Record query completion time
        query_completion_time = time.time()

        # Attach query completion time and wait_end_time to each result
        for result in batch_results:
            result["query_completion_time"] = query_completion_time
            result["query_start_time"] = embed_start  # Add wait_end_time

        return batch, batch_results

    def process_batch_generation(self, batch, batch_results):
        """Process generation for a batch sequentially"""
        print(f"Starting generation for batch of {len(batch)} questions")
        generation_start_time = time.time()

        prompts = []
        metadata_list = []

        # Prepare all prompts and metadata
        for question, result in zip(batch, batch_results):
            context = result.get("answer", "")
            query = question.question_text
            prompt = self.format_prompt(context, query)
            prompts.append(prompt)

            # Store metadata for retrieval
            metadata_list.append(
                {
                    "question": question,
                    "result": result,
                    "prompt": prompt,
                    "generation_start_time": time.time(),
                }
            )

        # Generate all outputs at once using VLLM
        outputs = self.vllm_model.generate(prompts, self.sampling_params)

        # Process results
        for output, metadata in zip(outputs, metadata_list):
            question = metadata["question"]
            result = metadata["result"]
            full_prompt = metadata["prompt"]
            generation_start_time = metadata["generation_start_time"]

            # Add generation time to timing stats
            generation_time = time.time() - generation_start_time
            log_timing(self.timing_stats, "generation_time", generation_time)

            # Get the model's output
            model_output = output.outputs[0].text.strip()

            # Combine prompt and response as the complete response
            complete_response = f"{full_prompt} {model_output}"

            # Store this as the llm_response
            result["llm_response"] = complete_response

            completion_time = time.time()

            # Store the result directly since we're operating sequentially
            self.results.append(
                {
                    "question": question,
                    "result": result,
                    "arrival_time": question.arrival_time,
                    "query_start_time": result.get("query_start_time", time.time()),
                    "query_completion_time": result.get("query_completion_time", time.time()),
                    "completion_time": completion_time,
                    "interval_id": getattr(question, "interval_id", 0),
                }
            )

            # Update interval completion count
            if hasattr(question, "interval_id"):
                interval_idx = question.interval_id
                if 0 <= interval_idx < len(self.interval_completion_count):
                    self.interval_completion_count[interval_idx] += 1

        generation_end_time = time.time()
        print(f"Completed generation for batch in {generation_end_time - generation_start_time:.2f}s")
        return len(batch)

    def export_timing_to_csv(self):
        """Export all timing information to a CSV file"""
        import csv

        csv_filename = f"{self.model_name}_{self.arrival_rates}_{self.rate_change_interval}_question_timings_efficient.csv"
        print(f"Exporting timing information to {csv_filename}")

        # Sort results by question arrival time
        sorted_results = sorted(self.results, key=lambda x: x["arrival_time"])

        with open(csv_filename, "w", newline="") as csvfile:
            fieldnames = [
                "question_id",
                "interval_id",
                "arrival_rate",
                "arrival_time",
                "query_start_time",  # Added wait_end_time
                "query_completion_time",
                "generation_completion_time",
                "batch_size",
                "resident_partitions",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for i, result in enumerate(sorted_results):
                question = result["question"]

                # Calculate relative times from base_time
                arrival_time_rel = result["arrival_time"] - self.base_time
                query_start_time_rel = (
                    result.get("query_start_time", result["arrival_time"]) - self.base_time
                )  # Added with fallback
                query_completion_time_rel = result["query_completion_time"] - self.base_time
                generation_completion_time_rel = result["completion_time"] - self.base_time

                # Get interval ID and corresponding arrival rate
                interval_id = getattr(question, "interval_id", 0)
                arrival_rate = self.arrival_rates[interval_id] if interval_id < len(self.arrival_rates) else 0

                writer.writerow(
                    {
                        "question_id": i,  # Use index as question ID
                        "interval_id": interval_id,
                        "arrival_rate": arrival_rate,
                        "arrival_time": f"{arrival_time_rel:.4f}",
                        "query_start_time": f"{query_start_time_rel:.4f}",  # Added wait_end_time
                        "query_completion_time": f"{query_completion_time_rel:.4f}",
                        "generation_completion_time": f"{generation_completion_time_rel:.4f}",
                        "batch_size": self.max_batch_size,  # Using max_batch_size for consistency
                        "resident_partitions": self.resident_partitions,
                    }
                )

        print(f"Successfully exported {len(sorted_results)} question records to {csv_filename}")

    def run(self):
        """Run processing with sequential query and generation for each batch, using dynamic batch sizes"""
        import threading

        next_question_idx = 0
        total_questions = len(self.questions)
        current_batch = []

        print("\nStarting sequential VLLM processing with dynamic batch size")
        print(f"Total questions: {total_questions}")
        print(f"Maximum batch size: {self.max_batch_size}")
        print(f"Dynamic batch sizing: arrival_rate * 4")

        # 创建问题到达监控线程
        arrival_monitor_stop = threading.Event()

        def arrival_monitor():
            """监控问题到达的线程函数"""
            monitor_idx = 0
            while not arrival_monitor_stop.is_set() and monitor_idx < total_questions:
                current_time = time.time()
                # 检查问题到达
                while monitor_idx < total_questions:
                    question = self.questions[monitor_idx]
                    # 如果问题到达时间未到则等待
                    if question.arrival_time > current_time:
                        break
                    # 打印到达信息
                    elapsed_time = question.arrival_time - self.base_time
                    print(f"Question arrived at {elapsed_time:.2f}s: {question.question_text[:50]}...")
                    monitor_idx += 1
                # 短暂休眠以避免CPU占用过高
                time.sleep(0.1)

        # 启动监控线程
        monitor_thread = threading.Thread(target=arrival_monitor)
        monitor_thread.daemon = True
        monitor_thread.start()

        try:
            while next_question_idx < total_questions:
                current_time = time.time()
                
                # 获取当前时间间隔和到达率
                current_interval = self.get_current_interval(current_time)
                current_arrival_rate = self.arrival_rates[current_interval]
                
                # 计算当前间隔的目标批次大小
                target_batch_size = min(int(current_arrival_rate * 4), self.max_batch_size)
                target_batch_size = max(1, target_batch_size)  # 确保至少为1
                
                # 收集问题直到达到目标批次大小或没有更多已到达的问题
                while (len(current_batch) < target_batch_size and 
                    next_question_idx < total_questions and 
                    self.questions[next_question_idx].arrival_time <= current_time):
                    current_batch.append(self.questions[next_question_idx])
                    next_question_idx += 1
                
                # 如果已收集一些问题，并且达到目标批次大小或没有更多已到达的问题
                if current_batch and (len(current_batch) >= target_batch_size or 
                                    next_question_idx >= total_questions or
                                    (next_question_idx < total_questions and 
                                    self.questions[next_question_idx].arrival_time > current_time)):
                    batch_size = len(current_batch)

                    # 记录批处理信息
                    print(f"\nCurrent interval: {current_interval}, Arrival rate: {current_arrival_rate}/min")
                    print(f"Target batch size: {target_batch_size} (calculated as arrival_rate * 4 = {current_arrival_rate * 4})")
                    print(f"Processing batch of {batch_size} questions sequentially")

                    # 第1步：处理此批次的查询
                    print(f"Step 1: Query processing for batch at {time.time() - self.base_time:.2f}s")
                    batch, batch_results = self.process_batch_queries(current_batch)
                    print(f"Query processing completed at {time.time() - self.base_time:.2f}s")

                    # 第2步：处理此批次的生成（等待直到完成）
                    print(f"Step 2: Generation processing for batch at {time.time() - self.base_time:.2f}s")
                    self.process_batch_generation(batch, batch_results)
                    print(f"Generation completed at {time.time() - self.base_time:.2f}s")

                    print(f"Completed {next_question_idx}/{total_questions} questions")
                    current_batch = []
                
                # 如果批次中没有问题，需要等待下一个问题到达
                if not current_batch and next_question_idx < total_questions:
                    wait_time = self.questions[next_question_idx].arrival_time - current_time
                    if wait_time > 0:
                        time.sleep(min(wait_time, 1.0))  # 最多等待1秒

        finally:
            # 停止监控线程
            arrival_monitor_stop.set()
            monitor_thread.join(timeout=1.0)

        # 打印最终统计信息并导出结果
        print("\n===== Final Statistics =====")
        print(f"Total questions processed: {len(self.results)}")
        print("============================\n")
        print("Exporting timing results to CSV...")
        self.export_timing_to_csv()

    def get_sorted_results(self):
        """Get results sorted by arrival time"""
        return sorted(self.results, key=lambda x: x["arrival_time"])

    @property
    def expected_duration(self):
        """Return expected total runtime (seconds)"""
        if self.questions:
            return self.questions[-1].arrival_time - self.base_time
        return 0.0
