from typing import List, Dict, Optional, Any
import time
import torch
import numpy as np
from tqdm import tqdm
from utils import Question, log_timing, batch_query
from vllm import LLM, SamplingParams
import queue
import threading


class VLLMProcessor:
    """
    Processor that uses VLLM for generation with iteration-level scheduling.
    Queries are processed in batches, while generation uses VLLM's scheduling capabilities.
    """

    def __init__(
        self,
        questions: List[Dict],
        batch_size: int,
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
        gpu_memory_utilization: float = 0.85,
        seed: int = 42,
        vllm_batch_size: Optional[int] = None,  # Max concurrent requests for VLLM
    ):
        self.questions = [
            Question(question_text=q["question"], doc_id=q["doc_id"], arrival_time=0.0) for q in questions
        ]
        self.batch_size = batch_size
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

        # New interval tracking variables
        self.interval_question_count = []
        self.interval_completion_count = []

        # VLLM related attributes
        self.vllm_model = None
        self.sampling_params = SamplingParams(
            temperature=0.0, max_tokens=max_new_tokens, stop=None  # Deterministic generation
        )

        # Queue for generation requests
        self.generation_queue = queue.Queue()
        self.completion_queue = queue.Queue()
        self.active_generations = 0
        self.generation_lock = threading.Lock()

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
                cpu_offload_gb=100,
                max_model_len=512,
            )
            print("VLLM model initialized successfully")
        except Exception as e:
            print(f"Error initializing VLLM model: {e}")
            raise

    def _generate_arrival_times(self):
        """Generate arrival times based on time-varying arrival rates"""
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

        # Use the expected number of questions for each interval without scaling
        questions_per_interval = [int(round(count)) for count in expected_questions_per_interval]

        # Ensure question count doesn't exceed available questions
        if len(self.questions) < total_expected_questions:
            print(
                f"Warning: Not enough questions available. Need {total_expected_questions} but only have {len(self.questions)}"
            )
            # Scale down question counts proportionally
            scale_factor = len(self.questions) / total_expected_questions
            questions_per_interval = [int(round(count * scale_factor)) for count in questions_per_interval]
            # Ensure at least one question per interval
            questions_per_interval = [max(1, count) for count in questions_per_interval]
            # Adjust total to not exceed available questions
            while sum(questions_per_interval) > len(self.questions):
                # Find the largest interval and reduce by one
                max_idx = questions_per_interval.index(max(questions_per_interval))
                questions_per_interval[max_idx] -= 1

        # Print final question allocation
        print(f"Final questions per interval: {questions_per_interval}")
        print(f"Total questions to be processed: {sum(questions_per_interval)}")

        # Initialize tracking arrays for questions by interval
        self.interval_question_count = questions_per_interval
        self.interval_completion_count = [0] * len(self.arrival_rates)

        # Only use required number of questions, not all questions
        used_questions = self.questions[: sum(questions_per_interval)]

        # Assign interval ID for each question
        question_index = 0
        for i, question_count in enumerate(questions_per_interval):
            for j in range(question_count):
                if question_index < len(used_questions):
                    # Set interval ID directly
                    used_questions[question_index].interval_id = i
                    # Set arrival time
                    interval_start = self.base_time + sum(interval_durations[:i])
                    # Distribute arrival times evenly
                    used_questions[question_index].arrival_time = (
                        interval_start + (j / question_count) * interval_durations[i]
                        if question_count > 0
                        else interval_start
                    )
                    question_index += 1

        # Only process required questions
        self.questions = used_questions

        # Sort questions by arrival time
        self.questions.sort(key=lambda x: x.arrival_time)

        # Validate generated timestamps
        if self.questions:
            min_arrival = min(q.arrival_time for q in self.questions)
            max_arrival = max(q.arrival_time for q in self.questions)
            print(f"Arrival times range: {min_arrival - self.base_time:.2f}s to {max_arrival - self.base_time:.2f}s")

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
        embed_start = time.time()
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

        return batch, batch_results

    def submit_to_vllm(self, batch, batch_results):
        """Submit a batch of requests to VLLM and get results"""
        prompts = []
        metadata_list = []

        # Prepare all prompts and metadata
        for question, result in zip(batch, batch_results):
            context = result.get("answer", "")
            query = question.question_text
            prompt = self.format_prompt(context, query)
            prompts.append(prompt)

            request_id = f"{question.doc_id}_{hash(question.question_text)}"

            # Store metadata for retrieval
            metadata_list.append(
                {
                    "question": question, 
                    "result": result,  # Keep the original result object
                    "prompt": prompt,  # Store the full prompt
                    "generation_start_time": time.time(), 
                    "request_id": request_id
                }
            )

        # Update active generations count
        with self.generation_lock:
            self.active_generations += len(prompts)

        # Generate all outputs at once
        outputs = self.vllm_model.generate(prompts, self.sampling_params)

        # Process outputs immediately
        for output, metadata in zip(outputs, metadata_list):
            question = metadata["question"]
            result = metadata["result"]  # Get the original result object
            full_prompt = metadata["prompt"]  # Get the stored prompt
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
            self.completion_queue.put(
                {
                    "question": question,
                    "result": result,  # Use the modified original result object
                    "arrival_time": question.arrival_time,
                    "completion_time": completion_time,
                }
            )

            # Update interval completion count
            if hasattr(question, "interval_id"):
                interval_idx = question.interval_id
                if 0 <= interval_idx < len(self.interval_completion_count):
                    self.interval_completion_count[interval_idx] += 1

        # Decrease active generation count
        with self.generation_lock:
            self.active_generations -= len(prompts)

    def generation_worker(self):
        """Worker thread for handling generation requests"""
        print("Starting VLLM generation worker thread")

        try:
            while True:
                # Check if there are items in the generation queue
                try:
                    batch, batch_results = self.generation_queue.get(timeout=0.1)
                except queue.Empty:
                    # No items in queue, check if we should exit
                    if self.should_stop_generation.is_set():
                        break
                    continue

                # Submit to VLLM and get results directly
                self.submit_to_vllm(batch, batch_results)

                # Mark this batch as processed
                self.generation_queue.task_done()

        except Exception as e:
            print(f"Error in generation worker: {e}")

        print("VLLM generation worker thread stopped")


    # Remove the check_completed_generations method since it relies on a non-existent function
    # And modify the run method to remove references to it
    def run(self):
        """Run processing with VLLM for generation"""
        import threading

        next_question_idx = 0
        total_questions = len(self.questions)
        current_batch = []
        current_interval_id = -1

        # Initialize tracking
        self.should_stop_generation = threading.Event()

        print("\nStarting VLLM processing with fixed query batch size and dynamic generation")
        print(f"Questions by interval: {self.interval_question_count}")
        print(f"Total questions: {total_questions}")
        print(f"Query batch size: {self.batch_size}")
        print(f"VLLM max concurrent generations: {self.vllm_batch_size}\n")

        # Create question arrival monitor thread
        arrival_monitor_stop = threading.Event()

        def arrival_monitor():
            """Thread function to monitor question arrivals"""
            monitor_idx = 0
            while not arrival_monitor_stop.is_set() and monitor_idx < total_questions:
                current_time = time.time()

                # Check for question arrivals
                while monitor_idx < total_questions:
                    question = self.questions[monitor_idx]

                    # Wait if question arrival time hasn't reached yet
                    if question.arrival_time > current_time:
                        break

                    # Question has arrived, print info
                    elapsed_time = question.arrival_time - self.base_time

                    print(
                        f"Question arrived at {elapsed_time:.2f}s: {question.question_text[:50]}... (Interval: {question.interval_id})"
                    )
                    monitor_idx += 1

                # Short sleep to avoid excessive CPU usage
                time.sleep(0.1)

        # Start question arrival monitor
        monitor_thread = threading.Thread(target=arrival_monitor)
        monitor_thread.daemon = True
        monitor_thread.start()

        # Start generation worker thread
        generation_thread = threading.Thread(target=self.generation_worker)
        generation_thread.daemon = True
        generation_thread.start()

        # Start completion collector thread
        def completion_collector():
            """Thread function to collect completed results"""
            while not self.should_stop_generation.is_set():
                try:
                    result = self.completion_queue.get(timeout=0.1)
                    self.results.append(result)
                    self.completion_queue.task_done()

                    # Print progress
                    completed = len(self.results)
                    if completed % 10 == 0 or completed == len(self.questions):
                        print(f"Completed {completed}/{len(self.questions)} questions")
                        print(f"Completions by interval: {self.interval_completion_count}")
                except queue.Empty:
                    # Just continue waiting
                    continue

        completion_thread = threading.Thread(target=completion_collector)
        completion_thread.daemon = True
        completion_thread.start()

        try:
            while next_question_idx < total_questions:
                # Check if a new interval has started
                if next_question_idx < total_questions:
                    next_question = self.questions[next_question_idx]
                    next_interval_id = next_question.interval_id

                    # If new interval and current batch has questions, process them
                    if current_interval_id != -1 and next_interval_id != current_interval_id and current_batch:
                        print(
                            f"\nProcessing final batch of interval {current_interval_id} with {len(current_batch)} questions"
                        )
                        batch, batch_results = self.process_batch_queries(current_batch)
                        self.generation_queue.put((batch, batch_results))
                        print(f"Queued batch for generation at {time.time() - self.base_time:.2f}s")
                        current_batch = []

                    # Update current interval
                    current_interval_id = next_interval_id

                # Collect questions for current interval until batch size is reached or interval ends
                while next_question_idx < total_questions:
                    next_question = self.questions[next_question_idx]

                    # Break if question belongs to a new interval
                    if next_question.interval_id != current_interval_id:
                        break

                    # Wait for question arrival time
                    current_time = time.time()
                    if next_question.arrival_time > current_time:
                        wait_time = next_question.arrival_time - current_time
                        time.sleep(min(wait_time, 1.0))

                    # Add question to current batch
                    current_batch.append(next_question)
                    next_question_idx += 1

                    # Process batch if full
                    if len(current_batch) >= self.batch_size:
                        print(
                            f"\nProcessing full batch of {len(current_batch)} questions (Interval: {current_interval_id})"
                        )
                        batch, batch_results = self.process_batch_queries(current_batch)
                        self.generation_queue.put((batch, batch_results))
                        print(f"Queued batch for generation at {time.time() - self.base_time:.2f}s")
                        print(f"Processed {next_question_idx}/{total_questions} questions")
                        current_batch = []
                        break  # Exit inner loop, check interval state again

                # Check if interval end reached, process final partial batch
                if next_question_idx < total_questions:
                    next_question = self.questions[next_question_idx]
                    if next_question.interval_id != current_interval_id and current_batch:
                        print(
                            f"\nProcessing final batch of interval {current_interval_id} with {len(current_batch)} questions"
                        )
                        batch, batch_results = self.process_batch_queries(current_batch)
                        self.generation_queue.put((batch, batch_results))
                        print(f"Queued batch for generation at {time.time() - self.base_time:.2f}s")
                        current_batch = []

                # Process any remaining questions
                if next_question_idx >= total_questions and current_batch:
                    print(
                        f"\nProcessing final remaining batch with {len(current_batch)} questions (Interval: {current_interval_id})"
                    )
                    batch, batch_results = self.process_batch_queries(current_batch)
                    self.generation_queue.put((batch, batch_results))
                    print(f"Queued final batch for generation at {time.time() - self.base_time:.2f}s")
                    current_batch = []

            # Wait for all generations to complete
            print("\nWaiting for all generations to complete...")
            self.generation_queue.join()

            # Give some time for final completions to be processed
            remaining_attempts = 10
            while remaining_attempts > 0 and len(self.results) < total_questions:
                print(f"Waiting for final completions: {len(self.results)}/{total_questions}")
                # Removed call to self.check_completed_generations()
                time.sleep(1.0)
                remaining_attempts -= 1

        finally:
            # Stop all threads
            arrival_monitor_stop.set()
            monitor_thread.join(timeout=1.0)

            self.should_stop_generation.set()
            generation_thread.join(timeout=1.0)
            completion_thread.join(timeout=1.0)

        # Print final statistics
        print("\n===== Final Statistics =====")
        print(f"Total questions processed: {len(self.results)}")
        print(f"Questions by interval: {self.interval_question_count}")
        print(f"Completions by interval: {self.interval_completion_count}")
        print("============================\n")

    def get_sorted_results(self):
        """Get results sorted by arrival time"""
        return sorted(self.results, key=lambda x: x["arrival_time"])

    @property
    def expected_duration(self):
        """Return expected total runtime (seconds)"""
        if self.questions:
            return self.questions[-1].arrival_time - self.base_time
        return 0.0
