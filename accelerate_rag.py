from typing import List, Dict
import time
import torch
import numpy as np
from tqdm import tqdm
from utils import Question, log_timing, batch_query, batch_generate_responses
from transformers import AutoModelForCausalLM


class AccelerateProcessor:
    """
    Sequential Processor that uses HuggingFace models for generation.
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
        model_name: str,
        tokenizer,
        collection,
        partition_names: List[str],
        timing_stats: Dict[str, List[float]],
        gpu_memory_gb: int,
        resident_partitions: int = 0,
        base_time: float = None,
        max_new_tokens: int = 32,
        seed: int = 42,
        dynagen: bool = False,
        env=None,
    ):
        self.questions = [
            Question(question_text=q["question"], doc_id=q["doc_id"], arrival_time=0.0) for q in questions
        ]
        self.max_batch_size = batch_size  # Renamed to max_batch_size
        self.arrival_rates = arrival_rates if arrival_rates else [16]  # Default single arrival rate
        self.rate_change_interval = rate_change_interval
        self.embedding_model = embedding_model
        self.model_name = model_name

        # Calculate memory requirements
        max_memory_per_batch = 0.5  # GiB per batch
        max_gpu_memory_size = gpu_memory_gb - (self.max_batch_size * max_memory_per_batch)

        # Initialize model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            max_memory={0: f"{max_gpu_memory_size}GiB"},
        )

        self.tokenizer = tokenizer
        self.collection = collection
        self.partition_names = partition_names
        self.timing_stats = timing_stats
        self.resident_partitions = resident_partitions
        self.base_time = base_time if base_time else time.time()
        self.results = []
        self.max_new_tokens = max_new_tokens

        # Add dynagen and env attributes if they exist in the original implementation
        self.dynagen = dynagen
        self.env = env

        # New interval tracking variables
        self.interval_question_count = []
        self.interval_completion_count = []

        np.random.seed(seed)

        # Generate arrival times
        self._generate_arrival_times()

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

        # Prepare all inputs for generation
        prompts = []
        metadata_list = []

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

        # Use batch_generate_responses for actual generation
        batch_responses, updated_timing_stats = batch_generate_responses(
            model=self.model,
            tokenizer=self.tokenizer,
            batch_results=batch_results,
            max_new_tokens=self.max_new_tokens,
            batch_size=len(batch),
            timing_stats=self.timing_stats,
            dynagen=self.dynagen,
            env=self.env if self.dynagen else None,
        )
        torch.cuda.empty_cache()
        self.timing_stats.update(updated_timing_stats)

        # Process results similar to VLLMProcessor
        for output, metadata in zip(batch_responses, metadata_list):
            question = metadata["question"]
            result = metadata["result"]
            # Add generation time to timing stats
            generation_time = time.time() - metadata["generation_start_time"]
            log_timing(self.timing_stats, "generation_time", generation_time)

            completion_time = time.time()

            # Store the result directly
            self.results.append(
                {
                    "question": question,
                    "result": output,  # Use the output from batch_generate_responses
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

        csv_filename = (
            f"{self.model_name}_{self.arrival_rates}_{self.rate_change_interval}_question_timings_accelerate.csv"
        )
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

        print("\nStarting sequential baseline processing with dynamic batch size")
        print(f"Total questions: {total_questions}")
        print(f"Maximum batch size: {self.max_batch_size}")
        print(f"Dynamic batch sizing: arrival_rate * 4")

        # Create question arrival monitoring thread
        arrival_monitor_stop = threading.Event()

        def arrival_monitor():
            """Thread function that monitors question arrivals"""
            monitor_idx = 0
            while not arrival_monitor_stop.is_set() and monitor_idx < total_questions:
                current_time = time.time()
                # Check for question arrivals
                while monitor_idx < total_questions:
                    question = self.questions[monitor_idx]
                    # If question arrival time hasn't come yet, wait
                    if question.arrival_time > current_time:
                        break
                    # Print arrival information
                    elapsed_time = question.arrival_time - self.base_time
                    print(f"Question arrived at {elapsed_time:.2f}s: {question.question_text[:50]}...")
                    monitor_idx += 1
                # Short sleep to avoid high CPU usage
                time.sleep(0.1)

        # Start monitoring thread
        monitor_thread = threading.Thread(target=arrival_monitor)
        monitor_thread.daemon = True
        monitor_thread.start()

        try:
            while next_question_idx < total_questions:
                current_time = time.time()

                # Get current interval and arrival rate
                current_interval = self.get_current_interval(current_time)
                current_arrival_rate = self.arrival_rates[current_interval]

                # Calculate current interval's target batch size
                target_batch_size = min(int(current_arrival_rate * 4), self.max_batch_size)
                target_batch_size = max(1, target_batch_size)  # Ensure at least 1

                # Collect questions until reaching target batch size or no more arrived questions
                while (
                    len(current_batch) < target_batch_size
                    and next_question_idx < total_questions
                    and self.questions[next_question_idx].arrival_time <= current_time
                ):
                    current_batch.append(self.questions[next_question_idx])
                    next_question_idx += 1

                # If some questions are collected, and reached target batch size or no more arrived questions
                if current_batch and (
                    len(current_batch) >= target_batch_size
                    or next_question_idx >= total_questions
                    or (
                        next_question_idx < total_questions
                        and self.questions[next_question_idx].arrival_time > current_time
                    )
                ):
                    batch_size = len(current_batch)

                    # Log batch processing info
                    print(f"\nCurrent interval: {current_interval}, Arrival rate: {current_arrival_rate}/min")
                    print(
                        f"Target batch size: {target_batch_size} (calculated as arrival_rate * 4 = {current_arrival_rate * 4})"
                    )
                    print(f"Processing batch of {batch_size} questions sequentially")

                    # Step 1: Process queries for this batch
                    print(f"Step 1: Query processing for batch at {time.time() - self.base_time:.2f}s")
                    batch, batch_results = self.process_batch_queries(current_batch)
                    print(f"Query processing completed at {time.time() - self.base_time:.2f}s")

                    # Step 2: Process generation for this batch (wait until complete)
                    print(f"Step 2: Generation processing for batch at {time.time() - self.base_time:.2f}s")
                    self.process_batch_generation(batch, batch_results)
                    print(f"Generation completed at {time.time() - self.base_time:.2f}s")

                    print(f"Completed {next_question_idx}/{total_questions} questions")
                    current_batch = []

                # If no questions in batch, need to wait for next question arrival
                if not current_batch and next_question_idx < total_questions:
                    wait_time = self.questions[next_question_idx].arrival_time - current_time
                    if wait_time > 0:
                        time.sleep(min(wait_time, 1.0))  # Wait at most 1 second

        finally:
            # Stop monitoring thread
            arrival_monitor_stop.set()
            monitor_thread.join(timeout=1.0)

        # Print final statistics and export results
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
