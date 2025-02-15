import time
from typing import List, Dict, Optional
from tqdm import tqdm
from utils import batch_query, batch_generate_responses, batch_query_qdrant
from pymilvus import Partition
from utils import Question

class DynOfflineProcessor:
    """Offline batch processor with dynamic configuration support"""

    def __init__(
        self,
        questions: List[Dict],
        batch_size: int,
        embedding_model,
        model,
        tokenizer,
        collection,
        partition_names: List[str],
        resident_partitions: int = 0,
        use_qdrant: bool = False,
        dynagen: bool = False,
        cache_gpu_percent: int = 0,
        cache_cpu_percent: int = 20,
        env: Optional[object] = None,
    ):
        self.questions = questions
        self.batch_size = batch_size
        self.embedding_model = embedding_model
        self.model = model
        self.tokenizer = tokenizer
        self.collection = collection
        self.partition_names = partition_names
        self.resident_partitions = resident_partitions
        self.use_qdrant = use_qdrant
        self.dynagen = dynagen
        self.env = env
        self.cache_gpu_percent = cache_gpu_percent
        self.cache_cpu_percent = cache_cpu_percent
        self.results = []
        self.base_time = time.time()

        # 保留详细的时间统计
        self.timing_breakdown = {
            "embedding_total": 0.0,
            "query_total": 0.0,
            "generation_total": 0.0,
            "total_processing": 0.0,
            "embedding_times": [],
            "query_times": [],
            "generation_times": [],
        }

    def update_resident_partitions(self, new_resident_partitions: int):
        """Update the number of resident partitions"""
        if self.use_qdrant:
            return

        # Keep track of currently loaded partitions
        if not hasattr(self, "loaded_partitions"):
            self.loaded_partitions = set(range(self.resident_partitions))

        # Calculate new desired partitions
        new_loaded = set(range(new_resident_partitions))

        # Only release partitions that aren't needed in new configuration
        to_release = self.loaded_partitions - new_loaded
        # Only load partitions that aren't already loaded
        to_load = new_loaded - self.loaded_partitions

        print(f"Currently loaded partitions: {self.loaded_partitions}")
        print(f"Releasing partitions: {to_release}")
        print(f"Loading partitions: {to_load}")

        if to_release:
            release_partition_names = [f"partition_{i}" for i in to_release]
            # self.collection.release(partition_names=release_partition_names)
            for partition_name in release_partition_names:
                partition = Partition(self.collection, partition_name)
                partition.release()

        if to_load:
            load_partition_names = [f"partition_{i}" for i in to_load]
            self.collection.load(partition_names=load_partition_names)

        self.loaded_partitions = new_loaded
        self.resident_partitions = new_resident_partitions

    def update_policy(
        self,
        cache_gpu_percent: int,
        cache_cpu_percent: int,
        new_batch_size: int,
        new_resident_partitions: Optional[int] = None,
    ):
        """Update cache configuration and resident partitions"""
        self.cache_gpu_percent = cache_gpu_percent
        self.cache_cpu_percent = cache_cpu_percent
        self.batch_size = new_batch_size
        if new_resident_partitions is not None:
            self.update_resident_partitions(new_resident_partitions)
        if self.dynagen:
            self.model.update_policy(cache_gpu_percent, cache_cpu_percent, new_batch_size)

    def _process_batch(self, batch: List[Dict]) -> List[Dict]:  
        """Process a single batch of questions"""
        arrival_time = time.time()
        query_texts = [q["question"] for q in batch]

        # 1. Embedding generation
        embed_start = time.time()
        query_embeddings = self.embedding_model.embed_documents(query_texts)
        embed_time = time.time() - embed_start
        self.timing_breakdown["embedding_times"].append(embed_time)
        self.timing_breakdown["embedding_total"] += embed_time

        # 2. Vector search
        query_start = time.time()
        if self.use_qdrant:
            batch_results, _ = batch_query_qdrant(
                client=self.collection,
                questions=batch,
                query_texts=query_texts,
                query_embeddings=query_embeddings,
                partition_names=self.partition_names,
                resident_partitions=self.resident_partitions,
            )
        else:
            batch_results, _ = batch_query(
                collection=self.collection,
                questions=batch,
                query_texts=query_texts,
                query_embeddings=query_embeddings,
                partition_names=self.partition_names,
                resident_partitions=self.resident_partitions,
            )
        query_time = time.time() - query_start
        self.timing_breakdown["query_times"].append(query_time)
        self.timing_breakdown["query_total"] += query_time

        # 3. Response generation
        gen_start = time.time()
        responses, _ = batch_generate_responses(
            model=self.model,
            tokenizer=self.tokenizer,
            batch_results=batch_results,
            max_new_tokens=128,
            batch_size=len(batch),
            dynagen=self.dynagen,
            env=self.env,
        )
        gen_time = time.time() - gen_start
        self.timing_breakdown["generation_times"].append(gen_time)
        self.timing_breakdown["generation_total"] += gen_time

        completion_time = time.time()

        # Format results to match main.py expectations
        formatted_results = []
        for i, response in enumerate(responses):
            formatted_results.append({
                'question': Question(
                    question_text=query_texts[i],
                    doc_id=batch[i].get('doc_id', ''),
                    arrival_time=arrival_time,
                    batch_id=len(self.results) + i
                ),
                'result': {
                    'llm_response': response['llm_response'],
                    'metadata': response['metadata']
                },
                'arrival_time': arrival_time,
                'completion_time': completion_time
            })

        self.results.extend(formatted_results)
        return formatted_results

    def _get_config_for_batch(self, batch_index: int, total_batches: int, configs: List[Dict]) -> Dict:
        """Determine which configuration to use for a given batch"""
        if not configs:
            return {}

        # Calculate absolute boundaries for each config
        boundaries = []
        current_boundary = 0
        remaining_batches = total_batches

        for i, config in enumerate(configs[:-1]):
            config_batches = config.get("num_batches", remaining_batches // (len(configs) - i))
            current_boundary += config_batches
            boundaries.append(current_boundary)
            remaining_batches -= config_batches

        # Find appropriate config based on batch index
        for i, boundary in enumerate(boundaries):
            if batch_index < boundary:
                return configs[i]

        return configs[-1]  # Use last config for remaining batches

    def run(self, experiment_config: Optional[List[Dict]] = None):
        """Execute offline batch processing with dynamic batch sizes"""
        print("\nStarting offline processing:")
        print(f"Total questions: {len(self.questions)}")

        total_start = time.time()

        if not experiment_config:
            # Process with default batch size if no config provided
            batches = [self.questions[i : i + self.batch_size] for i in range(0, len(self.questions), self.batch_size)]
            for batch in tqdm(batches, desc="Processing batches"):
                self._process_batch(batch)
        else:
            current_question_idx = 0
            remaining_questions = len(self.questions)

            # 处理每个配置指定的批次
            for config_idx, config in enumerate(experiment_config):
                current_batch_size = config.get("batch_size", self.batch_size)

                print(f"\nApplying configuration {config_idx + 1}:")
                print(f"- Current batch size: {current_batch_size}")
                print(f"- Remaining questions: {remaining_questions}")
                print(f"- Cache GPU: {config.get('cache_gpu_percent')}%")
                print(f"- Cache CPU: {config.get('cache_cpu_percent')}%")
                print(f"- Resident partitions: {config.get('resident_partitions')}")

                # Update system configuration
                self.update_policy(
                    config.get("cache_gpu_percent", self.cache_gpu_percent),
                    config.get("cache_cpu_percent", self.cache_cpu_percent),
                    current_batch_size,
                    config.get("resident_partitions", None),
                )

                # Process one batch with current configuration
                if remaining_questions >= current_batch_size:
                    batch = self.questions[current_question_idx : current_question_idx + current_batch_size]
                    self._process_batch(batch)
                    current_question_idx += current_batch_size
                    remaining_questions -= current_batch_size

            # 使用最后一个配置处理剩余的所有问题
            if remaining_questions > 0:
                last_config = experiment_config[-1]
                last_batch_size = last_config.get("batch_size", self.batch_size)
                
                print(f"\nProcessing remaining {remaining_questions} questions with last configuration:")
                print(f"- Maximum batch size: {last_batch_size}")
                print(f"- Cache GPU: {last_config.get('cache_gpu_percent')}%")
                print(f"- Cache CPU: {last_config.get('cache_cpu_percent')}%")
                print(f"- Resident partitions: {last_config.get('resident_partitions')}")

                # 使用最后的配置处理剩余的所有批次
                while remaining_questions > 0:
                    current_batch_size = min(last_batch_size, remaining_questions)
                    
                    # 使用实际的 batch size 更新策略
                    self.update_policy(
                        last_config.get("cache_gpu_percent", self.cache_gpu_percent),
                        last_config.get("cache_cpu_percent", self.cache_cpu_percent),
                        current_batch_size,  # 使用实际的 batch size
                        last_config.get("resident_partitions", None),
                    )
                    
                    batch = self.questions[current_question_idx : current_question_idx + current_batch_size]
                    self._process_batch(batch)
                    current_question_idx += current_batch_size
                    remaining_questions -= current_batch_size

        self.timing_breakdown["total_processing"] = time.time() - total_start
        # 在处理完成后打印时间统计
        self.print_timing_stats()

    def print_timing_stats(self):
        """Print detailed timing statistics with phase distributions"""
        total_time = sum(self.timing_breakdown["embedding_times"]) + \
                    sum(self.timing_breakdown["query_times"]) + \
                    sum(self.timing_breakdown["generation_times"])

        print("\nTiming Statistics:")
        print("=" * 50)

        # Phase timing breakdowns
        print("\nPhase timing breakdowns:")
        print(f"Embedding time: {sum(self.timing_breakdown['embedding_times']):.2f}s " \
              f"({100 * sum(self.timing_breakdown['embedding_times']) / total_time:.1f}%)")
        print(f"Query time: {sum(self.timing_breakdown['query_times']):.2f}s " \
              f"({100 * sum(self.timing_breakdown['query_times']) / total_time:.1f}%)")
        print(f"Generation time: {sum(self.timing_breakdown['generation_times']):.2f}s " \
              f"({100 * sum(self.timing_breakdown['generation_times']) / total_time:.1f}%)")
        print(f"Total processing time: {total_time:.2f}s")

        # Average times
        print("\nAverage times per batch:")
        if self.timing_breakdown["embedding_times"]:
            print(f"Embedding: {sum(self.timing_breakdown['embedding_times'])/len(self.timing_breakdown['embedding_times']):.2f}s")
        if self.timing_breakdown["query_times"]:
            print(f"Query: {sum(self.timing_breakdown['query_times'])/len(self.timing_breakdown['query_times']):.2f}s")
        if self.timing_breakdown["generation_times"]:
            print(f"Generation: {sum(self.timing_breakdown['generation_times'])/len(self.timing_breakdown['generation_times']):.2f}s")

    def get_timing_stats(self):
        """Return timing statistics"""
        return self.timing_breakdown

    def get_sorted_results(self):
        """Return all processed results sorted by arrival time"""
        return sorted(self.results, key=lambda x: x["arrival_time"])
