import time
from typing import List, Dict
from tqdm import tqdm
from utils import batch_query, batch_generate_responses, split_batch, Question


class OfflineProcessor:

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
        w_gpu_percent=0,
        w_cpu_percent=0,
        cache_gpu_percent=0,
        cache_cpu_percent=0,
        dynagen: bool = False,
        env=None,
    ):
        self.questions = questions
        self.batch_size = batch_size
        self.embedding_model = embedding_model
        self.model = model
        self.tokenizer = tokenizer
        self.collection = collection
        self.partition_names = partition_names
        self.resident_partitions = resident_partitions
        self.w_gpu_percent = w_gpu_percent
        self.w_cpu_percent = w_cpu_percent
        self.cache_gpu_percent = cache_gpu_percent
        self.cache_cpu_percent = cache_cpu_percent
        self.dynagen = dynagen
        self.env = env
        self.results = []
        self.base_time = time.time()

        # 专门记录各阶段的时间
        self.timing_breakdown = {
            "embedding_total": 0.0,
            "query_total": 0.0,
            "generation_total": 0.0,
            "total_processing": 0.0,
            "embedding_times": [],  # 记录每个batch的时间
            "query_times": [],
            "generation_times": [],
        }

    def _process_batch(self, batch: List[Dict]):
        arrival_time = time.time()
        """处理单个batch并记录各阶段时间"""
        query_texts = [q["question"] for q in batch]

        # 1. Embedding 阶段
        embed_start = time.time()
        query_embeddings = self.embedding_model.embed_documents(query_texts)
        embed_time = time.time() - embed_start
        self.timing_breakdown["embedding_times"].append(embed_time)
        self.timing_breakdown["embedding_total"] += embed_time

        batch_size_split, num_batches = split_batch(self.batch_size)
        print(f"batch_size_split: {batch_size_split}, num_batches: {num_batches}")
        print(f"cache_gpu_percent: {self.cache_gpu_percent}, cache_cpu_percent: {self.cache_cpu_percent}")
        self.model.update_policy(self.cache_gpu_percent, self.cache_gpu_percent, batch_size_split, num_batches)
        self.model.update_weight(self.w_gpu_percent, self.w_cpu_percent)
        # 2. Query 阶段
        query_start = time.time()

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

        # 3. Generation 阶段
        gen_start = time.time()
        responses, _ = batch_generate_responses(
            model=self.model,
            tokenizer=self.tokenizer,
            batch_results=batch_results,
            max_new_tokens=16,
            batch_size=len(batch),
            dynagen=self.dynagen,
            env=self.env,
        )
        gen_time = time.time() - gen_start
        self.timing_breakdown["generation_times"].append(gen_time)
        self.timing_breakdown["generation_total"] += gen_time
        completion_time = time.time()
        # 保存结果
        formatted_results = []
        for i, response in enumerate(responses):
            formatted_results.append(
                {
                    "question": Question(
                        question_text=query_texts[i],
                        doc_id=batch[i].get("doc_id", ""),
                        arrival_time=arrival_time,
                        batch_id=len(self.results) + i,
                    ),
                    "result": {"llm_response": response["llm_response"], "metadata": response["metadata"]},
                    "arrival_time": arrival_time,
                    "completion_time": completion_time,
                }
            )
        self.results.extend(formatted_results)

        return responses

    def run(self):
        """执行离线批处理"""
        print(f"\nStarting offline processing:")
        print(f"Total questions: {len(self.questions)}")
        print(f"Batch size: {self.batch_size}")

        total_start = time.time()

        # 释放所有分区（如果使用Milvus）
        if not self.resident_partitions:
            self.collection.release()

        # 按batch处理所有问题
        batches = [self.questions[i : i + self.batch_size] for i in range(0, len(self.questions), self.batch_size)]

        for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
            self._process_batch(batch)

        self.timing_breakdown["total_processing"] = time.time() - total_start

        # 打印详细的时间统计
        self._print_timing_stats()

    def _print_timing_stats(self):
        """打印详细的时间统计信息"""
        print("\nTiming Breakdown:")
        print(f"{'='*50}")
        print(f"Total Processing Time: {self.timing_breakdown['total_processing']:.2f}s")
        print(f"\nPhase-wise Statistics:")
        print(f"{'='*50}")

        # Embedding统计
        print("\nEmbedding Phase:")
        print(f"Total Time: {self.timing_breakdown['embedding_total']:.2f}s")
        print(
            f"Average Time per Batch: {sum(self.timing_breakdown['embedding_times'])/len(self.timing_breakdown['embedding_times']):.2f}s"
        )
        print(
            f"Percentage of Total: {(self.timing_breakdown['embedding_total']/self.timing_breakdown['total_processing'])*100:.1f}%"
        )

        # Query统计
        print("\nQuery Phase:")
        print(f"Total Time: {self.timing_breakdown['query_total']:.2f}s")
        print(
            f"Average Time per Batch: {sum(self.timing_breakdown['query_times'])/len(self.timing_breakdown['query_times']):.2f}s"
        )
        print(
            f"Percentage of Total: {(self.timing_breakdown['query_total']/self.timing_breakdown['total_processing'])*100:.1f}%"
        )

        # Generation统计
        print("\nGeneration Phase:")
        print(f"Total Time: {self.timing_breakdown['generation_total']:.2f}s")
        print(
            f"Average Time per Batch: {sum(self.timing_breakdown['generation_times'])/len(self.timing_breakdown['generation_times']):.2f}s"
        )
        print(
            f"Percentage of Total: {(self.timing_breakdown['generation_total']/self.timing_breakdown['total_processing'])*100:.1f}%"
        )

    def get_sorted_results(self):
        """获取所有处理结果"""
        return self.results

    def get_timing_stats(self):
        """获取时间统计信息"""
        return self.timing_breakdown
