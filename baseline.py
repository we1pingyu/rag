from typing import List, Dict
import time
import numpy as np
from tqdm import tqdm
from utils import Question, log_timing, batch_query, batch_generate_responses, batch_query_qdrant


class BaselineProcessor:

    def __init__(
        self,
        questions: List[Dict],
        batch_size: int,
        arrival_rate: float,  # 每分钟的问题数
        embedding_model,
        model,
        tokenizer,
        collection,
        partition_names: List[str],
        timing_stats: Dict[str, List[float]],
        resident_partitions: int = 0,
        base_time: float = None,
        use_qdrant: bool = False,
    ):
        self.questions = [
            Question(question_text=q["question"], doc_id=q["doc_id"], arrival_time=0.0) for q in questions
        ]
        self.batch_size = batch_size
        self.arrival_rate = arrival_rate
        self.embedding_model = embedding_model
        self.model = model
        self.tokenizer = tokenizer
        self.collection = collection
        self.partition_names = partition_names
        self.timing_stats = timing_stats
        self.resident_partitions = resident_partitions
        self.base_time = base_time or time.time()
        self.results = []
        self.use_qdrant = use_qdrant

        # 生成到达时间
        self._generate_arrival_times()

    def _generate_arrival_times(self):
        """使用泊松分布生成问题到达时间"""
        # 生成相邻到达的时间间隔（指数分布）
        intervals = np.random.exponential(60 / self.arrival_rate, len(self.questions))
        # 计算累积时间作为到达时间
        arrival_times = np.cumsum(intervals) + self.base_time

        # 更新每个问题的到达时间
        for q, t in zip(self.questions, arrival_times):
            q.arrival_time = t

        # 按到达时间排序
        self.questions.sort(key=lambda x: x.arrival_time)

    def process_batch(self, batch: List[Question]):
        """处理单个批次"""
        query_texts = [q.question_text for q in batch]
        questions_dict = [{"question": q.question_text, "doc_id": q.doc_id} for q in batch]

        # Release all partitions before processing new batch
        if not self.resident_partitions and not self.use_qdrant:
            self.collection.release()

        # Embedding generation
        embed_start = time.time()
        query_embeddings = self.embedding_model.embed_documents(query_texts)
        log_timing(self.timing_stats, "embedding_time", time.time() - embed_start)

        # Query phase
        if self.use_qdrant:
            batch_results, updated_timing_stats = batch_query_qdrant(
                client=self.collection,
                questions=questions_dict,
                query_texts=query_texts,
                query_embeddings=query_embeddings,
                partition_names=self.partition_names,
                timing_stats=self.timing_stats,
                resident_partitions=self.resident_partitions,
            )
        else:
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

        # Generation phase
        batch_responses, updated_timing_stats = batch_generate_responses(
            model=self.model,
            tokenizer=self.tokenizer,
            batch_results=batch_results,
            max_new_tokens=128,
            batch_size=len(batch),
            timing_stats=self.timing_stats,
        )
        self.timing_stats.update(updated_timing_stats)

        # Save results
        completion_time = time.time()
        for q, r in zip(batch, batch_responses):
            self.results.append(
                {
                    "question": q,
                    "result": r,
                    "arrival_time": q.arrival_time,  # 使用生成的到达时间
                    "completion_time": completion_time,
                }
            )

        return batch_responses

    def run(self):
        """运行串行处理，遵守到达率限制"""
        current_batch = []
        next_batch_start_time = None

        for q_idx, question in enumerate(tqdm(self.questions, desc="Processing questions")):
            # 等待直到问题的预定到达时间
            wait_time = question.arrival_time - time.time()
            if wait_time > 0:
                time.sleep(wait_time)

            current_batch.append(question)
            print(f"Question arrived at {time.time() - self.base_time:.2f}s")

            # 当积累了足够的问题或者是最后一批时处理batch
            if len(current_batch) >= self.batch_size or q_idx == len(self.questions) - 1:
                if current_batch:  # 确保batch不为空
                    print(f"\nProcessing batch of {len(current_batch)} questions")
                    self.process_batch(current_batch)
                    print(f"Completed batch at {time.time() - self.base_time:.2f}s")
                    current_batch = []

    def get_sorted_results(self):
        """获取按到达时间排序的结果"""
        return sorted(self.results, key=lambda x: x["arrival_time"])

    @property
    def expected_duration(self):
        """返回预期的总运行时间（秒）"""
        if self.questions:
            return self.questions[-1].arrival_time - self.base_time
        return 0.0
