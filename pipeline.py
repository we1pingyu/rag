import numpy as np
import time
from queue import Queue
from threading import Event, Thread, Lock
from typing import List, Dict, Optional, Callable
from utils import (
    Question,
    BatchResult,
    log_timing,
    batch_query,
    batch_generate_responses,
    batch_query_qdrant,
    build_qdrant_index,
)


class PipelineProcessor:

    def __init__(
        self,
        questions: List[Dict],
        batch_size: int,
        arrival_rate: float,
        embedding_model,
        model,
        tokenizer,
        collection,
        partition_names: List[str],
        timing_stats: Dict[str, List[float]],
        resident_partitions: int = 0,
        base_time: float = None,
        use_qdrant: bool = False,
        dynagen: bool = False,
        seed: int = 42,
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

        # 队列和状态管理
        self.question_queue = Queue()
        self.query_queue = Queue()  # 存储已完成query的batch
        self.stop_event = Event()
        self.query_lock = Lock()
        self.generation_lock = Lock()

        # 批次管理
        self.current_batch: List[Question] = []
        self.total_batches = (len(self.questions) + self.batch_size - 1) // self.batch_size
        self.completed_batches = 0

        np.random.seed(seed)
        self._generate_arrival_times()
        self.all_results = []
        self.results_lock = Lock()
        self.use_qdrant = use_qdrant
        self.dynagen = dynagen

    def _generate_arrival_times(self):
        """使用泊松分布生成问题到达时间"""
        intervals = np.random.exponential(60 / self.arrival_rate, len(self.questions))
        arrival_times = np.cumsum(intervals) + self.base_time

        for q, t in zip(self.questions, arrival_times):
            q.arrival_time = t

        self.questions.sort(key=lambda x: x.arrival_time)

    def _query_worker(self):
        """Execute query worker thread"""
        while not (self.stop_event.is_set() and self.question_queue.empty() and not self.current_batch):
            if self.question_queue.empty() and len(self.current_batch) < self.batch_size:
                time.sleep(0.1)
                continue

            # Collect questions into batch
            while not self.question_queue.empty() and len(self.current_batch) < self.batch_size:
                question = self.question_queue.get()
                self.current_batch.append(question)

            # Process a complete batch or final incomplete batch
            if len(self.current_batch) >= self.batch_size or (
                self.question_queue.empty() and self.current_batch and self.stop_event.is_set()
            ):
                with self.query_lock:
                    batch = self.current_batch
                    self.current_batch = []

                    # Execute query
                    query_texts = [q.question_text for q in batch]
                    questions_dict = [{"question": q.question_text, "doc_id": q.doc_id} for q in batch]

                    # Embedding generation
                    embed_start = time.time()
                    query_embeddings = self.embedding_model.embed_documents(query_texts)
                    log_timing(self.timing_stats, "embedding_time", time.time() - embed_start)

                    # Query execution based on selected backend
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

                    # Place query results in queue
                    batch_result = BatchResult(
                        questions=batch, query_embeddings=query_embeddings, query_results=batch_results
                    )
                    self.query_queue.put(batch_result)
                    print(f"Query completed for batch at {time.time() - self.base_time:.2f}s")


    def _generation_worker(self):
        """Execute generation worker thread"""
        while not (
            self.stop_event.is_set() and self.query_queue.empty() and self.completed_batches >= self.total_batches
        ):
            batch_result = self.query_queue.get(timeout=1.0)  # Wait for new batch results
            

            with self.generation_lock:
                # Execute generation
                batch_responses, updated_timing_stats = batch_generate_responses(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    batch_results=batch_result.query_results,
                    max_new_tokens=128,
                    batch_size=len(batch_result.questions),
                    timing_stats=self.timing_stats,
                    dynagen=self.dynagen,
                )
                self.timing_stats.update(updated_timing_stats)

                batch_result.generation_results = batch_responses
                self.completed_batches += 1

                with self.results_lock:
                    for q, r in zip(batch_result.questions, batch_responses):
                        self.all_results.append(
                            {
                                "question": q,
                                "result": r,
                                "arrival_time": q.arrival_time,
                                "completion_time": time.time(),
                            }
                        )

                print(f"Generation completed for batch at {time.time() - self.base_time:.2f}s")
                print(f"Completed {self.completed_batches}/{self.total_batches} batches")

            # except Exception as e:
            #     print(f"Error in generation worker: {str(e)}")

    def get_sorted_results(self):
        """获取按问题到达时间排序的所有结果"""
        with self.results_lock:
            sorted_results = sorted(self.all_results, key=lambda x: x["arrival_time"])
            return sorted_results

    def _question_arrival_simulator(self):
        """模拟问题到达的线程"""
        for question in self.questions:
            if self.stop_event.is_set():
                break

            current_time = time.time()
            wait_time = question.arrival_time - current_time
            if wait_time > 0:
                time.sleep(wait_time)

            self.question_queue.put(question)
            print(f"Question arrived at {time.time() - self.base_time:.2f}s: {question.question_text[:50]}...")

    def run(self):
        """运行流水线处理"""
        # 创建工作线程
        arrival_thread = Thread(target=self._question_arrival_simulator)
        query_thread = Thread(target=self._query_worker)
        generation_thread = Thread(target=self._generation_worker)

        try:
            # 启动所有线程
            arrival_thread.start()
            query_thread.start()
            generation_thread.start()

            # 等待问题到达完成
            arrival_thread.join()
            print("All questions have arrived")

            # 设置停止标志
            self.stop_event.set()

            # 等待所有处理完成
            query_thread.join()
            generation_thread.join()

            print("Pipeline processing completed")

        except KeyboardInterrupt:
            print("\nStopping pipeline manager...")
            self.stop_event.set()
            arrival_thread.join()
            query_thread.join()
            generation_thread.join()

    @property
    def total_questions(self):
        return len(self.questions)

    @property
    def expected_duration(self):
        """返回预期的总运行时间（秒）"""
        if self.questions:
            return self.questions[-1].arrival_time - self.base_time
        return 0.0
