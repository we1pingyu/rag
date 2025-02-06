import numpy as np
import time
from queue import Queue
from threading import Event, Thread
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass


@dataclass
class Question:
    """表示单个问题的数据类"""

    question_text: str
    doc_id: str
    arrival_time: float
    batch_id: Optional[int] = None


class WorkloadManager:
    def __init__(
        self,
        questions: List[Dict],
        batch_size: int,
        arrival_rate: float,  # 每分钟平均到达的问题数
        processing_func: Callable,
        timing_stats: Dict[str, List[float]],
        base_time: float,
        seed: int = 42,
    ):
        self.questions = [
            Question(question_text=q["question"], doc_id=q["doc_id"], arrival_time=0.0) for q in questions
        ]
        self.batch_size = batch_size
        self.arrival_rate = arrival_rate
        self.processing_func = processing_func
        self.question_queue = Queue()
        self.finished_queue = Queue()  # 新增：用于跟踪完成的batch数量
        self.current_batch: List[Question] = []
        self.stop_event = Event()
        self.timing_stats = timing_stats
        self.base_time = base_time
        np.random.seed(seed)

        # 计算预期的batch数量
        self.total_batches = (len(self.questions) + self.batch_size - 1) // self.batch_size
        self._generate_arrival_times()

    def _generate_arrival_times(self):
        """使用泊松分布生成问题到达时间"""
        intervals = np.random.exponential(60 / self.arrival_rate, len(self.questions))
        arrival_times = np.cumsum(intervals) + self.base_time

        for q, t in zip(self.questions, arrival_times):
            q.arrival_time = t

        self.questions.sort(key=lambda x: x.arrival_time)

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

    def _batch_processor(self):
        """处理批次的线程"""
        while not (self.stop_event.is_set() and self.question_queue.empty() and not self.current_batch):
            if self.question_queue.empty() and len(self.current_batch) < self.batch_size:
                time.sleep(0.1)
                continue

            if not self.question_queue.empty():
                question = self.question_queue.get()
                self.current_batch.append(question)

            # 当达到batch size或所有问题都已到达且current_batch不为空时处理
            if len(self.current_batch) >= self.batch_size or (
                self.question_queue.empty() and self.current_batch and self.stop_event.is_set()
            ):
                batch = self.current_batch
                self.current_batch = []

                try:
                    print(f"Starting to process batch at {time.time() - self.base_time:.2f}s")
                    self.processing_func(batch)
                    self.finished_queue.put(1)  # 标记一个batch完成
                    print(f"Completed batch at {time.time() - self.base_time:.2f}s")
                except Exception as e:
                    print(f"Error processing batch: {str(e)}")

    def run(self):
        """启动工作负载管理器并等待所有batch完成"""
        arrival_thread = Thread(target=self._question_arrival_simulator)
        processor_thread = Thread(target=self._batch_processor)

        try:
            arrival_thread.start()
            processor_thread.start()

            # 等待所有问题到达
            arrival_thread.join()
            print("All questions have arrived")

            # 设置停止标志，让处理器处理最后的batch
            self.stop_event.set()

            # 等待所有batch处理完成
            completed_batches = 0
            while completed_batches < self.total_batches:
                self.finished_queue.get()
                completed_batches += 1
                print(f"Completed {completed_batches}/{self.total_batches} batches")

            print("All batches have been processed")
            processor_thread.join()

        except KeyboardInterrupt:
            print("\nStopping workload manager...")
            self.stop_event.set()
            arrival_thread.join()
            processor_thread.join()

    @property
    def total_questions(self):
        return len(self.questions)

    @property
    def expected_duration(self):
        """返回预期的总运行时间（秒）"""
        if self.questions:
            return self.questions[-1].arrival_time
        return 0.0
