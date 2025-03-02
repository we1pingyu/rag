from typing import List, Dict
import time
import torch
import numpy as np
from tqdm import tqdm
from utils import Question, log_timing, batch_query, batch_generate_responses
from transformers import AutoModelForCausalLM

class BaselineProcessor:

    def __init__(
        self,
        questions: List[Dict],
        batch_size: int,
        arrival_rates: List[float],  # 每分钟的问题数（多个）
        rate_change_interval: int,  # 间隔时间（秒）
        embedding_model,
        model_name,
        tokenizer,
        collection,
        partition_names: List[str],
        timing_stats: Dict[str, List[float]],
        gpu_memory_gb: int,
        resident_partitions: int = 0,
        base_time: float = None,
        seed: int = 42,
    ):
        self.questions = [
            Question(question_text=q["question"], doc_id=q["doc_id"], arrival_time=0.0) for q in questions
        ]
        self.batch_size = batch_size
        self.arrival_rates = arrival_rates if arrival_rates else [16]  # 默认单一到达率
        self.rate_change_interval = rate_change_interval
        self.embedding_model = embedding_model
        max_memory_per_batch = 0.25  # GiB per batch
        # Calculate remaining memory after accounting for batch size
        max_gpu_memory_size = gpu_memory_gb - (self.batch_size * max_memory_per_batch)
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

        # 新增的间隔跟踪变量
        self.interval_question_count = []
        self.interval_completion_count = []

        np.random.seed(seed)

        # 生成到达时间
        self._generate_arrival_times()

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

    def process_batch(self, batch: List[Question]):
        """处理单个批次"""
        query_texts = [q.question_text for q in batch]
        questions_dict = [{"question": q.question_text, "doc_id": q.doc_id} for q in batch]

        # Release all partitions before processing new batch
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

        # Generation phase
        batch_responses, updated_timing_stats = batch_generate_responses(
            model=self.model,
            tokenizer=self.tokenizer,
            batch_results=batch_results,
            max_new_tokens=16,
            batch_size=len(batch),
            timing_stats=self.timing_stats,
            dynagen=self.dynagen,
            env=self.env if self.dynagen else None,
        )
        torch.cuda.empty_cache()
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

            # 更新区间完成计数
            if hasattr(q, "interval_id"):
                interval_idx = q.interval_id
                if 0 <= interval_idx < len(self.interval_completion_count):
                    self.interval_completion_count[interval_idx] += 1

        return batch_responses

    def run(self):
        """运行串行处理，按固定batch size处理，每个interval的最后一个不足batch size的batch单独处理"""
        import threading

        next_question_idx = 0
        total_questions = len(self.questions)
        current_batch = []
        current_interval_id = -1

        print("\nStarting baseline processing with fixed batch size")
        print(f"Questions by interval: {self.interval_question_count}")
        print(f"Total questions: {total_questions}")
        print(f"Batch size: {self.batch_size}\n")

        # 创建一个问题到达监控器线程
        arrival_monitor_stop = threading.Event()

        def arrival_monitor():
            """监控问题到达的线程函数"""
            monitor_idx = 0
            while not arrival_monitor_stop.is_set() and monitor_idx < total_questions:
                current_time = time.time()

                # 检查是否有问题到达
                while monitor_idx < total_questions:
                    question = self.questions[monitor_idx]

                    # 如果问题到达时间未到，等待
                    if question.arrival_time > current_time:
                        break

                    # 问题已经到达，打印信息
                    elapsed_time = question.arrival_time - self.base_time

                    print(
                        f"Question arrived at {elapsed_time:.2f}s: {question.question_text[:50]}... (Interval: {question.interval_id})"
                    )
                    monitor_idx += 1

                # 短暂睡眠以避免CPU过度使用
                time.sleep(0.1)

        # 启动问题到达监控器
        monitor_thread = threading.Thread(target=arrival_monitor)
        monitor_thread.daemon = True
        monitor_thread.start()

        try:
            while next_question_idx < total_questions:
                # 检查是否有新的区间开始
                if next_question_idx < total_questions:
                    next_question = self.questions[next_question_idx]
                    next_interval_id = next_question.interval_id

                    # 如果是新区间，并且当前批次有问题，处理它们
                    if current_interval_id != -1 and next_interval_id != current_interval_id and current_batch:
                        print(
                            f"\nProcessing final batch of interval {current_interval_id} with {len(current_batch)} questions"
                        )
                        self.process_batch(current_batch)
                        print(f"Completed batch at {time.time() - self.base_time:.2f}s")
                        print(f"Completion counts by interval: {self.interval_completion_count}")
                        current_batch = []

                    # 更新当前区间
                    current_interval_id = next_interval_id

                # 收集当前区间的问题，直到达到批次大小或者区间结束
                while next_question_idx < total_questions:
                    next_question = self.questions[next_question_idx]

                    # 如果问题属于新区间，跳出循环
                    if next_question.interval_id != current_interval_id:
                        break

                    # 等待问题到达
                    current_time = time.time()
                    if next_question.arrival_time > current_time:
                        wait_time = next_question.arrival_time - current_time
                        time.sleep(min(wait_time, 1.0))

                    # 添加问题到当前批次
                    current_batch.append(next_question)
                    next_question_idx += 1

                    # 如果批次已满，处理它
                    if len(current_batch) >= self.batch_size:
                        print(
                            f"\nProcessing full batch of {len(current_batch)} questions (Interval: {current_interval_id})"
                        )
                        self.process_batch(current_batch)
                        print(f"Completed batch at {time.time() - self.base_time:.2f}s")
                        print(f"Completion counts by interval: {self.interval_completion_count}")
                        print(f"Processed {next_question_idx}/{total_questions} questions")
                        current_batch = []
                        break  # 退出内循环，重新检查区间状态

                # 检查是否到达了区间末尾，处理最后不足batch size的批次
                if next_question_idx < total_questions:
                    next_question = self.questions[next_question_idx]
                    if next_question.interval_id != current_interval_id and current_batch:
                        print(
                            f"\nProcessing final batch of interval {current_interval_id} with {len(current_batch)} questions"
                        )
                        self.process_batch(current_batch)
                        print(f"Completed batch at {time.time() - self.base_time:.2f}s")
                        print(f"Completion counts by interval: {self.interval_completion_count}")
                        current_batch = []

                # 检查是否所有问题都已处理
                if next_question_idx >= total_questions and current_batch:
                    print(
                        f"\nProcessing final remaining batch with {len(current_batch)} questions (Interval: {current_interval_id})"
                    )
                    self.process_batch(current_batch)
                    print(f"Completed final batch at {time.time() - self.base_time:.2f}s")
                    print(f"Completion counts by interval: {self.interval_completion_count}")
                    current_batch = []

        finally:
            # 停止问题到达监控器线程
            arrival_monitor_stop.set()
            monitor_thread.join(timeout=1.0)  # 给1秒时间让线程优雅结束

        # 打印最终统计信息
        print("\n===== Final Statistics =====")
        print(f"Total questions processed: {sum(self.interval_completion_count)}")
        print(f"Questions by interval: {self.interval_question_count}")
        print(f"Completions by interval: {self.interval_completion_count}")
        print("============================\n")

    def get_sorted_results(self):
        """获取按到达时间排序的结果"""
        return sorted(self.results, key=lambda x: x["arrival_time"])

    @property
    def expected_duration(self):
        """返回预期的总运行时间（秒）"""
        if self.questions:
            return self.questions[-1].arrival_time - self.base_time
        return 0.0
