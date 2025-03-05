import argparse
import time
import resource
import pickle
import random
import torch
import os
import warnings
import shutil
from pathlib import Path
from pymilvus import connections, Collection
from baseline import BaselineProcessor
from active_profiling import ActiveProfilingProcessor
from dyn_pipeline import DynPipelineProcessor
from efficient_rag import VLLMProcessor
from offline import OfflineProcessor
from utils import build_index, get_milvus_memory_usage, calculate_latency_stats


os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="langchain")
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
from transformers import AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings

MAX_BATCH_SIZE = 65536

# Global constants
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-l6-v2"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pipeline RAG processing")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="LLM Model name")
    parser.add_argument("--total_questions", type=int, default=2000, help="Total number of questions to process")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of questions to process per batch")
    parser.add_argument("--persist_dir", type=str, default="trivia_data_milvus", help="Directory for persisted data")
    parser.add_argument("--dataset", type=str, default="trivia", help="Dataset to use for rag: nq or trivia or macro")
    parser.add_argument("--display_results", action="store_true", help="Whether to display final results")
    parser.add_argument("--cpu_memory_limit", type=int, default=166, help="CPU memory limit in GB")
    parser.add_argument("--gpu_memory_limit", type=int, default=12, help="GPU memory limit in GB")
    parser.add_argument("--resident_partitions", type=int, default=0, help="Number of resident partitions")
    parser.add_argument(
        "--arrival_rates",
        type=float,
        nargs="+",
        default=[8, 16, 32, 64],
        help="List of arrival rates to use in dynamic pipeline",
    )
    parser.add_argument(
        "--rate_change_interval", type=int, default=300, help="Interval in seconds between arrival rate changes"
    )
    parser.add_argument("--build_index", action="store_true", help="Whether to build Milvus index")
    parser.add_argument("--num_partitions", type=int, default=32, help="Number of partitions for index building")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--baseline", action="store_true", help="Run in baseline (serial) mode with HF transformer")
    mode_group.add_argument(
        "--dyn_pipeline", action="store_true", help="Run in dynamic pipeline mode with dynagen model"
    )
    mode_group.add_argument("--vllm", action="store_true", help="Run with VLLM for efficient generation")
    mode_group.add_argument("--active", action="store_true", help="Run in active profiling mode with dynagen model")
    mode_group.add_argument("--offline", action="store_true", help="Run in offline mode with dynagen model")
    parser.add_argument(
        "--percent",
        nargs="+",
        type=int,
        default=[0, 40, 0, 20],
        help="four numbers: w_gpu_percent, w_cpu_percent, cache_gpu_percent, cache_cpu_percent",
    )
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

    args = parser.parse_args()
    random.seed(args.seed)

    if args.build_index:
        print("\nBuilding index...")
        # 初始化embedding model用于构建索引
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            multi_process=True,
            show_progress=True,
            model_kwargs={"device": "cuda"},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": MAX_BATCH_SIZE,
            },
        )
        build_index(
            persist_directory=args.persist_dir,
            num_partitions=args.num_partitions,
            embedding_model=embedding_model,
            dataset=args.dataset,
        )
        print("Index building completed")

    print("Connecting to Milvus...")
    connections.connect(host="localhost", port="19530")
    collection = Collection(f"{args.dataset}_collection")
    collection.release()

    try:
        total_gpu_memory = 24
        fraction = args.gpu_memory_limit / total_gpu_memory
        print(f"Setting GPU memory fraction to {fraction}")
        torch.cuda.set_per_process_memory_fraction(fraction)

        collection.load(["partition_0"])
        print(f"Initial Milvus memory usage: {get_milvus_memory_usage():.2f} GB")
        partition_size_gb = get_milvus_memory_usage()
        available_cpu_mem = args.cpu_memory_limit - partition_size_gb * (args.resident_partitions + 1)
        # resource.setrlimit(resource.RLIMIT_AS, (int(available_cpu_mem * 1024**3), int(available_cpu_mem * 1024**3)))
        print(f"Set CPU available memory to {int(available_cpu_mem)} GB")

        # 加载 resident partitions
        if args.resident_partitions > 0:
            resident_partition_names = [f"partition_{i}" for i in range(args.resident_partitions)]
            collection.load(resident_partition_names)
            print(f"Loaded {args.resident_partitions} resident partitions: {resident_partition_names}")
        print("Loading models and initializing...")
        # Load LLM
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Load embedding model
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            multi_process=False,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Load metadata
        metadata_path = Path(args.persist_dir) / "metadata.pkl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
            all_questions = metadata["questions"]
            # random.shuffle(all_questions)
            partition_names = metadata["partition_names"]

        timing_stats = {}
        total_start_time = time.time()
        if args.dyn_pipeline:
            print("\nRunning in dynamic pipeline mode")
            processor = DynPipelineProcessor(
                questions=all_questions[: args.total_questions],
                batch_size=args.batch_size,
                arrival_rates=[8],
                embedding_model=embedding_model,
                model_name=args.model,
                tokenizer=tokenizer,
                collection=collection,
                partition_names=partition_names,
                timing_stats=timing_stats,
                resident_partitions=args.resident_partitions,
                base_time=total_start_time,
                rate_change_interval=600,
                partition_size_gb=partition_size_gb,
                total_cpu_gb=available_cpu_mem,
                gpu_memory_gb=args.gpu_memory_limit,
            )
        elif args.baseline:
            print("\nRunning in baseline (serial) mode")
            processor = BaselineProcessor(
                questions=all_questions[: args.total_questions],
                batch_size=args.batch_size,
                arrival_rates=[2],
                rate_change_interval=60,
                embedding_model=embedding_model,
                model_name=args.model,
                tokenizer=tokenizer,
                collection=collection,
                partition_names=partition_names,
                timing_stats=timing_stats,
                resident_partitions=args.resident_partitions,
                base_time=total_start_time,
                gpu_memory_gb=args.gpu_memory_limit,
            )
        elif args.vllm:
            print("\nRunning with VLLM processor")
            processor = VLLMProcessor(
                questions=all_questions[: args.total_questions],
                batch_size=args.batch_size,
                arrival_rates=[4],
                rate_change_interval=60,
                embedding_model=embedding_model,
                model_name=args.model,  # Pass model name instead of model instance
                tokenizer=tokenizer,
                collection=collection,
                partition_names=["partition_0"],
                timing_stats=timing_stats,
                resident_partitions=args.resident_partitions,
                base_time=total_start_time,
                max_new_tokens=16,  # Adjustable parameter
                gpu_memory_utilization=0.5,  # Adjustable parameter
                vllm_batch_size=args.batch_size,  # Can be different from query batch size
            )
        elif args.active:
            processor = ActiveProfilingProcessor(
                questions=all_questions,
                embedding_model=embedding_model,
                model_name=args.model,
                tokenizer=tokenizer,
                collection=collection,
                partition_names=partition_names,
                partition_size_gb=partition_size_gb,
                total_cpu_gb=available_cpu_mem,
                gpu_memory_gb=args.gpu_memory_limit,
            )
        elif args.offline:
            print("\nRunning in offline mode (batch processing)")
            processor = OfflineProcessor(
                questions=all_questions[: args.total_questions],
                batch_size=args.batch_size,
                embedding_model=embedding_model,
                model_name=args.model,
                tokenizer=tokenizer,
                collection=collection,
                partition_names=partition_names,
                w_gpu_percent=args.percent[0],
                w_cpu_percent=args.percent[1],
                cache_gpu_percent=args.percent[2],
                cache_cpu_percent=args.percent[3],
                resident_partitions=args.resident_partitions,
            )

        if not args.active and not args.offline:
            print(f"\nStarting processing:")
            print(f"Total questions: {len(all_questions[:args.total_questions])}")
            print(f"Batch size: {args.batch_size}")
            print(f"Expected duration: {processor.expected_duration:.2f} seconds")

        # Run pipeline
        if args.active:
            optimal_configs = processor.run()

        processor.run()

        if not args.active:
            sorted_results = processor.get_sorted_results()
            latency_stats = calculate_latency_stats(sorted_results)

            if args.display_results:
                print("\nProcessed Results:")
                print("=" * 80)
                sorted_results = processor.get_sorted_results()
                for idx, result in enumerate(sorted_results, 1):
                    print(f"\nQuestion {idx}/{len(sorted_results)}:")
                    print(f"Q: {result['question'].question_text}")
                    print(f"A: {result['result']['llm_response']}")
                    print(f"Retrieved Doc IDs: {result['result']['metadata']['doc_ids']}")
                    print(f"Arrival time: {result['arrival_time'] - processor.base_time:.2f}s")
                    print(f"Completion time: {result['completion_time'] - processor.base_time:.2f}s")
                    print(f"Processing time: {result['completion_time'] - result['arrival_time']:.2f}s")
                    print("-" * 40)

            print("\nLatency Statistics:")
            print(f"Average Latency: {latency_stats['average_latency']:.2f} seconds")
            print(f"90th Percentile Latency: {latency_stats['p90_latency']:.2f} seconds")
            print(f"Maximum Latency: {latency_stats['max_latency']:.2f} seconds")

            # Print final statistics
            total_time = time.time() - total_start_time
            print(f"\nTotal end-to-end time: {total_time:.3f}s")

    finally:
        print("\nCleaning up...")
        collection.release()
        connections.disconnect("default")
        if os.path.exists("./dynagen_offload_dir"):
            shutil.rmtree("./dynagen_offload_dir")
