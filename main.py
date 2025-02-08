import argparse
import time
import resource
from pathlib import Path
import pickle
import random
from pymilvus import connections, Collection
from pipeline_manager import PipelineManager
from baseline import BaselineProcessor
from utils import build_index, get_milvus_memory_usage, calculate_latency_stats, init_dynagen_model, build_qdrant_index
from qdrant_client import QdrantClient

import os
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="langchain")
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings

# Global constants
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-l6-v2"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pipeline RAG processing")
    parser.add_argument("--total_questions", type=int, default=32, help="Total number of questions to process")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of questions to process per batch")
    parser.add_argument("--persist_dir", type=str, default="rag_data_milvus", help="Directory for persisted data")
    parser.add_argument("--display_results", action="store_true", help="Whether to display final results")
    parser.add_argument("--cpu_memory_limit", type=int, default=64, help="CPU memory limit in GB")
    parser.add_argument("--resident_partitions", type=int, default=0, help="Number of resident partitions")
    parser.add_argument("--arrival_rate", type=float, default=16, help="Average number of questions arriving per minute")
    parser.add_argument("--build_index", action="store_true", help="Whether to build Milvus index")
    parser.add_argument("--num_partitions", type=int, default=10, help="Number of partitions for index building")
    parser.add_argument("--baseline", action="store_true", help="Run in baseline (serial) mode")
    parser.add_argument("--dynagen", action="store_true", help="Whether to use DynaGen for generation")
    parser.add_argument("--qdrant", action="store_true", help="Use Qdrant instead of Milvus")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    random.seed(args.seed)
    if args.build_index:
        print("\nBuilding index...")
        # 初始化embedding model用于构建索引
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            multi_process=True,
            model_kwargs={"device": "cuda"},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 65536,
                "show_progress_bar": True,
            },
        )
        if args.qdrant:
            client = build_qdrant_index(
                persist_directory=args.persist_dir, num_partitions=args.num_partitions, embedding_model=embedding_model
            )
        else:
            build_index(
                persist_directory=args.persist_dir, num_partitions=args.num_partitions, embedding_model=embedding_model
            )
        print("Index building completed")

    timing_stats = {}
    total_start_time = time.time()

    if args.qdrant:
        print("Connecting to Qdrant...")
        client = QdrantClient("localhost", port=6333)
    else:
        print("Connecting to Milvus...")
        connections.connect(host="localhost", port="19530")
        collection = Collection("nq_collection")
        collection.release()

    try:
        if not args.qdrant:
            collection.load(["partition_0"])
            print(f"Initial Milvus memory usage: {get_milvus_memory_usage():.2f} GB")
            available_cpu_mem = args.cpu_memory_limit - get_milvus_memory_usage() * (args.resident_partitions + 1)
            resource.setrlimit(resource.RLIMIT_AS, (int(available_cpu_mem * 1024**3), int(available_cpu_mem * 1024**3)))
            print(f"Set CPU available memory to {int(available_cpu_mem)} GB")

            # 加载 resident partitions
            if args.resident_partitions > 0:
                resident_partition_names = [f"partition_{i}" for i in range(args.resident_partitions)]
                collection.load(resident_partition_names)
                print(f"Loaded {args.resident_partitions} resident partitions: {resident_partition_names}")
        print("Loading models and initializing...")
        # Load LLM
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if args.dynagen:
            model = init_dynagen_model(model_name, tokenizer, args)

        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                max_memory={0: "14GiB"},
            )

        # Load embedding model
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            multi_process=True,
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

        if args.baseline:
            print("\nRunning in baseline (serial) mode")
            processor = BaselineProcessor(
                questions=all_questions[: args.total_questions],
                batch_size=args.batch_size,
                arrival_rate=args.arrival_rate,
                embedding_model=embedding_model,
                model=model,
                tokenizer=tokenizer,
                collection=client if args.qdrant else collection,
                partition_names=partition_names,
                timing_stats=timing_stats,
                resident_partitions=args.resident_partitions,
                base_time=total_start_time,
                use_qdrant=args.qdrant,
            )
        else:
            # Initialize pipeline manager
            processor = PipelineManager(
                questions=all_questions[: args.total_questions],
                batch_size=args.batch_size,
                arrival_rate=args.arrival_rate,
                embedding_model=embedding_model,
                model=model,
                tokenizer=tokenizer,
                collection=client if args.qdrant else collection,
                partition_names=partition_names,
                timing_stats=timing_stats,
                resident_partitions=args.resident_partitions,
                base_time=total_start_time,
                use_qdrant=args.qdrant,
            )

        print(f"\nStarting processing:")
        print(f"Total questions: {len(all_questions[:args.total_questions])}")
        print(f"Batch size: {args.batch_size}")
        if not args.baseline:
            print(f"Average arrival rate: {args.arrival_rate} questions/minute")
            print(f"Expected duration: {processor.expected_duration:.2f} seconds")

        # Run pipeline
        processor.run()
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
                print(f"Arrival time: {result['arrival_time'] - processor .base_time:.2f}s")
                print(f"Completion time: {result['completion_time'] - processor .base_time:.2f}s")
                print(f"Processing time: {result['completion_time'] - result['arrival_time']:.2f}s")
                print("-" * 40)

        print("\nLatency Statistics:")
        print(f"Average Latency: {latency_stats['average_latency']:.2f} seconds")
        print(f"90th Percentile Latency: {latency_stats['p90_latency']:.2f} seconds")
        print(f"Maximum Latency: {latency_stats['max_latency']:.2f} seconds")

        # Print final statistics
        total_time = time.time() - total_start_time
        # print("\nTiming Statistics:")
        # category_totals = {}
        # for key, times in timing_stats.items():
        #     category_totals[key] = sum(times)
        #     print(f"{key}:")
        #     print(f"  Total time: {category_totals[key]:.3f}s")
        #     print(f"  Percentage of total: {(category_totals[key]/total_time)*100:.1f}%")
        print(f"\nTotal end-to-end time: {total_time:.3f}s")

    finally:
        print("\nCleaning up...")
        if not args.qdrant:
            collection.release()
            connections.disconnect("default")
