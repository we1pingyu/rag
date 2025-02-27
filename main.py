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
from pipeline import PipelineProcessor
from baseline import BaselineProcessor
from offline import OfflineProcessor
from dyn_offline import DynOfflineProcessor
from active_profiling import ActiveProfilingProcessor
from utils import build_index, get_milvus_memory_usage, calculate_latency_stats, init_dynagen_model, build_qdrant_index
from qdrant_client import QdrantClient


os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="langchain")
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings

MAX_BATCH_SIZE = 65536

# Global constants
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-l6-v2"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pipeline RAG processing")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="LLM Model name")
    parser.add_argument("--total_questions", type=int, default=6, help="Total number of questions to process")
    parser.add_argument("--batch_size", type=int, default=2, help="Number of questions to process per batch")
    parser.add_argument("--persist_dir", type=str, default="trivia_data_milvus", help="Directory for persisted data")
    parser.add_argument("--dataset", type=str, default="trivia", help="Dataset to use for rag: nq or trivia or macro")
    parser.add_argument("--display_results", action="store_true", help="Whether to display final results")
    parser.add_argument("--cpu_memory_limit", type=int, default=180, help="CPU memory limit in GB")
    parser.add_argument("--gpu_memory_limit", type=int, default=12, help="GPU memory limit in GB")
    parser.add_argument("--resident_partitions", type=int, default=0, help="Number of resident partitions")
    parser.add_argument("--arrival_rate", type=float, default=16, help="Number of questions arriving per minute")
    parser.add_argument("--build_index", action="store_true", help="Whether to build Milvus index")
    parser.add_argument("--num_partitions", type=int, default=32, help="Number of partitions for index building")
    parser.add_argument("--dynagen", action="store_true", help="Whether to use DynaGen for generation")
    parser.add_argument("--qdrant", action="store_true", help="Use Qdrant instead of Milvus")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--baseline", action="store_true", help="Run in baseline (serial) mode")
    parser.add_argument("--offline", action="store_true", help="Run in offline mode")
    parser.add_argument("--dyn_offline", action="store_true", help="Run in offline mode with dynamic configuration")
    parser.add_argument("--active", action="store_true", help="Run in active profiling mode")
    parser.add_argument(
        "--percent",
        nargs="+",
        type=int,
        default=[0, 11, 66, 0],
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
        if args.qdrant:
            client = build_qdrant_index(
                persist_directory=args.persist_dir, num_partitions=args.num_partitions, embedding_model=embedding_model
            )
        else:
            build_index(
                persist_directory=args.persist_dir,
                num_partitions=args.num_partitions,
                embedding_model=embedding_model,
                dataset=args.dataset,
            )
        print("Index building completed")

    if args.qdrant:
        print("Connecting to Qdrant...")
        client = QdrantClient("localhost", port=6333)
    else:
        print("Connecting to Milvus...")
        connections.connect(host="localhost", port="19530")
        collection = Collection(f"{args.dataset}_collection")
        collection.release()

    try:
        if not args.qdrant:
            total_gpu_memory = 24
            fraction = args.gpu_memory_limit / total_gpu_memory
            print(f"Setting GPU memory fraction to {fraction}")
            torch.cuda.set_per_process_memory_fraction(fraction)

            collection.load(["partition_0"])
            print(f"Initial Milvus memory usage: {get_milvus_memory_usage():.2f} GB")
            partition_size_gb = get_milvus_memory_usage()
            available_cpu_mem = args.cpu_memory_limit - partition_size_gb * (args.resident_partitions + 1)
            resource.setrlimit(resource.RLIMIT_AS, (int(available_cpu_mem * 1024**3), int(available_cpu_mem * 1024**3)))
            print(f"Set CPU available memory to {int(available_cpu_mem)} GB")

            # 加载 resident partitions
            if args.resident_partitions > 0:
                resident_partition_names = [f"partition_{i}" for i in range(args.resident_partitions)]
                collection.load(resident_partition_names)
                print(f"Loaded {args.resident_partitions} resident partitions: {resident_partition_names}")
        print("Loading models and initializing...")
        # Load LLM
        model_name = args.model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if args.dynagen:
            print("\nInitializing DynaGen model...")
            model, model_config, env = init_dynagen_model(model_name, tokenizer, args)
            dummy_text = ["Hello world"] * 4  # Simple warmup text
            dummy_input = tokenizer(
                dummy_text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512,
                padding_side="left",
            ).input_ids
            # Perform warmup
            model.generate(
                dummy_input,
                do_sample=False,
                max_new_tokens=1,
                pad_token_id=tokenizer.pad_token_id,
            )
            torch.cuda.empty_cache()
        else:
            print("\nInitializing HF model...")
            max_memory_per_batch = 1.5  # GiB per batch
            total_memory = args.gpu_memory_limit  # Total available memory in GiB

            # Calculate remaining memory after accounting for batch size
            max_memory_size = total_memory - (args.batch_size * max_memory_per_batch)

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                max_memory={0: f"{max_memory_size}GiB"},
            )

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
        if args.offline:
            print("\nRunning in offline mode (batch processing)")
            processor = OfflineProcessor(
                questions=all_questions[: args.total_questions],
                batch_size=args.batch_size,
                embedding_model=embedding_model,
                model=model,
                tokenizer=tokenizer,
                collection=collection,
                partition_names=["partition_0"],
                w_gpu_percent=args.percent[0],
                w_cpu_percent=args.percent[1],
                cache_gpu_percent=args.percent[2],
                cache_cpu_percent=args.percent[3],
                resident_partitions=args.resident_partitions,
                dynagen=args.dynagen,
                env=env,
            )
        elif args.active:
            processor = ActiveProfilingProcessor(
                questions=all_questions,
                embedding_model=embedding_model,
                model=model,
                tokenizer=tokenizer,
                collection=collection,
                partition_names=partition_names,
                partition_size_gb=partition_size_gb,
                model_config=model_config,
                total_cpu_gb=available_cpu_mem,
                gpu_memory_gb=args.gpu_memory_limit,
                env=env,
            )

        elif args.dyn_offline:
            print("\nRunning in offline mode with dynamic configuration")
            processor = DynOfflineProcessor(
                questions=all_questions[: args.total_questions],
                batch_size=args.batch_size,
                embedding_model=embedding_model,
                model=model,
                tokenizer=tokenizer,
                collection=client if args.qdrant else collection,
                partition_names=partition_names,
                resident_partitions=args.resident_partitions,
                use_qdrant=args.qdrant,
                dynagen=args.dynagen,
                cache_gpu_percent=args.percent[2],
                cache_cpu_percent=args.percent[3],
                env=env,
            )
        elif args.baseline:
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
                dynagen=args.dynagen,
                env=env if args.dynagen else None,
            )
        else:
            # Initialize pipeline manager
            processor = PipelineProcessor(
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
                dynagen=args.dynagen,
            )

        if not args.active:
            print(f"\nStarting processing:")
            print(f"Total questions: {len(all_questions[:args.total_questions])}")
            print(f"Batch size: {args.batch_size}")
        if not args.offline and not args.dyn_offline and not args.active:
            print(f"Average arrival rate: {args.arrival_rate} questions/minute")
            print(f"Expected duration: {processor.expected_duration:.2f} seconds")

        # Run pipeline
        if args.active:
            optimal_configs = processor.run()
        elif args.dyn_offline:
            processor.run(
                experiment_config=[
                    {
                        "batch_size": args.batch_size,
                        "cache_gpu_percent": 80,
                        "cache_cpu_percent": 20,
                        "resident_partitions": 1,  # New parameter
                    },
                    {
                        "batch_size": args.batch_size + 1,
                        "cache_gpu_percent": 50,
                        "cache_cpu_percent": 50,
                        "resident_partitions": 2,  # Will release partitions 4 and 3
                    },
                    {
                        "batch_size": args.batch_size + 2,
                        "cache_gpu_percent": 0,
                        "cache_cpu_percent": 20,
                        "resident_partitions": 3,  # Will load partitions 6 and 5
                    },
                ]
            )
        else:
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
        if args.dynagen:
            if os.path.exists("./dynagen_offload_dir"):
                shutil.rmtree("./dynagen_offload_dir")
            env.close_copy_threads()
