import os
import pickle
import gzip
import json
import functools
import warnings
import torch
import shutil
import gc
import psutil
import time
import numpy as np
import argparse
import resource
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional, List, Dict
from tqdm import tqdm
from workload import WorkloadManager, Question

# Milvus imports
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="langchain")
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Global constants
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-l6-v2"
EMBEDDING_DIM = 384  # Dimension for all-MiniLM-l6-v2
MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]


####################################
# Document Processing Functions
####################################
def process_doc_initializer(tokenizer_name, chunk_size):
    global text_splitter
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )


def batch_process_docs(docs_batch):
    global text_splitter
    results = []
    for doc in docs_batch:
        results.extend(text_splitter.split_documents([doc]))
    return results


def split_documents(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,
    batch_size: int = 100,
) -> List[LangchainDocument]:
    """Split documents using multiprocessing"""
    num_processes = os.cpu_count() or 1
    process_doc_partial = functools.partial(batch_process_docs)
    batches = [knowledge_base[i : i + batch_size] for i in range(0, len(knowledge_base), batch_size)]
    docs_processed = []

    with ProcessPoolExecutor(
        max_workers=num_processes, initializer=process_doc_initializer, initargs=(tokenizer_name, chunk_size)
    ) as executor:
        results = list(
            tqdm(executor.map(process_doc_partial, batches), total=len(batches), desc="Splitting documents in batches")
        )
        for result in results:
            docs_processed.extend(result)
    return docs_processed


####################################
# Data Loading Functions
####################################
def load_nq_data(file_path: str, max_docs: int = None):
    """Load Natural Questions dataset"""
    docs = []
    questions = []
    with gzip.open(file_path, "rt") as f:
        for i, line in enumerate(tqdm(f, desc="Loading NQ data")):
            if max_docs and i >= max_docs:
                break
            if line:
                doc = json.loads(line)
                docs.append(LangchainDocument(page_content=doc["document_text"], metadata={"source": f"nq_doc_{i}"}))
                questions.append({"question": doc["question_text"] + "?", "doc_id": f"nq_doc_{i}"})
    return docs, questions


####################################
# Milvus Index Building
####################################
def create_milvus_collection(collection_name: str):
    """Create Milvus collection with the specified schema"""
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="partition_id", dtype=DataType.INT64),  # Changed from shard_id
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
    ]

    schema = CollectionSchema(fields=fields, description="Document collection for RAG")
    collection = Collection(name=collection_name, schema=schema)

    index_params = {"metric_type": "IP", "index_type": "FLAT"}
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection


def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process()
    return process.memory_info().rss / (1024**3)


def log_timing(timing_dict, key, duration):
    """Log timing information"""
    if key not in timing_dict:
        timing_dict[key] = []
    timing_dict[key].append(duration)


def get_milvus_memory_usage():
    total_rss = 0
    # 遍历所有进程
    for proc in psutil.process_iter(attrs=["pid", "cmdline", "memory_info"]):
        try:
            cmdline = proc.info["cmdline"]
            # 判断命令行中是否包含 "milvus"（根据实际情况可以修改匹配条件）
            if cmdline and "milvus run standalone" in " ".join(cmdline):
                # memory_info().rss 返回单位为字节的常驻内存
                total_rss += proc.info["memory_info"].rss
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return total_rss / (1024**3)


def build_index(persist_directory: Optional[str] = None, num_partitions: int = 10):
    """Build and populate a Milvus index with partitions"""
    # Connect to Milvus
    connections.connect(host="localhost", port="19530")

    collection_name = "nq_collection"
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    collection = create_milvus_collection(collection_name)
    print(f"Created Milvus collection: {collection_name}")

    # Create partitions
    partition_names = []
    for i in range(num_partitions):
        partition_name = f"partition_{i}"
        collection.create_partition(partition_name)
        partition_names.append(partition_name)
    print(f"Created {num_partitions} partitions")

    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Load documents
    RAW_KNOWLEDGE_BASE, questions = load_nq_data("v1.0-simplified_simplified-nq-train.jsonl.gz", max_docs=None)
    print(f"Loaded {len(RAW_KNOWLEDGE_BASE)} raw documents.")

    # Process documents in partitions
    total_docs = len(RAW_KNOWLEDGE_BASE)
    docs_per_partition = (total_docs + num_partitions - 1) // num_partitions
    print(f"Processing {total_docs} documents in {num_partitions} partitions ({docs_per_partition} docs per partition)")

    for partition_idx in range(num_partitions):
        print(f"\nProcessing partition {partition_idx + 1}/{num_partitions}")
        partition_name = partition_names[partition_idx]

        # Get partition documents
        start_idx = partition_idx * docs_per_partition
        if start_idx >= total_docs:
            break
        end_idx = min((partition_idx + 1) * docs_per_partition, total_docs)
        partition_docs = RAW_KNOWLEDGE_BASE[start_idx:end_idx]

        # Process documents
        docs_processed = split_documents(
            chunk_size=512, knowledge_base=partition_docs, tokenizer_name=EMBEDDING_MODEL_NAME
        )
        print(f"Partition {partition_idx + 1}: split into {len(docs_processed)} chunks")

        # Prepare data for insertion
        texts = [doc.page_content for doc in docs_processed]
        doc_ids = [f"doc_{partition_idx}_{i}" for i in range(len(texts))]
        partition_ids = [partition_idx] * len(texts)  # Changed from shard_ids

        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} documents ...")
        embeddings = embedding_model.embed_documents(texts)

        # Insert data in batches
        batch_size = 1000
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Inserting into partition {partition_name}"):
            end_batch = min(i + batch_size, len(texts))
            entities = [
                doc_ids[i:end_batch],
                texts[i:end_batch],
                partition_ids[i:end_batch],  # Changed from shard_ids
                embeddings[i:end_batch],
            ]
            try:
                collection.insert(entities, partition_name=partition_name)
            except Exception as e:
                print(f"Error inserting batch into partition {partition_name}: {str(e)}")
                continue

        # Clean up memory
        del docs_processed, texts, doc_ids, embeddings
        torch.cuda.empty_cache()

        print(f"Completed partition {partition_idx + 1}, collection now has {collection.num_entities} documents")

    # Save questions and partition info
    save_path = Path(persist_directory)
    save_path.mkdir(parents=True, exist_ok=True)

    metadata = {"questions": questions, "partition_names": partition_names}

    with open(save_path / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print(f"Saved metadata to {save_path / 'metadata.pkl'}")

    connections.disconnect("default")


def batch_query(
    collection: Collection,
    questions: List[Dict],
    query_texts: List[str],
    query_embeddings,
    partition_names: Optional[List[str]] = None,
    search_params: Optional[dict] = None,
    timing_stats: Optional[Dict[str, List[float]]] = None,
    resident_partitions: int = 0,
):
    """Batch query with support for resident partitions"""
    if timing_stats is None:
        timing_stats = {}

    try:
        aggregated_results = [[] for _ in range(len(query_texts))]

        # Track which partitions are already loaded
        loaded_partitions = set()

        # Pre-load resident partitions
        if resident_partitions > 0:
            resident_partition_names = partition_names[:resident_partitions]
            load_start = time.time()
            collection.load(partition_names=resident_partition_names)
            loaded_partitions.update(resident_partition_names)
            log_timing(timing_stats, "partition_load_time", time.time() - load_start)

        # Process each partition
        for partition_name in partition_names:
            # Load partition if it's not already resident
            if partition_name not in loaded_partitions:
                load_start = time.time()
                collection.load(partition_names=[partition_name])
                log_timing(timing_stats, "partition_load_time", time.time() - load_start)

            search_start = time.time()
            partition_search_results = collection.search(
                data=query_embeddings,
                anns_field="embedding",
                param=search_params if search_params else {"metric_type": "IP"},
                limit=2,
                output_fields=["text", "doc_id"],
                partition_names=[partition_name],
            )
            log_timing(timing_stats, "search_time", time.time() - search_start)

            # Only release non-resident partitions
            if partition_name not in loaded_partitions and partition_name not in partition_names[:resident_partitions]:
                collection.release(partition_names=[partition_name])

            # Aggregate results
            for i, hits in enumerate(partition_search_results):
                aggregated_results[i].extend(hits)

        results_start = time.time()
        final_results = []
        for idx, hits in enumerate(aggregated_results):
            hits.sort(key=lambda x: x.distance, reverse=True)
            top_hits = hits[:3]
            docs = [hit.entity.get("text") for hit in top_hits]
            context = " ".join(docs)
            final_results.append(
                {
                    "query": query_texts[idx],
                    "answer": context,
                    "metadata": {
                        "doc_ids": [hit.entity.get("doc_id") for hit in top_hits],
                        "searched_partitions": partition_names,
                    },
                    "original_doc_id": questions[idx]["doc_id"],
                    "num_docs_retrieved": len(docs),
                }
            )
        log_timing(timing_stats, "result_processing_time", time.time() - results_start)
        return final_results, timing_stats

    except Exception as e:
        print(f"Error in batch_query: {str(e)}")
        raise


def batch_generate_responses(
    model,
    tokenizer,
    batch_results: List[Dict],
    max_new_tokens: int = 500,
    batch_size: int = 4,
    timing_stats: Optional[Dict[str, List[float]]] = None,
) -> List[Dict]:
    """根据检索结果生成回答（保持原有逻辑）"""
    if timing_stats is None:
        timing_stats = {}

    prep_start = time.time()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    batch_prompts = []
    for result in batch_results:
        context = result["answer"] if result["answer"] else "No relevant context found."
        prompt = f"""
        Use the following context to answer the question. Provide a clear and concise answer that is specific and relevant.
        Question: {result['query']}
        Context: {context}
        Answer:
        """
        batch_prompts.append(prompt)
    log_timing(timing_stats, "prompt_prep_time", time.time() - prep_start)

    all_responses = []
    for i in range(0, len(batch_prompts), batch_size):
        batch = batch_prompts[i : i + batch_size]
        tokenize_start = time.time()
        inputs = tokenizer(
            batch, return_tensors="pt", padding="max_length", truncation=True, max_length=1024, padding_side="left"
        ).to(model.device)
        log_timing(timing_stats, "tokenization_time", time.time() - tokenize_start)

        generate_start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        log_timing(timing_stats, "generation_time", time.time() - generate_start)

        decode_start = time.time()
        batch_responses = [tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]
        log_timing(timing_stats, "decoding_time", time.time() - decode_start)

        all_responses.extend(batch_responses)

    compile_start = time.time()
    for result, response in zip(batch_results, all_responses):
        result["llm_response"] = response
    log_timing(timing_stats, "results_compilation_time", time.time() - compile_start)

    return batch_results, timing_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch query and response generation for RAG")
    parser.add_argument("--total_questions", type=int, default=32, help="Total number of questions to process")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of questions to process per batch")
    parser.add_argument("--persist_dir", type=str, default="rag_data_milvus", help="Directory for persisted data")
    parser.add_argument("--display_results", action="store_true", help="Whether to display final results")
    parser.add_argument("--cpu_memory_limit", type=int, default=64, help="CPU memory limit in GB")
    parser.add_argument("--build_index", action="store_true", help="Whether to build Milvus index")
    parser.add_argument("--resident_partitions", type=int, default=0, help="Number of resident partitions")
    parser.add_argument(
        "--arrival_rate", type=float, default=8, help="Average number of questions arriving per minutes"
    )
    args = parser.parse_args()
    persist_dir = args.persist_dir

    if args.build_index:
        build_index(persist_directory=persist_dir)

    cpu_mem_limit_bytes = args.cpu_memory_limit
    timing_stats = {}

    print("Connecting to Milvus...")
    connections.connect(host="localhost", port="19530")
    collection = Collection("nq_collection")
    collection.release()

    try:
        total_start_time = time.time()
        # Initial collection setup and memory management
        collection.load(["partition_0"])
        print(f"Initial Milvus memory usage: {get_milvus_memory_usage():.2f} GB")
        available_cpu_mem = cpu_mem_limit_bytes - get_milvus_memory_usage() * (args.resident_partitions + 1)
        resource.setrlimit(resource.RLIMIT_AS, (available_cpu_mem * 1024**3, available_cpu_mem * 1024**3))
        print(f"Set CPU available memory to {int(available_cpu_mem)} GB")

        if args.resident_partitions > 0:
            resident_partition_names = [f"partition_{i}" for i in range(args.resident_partitions)]
            collection.load(resident_partition_names)
            print(f"Loaded {args.resident_partitions} resident partitions: {resident_partition_names}")

        # Load models
        print("Loading language model...")
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        load_llm_start = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            max_memory={0: "20GiB"},
        )
        log_timing(timing_stats, "llm_load_time", time.time() - load_llm_start)

        print("Loading embedding model...")
        load_embedding_model_start = time.time()
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            multi_process=True,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        log_timing(timing_stats, "embedding_model_load_time", time.time() - load_embedding_model_start)

        # Load metadata
        metadata_path = Path(persist_dir) / "metadata.pkl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
            all_questions = metadata["questions"]
            partition_names = metadata["partition_names"]

        def process_question_batch(batch: List[Question], timing_stats: Dict[str, List[float]]):
            """处理一批问题的函数"""
            query_texts = [q.question_text for q in batch]
            questions_dict = [{"question": q.question_text, "doc_id": q.doc_id} for q in batch]

            # Release all partitions before processing new batch
            if not args.resident_partitions:
                collection.release()

            # Embedding generation
            embed_start = time.time()
            query_embeddings = embedding_model.embed_documents(query_texts)
            log_timing(timing_stats, "embedding_time", time.time() - embed_start)

            # Query phase
            batch_results, updated_timing_stats = batch_query(
                collection=collection,
                questions=questions_dict,
                query_texts=query_texts,
                query_embeddings=query_embeddings,
                partition_names=partition_names,
                timing_stats=timing_stats,
                resident_partitions=args.resident_partitions,
            )
            timing_stats.update(updated_timing_stats)

            # Generation phase
            batch_responses, updated_timing_stats = batch_generate_responses(
                model=model,
                tokenizer=tokenizer,
                batch_results=batch_results,
                max_new_tokens=128,
                batch_size=len(batch),
                timing_stats=timing_stats,
            )
            timing_stats.update(updated_timing_stats)

            if args.display_results:
                for i, result in enumerate(batch_responses):
                    print(f"\nQuery {i + 1}:")
                    print(f"Q: {result['query']}")
                    print(f"A: {result['llm_response']}")
                    print(f"Retrieved Doc IDs: {result['metadata']['doc_ids']}")

        def process_batch_wrapper(batch: List[Question]):
            return process_question_batch(batch, timing_stats)

        # Initialize and run workload manager
        print("\nInitializing workload manager...")
        workload = WorkloadManager(
            questions=all_questions[: args.total_questions],
            batch_size=args.batch_size,
            arrival_rate=args.arrival_rate,
            processing_func=process_batch_wrapper,
            timing_stats=timing_stats,
            base_time=total_start_time,  # 使用程序启动时间作为基准时间
        )

        print(f"\nStarting workload simulation:")
        print(f"Total questions: {workload.total_questions}")
        print(f"Batch size: {args.batch_size}")
        print(f"Average arrival rate: {args.arrival_rate} questions/minutes")
        print(f"Expected duration: {workload.expected_duration:.2f} seconds")

        workload.run()

        # Final statistics
        total_time = time.time() - total_start_time
        print("\nTiming Statistics:")
        category_totals = {}
        for key, times in timing_stats.items():
            category_totals[key] = np.sum(times)
            print(f"{key}:")
            print(f"  Total time: {category_totals[key]:.3f}s")
            print(f"  Percentage of total: {(category_totals[key]/total_time)*100:.1f}%")
        print(f"\nTotal end-to-end time: {total_time:.3f}s")
        measured_time = sum(category_totals.values())
        coverage_percentage = (measured_time / total_time) * 100
        print(f"Timing coverage: {coverage_percentage:.1f}% of total time accounted for")

    finally:
        print("\nCleaning up...")
        collection.release()
        connections.disconnect("default")
