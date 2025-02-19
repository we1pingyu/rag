import time
import gzip
import json
import pickle
import psutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor
import functools
from tqdm import tqdm
import torch
import os
import gc
from dataclasses import dataclass
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from pymilvus import Collection, utility, CollectionSchema, FieldSchema, DataType, connections, Partition
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from dynagen.compression import CompressionConfig
from dynagen.llama_config import LlamaConfig, get_llama_config
from dynagen.pytorch_backend import LlamaTorchDevice, TorchDisk, get_torch_mixed_device_mem_manager
from dynagen.dynagen_opt import Policy
from dynagen.utils import ExecutionEnv, GB, str2bool
from dynagen.dynagen_llama import LlamaLM
from datasets import load_dataset, get_dataset_split_names

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
    "</p>",
    "</div>",
    "</section>",
    "<br>",
    "<br/>",
    "\t",
]


def get_memory_usage() -> Dict[str, float]:
    """Get current process memory usage in GB"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    return {
        "rss": memory_info.rss / (1024**3),  # Resident Set Size in GB
        "vms": memory_info.vms / (1024**3),  # Virtual Memory Size in GB
    }


def print_memory_usage(prefix: str = "") -> None:
    """Print current memory usage with optional prefix"""
    memory_info = get_memory_usage()
    print(f"{prefix}Memory Usage - RSS: {memory_info['rss']:.2f} GB, VMS: {memory_info['vms']:.2f} GB")


def init_dynagen_model(model_name, tokenizer, args):
    """Initialize dynagen Llama model with specified configuration"""
    gpu = LlamaTorchDevice("cuda:0")
    cpu = LlamaTorchDevice("cpu")
    disk = TorchDisk("./dynagen_offload_dir")
    env = ExecutionEnv(
        gpu=gpu, cpu=cpu, disk=disk, mixed=get_torch_mixed_device_mem_manager("default", [gpu, cpu, disk])
    )

    # Configure dynagen policy
    policy = Policy(
        gpu_batch_size=args.batch_size,
        num_gpu_batches=1,
        w_gpu_percent=args.percent[0],  # Store all weights on GPU
        w_cpu_percent=args.percent[1],
        cache_gpu_percent=args.percent[2],  # Store all cache on GPU
        cache_cpu_percent=args.percent[3],
        act_gpu_percent=100,  # Store all activations on GPU
        act_cpu_percent=0,
        overlap=True,
        sep_layer=True,
        pin_weight=False,
        cpu_cache_compute=False,
        attn_sparsity=1.0,
        compress_weight=False,
        comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False),
        compress_cache=False,
        comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False),
    )

    # Get Llama configuration
    llama_config = get_llama_config(model_name, pad_token_id=tokenizer.eos_token_id)

    # Initialize LlamaLM model
    model = LlamaLM(config=llama_config, env=env, path="~/llama_weights", policy=policy)

    return model, llama_config, env


@dataclass
class Question:
    """表示单个问题的数据类"""

    question_text: str
    doc_id: str
    arrival_time: float
    batch_id: Optional[int] = None


@dataclass
class BatchResult:
    """表示一个批次的查询结果"""

    questions: List[Question]
    query_embeddings: Optional[torch.Tensor] = None
    query_results: Optional[List[Dict]] = None
    generation_results: Optional[List[Dict]] = None


def process_doc_initializer(tokenizer_name, chunk_size):
    """初始化文档处理器"""
    global text_splitter, tokenizer
    # 先加载 tokenizer，再创建 splitter
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )


def batch_process_docs(docs_batch):
    """处理一批文档"""
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
    num_processes = os.cpu_count()  # 限制最大进程数
    process_doc_partial = functools.partial(batch_process_docs)
    batches = [knowledge_base[i : i + batch_size] for i in range(0, len(knowledge_base), batch_size)]

    with ProcessPoolExecutor(
        max_workers=num_processes, initializer=process_doc_initializer, initargs=(tokenizer_name, chunk_size)
    ) as executor:
        results = list(tqdm(executor.map(process_doc_partial, batches), total=len(batches), desc="Splitting documents"))
        return [doc for batch_result in results for doc in batch_result]


def load_nq(file_path: str, max_docs: int = 1000):
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


def load_marco(max_docs: int = None) -> Tuple[List[LangchainDocument], List[Dict]]:
    configs = ["v2.1"]
    docs = []
    questions = []
    doc_counter = 0

    for config in configs:
        print(f"\nLoading config: {config}")

        # Get all splits for this configuration
        splits = get_dataset_split_names("microsoft/ms_marco", config)
        print(f"Found splits: {splits}")

        for split in splits:
            print(f"\nProcessing split: {split}")

            # Load the dataset for this config and split
            try:
                dataset = load_dataset("microsoft/ms_marco", config, split=split)
            except Exception as e:
                print(f"Error loading {config}/{split}: {e}")
                continue

            # Create a progress bar for this split
            iterator = tqdm(
                enumerate(dataset),
                total=len(dataset) if max_docs is None else min(max_docs - len(docs), len(dataset)),
                desc=f"Loading {config}/{split}",
            )

            for i, example in iterator:
                # Check if we've reached max_docs
                if max_docs and len(docs) >= max_docs:
                    print(f"Reached max_docs limit of {max_docs}")
                    return docs, questions

                if example["passages"]["passage_text"]:
                    # Create a unique doc_id for each question
                    doc_id = f"macro_doc_{doc_counter}"
                    doc_counter += 1

                    # Combine all search contexts into one document
                    combined_context = " ".join(example["passages"]["passage_text"])
                    # Create LangchainDocument
                    doc = LangchainDocument(page_content=combined_context, metadata={"source": doc_id})
                    docs.append(doc)
                    # Store question with reference to doc_id
                    questions.append({"question": example["query"], "doc_id": doc_id})

    print(f"\nFinal dataset statistics:")
    print(f"Total documents loaded: {len(docs)}")
    print(f"Total questions loaded: {len(questions)}")
    print(f"Documents per config:")
    for config in configs:
        config_docs = len([d for d in docs if d.metadata["config"] == config])
        print(f"  {config}: {config_docs}")

    return docs, questions


def load_trivia(max_docs: int = 1000) -> Tuple[List[LangchainDocument], List[Dict]]:
    configs = ["rc", "rc.web", "rc.wikipedia", "unfiltered"]
    docs = []
    questions = []
    doc_counter = 0

    for config in configs:
        print(f"\nLoading config: {config}")

        # Get all splits for this configuration
        splits = get_dataset_split_names("mandarjoshi/trivia_qa", config)
        print(f"Found splits: {splits}")

        for split in splits:
            print(f"\nProcessing split: {split}")

            # Load the dataset for this config and split
            try:
                dataset = load_dataset("mandarjoshi/trivia_qa", config, split=split)
            except Exception as e:
                print(f"Error loading {config}/{split}: {e}")
                continue

            # Create a progress bar for this split
            iterator = tqdm(
                enumerate(dataset),
                total=len(dataset) if max_docs is None else min(max_docs - len(docs), len(dataset)),
                desc=f"Loading {config}/{split}",
            )

            for i, example in iterator:
                # Check if we've reached max_docs
                if max_docs and len(docs) >= max_docs:
                    print(f"Reached max_docs limit of {max_docs}")
                    return docs, questions

                if example["search_results"]["search_context"]:
                    # Create a unique doc_id for each question
                    doc_id = f"trivia_doc_{doc_counter}"
                    doc_counter += 1

                    # Combine all search contexts into one document
                    combined_context = " ".join(example["search_results"]["search_context"])
                    # Create LangchainDocument
                    doc = LangchainDocument(page_content=combined_context, metadata={"source": doc_id})
                    docs.append(doc)
                    # Store question with reference to doc_id
                    questions.append({"question": example["question"], "doc_id": doc_id})

    print(f"\nFinal dataset statistics:")
    print(f"Total documents loaded: {len(docs)}")
    print(f"Total questions loaded: {len(questions)}")
    print(f"Documents per config:")
    for config in configs:
        config_docs = len([d for d in docs if d.metadata["config"] == config])
        print(f"  {config}: {config_docs}")

    return docs, questions


def create_milvus_collection(collection_name: str):
    """Create Milvus collection with schema"""
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000),
        FieldSchema(name="partition_id", dtype=DataType.INT64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
    ]

    schema = CollectionSchema(fields=fields, description="Document collection for RAG")
    collection = Collection(name=collection_name, schema=schema)

    index_params = {"metric_type": "IP", "index_type": "FLAT"}
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection


def get_milvus_memory_usage():
    total_rss = 0
    # 遍历所有进程
    for proc in psutil.process_iter(attrs=["pid", "cmdline", "memory_info"]):
        cmdline = proc.info["cmdline"]
        # 判断命令行中是否包含 "milvus"（根据实际情况可以修改匹配条件）
        if cmdline and "milvus run standalone" in " ".join(cmdline):
            # memory_info().rss 返回单位为字节的常驻内存
            total_rss += proc.info["memory_info"].rss
    return total_rss / (1024**3)


def calculate_latency_stats(results):
    """计算延迟统计信息"""
    # 计算每个查询的延迟
    latencies = []
    for result in results:
        latency = result["completion_time"] - result["arrival_time"]
        latencies.append(latency)

    # 计算统计信息
    avg_latency = np.mean(latencies)
    p90_latency = np.percentile(latencies, 90)
    max_latency = np.max(latencies)

    return {
        "average_latency": avg_latency,
        "p90_latency": p90_latency,
        "max_latency": max_latency,
        "individual_latencies": latencies,
    }


def merge_datasets(
    nq_data: Tuple[List[LangchainDocument], List[Dict]], trivia_data: Tuple[List[LangchainDocument], List[Dict]]
) -> Tuple[List[LangchainDocument], List[Dict]]:
    """Merge Natural Questions and TriviaQA datasets.

    Args:
        nq_data: Tuple of (documents, questions) from load_nq_data
        trivia_data: Tuple of (documents, questions) from load_trivia

    Returns:
        Tuple containing:
        - List[LangchainDocument]: Combined list of documents
        - List[Dict]: Combined list of questions with updated doc IDs
    """
    # Unpack input data
    nq_docs, nq_questions = nq_data
    trivia_docs, trivia_questions = trivia_data

    # Initialize combined lists
    combined_docs = []
    combined_questions = []

    print(f"Processing {len(nq_docs)} NQ documents and {len(trivia_docs)} TriviaQA documents")

    # Add NQ documents and questions (keep original IDs)
    combined_docs.extend(nq_docs)
    combined_questions.extend(nq_questions)

    # Create new IDs for TriviaQA documents to avoid conflicts
    doc_id_mapping = {}

    # Process TriviaQA documents
    for i, doc in enumerate(trivia_docs):
        old_doc_id = doc.metadata["source"]
        new_doc_id = f"combined_trivia_doc_{i}"  # Create new unique ID
        doc_id_mapping[old_doc_id] = new_doc_id

        # Update document metadata
        doc.metadata["source"] = new_doc_id
        doc.metadata["original_dataset"] = "trivia_qa"
        combined_docs.append(doc)

    # Process TriviaQA questions and update doc_ids
    for question in trivia_questions:
        old_doc_id = question["doc_id"]
        new_question = {
            "question": question["question"],
            "doc_id": doc_id_mapping[old_doc_id],
            "original_dataset": "triviaqa",
        }
        combined_questions.append(new_question)

    # Add dataset source information to NQ documents and questions
    for doc in nq_docs:
        doc.metadata["original_dataset"] = "natural_questions"

    for question in nq_questions:
        question["original_dataset"] = "natural_questions"

    print(f"Combined dataset contains:")
    print(f"- {len(combined_docs)} documents")
    print(f"- {len(combined_questions)} questions")

    # Verify all doc_ids in questions have corresponding documents
    doc_ids = {doc.metadata["source"] for doc in combined_docs}
    question_doc_ids = {q["doc_id"] for q in combined_questions}
    missing_docs = question_doc_ids - doc_ids

    if missing_docs:
        print(f"Warning: Found {len(missing_docs)} questions with missing documents")

    return combined_docs, combined_questions


def build_index(
    persist_directory: Optional[str] = None,
    dataset: Optional[str] = None,
    num_partitions: int = 10,
    embedding_model=None,
    max_docs: int = 1000,
):
    """Build and populate a Milvus index with partitions"""
    # Connect to Milvus
    connections.connect(host="localhost", port="19530")

    collection_name = f"{dataset}_collection"
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

    # Load documents
    if dataset == "nq":
        RAW_KNOWLEDGE_BASE, questions = load_nq("v1.0-simplified_simplified-nq-train.jsonl.gz", max_docs=max_docs)
    elif dataset == "trivia":
        RAW_KNOWLEDGE_BASE, questions = load_trivia(max_docs=max_docs)
    elif dataset == "macro":
        RAW_KNOWLEDGE_BASE, questions = load_marco(max_docs=max_docs)
    else:
        RAW_KNOWLEDGE_BASE, questions = merge_datasets(
            load_trivia(max_docs=max_docs), load_nq("v1.0-simplified_simplified-nq-train.jsonl.gz", max_docs=max_docs)
        )
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
            chunk_size=256, knowledge_base=partition_docs, tokenizer_name=EMBEDDING_MODEL_NAME
        )
        print(f"Partition {partition_idx + 1}: split into {len(docs_processed)} chunks")

        # Prepare data for insertion
        texts = [doc.page_content for doc in docs_processed]
        doc_ids = [f"doc_{partition_idx}_{i}" for i in range(len(texts))]
        partition_ids = [partition_idx] * len(texts)

        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} documents ...")
        start_time = time.time()
        embeddings = embedding_model.embed_documents(texts)
        print(f"Embeddings generated in {time.time() - start_time:.2f} seconds")

        # Insert data in batches
        batch_size = 1000
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Inserting into partition {partition_name}"):
            end_batch = min(i + batch_size, len(texts))
            entities = [
                doc_ids[i:end_batch],
                texts[i:end_batch],
                partition_ids[i:end_batch],
                embeddings[i:end_batch],
            ]
            collection.insert(entities, partition_name=partition_name)

        # Clean up memory
        del docs_processed, texts, doc_ids, embeddings
        gc.collect()
        # torch.cuda.empty_cache()

        print(f"Completed partition {partition_idx + 1}, collection now has {collection.num_entities} documents")

    # Save questions and partition info
    save_path = Path(persist_directory)
    save_path.mkdir(parents=True, exist_ok=True)

    metadata = {"questions": questions, "partition_names": partition_names}
    with open(save_path / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print(f"Saved metadata to {save_path / 'metadata.pkl'}")

    connections.disconnect("default")
    return collection


def log_timing(timing_dict: Dict[str, List[float]], key: str, duration: float):
    """记录时间信息"""
    if key not in timing_dict:
        timing_dict[key] = []
    timing_dict[key].append(duration)


def batch_query(
    collection,
    questions: List[Dict],
    query_texts: List[str],
    query_embeddings,
    partition_names: Optional[List[str]] = None,
    search_params: Optional[dict] = None,
    timing_stats: Optional[Dict[str, List[float]]] = None,
    resident_partitions: int = 0,
):
    """Execute batch queries"""
    if timing_stats is None:
        timing_stats = {}

    aggregated_results = [[] for _ in range(len(query_texts))]

    # Process each partition
    for partition_name in tqdm(partition_names, desc="Quering"):
        # Only load if it's not a resident partition
        if partition_name not in partition_names[:resident_partitions]:
            load_start = time.time()
            collection.load(partition_names=[partition_name])
            print(f"Loaded partition {partition_name} {get_milvus_memory_usage():.2f} GB")
            log_timing(timing_stats, "partition_load_time", time.time() - load_start)

        search_start = time.time()
        partition_search_results = collection.search(
            data=query_embeddings,
            anns_field="embedding",
            param=search_params if search_params else {"metric_type": "IP"},
            limit=5,
            output_fields=["text", "doc_id"],
            partition_names=[partition_name],
        )
        log_timing(timing_stats, "search_time", time.time() - search_start)

        # Release non-resident partitions after search
        if partition_name not in partition_names[:resident_partitions]:
            partition = Partition(collection, partition_name)
            partition.release()

        for i, hits in enumerate(partition_search_results):
            aggregated_results[i].extend(hits)

    results_start = time.time()
    final_results = []
    for idx, hits in enumerate(aggregated_results):
        hits.sort(key=lambda x: x.distance, reverse=True)
        top_hits = hits[:5]
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


def batch_generate_responses(
    model,
    tokenizer,
    batch_results: List[Dict],
    max_new_tokens: int = 500,
    batch_size: int = 4,
    timing_stats: Optional[Dict[str, List[float]]] = None,
    dynagen: bool = True,
    env=None,
) -> List[Dict]:
    """生成批量回答"""
    if timing_stats is None:
        timing_stats = {}

    prep_start = time.time()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    batch_prompts = []
    for result in batch_results:
        context = result["answer"] if result["answer"] else "No relevant context found."
        prompt = f"""
        Answer the following question with a concise response (no more than 20 words) that directly uses the provided context. If the context lacks relevant information, state that you cannot answer.

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
        if dynagen:
            inputs = tokenizer(
                batch, return_tensors="pt", padding="max_length", truncation=True, max_length=512, padding_side="left"
            ).input_ids
        else:
            inputs = tokenizer(
                batch, return_tensors="pt", padding="max_length", truncation=True, max_length=512, padding_side="left"
            ).to(model.device)
        log_timing(timing_stats, "tokenization_time", time.time() - tokenize_start)

        generate_start = time.time()
        if dynagen:
            outputs = model.generate(
                inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        else:
            print("Generating ...")
            outputs = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            print("generation_time: ", time.time() - generate_start)
        log_timing(timing_stats, "generation_time", time.time() - generate_start)

        decode_start = time.time()
        batch_responses = [tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]
        log_timing(timing_stats, "decoding_time", time.time() - decode_start)

        all_responses.extend(batch_responses)

    compile_start = time.time()
    for result, response in zip(batch_results, all_responses):
        result["llm_response"] = response
    log_timing(timing_stats, "results_compilation_time", time.time() - compile_start)
    # if dynagen:
    #     env.close_copy_threads()
    return batch_results, timing_stats


def create_qdrant_collection(collection_name: str, vector_size: int = 384):
    """Create Qdrant collection with schema"""
    client = QdrantClient("localhost", port=6333)

    # Create collection
    client.recreate_collection(
        collection_name=collection_name, vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )

    return client


def build_qdrant_index(persist_directory: Optional[str] = None, num_partitions: int = 10, embedding_model=None):
    """Build and populate a Qdrant index with partitions"""
    client = create_qdrant_collection("nq_collection")
    print(f"Created Qdrant collection: nq_collection")

    # Load documents
    RAW_KNOWLEDGE_BASE, questions = load_nq("v1.0-simplified_simplified-nq-train.jsonl.gz", max_docs=None)
    print(f"Loaded {len(RAW_KNOWLEDGE_BASE)} raw documents.")

    # Process documents in partitions
    total_docs = len(RAW_KNOWLEDGE_BASE)
    docs_per_partition = (total_docs + num_partitions - 1) // num_partitions
    partition_names = [f"partition_{i}" for i in range(num_partitions)]

    print(f"Processing {total_docs} documents in {num_partitions} partitions ({docs_per_partition} docs per partition)")

    for partition_idx in range(num_partitions):
        print(f"\nProcessing partition {partition_idx + 1}/{num_partitions}")

        # Get partition documents
        start_idx = partition_idx * docs_per_partition
        if start_idx >= total_docs:
            break
        end_idx = min((partition_idx + 1) * docs_per_partition, total_docs)
        partition_docs = RAW_KNOWLEDGE_BASE[start_idx:end_idx]

        # Process documents
        docs_processed = split_documents(
            chunk_size=256, knowledge_base=partition_docs, tokenizer_name=EMBEDDING_MODEL_NAME
        )
        print(f"Partition {partition_idx + 1}: split into {len(docs_processed)} chunks")

        # Prepare data
        texts = [doc.page_content for doc in docs_processed]
        doc_ids = [f"doc_{partition_idx}_{i}" for i in range(len(texts))]

        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} documents ...")
        embeddings = embedding_model.embed_documents(texts)
        global_start_id = partition_idx * docs_per_partition

        # Insert data in batches
        batch_size = 1000
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Inserting into partition {partition_idx}"):
            end_batch = min(i + batch_size, len(texts))

            points = [
                PointStruct(
                    id=global_start_id + i + idx,
                    vector=embedding,  # embedding is already a list
                    payload={"text": text, "doc_id": doc_id, "partition_id": partition_idx},
                )
                for idx, (text, doc_id, embedding) in enumerate(
                    zip(texts[i:end_batch], doc_ids[i:end_batch], embeddings[i:end_batch])
                )
            ]

            client.upsert(collection_name="nq_collection", points=points)

        # Clean up memory
        del docs_processed, texts, doc_ids, embeddings
        torch.cuda.empty_cache()

        print(f"Completed partition {partition_idx + 1}")

    # Save questions and partition info
    save_path = Path(persist_directory)
    save_path.mkdir(parents=True, exist_ok=True)

    metadata = {"questions": questions, "partition_names": partition_names}
    with open(save_path / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print(f"Saved metadata to {save_path / 'metadata.pkl'}")

    return client


def batch_query_qdrant(
    client: QdrantClient,
    questions: List[Dict],
    query_texts: List[str],
    query_embeddings,
    partition_names: Optional[List[str]] = None,
    timing_stats: Optional[Dict[str, List[float]]] = None,
    resident_partitions: int = 0,
):
    """Execute batch query using Qdrant"""
    if timing_stats is None:
        timing_stats = {}

    results_start = time.time()
    final_results = []

    for idx, (query_embedding, query_text) in enumerate(zip(query_embeddings, query_texts)):

        # Search across all partitions
        search_results = client.search(
            collection_name="nq_collection", query_vector=query_embedding, limit=5, with_payload=True
        )
        # Process results
        docs = [hit.payload["text"] for hit in search_results]
        context = " ".join(docs)

        final_results.append(
            {
                "query": query_text,
                "answer": context,
                "metadata": {
                    "doc_ids": [hit.payload["doc_id"] for hit in search_results],
                    "searched_partitions": partition_names,
                },
                "original_doc_id": questions[idx]["doc_id"],
                "num_docs_retrieved": len(docs),
            }
        )

    log_timing(timing_stats, "result_processing_time", time.time() - results_start)
    return final_results, timing_stats
