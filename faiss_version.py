import time
import warnings
import pickle
import torch
import datasets
import json
import os
import gzip
import functools
import zstandard as zstd
import numpy as np
from typing import Optional, List, Tuple
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoConfig
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)  # Transformers 的部分警告属于 UserWarning
warnings.filterwarnings("ignore", module="langchain")
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-l6-v2"
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


def process_doc_initializer(tokenizer_name, chunk_size):
    global text_splitter
    # from langchain.text_splitter import RecursiveCharacterTextSplitter
    # from transformers import AutoTokenizer

    # 每个进程内初始化 text_splitter
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
    """Split documents into chunks with batching"""
    num_processes = os.cpu_count()

    # 初始化分词器
    process_doc_partial = functools.partial(batch_process_docs)

    # 将文档划分为批次
    batches = [knowledge_base[i : i + batch_size] for i in range(0, len(knowledge_base), batch_size)]

    docs_processed = []
    with ProcessPoolExecutor(
        max_workers=num_processes, initializer=process_doc_initializer, initargs=(tokenizer_name, chunk_size)
    ) as executor:
        results = list(
            tqdm(
                executor.map(process_doc_partial, batches),
                total=len(batches),
                desc="Splitting documents in batches",
            )
        )
        for result in results:
            docs_processed.extend(result)

    return docs_processed


def save_knowledge_base(knowledge_base, save_path):
    """Save RAW_KNOWLEDGE_BASE to disk"""
    save_path = Path(save_path)
    with open(save_path / "knowledge_base.pkl", "wb") as f:
        pickle.dump(knowledge_base, f)


def save_vector_database(vector_db, save_path):
    """Save FAISS index, docstore, and index_to_docstore_id to disk."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # 保存 FAISS 索引
    print("Saving FAISS index...")
    faiss.write_index(vector_db.index, str(save_path / "faiss.index"))

    # 保存 docstore
    print("Saving docstore...")
    with open(save_path / "docstore.pkl", "wb") as f:
        pickle.dump(vector_db.docstore, f)

    # 保存 index_to_docstore_id
    print("Saving index_to_docstore_id...")
    with open(save_path / "index_to_docstore_id.pkl", "wb") as f:
        pickle.dump(vector_db.index_to_docstore_id, f)


def load_knowledge_base(load_path):
    """Load RAW_KNOWLEDGE_BASE from disk"""
    load_path = Path(load_path)
    with open(load_path / "knowledge_base.pkl", "rb") as f:
        return pickle.load(f)


def load_vector_db_shard_faiss(shard_path: str, embedding_model, use_mmap=True) -> FAISS:
    """
    从指定 shard_path 加载 FAISS 索引（采用内存映射）、docstore 以及 index_to_docstore_id，
    返回一个 FAISS 对象。
    """
    shard_path = Path(shard_path)
    print(f"[load_vector_db_shard_faiss] Loading shard from {shard_path} ...")

    t0 = time.time()
    io_flag = faiss.IO_FLAG_MMAP if use_mmap else 0
    index = faiss.read_index(str(shard_path / "faiss.index"), io_flag)
    t1 = time.time()
    print(f"[load_vector_db_shard_faiss] Loaded FAISS index in {t1 - t0:.2f} seconds.")

    with open(shard_path / "docstore.pkl", "rb") as f:
        docstore = pickle.load(f)
    with open(shard_path / "index_to_docstore_id.pkl", "rb") as f:
        index_to_docstore_id = pickle.load(f)

    vector_db = FAISS(
        embedding_function=embedding_model.embed_query,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
        distance_strategy=DistanceStrategy.COSINE,
    )
    return vector_db


def load_nq_data(file_path: str, max_docs: int = None):
    """Load Natural Questions dataset and separate documents/questions"""
    docs = []
    questions = []

    with gzip.open(file_path, "rt") as f:
        for i, line in enumerate(tqdm(f, desc="Loading NQ data")):
            if max_docs and i >= max_docs:
                break
            if line:
                doc = json.loads(line)
                docs.append(
                    LangchainDocument(
                        page_content=doc["document_text"],
                        metadata={"source": f"nq_doc_{i}"},
                    )
                )
                questions.append({"question": doc["question_text"] + "?", "doc_id": f"nq_doc_{i}"})
    return docs, questions


def build_index_shard_faiss():
    """
    将原始文档分成大约 1/10 的块，
    对每一块先进行多进程分词，再用 FAISS.from_documents 构建索引，
    构建完一块后立刻保存到对应的子目录，并释放内存。
    """
    # 加载原始数据（可根据需要设定 max_docs）
    RAW_KNOWLEDGE_BASE, questions = load_nq_data("v1.0-simplified_simplified-nq-train.jsonl.gz", max_docs=1000)
    total_docs = len(RAW_KNOWLEDGE_BASE)
    print(f"[build_index_shard_faiss] Total docs: {total_docs}")

    # 以大约 1/10 为一片
    shard_count = 20
    docs_per_shard = total_docs // shard_count + 1

    shards_dir = Path("rag_data_faiss_shard")
    shards_dir.mkdir(parents=True, exist_ok=True)

    # 初始化嵌入模型
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": False},  # FAISS内部会处理距离
    )

    for i in range(shard_count):
        start_idx = i * docs_per_shard
        if start_idx >= total_docs:
            break
        end_idx = min((i + 1) * docs_per_shard, total_docs)
        chunk_docs = RAW_KNOWLEDGE_BASE[start_idx:end_idx]
        print(f"[build_index_shard_faiss] Shard {i}: processing docs {start_idx} to {end_idx} (size={len(chunk_docs)})")

        # 对当前分片做多进程分词
        splitted_docs = split_documents(
            chunk_size=512, knowledge_base=chunk_docs, tokenizer_name=EMBEDDING_MODEL_NAME, batch_size=100
        )
        print(f"[build_index_shard_faiss] Shard {i}: splitted docs count = {len(splitted_docs)}")

        # 利用 FAISS.from_documents 构建向量数据库
        shard_vector_db = FAISS.from_documents(
            splitted_docs, embedding_model, distance_strategy=DistanceStrategy.COSINE
        )

        # 保存当前分片：在子目录 shard_i 下保存 FAISS 索引、docstore、index_to_docstore_id
        shard_dir = shards_dir / f"shard_{i}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        save_vector_database(shard_vector_db, str(shard_dir))

        # 释放内存：删除当前分片的中间变量，并调用 torch.cuda.empty_cache()
        del chunk_docs, splitted_docs, shard_vector_db
        torch.cuda.empty_cache()

    # 最后统一保存 questions
    with open(shards_dir / "questions.pkl", "wb") as f:
        pickle.dump(questions, f)

    print("[build_index_shard_faiss] Done building all FAISS shards.")


def print_gpu_memory_usage(step_name: str):
    """打印当前 GPU 的显存占用"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2  # 转换为 MB
        cached = torch.cuda.memory_reserved() / 1024**2  # 转换为 MB
        print(f"[GPU MEMORY] {step_name}: Allocated: {allocated:.2f} MB, Reserved: {cached:.2f} MB")
    else:
        print(f"[GPU MEMORY] {step_name}: No GPU available")


def save_data(docs, questions, save_path):
    """Save both documents and questions"""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    with open(save_path / "questions.pkl", "wb") as f:
        pickle.dump(questions, f)


def batch_similarity_search_shard_faiss(shards_dir: str, queries: List[str], embedding_model, k=3, use_mmap=True):
    """
    对指定目录下的所有分片进行检索，并合并每个分片的 Top K 候选，最终返回全局 Top K。
    返回格式：List[List[Document]]，每个查询对应一个文档列表。
    """
    shards_dir = Path(shards_dir)
    shard_paths = sorted(
        [p for p in shards_dir.iterdir() if p.is_dir() and p.name.startswith("shard_")], key=lambda p: p.name
    )
    print(f"[batch_similarity_search_shard_faiss] Found {len(shard_paths)} shards.")

    # 生成查询向量（假设 embedding_model.embed_documents 返回 numpy 数组或列表）
    query_embeddings = embedding_model.embed_documents(queries)
    query_embeddings = np.array(query_embeddings, dtype=np.float32)
    if isinstance(query_embeddings, list):
        query_embeddings = np.array(query_embeddings)
    M = len(queries)
    all_candidates = [[] for _ in range(M)]

    # 遍历每个分片进行检索
    for spath in shard_paths:
        vector_db = load_vector_db_shard_faiss(str(spath), embedding_model, use_mmap=use_mmap)
        print(f"[batch_similarity_search_shard_faiss] Searching in {spath.name} ...")
        distances, indices = vector_db.index.search(query_embeddings, k)
        for i in range(M):
            for j in range(k):
                idx = indices[i][j]
                if idx != -1:
                    doc_id = str(vector_db.index_to_docstore_id[idx])
                    # 此处假设 docstore 是 dict 类型
                    doc = vector_db.docstore.search(doc_id)
                    if doc is not None:
                        all_candidates[i].append((distances[i][j], doc))
        del vector_db  # 释放分片资源
    # 全局排序取每个查询的 Top K
    final_results = []
    for i in range(M):
        sorted_candidates = sorted(all_candidates[i], key=lambda x: x[0], reverse=True)
        top_k_docs = [doc for (score, doc) in sorted_candidates[:k]]
        final_results.append(top_k_docs)
    return final_results


def batch_generate_responses(model, tokenizer, batch_queries, batch_retrieved_docs, max_new_tokens=500):
    """批量生成回答"""
    batch_prompts = []
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    for query, retrieved_docs in zip(batch_queries, batch_retrieved_docs):
        retrieved_docs_text = [doc.page_content for doc in retrieved_docs]
        context = "".join([f"Document {str(i)}:" + doc for i, doc in enumerate(retrieved_docs_text)])

        # 构造批量的 final_prompt
        final_prompt = f"""
        Use the following context to answer the question. Be as specific and relevant as possible.
        Question:{query}
        Context:{context}
        Answer:
        """
        batch_prompts.append(final_prompt)

    # 批量 tokenization
    inputs = tokenizer(
        batch_prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=1024, padding_side="left"
    ).to(model.device)
    print(inputs["input_ids"].shape)

    # 批量生成
    outputs = model.generate(
        **inputs,
        do_sample=False,
        temperature=0.2,
        repetition_penalty=1.1,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )

    # 解码所有生成的输出
    all_responses = [tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]
    return all_responses


def batch_query_shard_faiss(queries: List[str], embedding_model, batch_size: int = 32, use_mmap=True):
    """
    使用分片化的 FAISS 索引进行批量查询：
      - 每个分片采用内存映射加载，
      - 依次检索后合并结果，
      - 调用生成模型生成回答。
    """
    shards_dir = Path("rag_data_faiss_shard")

    READER_MODEL_NAME = "meta-llama/Llama-3.2-3B"
    model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)

    all_answers = []
    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i : i + batch_size]
        batch_docs = batch_similarity_search_shard_faiss(
            str(shards_dir), batch_queries, embedding_model, k=1, use_mmap=use_mmap
        )
        batch_responses = batch_generate_responses(model, tokenizer, batch_queries, batch_docs)
        for q, resp, docs in zip(batch_queries, batch_responses, batch_docs):
            context = "\n".join(doc.page_content for doc in docs)
            all_answers.append({"query": q, "answer": resp, "context": context})
    return all_answers


if __name__ == "__main__":
    # First time: build and save index
    # build_index_shard_faiss()

    # 查询时：加载 questions，并使用分片检索
    with open(Path("rag_data_faiss_shard") / "questions.pkl", "rb") as f:
        questions = pickle.load(f)

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 例如查询前 4 个问题
    queries = [q["question"] for q in questions[:4]]
    print(f"Processing {len(queries)} queries using shard FAISS ...")
    answers = batch_query_shard_faiss(queries, embedding_model, batch_size=4, use_mmap=False)

    for i, result in enumerate(answers):
        print(f"Result {i + 1}: {result['answer']}")
