import time
import warnings
import pickle
import faiss_version
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


def process_doc(doc, tokenizer_name, chunk_size):
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )
    return text_splitter.split_documents([doc])


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
    num_processes = min(os.cpu_count(), 64)

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
    faiss_version.write_index(vector_db.index, str(save_path / "faiss.index"))

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


def load_vector_database(load_path, embedding_model):
    """Load FAISS index (optionally to GPU), docstore, and index_to_docstore_id."""
    load_path = Path(load_path)

    t0 = time.time()
    # 1) 读取 faiss.index
    print("Loading FAISS index...")
    index = faiss_version.read_index(str(load_path / "faiss.index"))
    print(f"Index type: {type(index)}")

    # 若可用 GPU, 则将 faiss 索引搬到 GPU
    print("Moving FAISS index to GPU...")
    if torch.cuda.is_available():
        res = faiss_version.StandardGpuResources()
        res.setTempMemory(1024 * 1024 * 1024)
        index = faiss_version.index_cpu_to_gpu(res, 0, index)
    t1 = time.time()
    print(f"Index type: {type(index)}")

    print_gpu_memory_usage("After moving FAISS index to GPU")
    print(f"[TIMING] Loding FAISS index to GPU took {t1 - t0:.4f} seconds")

    # 2) 读取 docstore
    print("Loading docstore...")
    t0 = time.time()
    with open(load_path / "docstore.pkl", "rb") as f:
        docstore = pickle.load(f)
    t1 = time.time()
    print(f"[TIMING] Loading docstore took {t1 - t0:.4f} seconds")

    # 3) 读取 index_to_docstore_id
    print("Loading index_to_docstore_id...")
    t0 = time.time()
    with open(load_path / "index_to_docstore_id.pkl", "rb") as f:
        index_to_docstore_id = pickle.load(f)
    t1 = time.time()
    print(f"[TIMING] Loading index_to_docstore_id took {t1 - t0:.4f} seconds")

    # 4) 初始化 FAISS 对象
    vector_db = FAISS(
        embedding_function=embedding_model.embed_query,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
        distance_strategy=DistanceStrategy.COSINE,
    )
    return vector_db


def load_oscar_data(file_path: str, max_docs: int = 1000):
    """Load OSCAR dataset from .zst file"""
    docs = []
    with open(file_path, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text = reader.read().decode("utf-8")
            for i, line in enumerate(tqdm(text.split("\n"))):
                # if i >= max_docs:
                #     break
                if line:
                    doc = json.loads(line)
                    docs.append(LangchainDocument(page_content=doc["content"], metadata={"source": f"oscar_doc_{i}"}))
    return docs


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


def build_index():
    """Build and save knowledge base and vector database"""

    # RAW_KNOWLEDGE_BASE = load_oscar_data("en_meta_part_1.jsonl.zst")
    RAW_KNOWLEDGE_BASE, questions = load_nq_data("v1.0-simplified_simplified-nq-train.jsonl.gz", max_docs=100000)

    docs_processed = split_documents(
        512,
        RAW_KNOWLEDGE_BASE,
        tokenizer_name=EMBEDDING_MODEL_NAME,
    )

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": False},
    )

    print("Building vector database...")
    KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
        docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )

    save_path = Path("rag_data")
    # save_knowledge_base(RAW_KNOWLEDGE_BASE, save_path)
    save_data(RAW_KNOWLEDGE_BASE, questions, save_path)
    save_vector_database(KNOWLEDGE_VECTOR_DATABASE, save_path)


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


def batch_similarity_search(vector_db, queries: List[str], embedding_model, k=1):
    """使用 FAISS 批量查询进行相似度搜索"""
    import numpy as np

    # 确保 queries 是一个字符串列表
    if not all(isinstance(query, str) for query in queries):
        raise ValueError("All queries must be strings.")

    # 使用 embed_documents 获取批量查询嵌入
    query_embeddings = embedding_model.embed_documents(queries)  # Shape: (num_queries, embedding_dim)

    # 将 query_embeddings 转换为 numpy.ndarray
    if isinstance(query_embeddings, list):
        query_embeddings = np.array(query_embeddings)

    # 使用 FAISS 的批量查询功能
    distances, indices = vector_db.index.search(query_embeddings, k)

    # 将结果转换为文档列表
    results = []
    for query_idx in range(len(queries)):
        docs = []
        for idx in indices[query_idx]:
            if idx != -1:
                doc_id = str(vector_db.index_to_docstore_id[idx])
                # 使用 get_document 方法获取文档
                doc = vector_db.docstore.search(doc_id)
                if doc:
                    docs.append(doc)
        results.append(docs)

    return results


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
        """
        batch_prompts.append(final_prompt)

    # 批量 tokenization
    inputs = tokenizer(
        batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512, padding_side="left"
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


def batch_query(queries: List[str], embedding_model, batch_size: int = 32):
    """批量查询"""
    load_path = Path("rag_data")

    # Step 1: Load vector database
    vector_db = load_vector_database(load_path, embedding_model)
    print_gpu_memory_usage("After loading vector database")

    # Step 2: Load language model and tokenizer
    READER_MODEL_NAME = "meta-llama/Llama-3.2-3B"
    model = AutoModelForCausalLM.from_pretrained(
        READER_MODEL_NAME,
        device_map="cpu",
    )
    print(f"Model loaded on devices: {model.hf_device_map}")
    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)
    print_gpu_memory_usage("After loading language model and tokenizer")

    # Step 3: Batch processing
    all_answers = []
    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i : i + batch_size]
        print(f"\nProcessing batch {i // batch_size + 1} with {len(batch_queries)} queries...")

        # Batch similarity search
        t0 = time.time()
        batch_retrieved_docs = batch_similarity_search(vector_db, batch_queries, embedding_model, k=3)
        t1 = time.time()
        print(f"[TIMING] batch_similarity_search took {t1 - t0:.4f} seconds")

        # Batch response generation
        t0 = time.time()
        batch_responses = batch_generate_responses(model, tokenizer, batch_queries, batch_retrieved_docs)
        t1 = time.time()
        print(f"[TIMING] batch_generate_responses took {t1 - t0:.4f} seconds")

        # 保存结果
        for query, response, retrieved_docs in zip(batch_queries, batch_responses, batch_retrieved_docs):
            retrieved_docs_text = [doc.page_content for doc in retrieved_docs]
            context = "\nExtracted documents:\n"
            context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)])

            all_answers.append({"query": query, "answer": response, "context": context})

    return all_answers


if __name__ == "__main__":
    # First time: build and save index
    # build_index()

    # Query time: load and query
    # Load saved questions
    with open(Path("rag_data") / "questions.pkl", "rb") as f:
        questions = pickle.load(f)

    # 初始化嵌入模型
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 提取要查询的问题
    queries = [q["question"] for q in questions[:4]]  # 取前 64 个问题
    print(f"Processing {len(queries)} queries...")

    # 批量查询和回答生成
    answers = batch_query(queries, embedding_model, batch_size=4)

    # 打印答案
    for i, result in enumerate(answers):
        print(f"Result {i + 1}: {result['answer']}")
