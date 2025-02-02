import time
import warnings
import pickle
import faiss
import torch
import datasets
import json
import os
import gzip
import functools
import zstandard as zstd
import numpy as np
from typing import Optional, List, Tuple, Dict
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
import shutil

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


@dataclass
class Document:
    """Document class to replace LangchainDocument for better SQLite integration"""

    doc_id: str
    content: str
    metadata: dict

    @classmethod
    def from_langchain(cls, doc: LangchainDocument, doc_id: str):
        return cls(doc_id=doc_id, content=doc.page_content, metadata=doc.metadata)

    def to_langchain(self) -> LangchainDocument:
        return LangchainDocument(page_content=self.content, metadata=self.metadata)


def clean_data_directory(data_dir: Path):
    """
    Safely clean up the data directory

    Args:
        data_dir: Path to the data directory to clean
    """
    try:
        if data_dir.exists():
            print(f"Cleaning existing data directory: {data_dir}")
            # 删除所有分片目录
            for shard_dir in data_dir.glob("shard_*"):
                if shard_dir.is_dir():
                    shutil.rmtree(shard_dir)
                    print(f"Removed shard directory: {shard_dir}")

            # 删除 SQLite 数据库文件
            db_file = data_dir / "documents.db"
            if db_file.exists():
                db_file.unlink()
                print(f"Removed database file: {db_file}")

            # 删除问题文件
            questions_file = data_dir / "questions.pkl"
            if questions_file.exists():
                questions_file.unlink()
                print(f"Removed questions file: {questions_file}")

            # 删除其他可能存在的数据文件
            for file in data_dir.glob("*.pkl"):
                file.unlink()
                print(f"Removed file: {file}")

            # 如果目录为空，删除目录
            if not any(data_dir.iterdir()):
                data_dir.rmdir()
                print(f"Removed empty directory: {data_dir}")
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")
        raise


class SQLiteDocumentStore:
    def __init__(self, db_path: str):
        """Initialize SQLite document store

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
            """
            )
            conn.commit()

    def add_documents(self, documents: List[Document]):
        """Add multiple documents to the store"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO documents (doc_id, content, metadata) VALUES (?, ?, ?)",
                [(doc.doc_id, doc.content, json.dumps(doc.metadata)) for doc in documents],
            )
            conn.commit()

    def get_documents(self, doc_ids: List[str]) -> List[Optional[Document]]:
        """Retrieve multiple documents by their IDs"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT doc_id, content, metadata FROM documents WHERE doc_id IN ({})".format(
                    ",".join("?" * len(doc_ids))
                ),
                doc_ids,
            )
            results = cursor.fetchall()

            # Create a mapping of doc_id to Document
            doc_map = {row[0]: Document(doc_id=row[0], content=row[1], metadata=json.loads(row[2])) for row in results}

            # Return documents in the same order as requested
            return [doc_map.get(doc_id) for doc_id in doc_ids]

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Retrieve a single document by ID"""
        docs = self.get_documents([doc_id])
        return docs[0] if docs else None


def embed_documents_with_progress(embedding_model, texts, batch_size=32):
    """
    对 texts 进行分批 embedding，并显示一个 tqdm 进度条。
    embedding_model: 你的 HuggingFaceEmbeddings 或自定义模型
    texts: List[str]
    batch_size: 每批处理多少条
    """
    all_embeddings = []
    num_texts = len(texts)

    for start_idx in tqdm(range(0, num_texts, batch_size), desc="Embedding texts"):
        end_idx = start_idx + batch_size
        batch_texts = texts[start_idx:end_idx]

        # 调用原先的 embed_documents
        batch_embeddings = embedding_model.embed_documents(batch_texts)
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


class shardVectorStore:
    def __init__(self, embeddings: torch.Tensor, index_to_docstore_id: Dict[int, str], device: str = "cuda"):
        """
        Args:
            embeddings: (N, D) tensor of embeddings
            index_to_docstore_id: Mapping from embedding index to document ID
            device: Device to store embeddings on
        """
        self.device = device
        self.embeddings = embeddings.to(self.device)
        self.index_to_docstore_id = index_to_docstore_id

        # Normalize embeddings for cosine similarity
        norms = self.embeddings.norm(dim=1, keepdim=True)
        self.embeddings.div_(norms.add_(1e-8))

    @classmethod
    def from_documents(cls, docs: List[Document], embedding_model, device: str = "cuda") -> "shardVectorStore":
        """Create vector store from documents"""
        # Generate embeddings
        texts = [doc.content for doc in docs]
        print("Embedding documents...")
        embeddings_list = embed_documents_with_progress(embedding_model, texts, batch_size=len(texts))
        embeddings_tensor = torch.tensor(embeddings_list, dtype=torch.float32)

        # Create index mapping
        index_to_docstore_id = {i: doc.doc_id for i, doc in enumerate(docs)}

        return cls(embeddings_tensor, index_to_docstore_id, device)

    def search(self, query_embeddings: torch.Tensor, k: int = 3):
        """Search for similar documents"""
        query_embeddings = query_embeddings.to(self.device)
        query_norms = query_embeddings.norm(p=2, dim=1, keepdim=True)
        query_normed = query_embeddings / (query_norms + 1e-8)

        similarities = torch.matmul(query_normed, self.embeddings.transpose(0, 1))
        distances, indices = torch.topk(similarities, k, dim=1)

        return distances.detach().cpu().numpy(), indices.detach().cpu().numpy()


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


def save_shard(vector_store: shardVectorStore, doc_store: SQLiteDocumentStore, shard_path: Path):
    """Save a vector store shard and its documents"""
    # Save embeddings
    embeddings_np = vector_store.embeddings.cpu().numpy()
    np.save(shard_path / "embeddings.npy", embeddings_np)

    # Save index mapping
    with open(shard_path / "index_to_docstore_id.pkl", "wb") as f:
        pickle.dump(vector_store.index_to_docstore_id, f)


def load_shard(shard_path: Path, device="cpu") -> shardVectorStore:
    """Load a vector store shard"""
    embeddings_np = np.load(shard_path / "embeddings.npy")
    embeddings_tensor = torch.tensor(embeddings_np, dtype=torch.float32, device=device)

    with open(shard_path / "index_to_docstore_id.pkl", "rb") as f:
        index_to_docstore_id = pickle.load(f)

    return shardVectorStore(embeddings_tensor, index_to_docstore_id, device=device)


def build_index_with_sqlite():
    """Build shard index with SQLite document store"""
    # Load raw data
    RAW_KNOWLEDGE_BASE, questions = load_nq_data("v1.0-simplified_simplified-nq-train.jsonl.gz")
    total_docs = len(RAW_KNOWLEDGE_BASE)
    print(f"Total docs: {total_docs}")

    # Initialize stores
    shards_dir = Path("rag_data_torch_sqlite")
    clean_data_directory(shards_dir)
    shards_dir.mkdir(parents=True, exist_ok=True)

    doc_store = SQLiteDocumentStore(str(shards_dir / "documents.db"))

    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-l6-v2",
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Process in shards
    shard_count = 10
    docs_per_shard = total_docs // shard_count + 1

    for i in range(shard_count):
        start_idx = i * docs_per_shard
        if start_idx >= total_docs:
            break
        end_idx = min((i + 1) * docs_per_shard, total_docs)

        # Process documents for this shard
        chunk_docs = RAW_KNOWLEDGE_BASE[start_idx:end_idx]
        print(f"Shard {i}: Processing docs {start_idx}..{end_idx} (size={len(chunk_docs)})")

        # Split documents
        splitted_docs = split_documents(
            chunk_size=512,
            knowledge_base=chunk_docs,
            tokenizer_name=EMBEDDING_MODEL_NAME,
            batch_size=100,
        )
        print(f"Shard {i} => split into {len(splitted_docs)} chunks")

        # Convert to our Document format and store in SQLite
        documents = [Document.from_langchain(doc, f"doc_{i}_{j}") for j, doc in enumerate(splitted_docs)]
        doc_store.add_documents(documents)

        # Build vector store for this shard
        vector_store = shardVectorStore.from_documents(documents, embedding_model, device="cpu")

        # Save shard
        shard_dir = shards_dir / f"shard_{i}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        save_shard(vector_store, doc_store, shard_dir)

        # Clean up
        del chunk_docs, splitted_docs, vector_store
        torch.cuda.empty_cache()

    # Save questions
    with open(shards_dir / "questions.pkl", "wb") as f:
        pickle.dump(questions, f)

    print("Finished building index with SQLite storage")


def batch_similarity_search_with_sqlite(
    shards_dir: Path, doc_store: SQLiteDocumentStore, queries: List[str], embedding_model, device="cpu", k=3
):
    """
    Search across shards and retrieve documents from SQLite
    """
    # Generate query embeddings
    query_embeddings_np = embedding_model.embed_documents(queries)
    query_embeddings_tensor = torch.tensor(query_embeddings_np, dtype=torch.float32, device=device)
    M = len(queries)

    # Get all shard paths
    shard_paths = sorted(
        [p for p in shards_dir.iterdir() if p.is_dir() and p.name.startswith("shard_")], key=lambda p: p.name
    )

    # Store (similarity, doc_id) for each query
    all_results = [[] for _ in range(M)]

    # Search each shard
    for spath in shard_paths:
        vector_store = load_shard(spath, device=device)
        distances, indices = vector_store.search(query_embeddings_tensor, k=k)

        # Collect results
        for i in range(M):
            for j in range(k):
                sim_score = distances[i][j]
                idx_in_shard = int(indices[i][j])
                doc_id = vector_store.index_to_docstore_id[idx_in_shard]
                all_results[i].append((sim_score, doc_id))

        del vector_store
        torch.cuda.empty_cache()

    # Get top-k results for each query
    final_results = []
    for i in range(M):
        # Sort by similarity score
        candidates = sorted(all_results[i], key=lambda x: x[0], reverse=True)
        top_k_doc_ids = [doc_id for (_, doc_id) in candidates[:k]]

        # Retrieve documents from SQLite
        top_k_docs = doc_store.get_documents(top_k_doc_ids)
        final_results.append([doc.to_langchain() for doc in top_k_docs if doc])

    return final_results


def batch_generate_responses(model, tokenizer, batch_queries, batch_retrieved_docs, max_new_tokens=500):
    """批量生成回答"""
    batch_prompts = []
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    for query, retrieved_docs in zip(batch_queries, batch_retrieved_docs):
        retrieved_docs_text = [doc.page_content for doc in retrieved_docs]
        context = " ".join(retrieved_docs_text)

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
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )

    # 解码所有生成的输出
    all_responses = [tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]
    return all_responses


def batch_query_with_sqlite(queries: List[str], embedding_model, device="cpu", k=3, batch_size=32):
    """
    Main query function using SQLite document store
    """
    shards_dir = Path("rag_data_torch_sqlite")
    doc_store = SQLiteDocumentStore(str(shards_dir / "documents.db"))

    # Initialize language model
    READER_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
    model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)

    all_answers = []
    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i : i + batch_size]

        # Retrieve relevant documents
        t0 = time.time()
        batch_docs_list = batch_similarity_search_with_sqlite(
            shards_dir, doc_store, batch_queries, embedding_model, device=device, k=k
        )
        print(f"Batch {i} - Retrieval time: {time.time() - t0:.2f}s")

        # Generate answers
        t0 = time.time()
        batch_responses = batch_generate_responses(model, tokenizer, batch_queries, batch_docs_list)
        print(f"Batch {i} - Generation time: {time.time() - t0:.2f}s")

        # Combine results
        for q, resp, docs in zip(batch_queries, batch_responses, batch_docs_list):
            all_answers.append({"query": q, "answer": resp, "context": "\n".join(d.page_content for d in docs)})

    return all_answers


# Example usage
if __name__ == "__main__":
    # First time: build index
    build_index_with_sqlite()

    # Query time
    shards_dir = Path("rag_data_torch_sqlite")
    with open(shards_dir / "questions.pkl", "rb") as f:
        questions = pickle.load(f)

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-l6-v2",
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    queries = [q["question"] for q in questions[:4]]
    answers = batch_query_with_sqlite(queries, embedding_model, device="cpu", k=3, batch_size=4)

    for i, result in enumerate(answers):
        print(f"\nQuery {i + 1}:")
        print(f"Q: {result['query']}")
        print(f"A: {result['answer']}")
        print("-" * 80)
