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


class MyTorchVectorDatabase:
    def __init__(
        self,
        embeddings: torch.Tensor,
        docstore: Dict[str, LangchainDocument],
        index_to_docstore_id: Dict[int, str],
        device: str = "cuda",
    ):
        """
        embeddings: (N, D) 的 torch.Tensor
        docstore: doc_id -> Document
        index_to_docstore_id: index -> doc_id
        device: "cuda" 或 "cpu"
        """
        self.device = device

        # 把所有向量搬到指定 device（GPU/CPU）
        self.embeddings = embeddings.to(self.device)
        self.docstore = docstore
        self.index_to_docstore_id = index_to_docstore_id

        # 如果要用余弦相似度，则对向量做归一化
        # 注意：要保证 embeddings 不是全0，否则 norm=0
        norms = self.embeddings.norm(dim=1, keepdim=True)
        self.embeddings.div_(norms.add_(1e-8))

    @classmethod
    def from_documents(
        cls, docs: List[LangchainDocument], embedding_model, device: str = "cuda"
    ) -> "MyTorchVectorDatabase":
        """
        构建向量数据库:
          1) 对 docs 分块后文本 用 embedding_model.embed_documents() 生成向量
          2) 转成 torch.Tensor，搬到 GPU/CPU
          3) 建立 docstore 与 index_to_docstore_id
        """
        # 1) 取所有文档文本
        texts = [doc.page_content for doc in docs]

        # 2) 批量生成向量
        #    如果 embedding_model 返回 np.ndarray 或 List[List[float]],
        #    需要转成 torch.Tensor
        print("Embedding documents...")
        embeddings_list = embed_documents_with_progress(
            embedding_model, texts, batch_size=int(len(texts) / 10)
        )  # => shape (N, D) in NumPy
        embeddings_tensor = torch.tensor(embeddings_list, dtype=torch.float32)

        # 3) 构建 docstore、index_to_docstore_id
        docstore = {}
        index_to_docstore_id = {}
        for i, doc in enumerate(docs):
            doc_id = f"doc_{i}"
            docstore[doc_id] = doc
            index_to_docstore_id[i] = doc_id

        return cls(embeddings_tensor, docstore, index_to_docstore_id, device)

    def search(self, query_embeddings: torch.Tensor, k: int = 3):
        """
        给定查询向量 query_embeddings (M, D) (Torch Tensor)，返回 (distances, indices)
          distances: shape (M, k)，为余弦相似度
          indices:   shape (M, k)
        """
        # 1) 把查询向量搬到同样的 device
        query_embeddings = query_embeddings.to(self.device)

        # 2) 做归一化
        query_norms = query_embeddings.norm(p=2, dim=1, keepdim=True)
        query_normed = query_embeddings / (query_norms + 1e-8)

        # 3) 计算相似度 = q_normed @ e_normed.T   [M, N]
        similarities = torch.matmul(query_normed, self.embeddings.transpose(0, 1))
        print(query_normed.device, self.embeddings.device)
        # similarities.shape = (M, N)

        # 4) 利用 torch.topk 获取前 k 个最大值(相似度高=最相似)
        #    distances.shape = (M, k), indices.shape = (M, k)
        distances, indices = torch.topk(similarities, k, dim=1)

        # 5) 返回 CPU 上的 numpy 数组 或者保持 Torch Tensor 也可以
        # 如果后面需要再在 GPU 上做别的操作，就别 .cpu()
        return distances.detach().cpu().numpy(), indices.detach().cpu().numpy()


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


def save_vector_database(vector_db: MyTorchVectorDatabase, save_path: str):
    """
    保存 embeddings、docstore、index_to_docstore_id。
    注意：embedding 是 torch.Tensor，存为 .pt/.bin 等 PyTorch格式 或 npy也行（先转回CPU np.array）
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    print("Saving embeddings to CPU npy for demonstration...")
    # 要存成 numpy，先移到 CPU 并转 np.array
    embeddings_np = vector_db.embeddings.cpu().numpy()
    np.save(save_path / "embeddings.npy", embeddings_np)

    print("Saving docstore...")
    with open(save_path / "docstore.pkl", "wb") as f:
        pickle.dump(vector_db.docstore, f)

    print("Saving index_to_docstore_id...")
    with open(save_path / "index_to_docstore_id.pkl", "wb") as f:
        pickle.dump(vector_db.index_to_docstore_id, f)


def load_knowledge_base(load_path):
    """Load RAW_KNOWLEDGE_BASE from disk"""
    load_path = Path(load_path)
    with open(load_path / "knowledge_base.pkl", "rb") as f:
        return pickle.load(f)


def load_vector_database(load_path: str, device: str = "cuda") -> MyTorchVectorDatabase:
    """
    从磁盘加载 MyTorchVectorDatabase
    """
    load_path = Path(load_path)

    print("Loading embeddings...")
    t0 = time.time()
    embeddings_np = np.load(load_path / "embeddings.npy")
    # 转回 torch.Tensor 并放到指定 device
    embeddings_tensor = torch.tensor(embeddings_np, dtype=torch.float32, device=device)
    t1 = time.time()
    print(f"[TIMING] Loading embeddings took {t1 - t0:.4f} seconds")

    print("Loading docstore...")
    t0 = time.time()
    with open(load_path / "docstore.pkl", "rb") as f:
        docstore = pickle.load(f)
    t1 = time.time()
    print(f"[TIMING] Loading docstore took {t1 - t0:.4f} seconds")

    print("Loading index_to_docstore_id...")
    t0 = time.time()
    with open(load_path / "index_to_docstore_id.pkl", "rb") as f:
        index_to_docstore_id = pickle.load(f)
    t1 = time.time()
    print(f"[TIMING] Loading index_to_docstore_id took {t1 - t0:.4f} seconds")

    vector_db = MyTorchVectorDatabase(embeddings_tensor, docstore, index_to_docstore_id, device=device)
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


def build_index():
    RAW_KNOWLEDGE_BASE, questions = load_nq_data("v1.0-simplified_simplified-nq-train.jsonl.gz")

    docs_processed = split_documents(
        512,
        RAW_KNOWLEDGE_BASE,
        tokenizer_name=EMBEDDING_MODEL_NAME,
    )

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-l6-v2",
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 原本是使用:
    # KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(docs_processed, embedding_model)
    # save_vector_database(KNOWLEDGE_VECTOR_DATABASE, save_path)

    # 改为：
    KNOWLEDGE_VECTOR_DATABASE = MyTorchVectorDatabase.from_documents(docs_processed, embedding_model, device="cpu")
    save_vector_database(KNOWLEDGE_VECTOR_DATABASE, "rag_data_torch")

    # 同时保存原始文档和问题
    save_data(RAW_KNOWLEDGE_BASE, questions, "rag_data_torch")


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


def batch_similarity_search(vector_db: MyTorchVectorDatabase, queries, embedding_model, k=3):
    """
    给定一个向量数据库+查询字符串列表 queries，用embedding_model获取查询向量后，
    通过vector_db.search拿到 Top K 相似文档
    """
    # 1) 生成查询向量 (M, D) -> Torch Tensor
    #  如果 embedding_model 返回 numpy，就手动转换
    query_embeddings_np = embedding_model.embed_documents(queries)  # shape (M, D)
    query_embeddings_tensor = torch.tensor(query_embeddings_np, dtype=torch.float32)

    # 2) 调用向量数据库的 search
    distances, indices = vector_db.search(query_embeddings_tensor, k=k)

    # 3) 把检索到的文档拿出来
    results = []
    for i in range(len(queries)):
        docs = []
        for idx in indices[i]:
            doc_id = vector_db.index_to_docstore_id[int(idx)]
            doc = vector_db.docstore[doc_id]
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


def batch_query(queries, embedding_model, device="cuda", batch_size=32):
    # 1) 加载自定义Torch向量数据库
    vector_db = load_vector_database("rag_data_torch", device=device)

    # 2) 加载语言模型 (放GPU或CPU)
    READER_MODEL_NAME = "meta-llama/Llama-3.2-3B"
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME, device_map="cuda")
    t1 = time.time()
    print(f"[TIMING] Loading model took {t1 - t0:.4f} seconds")
    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)

    all_answers = []
    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i : i + batch_size]

        # 相似度检索
        t0 = time.time()
        batch_docs = batch_similarity_search(vector_db, batch_queries, embedding_model, k=3)
        t1 = time.time()
        print(f"[TIMING] Similarity search took {t1 - t0:.4f} seconds")

        # 生成回答（略），调用 batch_generate_responses(...) 等
        t0 = time.time()
        batch_responses = batch_generate_responses(model, tokenizer, batch_queries, batch_docs)
        t1 = time.time()
        print(f"[TIMING] Response generation took {t1 - t0:.4f} seconds")

        # 整理输出
        for q, resp, docs in zip(batch_queries, batch_responses, batch_docs):
            all_answers.append({"query": q, "answer": resp, "context": "\n".join(doc.page_content for doc in docs)})

    return all_answers


if __name__ == "__main__":
    # 第一次执行时：构建并保存索引
    build_index()

    # 推理时：
    with open(Path("rag_data_torch") / "questions.pkl", "rb") as f:
        questions = pickle.load(f)

    # 初始化Embedding
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-l6-v2",
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 选取若干问题做测试
    queries = [q["question"] for q in questions[:4]]

    answers = batch_query(queries, embedding_model, device="cpu", batch_size=4)

    for i, result in enumerate(answers):
        print(f"Result {i + 1}: {result['answer']}")
