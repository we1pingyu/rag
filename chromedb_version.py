import os
import pickle
import gzip
import json
import functools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional, List, Dict
from tqdm import tqdm
from transformers import AutoTokenizer
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import shutil

# 全局常量
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


####################################
# 分词与预处理函数
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
    """使用多进程将文档分块"""
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
# 数据加载函数（以 Natural Questions 为例）
####################################
def load_nq_data(file_path: str, max_docs: int = None):
    """加载 Natural Questions 数据集，返回文档列表和问题列表"""
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


class EmbeddingFunctionWrapper:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        input = [text.replace("\n", " ") for text in input]
        return self.embedding_model.embed_documents(input)


####################################
# 构建索引，并存储到 ChromaDB（Flat 索引，内存加载）
####################################
def build_index(persist_directory: Optional[str] = None):
    """
    Build and populate a ChromaDB index.

    Args:
        persist_directory: If provided, data will be persisted to this directory.
                         If None, data will be stored in memory.

    Returns:
        tuple: (client, collection, embedding_model)
    """
    if Path(persist_directory).exists():
        shutil.rmtree(persist_directory)

    # 1. Load and process documents
    RAW_KNOWLEDGE_BASE, questions = load_nq_data("v1.0-simplified_simplified-nq-train.jsonl.gz", max_docs=10)
    print(f"Loaded {len(RAW_KNOWLEDGE_BASE)} raw documents.")

    docs_processed = split_documents(
        chunk_size=512, knowledge_base=RAW_KNOWLEDGE_BASE, tokenizer_name=EMBEDDING_MODEL_NAME
    )
    print(f"After splitting, obtained {len(docs_processed)} document chunks.")
    for doc in docs_processed:
        print(f"Document: {doc.page_content}...")

    # 2. Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    embedding_fn = EmbeddingFunctionWrapper(embedding_model)

    # 3. Initialize ChromaDB client
    if persist_directory:
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=persist_directory)
        print(f"Created persistent client with directory: {persist_directory}")
    else:
        client = chromadb.Client()
        print("Created in-memory client")

    # 4. Create or get collection
    collection = client.get_or_create_collection(
        name="nq_collection",
        embedding_function=embedding_fn,
    )

    # 5. Prepare data for insertion
    texts = [doc.page_content for doc in docs_processed]
    ids = [f"doc_{i}" for i in range(len(texts))]
    metadatas = [doc.metadata for doc in docs_processed]

    # 6. Add documents to collection
    batch_size = int(len(texts) / 10)  # Reduce batch size to avoid memory issues
    for i in range(0, len(texts), batch_size):
        end_idx = min(i + batch_size, len(texts))
        collection.add(documents=texts[i:end_idx], metadatas=metadatas[i:end_idx], ids=ids[i:end_idx])
        print(f"Added batch {i//batch_size + 1} of {(len(texts) + batch_size - 1)//batch_size}")

    print(f"Collection now has {collection.count()} documents.")

    # 7. Save questions
    save_path = Path(persist_directory) if persist_directory else Path("in_memory_data")
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "questions.pkl", "wb") as f:
        pickle.dump(questions, f)
    print(f"Saved questions to {save_path / 'questions.pkl'}")

    return client, collection


####################################
# 批量查询函数（从 questions.pkl 中加载问题，并批量查询）
####################################
def batch_query(embedding_model, batch_size: int = 4, persist_directory: str = "rag_data_chroma"):
    """
    Perform batch queries on the ChromaDB collection.

    Args:
        embedding_model: HuggingFaceEmbeddings model instance
        batch_size: Number of questions to process in each batch (default: 4)
        persist_directory: Directory where ChromaDB data is stored

    Returns:
        list: List of dictionaries containing query-answer pairs
    """
    try:
        # Load questions from pickle file
        questions_path = Path(persist_directory) / "questions.pkl"
        if not questions_path.exists():
            raise FileNotFoundError(f"Questions file not found at {questions_path}")

        with open(questions_path, "rb") as f:
            questions = pickle.load(f)
            print(f"Loaded {len(questions)} questions from {questions_path}")

        # Take only the first batch_size questions
        questions = questions[:batch_size]
        query_texts = [q["question"] for q in questions]
        print(f"Processing {len(query_texts)} questions in batch")

        # Initialize embedding function and get collection
        embedding_fn = EmbeddingFunctionWrapper(embedding_model)
        client = chromadb.PersistentClient(path=persist_directory)
        try:
            collection = client.get_collection(name="nq_collection", embedding_function=embedding_fn)
        except Exception as e:
            raise ValueError(f"Failed to get collection: {str(e)}")

        # Perform batch query
        batch_result = collection.query(
            query_texts=query_texts,
            n_results=3,  # 增加返回的文档数量
        )

        # 处理和合并结果
        results = []
        for idx, (query, docs, metadata) in enumerate(
            zip(query_texts, batch_result["documents"], batch_result["metadatas"])
        ):
            # 合并多个检索到的文档
            context = " ".join(docs)

            # 记录完整的检索信息
            results.append(
                {
                    "query": query,
                    "answer": context,
                    "metadata": metadata,
                    "original_doc_id": questions[idx]["doc_id"],
                    "num_docs_retrieved": len(docs),
                }
            )

        print(f"Successfully processed {len(results)} queries")
        return results

    except Exception as e:
        print(f"Error in batch_query: {str(e)}")
        raise


def batch_generate_responses(
    model, tokenizer, batch_results: List[Dict], max_new_tokens: int = 500, batch_size: int = 4
) -> List[Dict]:
    """
    Generate responses for a batch of queries using retrieved documents from ChromaDB.

    Args:
        model: The language model for generation
        tokenizer: The tokenizer corresponding to the model
        batch_results: List of results from batch_query function
        max_new_tokens: Maximum number of tokens to generate
        batch_size: Size of batches to process

    Returns:
        List[Dict]: Original results with added LLM responses
    """
    # Set padding token if not set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Prepare prompts for each query-context pair
    batch_prompts = []
    for result in batch_results:
        # Format context from the retrieved document
        context = result["answer"] if result["answer"] else "No relevant context found."

        # Construct prompt
        final_prompt = f"""
        Use the following context to answer the question. Be as specific and relevant as possible.
        Question:{result['query']}
        Context:{context}
        Answer:
        """
        batch_prompts.append(final_prompt)

    # Process in smaller batches to manage memory
    all_responses = []
    for i in range(0, len(batch_prompts), batch_size):
        batch = batch_prompts[i : i + batch_size]

        # Tokenize batch
        inputs = tokenizer(
            batch, return_tensors="pt", padding="max_length", truncation=True, max_length=1024, padding_side="left"
        ).to(model.device)

        print(f"Processing batch {i//batch_size + 1}, input shape: {inputs['input_ids'].shape}")

        # Generate responses
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode responses
        batch_responses = [tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]
        all_responses.extend(batch_responses)

    # Add responses to results
    for result, response in zip(batch_results, all_responses):
        result["llm_response"] = response

    return batch_results


# Example usage in main:
if __name__ == "__main__":
    persist_dir = "rag_data_chroma"
    # Build index
    build_index(persist_directory=persist_dir)

    # Initialize models
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Or your preferred model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    try:
        # Get retrieved documents
        retrieval_results = batch_query(embedding_model, batch_size=4, persist_directory=persist_dir)

        # Generate responses
        final_results = batch_generate_responses(
            model=model, tokenizer=tokenizer, batch_results=retrieval_results, max_new_tokens=500, batch_size=4
        )

        # Display results
        print("\nFinal Results:")
        for i, result in enumerate(final_results, 1):
            print(f"\nQuery {i}:")
            print(f"Q: {result['query']}")
            print(f"A: {result['llm_response']}")
            print(f"Retrieved Doc ID: {result['original_doc_id']}")

    except Exception as e:
        print(f"Error during processing: {str(e)}")
