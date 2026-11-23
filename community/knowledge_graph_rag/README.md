# Knowledge Graphs for RAG with NVIDIA AI Foundation Models and Endpoints

This repository implements a GPU-accelerated pipeline for creating and querying knowledge graphs using Retrieval-Augmented Generation (RAG). Our approach leverages NVIDIA's AI technologies and RAPIDS ecosystem to process large-scale datasets efficiently.

## Overview

This project demonstrates:
- Creation of knowledge graphs from various document sources
- Provides a simple script to download research papers from Arxiv for a given topic
- GPU-accelerated graph processing and analysis using NVIDIA's RAPIDS Graph Analytics library (cuGraph): https://github.com/rapidsai/cugraph
- Hybrid semantic search combining keyword and dense vector approaches
- Integration of knowledge graphs into RAG workflows
- Visualization of the knowledge graph through [Gephi-Lite](https://github.com/gephi/gephi-lite), an open-source web app for visualization of large graphs
- Comprehensive evaluation metrics using NVIDIA's Nemotron-4 340B model for synthetic data generation and reward scoring

## Technologies Used

- **Frontend**: Streamlit
- **Graph Representation and Optimization**: cuGraph (RAPIDS), NetworkX
- **Vector Database**: Milvus
- **LLM Models**:
  - NVIDIA AI Playground hosted models for graph creation and querying, providing numerous instruct-fine-tuned options
  - NVIDIA AI Playground hosted Nemotron-4 340B model for synthetic data generation and evaluation reward scoring

## Architecture Diagram

Here is how the ingestion system is designed, by leveraging a high throughput hosted LLM deployment which can process multiple document chunks in parallel. The LLM can optionally be fine-tuned for triple extraction, thereby requiring a shorter prompt and enabling greater accuracy and optimized inference.

```mermaid
graph TD
    A[Document Collection] --> B{Document Splitter}
    B --> |Chunk 1| C1[LLM Stream 1]
    B --> |Chunk 2| C2[LLM Stream 2]
    B --> |Chunk 3| C3[LLM Stream 3]
    B --> |...| C4[...]
    B --> |Chunk N| C5[LLM Stream N]
    C1 --> D[Response Parser<br>and Aggregator]
    C2 --> D
    C3 --> D
    C4 --> D
    C5 --> D
    D --> E[GraphML Generator]
    E --> F[Single GraphML File]
```

Here's how the inference system is designed, incorporating both hybrid dense-vector search and sparse keyword-based search, reranking, and Knowledge Graph for multi-hop search:

```mermaid
graph LR
    E(User Query) --> A(FRONTEND<br/>Chat UI<br/>Streamlit)
    A --Dense-Sparse<br>Retrieval--> C(Milvus Vector DB)
    A --Multi-hop<br>Search--> F(Knowledge Graph <br> with cuGraph)
    C --Hybrid<br>Chunks--> X(Reranker)
    X -- Augmented<br/>Prompt--> B((Hosted LLM API<br/>NVIDIA AI Playground))
    F -- Graph Context<br>Triples--> B
    B --> D(Streaming<br/>Chat Response)
```

This architecture shows how the user query is processed through both the Milvus Vector DB for traditional retrieval and the Knowledge Graph with cuGraph for multi-hop search. The results from both are then used to augment the prompt sent to the NVIDIA AI Playground backend.

## 2026 Update: Running This Example Today

The setup steps below date from mid-2024. In November 2026 the example no longer ran as written: its hosted models were retired, its pinned packages no longer install, and two parsing steps broke with newer models. This fork contains the fixes. It was tested end to end on 2026-11-23 (Python 3.12, Milvus 2.6.27, NVIDIA GB10 / aarch64): 5 arXiv papers → 898 chunks → 3,853 triples → a graph of 3,197 entities.

### What changed and why

| Problem | Fix |
|---|---|
| The models the code names are retired (HTTP 410/404): `mistralai/mixtral-8x7b-instruct-v0.1`, `meta/llama3-70b-instruct`, `nvidia/nemotron-4-340b-instruct`, `nvidia/nemotron-4-340b-reward`, `nvidia/nv-embed-v1` | New `backend/utils/nvidia_llm.py`: default model `nvidia/nemotron-3.5-super-vl-120b-a12b`; RAGAS embeddings use `nvidia/nemotron-3.5-embed-1b` |
| The UI's model list comes from `langchain-nvidia-ai-endpoints` 0.1.2's built-in 2024 table (16 dead models), and the `"instruct" in model.id` filter matches almost nothing today | `live_chat_models()` lists the live API catalog's chat models, default first |
| Today's Nemotron models "think" first: with `max_tokens=1024`, the reasoning used up the budget and `content` came back empty (0/20 test chunks parsed) | `ChatNVIDIA` subclass sends `chat_template_kwargs={"enable_thinking": False}` (20/20 parsed, ~0.95 s per chunk) |
| Models often wrap JSON or Python lists in Markdown code fences, so `json.loads` / `ast.literal_eval` fail (the chat page then silently drops all graph context) | `strip_fences()` before parsing triples and entity lists |
| `requirements.txt` no longer installs on Python 3.12 (numpy 1.24 has no wheels; several imported packages are missing) | `requirements-2026.txt`; each change is explained in its header |
| `python server.py` crashes (`TypeError ... loop_factory`, ragas' nest_asyncio vs current uvicorn) | Start the backend with the `uvicorn` CLI |
| The upload page sometimes shows "Received invalid progress data." and the progress bar stops (the backend rewrites `progress.txt` while the frontend reads it) | `update_progress` writes a temporary file and renames it atomically |
| The Q&A slider on the Evaluation page starts at 10 | It now starts at 1 |

The Nemotron-4 340B reward model has no hosted successor. On the Evaluation page the "Nemotron" scoring therefore fails with 410, while LLM-as-a-judge and RAGAS run with the substitute models above. The results chart further down is the original authors' published result.

### Quick start (2026)

```bash
git clone https://github.com/myli5/GenerativeAIExamples.git && cd GenerativeAIExamples/community/knowledge_graph_rag

# API key: load it from a private file rather than typing it into a shared or recorded terminal
export NVIDIA_API_KEY="nvapi-..."

# Python environment (uv is fast; plain venv + pip also works)
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements-2026.txt
chmod -R go-w .venv          # NLTK refuses group-writable data folders

# Milvus standalone, ports bound to localhost
docker compose -f milvus-standalone-local.yml up -d
curl -s http://localhost:9091/healthz      # -> OK

# .env (DATA_DIR must be an absolute path)
mkdir -p backend/data
printf "DATA_DIR=$PWD/backend/data\nBACKEND_URL=http://localhost:8000\n" > .env

# Papers to build the graph from (any folder of PDFs inside backend/ works)
python backend/utils/download_papers.py -s 'abs:"knowledge graph" AND abs:"retrieval-augmented generation"' -sd 2024-05-24 -ed 2024-05-31 -n 5 -d backend/papers

# Terminal 1: backend
cd backend && uvicorn server:app --host 127.0.0.1 --port 8000
# Terminal 2: frontend
cd frontend && streamlit run ui.py
```

If port 8000 is taken, run uvicorn on another port and set `BACKEND_URL` to match. In the UI, pick the model, select the `papers` folder and click **Process Documents**. Then use the chat page with **Use knowledge graph** ticked, and the visualization page to open `backend/data/knowledge_graph.graphml` in Gephi Lite.

The README mentions cuGraph, but the code builds and queries the graph with plain NetworkX. `backend/utils/cugraph_bench.py` runs the same NetworkX calls with the nx-cugraph GPU backend. On a GB10 with nx-cugraph-cu13 26.10, the top-5 rankings were identical and the speed-ups were:

| Graph | PageRank | Betweenness centrality |
|---|---|---|
| Our graph (3,197 nodes) | ×1.7 | ×366 |
| Synthetic graph (2M edges) | ×76 | ×194 |

Known limitations that remain:
- **Duplicate entities.** The extracted graph keeps entities that differ only in case (for example "DepsRAG" and "DEPSRAG").
- **Exact-name lookup.** Graph lookup in the chat matches entity names exactly, so a query can get zero triples even though the graph holds the facts.

## Setup Steps (original, 2024)

Follow these steps to get the chatbot up and running in less than 5 minutes:

### 1. Clone this repository to a Linux machine

```bash
git clone https://github.com/NVIDIA/GenerativeAIExamples/ && cd GenerativeAIExamples/community/knowledge_graph_rag
```

### 2. Get an NVIDIA AI Playground API Key

```bash
export NVIDIA_API_KEY="nvapi-*******************"
```

If you don't have an API key, follow [these instructions](https://github.com/NVIDIA/GenerativeAIExamples/blob/main/docs/api-catalog.md#get-an-api-key-for-the-accessing-models-on-the-api-catalog) to sign up for an NVIDIA AI Foundation developer account and obtain access.

### 3. Create a Python virtual environment and activate it

Refer to the .env.example include the data directory to point to data folder you create inside the backend folder: backend/data. Also set the backend URL.

### 4. Create a Python virtual environment and activate it

```bash
cd knowledge_graph_rag
pip install virtualenv
python3 -m virtualenv venv
source venv/bin/activate
```

### 5. Install the required packages

```bash
pip install -r requirements.txt
```

### 6. Setup a hosted Milvus vector database

Follow the instructions [here](https://milvus.io/docs/install_standalone-docker.md) to deploy a hosted Milvus instance for the vector database backend. Note that it must be Milvus 2.4 or better to support [hybrid search](https://milvus.io/docs/multi-vector-search.md). We do not support disabling this feature for previous versions of Milvus as of now.

### 7. Launch the Streamlit frontend and FAST API backend

```bash
cd knowledge_graph_rag
cd backend
python server.py

```
In another terminal,
```bash
cd knowledge_graph_rag
cd frontend
streamlit run ui.py

```

Open the URL in your browser to access the UI and chatbot!

### 8. Upload Docs and Train Model

Upload your own documents to a folder, or use an existing folder for the knowledge graph creation. Note that the implementation currently focuses on text from PDFs only. It can be extended to other text file formats using the Unstructured.io data loader in LangChain.

## Pipeline Components

1. **Data Ingestion**:
   - ArXiv paper downloader
   - Arbitrary document folder ingestion
2. **Knowledge Graph Creation**:
   - Uses the API Catalog models through the LangChain NVIDIA AI Endpoints interface
3. **Graph Representation**: cuGraph + RAPIDS + NetworkX
4. **Semantic Search**: Milvus 2.4.x for hybrid (keyword + dense vector) search
5. **RAG Integration**: Custom workflow incorporating knowledge graph retrieval
6. **Evaluation**: Comparison of different RAG approaches using Nemotron-4 340B model

## Evaluation Metrics

We've implemented comprehensive evaluation metrics using NVIDIA's Nemotron-4 340B model, which is designed for synthetic data generation and reward scoring. Our evaluation compares different RAG approaches across five key attributes:

1. **Helpfulness**: Overall helpfulness of the response to the prompt.
2. **Correctness**: Inclusion of all pertinent facts without errors.
3. **Coherence**: Consistency and clarity of expression.
4. **Complexity**: Intellectual depth required to write the response.
5. **Verbosity**: Amount of detail included in the response, relative to what is asked for in the prompt.

## Evaluation Results

We compared four RAG approaches on a small representative dataset using the NeMoTron-340B reward model:

![Evaluation Results](viz.png)

Key takeaways:
- Graph RAG significantly outperforms traditional Text RAG.
- Combined Text and Graph RAG shows promise but doesn't consistently beat the ground truth yet. This may be due to the way we structure the augmented prompt for the LLM and needs more experimentation.
- Our approach improves on verbosity and coherence compared to ground truth.

While we're not beating long-context ground truth across the board, these results show the potential of integrating knowledge graphs into RAG systems. We're particularly excited about the improvements in verbosity and coherence. Next steps include refining how we combine text and graph retrieval to get the best of both worlds.

## Component Swapping

All components are designed to be swappable. Here are some options:

- **Frontend**: The current Streamlit implementation can be replaced with other web frameworks.
- **Retrieval**: The embedding model and reranker model being used for semantic search can be swapped to use other models for higher performance. The number of entities retrieved prior to reranking can also be changed. The chunk size for documents can be changed.
- **Vector DB**: While we use Milvus, it can be replaced with options like ChromaDB, Pinecone, FAISS, etc. Milvus is designed to be highly performant and scale on GPU infrastructure.
- **Backend**:
  - Cloud Hosted: Currently uses NVIDIA AI Playground APIs, but can be deployed in a private DGX Cloud or AWS/Azure/GCP with NVIDIA GPUs and LLMs.
  - On-Prem/Locally Hosted: Smaller models like Llama2-7B or Mistral-7B can be run locally with appropriate hardware. Fine-tuning can also be done for the purpose of a specific model designed for triple extraction for a given use-case.

## Future Work

- Dynamic information incorporation into knowledge graphs (continuous update of knowledge graphs)
- Further refinement of evaluation metrics and combined semantic-graphRAG pipeline
- Investigating the impact of different graph structures and queries on RAG performance (single/multi-hop retrieval, BFS/DFS, etc)
- Expanding support for various document types and formats (multimodal RAG with knowledge graphs)
- Fine-tuning the Nemotron-4 340B model for domain-specific evaluations

## Contributing

Please create a merge request to this repository, our team appreciates any and all contributions that add features! We will review and get back as soon as possible.

## Acknowledgements

This project utilizes NVIDIA's AI technologies, including the Nemotron-4 340B model, and the RAPIDS ecosystem. We thank the open-source community for their invaluable contributions to the tools and libraries used in this project.