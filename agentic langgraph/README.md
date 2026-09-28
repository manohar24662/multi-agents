# Agentic LangGraph

`agentic-langgraph` is a command-line multi-agent workflow built with
[LangGraph](https://langchain-ai.github.io/langgraph/). Given one task, it creates a
short plan, sends work to specialist agents, optionally retrieves relevant private
documents from Qdrant, and returns a final response.

The project currently uses Groq for generation and a local SentenceTransformer model
for RAG embeddings. No NVIDIA or Gemini model is used by the workflow.

## Contents

- [What the application does](#what-the-application-does)
- [Architecture and workflow](#architecture-and-workflow)
- [Requirements and installation](#requirements-and-installation)
- [Configuration](#configuration)
- [Running a task](#running-a-task)
- [RAG: ingesting and querying documents](#rag-ingesting-and-querying-documents)
- [Chunking strategy](#chunking-strategy)
- [Testing and graph visualisation](#testing-and-graph-visualisation)
- [Project layout](#project-layout)
- [Troubleshooting and limitations](#troubleshooting-and-limitations)

## What the application does

For a request such as `tell me about Durga Manohar`, the workflow can:

1. Create a compact plan for the task.
2. Retrieve relevant chunks from your private Qdrant collection, when RAG is enabled.
3. Research the request using the retrieved context.
4. Analyse the research and document context.
5. Write a final, formatted response.

RAG retrieval is source-preserving: the prompts label each retrieved passage with its
original filename and, for PDFs, its page number. The writer is instructed not to
invent information beyond the supplied context.

## Architecture and workflow

```text
User task
   |
   v
Planner -> Supervisor -> Researcher -> Supervisor -> Analyst -> Supervisor -> Writer -> End
              |
              +-- optional Groq routing, otherwise local rules

First RAG use: Qdrant -> source-labelled context in shared state -> worker prompts
```

The actual worker sequence is dynamic. The planner proposes work, while the supervisor
chooses among the available agents. One remaining worker is chosen deterministically.
When `SUPERVISOR_USE_LLM=true`, Groq selects among the other available agents. If that
request fails or returns an invalid route, local rules choose the next worker.

| Component | Responsibility |
| --- | --- |
| `planner_agent.py` | Builds a numbered plan using the user task. |
| `supervisor_agent.py` | Routes to an available worker and enforces the iteration limit. |
| `research_agent.py` | Produces concise research and includes retrieved private context. |
| `analysist_agent.py` | Produces implications, risks, patterns, and recommendations. The filename retains the existing `analysist` spelling. |
| `writer_agent.py` | Combines task, research, analysis, and RAG context into the final response. |
| `llm.py` | Lazily configures the primary and fallback Groq clients. |
| `graph.py` / `router.py` | Defines the LangGraph nodes, conditional routes, and terminal conditions. |
| `state.py` | Declares the shared workflow state, including plan, research, analysis, report, error, and iteration fields. |

Each worker runs once in a normal workflow. The writer marks the task complete, which
ends the graph. `--max-iterations` limits supervisor decisions (default: `10`) to
prevent an accidental loop.

There is also an `agents/operations/` placeholder. It is **not connected to the graph**
and has no enabled tools, so it cannot perform system actions.

## Requirements and installation

- Python 3.14 or newer (as declared in `pyproject.toml`)
- [uv](https://docs.astral.sh/uv/)
- A Groq API key
- Optional: a Qdrant Cloud collection and API key for RAG

Install the locked dependency set from the repository root:

```powershell
uv sync --locked --link-mode copy
```

`--link-mode copy` is useful in this Windows + OneDrive workspace because hard-linking
can fail there. `requirements.txt` only installs this project itself; `pyproject.toml`
and `uv.lock` are the dependency sources of truth.

## Configuration

Create a `.env` file in the repository root. Keep it private: it contains API keys and
must not be committed to Git.

### Groq settings

```dotenv
# Required
GROQ_API_KEY=your_primary_groq_key

# Optional primary-model settings
GROQ_MODEL=openai/gpt-oss-20b
GROQ_MAX_TOKENS=4096

# Optional fallback. The fallback key may be a second Groq key.
# If GROQ_FALLBACK_API_KEY is omitted, GROQ_API_KEY is reused.
GROQ_FALLBACK_API_KEY=your_second_groq_key
GROQ_FALLBACK_MODEL=openai/gpt-oss-120b

# Optional Groq-based Supervisor routing. false (default) uses local rules.
SUPERVISOR_USE_LLM=false
```

All generation calls use temperature `0`. If a primary Groq request raises an error
(for example `429 Too Many Requests`) or returns no visible text, the relevant worker
automatically tries `GROQ_FALLBACK_MODEL`. If both attempts fail, the workflow ends
with an explanatory error and exit code `1`.

`GROQ_MAX_TOKENS` is the maximum completion size for **each** model request, not a
total account token allowance. Provider rate limits and quota still apply. Separate
primary and fallback keys can have separate provider quotas when they belong to
different Groq accounts or projects.

### RAG settings

```dotenv
# Required for indexing and retrieval
QDRANT_CLUSTER_ENDPOINT=https://your-cluster.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key

# Optional
QDRANT_COLLECTION=agentic_langgraph_documents
RAG_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RAG_EMBEDDING_DEVICE=cpu

# Keep false until at least one successful ingestion.
RAG_ENABLED=false

```

The embedding model is downloaded from Hugging Face the first time it is needed and
then cached locally. An optional `HF_TOKEN` can reduce Hugging Face anonymous-request
warnings and improve Hub rate limits. With RAG disabled or unconfigured, retrieval
returns no context and the normal agent workflow still runs.

## Running a task

From the repository root:

```powershell
uv run python -m agentic_langgraph.main 'Explain the benefits and risks of solar energy'
```

Or use the installed project command:

```powershell
uv run agentic-langgraph 'Explain the benefits and risks of solar energy'
```

Set a lower safety limit while testing:

```powershell
uv run python -m agentic_langgraph.main 'Tell me about Durga Manohar' --max-iterations 4
```

PowerShell accepts single quotes around the task. They also avoid confusing terminal
line wrapping while you type a longer question. A successful run prints `FINAL
RESPONSE`, the current timestamp, the task, and the writer's answer.

## RAG: ingesting and querying documents

### Supported document formats

| Format | Handling | Metadata |
| --- | --- | --- |
| `.txt` | Loads the full file as text. | Source, title, page `1` |
| `.md`, `.markdown` | Splits at Markdown heading hierarchy before semantic chunking. | Source, title, heading path, page `1` |
| text-based `.pdf` | Extracts text per PDF page. | Source, title, page |
| `.docx` | Extracts non-empty paragraphs. | Source, title, page `1` |

Scanned/image-only PDFs have no extractable text with `pypdf`; run OCR first or add an
OCR-capable loader. A damaged PDF may also fail before indexing.

### Index documents

Place documents in any folder, then run one explicit ingestion command. For example:

```powershell
uv run python -m agentic_langgraph.RAG_pipline.data_ingestion.processor `
  'src\agentic_langgraph\RAG_pipline\data_ingestion\files'
```

To ingest one file instead, provide that file path. You can customise the chunk target:

```powershell
uv run python -m agentic_langgraph.RAG_pipline.data_ingestion.processor `
  'C:\path\to\documents' --min-chunk-tokens 400 --max-chunk-tokens 512
```

Indexing is the only workflow action that writes to Qdrant. It embeds final chunks,
creates the configured collection when needed, and upserts the points using cosine
similarity. After a successful command, set `RAG_ENABLED=true`, then run normal tasks.

### Verify retrieval

`test_retrieval.py` is a small live diagnostic rather than an automated unit test. It
queries Qdrant and prints retrieved text, source, and similarity score:

```powershell
uv run python test_retrieval.py
```

Edit its query before running it if needed. It may print private document content, so
do not paste its output into public logs.

### How the workflow uses RAG

When enabled, the first worker that needs document context retrieves up to four candidate
chunks and creates a single RAG packet in LangGraph state. Researcher, Analyst, and
Writer reuse that packet, so a normal full run performs **one** Qdrant search and query
embedding instead of three.

Retrieved chunks are passed to the agents with source, page, and heading labels. The
prompts tell the agents to treat document text as untrusted data rather than instructions;
this is a prompt-level precaution, not an automatic security filter.

Retrieval errors are deliberately non-fatal: they are logged as `RAG retrieval was
skipped`, and the agents continue without private context.

## Chunking strategy

The chunker is in `RAG_pipline/data_ingestion/chunking/text_chunker.py`.

1. Markdown is divided by ATX (`#`) and Setext headings while ignoring fenced code
   blocks. Heading paths such as `Guide > Vector Search` are retained.
2. Each section is recursively divided into paragraphs, sentences, clauses, and then
   words only when necessary.
3. The embedding-model tokenizer measures token counts. The target is 400--512 tokens.
   Smaller chunks are intentionally retained when a document section itself is short or
   an exact minimum-size partition is impossible.
4. For an oversized section, every sentence receives an embedding. Candidate boundaries
   favour the least-similar adjacent sentences, avoiding a split through closely
   related material where possible.
5. Every final chunk retains `source`, `title`, `heading`, `page`, and `chunk_index`.
   The final chunks are embedded again before Qdrant upsert.

The default `BAAI/bge-small-en-v1.5` embedding model supports the 512-token upper
bound. Do not substitute a model with a shorter input limit unless you also lower
`--max-chunk-tokens`.

## Testing and graph visualisation

Run the fast, offline unit tests:

```powershell
uv run python -m unittest test_llm.py test_workflow.py test_chunking.py test_rag_context.py
```

These tests mock Groq, Qdrant, and embeddings; they do not consume API quota. They
cover Groq configuration/fallback, graph paths and error handling, RAG prompt
inclusion, shared RAG-packet reuse and routing, and Markdown semantic chunk metadata.

To render the graph:

```powershell
uv run python -m agentic_langgraph.scripts.debug_visualize
```

The script prints an ASCII graph when available and writes `graph.png` in the current
working directory when Mermaid PNG rendering succeeds.

## Project layout

```text
src/agentic_langgraph/
|-- main.py                         # CLI entry point
|-- graph.py                        # LangGraph construction
|-- router.py                       # Conditional graph routes
|-- state.py                        # Shared state schema
|-- agents/
|   |-- llm.py                      # Groq primary/fallback clients
|   |-- planner_agent.py
|   |-- supervisor_agent.py
|   |-- research_agent.py
|   |-- analysist_agent.py
|   |-- writer_agent.py
|   `-- operations/                 # Unwired safe placeholder
|-- RAG_pipline/
|   |-- data_ingestion/
|   |   |-- loader/                 # TXT/Markdown/PDF/DOCX extraction
|   |   |-- chunking/               # Heading-aware semantic chunks
|   |   `-- processor.py            # Ingestion CLI
|   `-- services/retrieval/
|       |-- embeddings.py           # Local BGE embeddings
|       `-- qdrant_store.py         # Qdrant indexing/retrieval
|       `-- rag_context.py          # Shared source-labelled RAG packet
`-- scripts/debug_visualize.py

test_llm.py                         # Offline LLM tests
test_workflow.py                    # Offline graph tests
test_chunking.py                    # Offline chunker tests
test_rag_context.py                 # Offline shared RAG-packet tests
test_retrieval.py                   # Live Qdrant diagnostic
```

The folder name `RAG_pipline` is intentionally shown exactly as it appears in the
current imports, despite the spelling. Renaming it requires updating every import.

## Troubleshooting and limitations

| Symptom | Meaning and action |
| --- | --- |
| `GROQ_API_KEY is missing` | Add the primary key to `.env` and start a new command. |
| `429 Too Many Requests` followed by a response | The primary model was rate-limited; the fallback completed successfully. No action is required unless it happens frequently. |
| Both Groq models fail | Wait for provider availability/quota, check both keys/models, then retry. The workflow cannot create a response without a model. |
| First RAG request is slow | The BGE embedding model is downloading/loading. Later retrievals in the same process are faster. |
| Hugging Face anonymous warning | Optional: set `HF_TOKEN`; the warning does not stop RAG. |
| `RAG retrieval was skipped` | Qdrant may be disabled, missing configuration, unavailable, or empty. The workflow continues without RAG. |
| Retrieved text asks the model to ignore rules | Retrieved passages are source-labelled and described as untrusted data in prompts, but there is no automated passage screening. Index trusted documents and verify answers against their sources. |
| No content from a PDF | The PDF is scanned/image-only, encrypted, or malformed. OCR or a different source document is needed. |
| `No readable ... documents` during ingestion | Verify the path and use supported formats. |
| RAG returns old passages after document deletion/shrinking | Ingestion upserts deterministic IDs but does not currently remove obsolete Qdrant points. Use a new collection or manually clean affected points before relying on replaced documents. |

Treat Qdrant as private storage: source paths and chunk text are uploaded during
ingestion. Only ingest material you are authorised to store there, and avoid sharing
terminal output that contains retrieved private information.
