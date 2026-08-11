# RAG Pipeline Fundamentals: A Beginner's Guide

This guide explains the ideas behind this project's Retrieval-Augmented
Generation (RAG) pipeline. It is intended for readers who are new to RAG and
want to understand not only which components exist, but why they exist and how
they work together.

## 1. What problem does RAG solve?

A language model knows patterns learned during training, but it does not
automatically know the latest or private facts in Jira, Confluence, PDFs, or
other project sources. RAG adds a retrieval step before answer generation:

```text
User question
    -> retrieve relevant project evidence
    -> give that evidence to the answer model
    -> generate an answer with citations, or refuse when evidence is insufficient
```

Retrieval does not prove that an answer is correct. Its job is to place the
best available evidence in front of the answer model. Grounding and citation
validation then reduce the risk that the model answers from unsupported
assumptions.

## 2. Why is it called a pipeline?

The service has ingest and query interfaces, but each interface coordinates a
sequence of internal stages. The output of one stage becomes the input of the
next, which is why the whole system is called a pipeline.

There are two main paths:

```text
Indexing path: source -> canonical document -> chunks -> search indexes
Query path:    question -> retrieval -> fusion -> evidence -> answer
```

The API interface is the entry point. The pipeline describes the processing
that happens behind that interface.

## 3. The indexing path

### 3.1 Canonical documents

Jira issues, Confluence pages, and PDFs arrive in different shapes. The service
first converts each source into a common document representation containing
stable identity, text, source information, and metadata.

Typical fields include:

```text
document_id, source_type, source_url, title, content, metadata
```

Stable identity matters because the service must update, delete, deduplicate,
and cite the same document consistently across multiple indexes.

### 3.2 Chunking: deciding the retrieval unit

Documents can be too long and contain too many topics to search as one item.
Chunking divides a document into smaller, meaningful passages:

```text
Document
    -> chunk 0
    -> chunk 1
    -> chunk 2
```

Each chunk receives a stable `chunk_id`. A chunk is the normal unit returned by
retrieval and supplied to the answer model.

Chunk size is a trade-off:

- A very large chunk can mix unrelated topics and weaken both lexical and
  semantic relevance.
- A very small chunk can separate a fact from the context needed to understand
  it.
- Overlap or structure-aware boundaries can reduce information loss at chunk
  edges.

### 3.3 Chunking is not the same as tokenization

Chunking divides a document into passages. Tokenization divides one passage
into model- or index-readable units.

```text
Document
    -> chunking
Chunks
    -> tokenization
Tokens
```

This project uses tokenization for different purposes:

- SQLite FTS5 tokenization creates searchable terms for lexical retrieval.
- The OpenAI embedding model's tokenizer converts text into model input tokens.
- A chunker may also count tokens to keep each passage inside a size limit.

These tokenizers can follow different rules. For example, an identifier such
as `KESTREL-741` may be split into multiple tokens. That is one reason the
service also preserves exact identifiers as metadata and supports deterministic
Jira-key lookup.

## 4. Dual indexing

The shared chunker produces each chunk once, then sends the same chunk identity
to two search indexes:

```text
                         +-> OpenAI embedding -> ChromaDB vector index
Canonical document
    -> shared chunks ----+
                         +-> SQLite FTS5 lexical index
```

This is called **dual indexing**. It does not mean that two unrelated versions
of the document are created. It means that the same canonical chunks receive
two different searchable representations:

- a lexical representation for words, identifiers, and exact terminology;
- a vector representation for meaning and paraphrases.

The indexes must share stable `document_id` and `chunk_id` values so results
from both paths can be deduplicated and fused.

Existing data that was written only to ChromaDB must be reingested after FTS5
is enabled. A vector index cannot act as a lexical index automatically.

## 5. Lexical retrieval: FTS5 and BM25

### 5.1 What FTS5 provides

SQLite FTS5 is a full-text search engine built into SQLite. It tokenizes
searchable fields and builds an inverted index.

Normal document-oriented storage looks like this:

```text
chunk-0 -> session, expire, inactivity
chunk-1 -> session, security, policy
chunk-2 -> inactivity, monitoring
```

An inverted index reverses that relationship:

```text
session    -> chunk-0, chunk-1
inactivity -> chunk-0, chunk-2
```

It works like the keyword index at the back of a book. The engine can look up
the chunks associated with a query term instead of scanning every chunk's full
text.

FTS5 is not merely a file format. In this project it is the embedded lexical
search engine that supplies tokenization, the inverted index, matching, and the
statistics needed for ranking.

### 5.2 What BM25 provides

BM25 is a relevance-ranking algorithm, not a database. It ranks lexical
matches using three main ideas:

1. Rare query terms are usually more informative than common terms.
2. Repeated occurrences help, but their benefit eventually saturates.
3. Long passages are normalized so they do not win merely because they contain
   more words.

An approximate mental model is:

```text
BM25 relevance
    = term rarity
    x term concentration in this chunk
    x document-length normalization
```

FTS5 and BM25 are related but are not the same thing. BM25 can be implemented
by many search engines. FTS5 is useful for this PoC because it already provides
an inverted index and a built-in `bm25()` ranking function without requiring a
separate Elasticsearch or OpenSearch service.

During ingestion, FTS5 builds the index and records term statistics. It cannot
calculate relevance for a question that does not exist yet. At query time it
matches the question's terms and calculates the BM25 ranking for that question.

This project gives more weight to matches in important fields. The defaults are
10 for an issue key, 5 for a title, and 1 for content. A Jira key in an
identifier field should therefore carry more lexical importance than an
incidental occurrence in a long body.

Lexical retrieval is especially useful for:

- Jira keys and document identifiers;
- error codes and version numbers;
- product names and uncommon terminology;
- queries that use the same words as the source.

## 6. Semantic retrieval: tokens, embeddings, and ChromaDB

### 6.1 From text to an embedding

For each document chunk, the embedding model performs a process conceptually
similar to this:

```text
Chunk text
    -> tokenizer
    -> token IDs
    -> contextual model representations
    -> pooling/projection
    -> one fixed-size chunk vector
```

The tokenizer only translates text into the model's numerical vocabulary. It
does not create semantic understanding by itself. The trained embedding model
uses the relationships among all tokens in context to produce the semantic
vector.

Individual vector dimensions are not normally human-readable labels such as
`timeout` or `authentication`. Meaning is distributed across many dimensions.
Texts with similar meanings tend to appear near each other in that learned
vector space.

### 6.2 Query embeddings

A normal, focused user question is usually not chunked. The entire question is
converted into one query embedding:

```text
Ingestion:
document chunk -> embedding model -> chunk vector -> ChromaDB

Query:
user question  -> same embedding model -> query vector
                                      -> compare with chunk vectors
```

The ingest and query paths must use compatible embedding models because their
vectors must live in the same learned space. Changing the embedding model
normally requires re-embedding the stored chunks.

Vector search can connect paraphrases that share meaning without sharing many
words. For example:

```text
Question: How long before an inactive user must authenticate again?
Chunk:    User sessions expire after seventeen minutes of inactivity.
```

The lexical overlap is incomplete, but their semantic vectors may be close.

Semantic retrieval is useful for:

- natural-language questions;
- paraphrases and synonyms;
- conceptually related text with limited word overlap.

It is less reliable for opaque identifiers whose value comes from exact
characters rather than semantic meaning.

## 7. The query path

For a normal question, this project's hybrid query path is:

```text
User query
    -> deterministic hints and explicit filters
    -> FTS5/BM25 lexical retrieval
    -> ChromaDB vector retrieval
    -> exact Jira-key lookup when applicable
    -> RRF fusion and deduplication
    -> optional reranking
    -> evidence selection
    -> /retrieve results or /query grounded generation
```

### 7.1 Exact and metadata retrieval

Not every question should depend on fuzzy relevance. If a query contains a
known Jira key such as `AUTH-101`, deterministic lookup can find that identity
directly. Metadata filters can restrict retrieval to fields such as project,
status, or source type.

These paths complement lexical and semantic retrieval:

| Method | Best suited to |
|---|---|
| Exact lookup | Stable IDs such as Jira keys |
| Metadata filtering | Project, status, type, and other structured constraints |
| BM25 | Exact words, rare terms, codes, and names |
| Vector search | Meaning, paraphrases, and conceptual similarity |

### 7.2 Why RRF is needed

BM25 scores and vector distances do not have the same scale or meaning. Adding
their raw numbers would be arbitrary. Reciprocal Rank Fusion (RRF) combines
their ranked positions instead:

```text
RRF score(candidate) = sum of 1 / (k + rank in each result list)
```

A chunk that ranks well in both lexical and vector search receives support from
both channels. A strong result from only one channel can still survive. Results
are deduplicated using stable chunk identity.

RRF improves rank fusion; it does not determine whether the evidence is true,
split a complex question, or guarantee that every user intent is covered.

### 7.3 Optional reranking and evidence selection

RRF uses ranks from the retrieval systems. An optional reranker can examine the
full query and candidate passages together and produce a more query-specific
ordering. In this PoC, reranking is optional and failures fall back to the RRF
order.

Evidence selection then chooses a small, bounded set for answer generation.
The `/retrieve` endpoint returns retrieval evidence without generating an
answer. The `/query` endpoint additionally asks the configured answer model to
use that evidence and validates its citation references.

## 8. One query vector and multi-intent questions

The default mental model is:

```text
one focused query -> one query embedding -> several relevant document chunks
```

Returning several chunks matters because a complete answer can require evidence
from adjacent passages or multiple documents.

A single question can nevertheless contain several unrelated intents:

```text
What is the session timeout, who approves password resets,
and which deployment token should be used?
```

One vector must compress all three topics, and no document chunk may cover all
of them. The goal should not be to force every query part to match one chunk.
The goal is to retrieve evidence that covers each information need.

A future query-decomposition flow could do this:

```text
Multi-intent question
    -> subquery 1 -> query vector 1 -> retrieval results
    -> subquery 2 -> query vector 2 -> retrieval results
    -> subquery 3 -> query vector 3 -> retrieval results
    -> merge evidence and synthesize the answer
```

Decomposition adds model calls, latency, cost, merge logic, and the risk of
changing the user's intent. It is intentionally not part of this PoC. A useful
current practice is to ask one primary question at a time and introduce
decomposition only when evaluation shows a recurring multi-intent coverage
problem.

## 9. A worked example

Assume the service ingests this Jira issue:

```text
document_id: jira:AUTH-101
title: Authentication session timeout
content: User sessions expire after seventeen minutes of inactivity.
```

The indexing path performs the following work:

1. Create a canonical Jira document.
2. Produce a chunk such as `jira:AUTH-101:chunk:0`.
3. Write its searchable fields to FTS5, which builds the lexical index.
4. Generate its embedding and store the vector plus metadata in ChromaDB.

Now consider two questions.

For `Summarize AUTH-101`:

- deterministic Jira-key lookup is strong;
- BM25 benefits from the rare identifier and weighted issue-key field;
- vector retrieval may add related semantic context.

For `How long before an inactive user must authenticate again?`:

- vector retrieval can connect the paraphrase to session expiration;
- BM25 may contribute matches such as `inactive` or `inactivity`, depending on
  tokenization;
- RRF combines both ranked result lists;
- evidence selection passes the best chunks to the answer stage.

The expected grounded answer should state seventeen minutes and cite only the
server-known evidence. If the model or validator cannot safely produce that
grounded output, retrieval can still be successful even though `/query`
refuses or reports validation failure.

## 10. Common misconceptions

### "FTS5 and BM25 are the same thing."

No. FTS5 is the SQLite full-text search engine; BM25 is the ranking algorithm
used for lexical matches. FTS5 happens to include a BM25 implementation.

### "BM25 is calculated during ingestion."

No. Ingestion prepares the inverted index and statistics. BM25 relevance is
calculated for a particular query at query time.

### "Chunking and tokenization are the same."

No. Chunking selects passages; tokenization converts a passage into terms or
model input units.

### "The query is split into chunks just like a document."

Normally no. A focused query produces one query embedding. Long or multi-intent
questions may require an explicit decomposition strategy.

### "The nearest vector result guarantees the correct answer."

No. Vector similarity is a retrieval signal, not proof. Hybrid retrieval,
evidence selection, grounded generation, citation validation, and evaluation
address different parts of the correctness problem.

### "RRF understands the query."

No. RRF combines ranked lists. It does not perform semantic reasoning or
validate factual support.

## 11. A compact mental model

```text
INDEXING
Source documents
    -> canonical representation
    -> shared chunking
    -> FTS5 terms and inverted index
    -> embedding vectors in ChromaDB

QUERYING
Focused question
    -> one query embedding
    -> lexical + vector + exact/metadata retrieval
    -> RRF and optional reranking
    -> selected evidence
    -> retrieval response or grounded answer
```

Remember these short definitions:

- **Chunking** decides which passage can be retrieved.
- **Tokenization** decides how text becomes searchable terms or model input.
- **FTS5** provides the embedded lexical index.
- **BM25** ranks lexical matches.
- **Embeddings** represent contextual meaning in a vector space.
- **ChromaDB** stores and searches document chunk vectors.
- **Dual indexing** gives the same chunks lexical and vector representations.
- **RRF** combines incomparable ranked lists using rank positions.
- **Query decomposition** turns a multi-intent question into focused subqueries.
- **Grounding** requires generated claims to be supported by retrieved evidence.

## 12. Where to go next

For implementation details, configuration, failure behavior, and PoC acceptance
criteria, read the
[Hybrid Retrieval and Grounded Answers PoC Design](../superpowers/specs/2026-08-04-hybrid-grounded-rag-poc-design.md).

When evaluating a retrieval change, ask these questions separately:

1. Did the correct evidence enter the candidate set?
2. Did it rank early enough to be selected?
3. Did the selected evidence cover the whole information need?
4. Did the answer use only that evidence?
5. Did the citations map back to trusted chunks and sources?

Separating these questions makes it easier to diagnose whether a problem comes
from chunking, lexical retrieval, vector retrieval, fusion, reranking, evidence
selection, or generation.
