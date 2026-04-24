from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, List

STOPWORDS = {
    "a", "an", "the", "and", "or", "to", "of", "for", "in", "on", "at", "is", "are", "be", "do", "does",
    "i", "we", "you", "can", "what", "when", "how", "with", "from", "by", "my", "our", "your", "it",
    "this", "that", "as", "if", "will", "may", "should", "about", "before", "after", "than"
}

try:  # Optional LangChain-powered RAG stack
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains.retrieval import create_retrieval_chain
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from langchain_chroma import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_RAG_AVAILABLE = True
except Exception:  # pragma: no cover - graceful fallback when optional deps are absent
    @dataclass
    class Document:  # type: ignore[override]
        page_content: str
        metadata: dict[str, Any]

    ChatPromptTemplate = None  # type: ignore
    create_stuff_documents_chain = None  # type: ignore
    create_retrieval_chain = None  # type: ignore
    ChatGoogleGenerativeAI = None  # type: ignore
    GoogleGenerativeAIEmbeddings = None  # type: ignore
    Chroma = None  # type: ignore
    RecursiveCharacterTextSplitter = None  # type: ignore
    LANGCHAIN_RAG_AVAILABLE = False


RAG_PROMPT = """You are a customer support assistant for a chemical supplier.
Answer the user's question only from the retrieved internal knowledge base.
If the retrieved context is insufficient, say so clearly and recommend escalation to a human rep.
Do not invent policies, guarantees, or regulatory claims.

Retrieved context:
{context}

Question:
{input}
"""


@dataclass
class DocChunk:
    source: str
    heading: str
    text: str
    score: float = 0.0


class FAQKnowledgeBase:
    """Hybrid FAQ knowledge base.

    Modes:
    - `langchain_rag`: uses markdown -> text splitter -> embeddings -> Chroma -> retrieval chain
    - `keyword_fallback`: lightweight local retrieval with no external dependencies or API keys

    The class automatically upgrades itself to real RAG when optional packages and a
    Gemini API key are available. Otherwise it keeps the existing deterministic behavior.
    """

    def __init__(
        self,
        docs_dir: str | Path,
        persist_dir: str | Path | None = None,
        prefer_rag: bool = True,
        model_name: str | None = None,
        embedding_model: str | None = None,
    ):
        self.docs_dir = Path(docs_dir)
        self.persist_dir = Path(persist_dir) if persist_dir else self.docs_dir.parent / "vectorstore" / "faq_kb"
        self.model_name = model_name or os.getenv("GOOGLE_GENAI_MODEL", "gemini-2.5-flash")
        self.embedding_model = embedding_model or os.getenv("GOOGLE_GENAI_EMBEDDING_MODEL", "models/text-embedding-004")
        self.chunks = self._load_docs()
        self.documents = self._build_documents()
        self.mode = "keyword_fallback"
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        self.last_error: str | None = None
        self.index_metadata_path = self.persist_dir / "index_meta.json"

        if prefer_rag:
            self._try_initialize_rag()

    def _load_docs(self) -> List[DocChunk]:
        chunks: List[DocChunk] = []
        for path in sorted(self.docs_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            sections = re.split(r"\n(?=##?\s)", text)
            for section in sections:
                section = section.strip()
                if not section:
                    continue
                lines = section.splitlines()
                heading = lines[0].lstrip("# ").strip() if lines else path.stem
                body = "\n".join(lines[1:]).strip() if len(lines) > 1 else section
                if body:
                    chunks.append(DocChunk(source=path.name, heading=heading, text=body))
        return chunks

    def _build_documents(self) -> list[Document]:
        docs: list[Document] = []
        if not self.chunks:
            return docs
        for idx, chunk in enumerate(self.chunks, start=1):
            docs.append(
                Document(
                    page_content=f"{chunk.heading}\n\n{chunk.text}",
                    metadata={
                        "source": chunk.source,
                        "heading": chunk.heading,
                        "chunk_id": idx,
                    },
                )
            )
        return docs

    def _content_signature(self) -> str:
        hasher = hashlib.sha256()
        for path in sorted(self.docs_dir.glob("*.md")):
            hasher.update(path.name.encode("utf-8"))
            hasher.update(path.read_bytes())
        return hasher.hexdigest()

    def _index_is_current(self) -> bool:
        if not self.index_metadata_path.exists():
            return False
        try:
            payload = json.loads(self.index_metadata_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        return payload.get("content_signature") == self._content_signature()

    def _write_index_metadata(self) -> None:
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "content_signature": self._content_signature(),
            "document_count": len(self.documents),
            "model_name": self.model_name,
            "embedding_model": self.embedding_model,
        }
        self.index_metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _rag_ready(self) -> bool:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        return bool(LANGCHAIN_RAG_AVAILABLE and api_key and self.documents)

    def _build_vectorstore(self, embeddings: Any) -> Any:
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        vectorstore = Chroma(
            collection_name="chemical_support_faq",
            embedding_function=embeddings,
            persist_directory=str(self.persist_dir),
        )

        needs_rebuild = not self._index_is_current()
        if needs_rebuild:
            # Avoid duplicate chunks when rebuilding because the docs changed.
            try:
                vectorstore.delete_collection()
            except Exception:
                pass
            vectorstore = Chroma(
                collection_name="chemical_support_faq",
                embedding_function=embeddings,
                persist_directory=str(self.persist_dir),
            )
            splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
            split_docs = splitter.split_documents(self.documents)
            vectorstore.add_documents(split_docs)
            self._write_index_metadata()
        return vectorstore

    def _try_initialize_rag(self) -> None:
        if not self._rag_ready():
            return
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model=self.embedding_model)
            self.vectorstore = self._build_vectorstore(embeddings)
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})

            llm = ChatGoogleGenerativeAI(model=self.model_name, temperature=0.1)
            prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
            combine_chain = create_stuff_documents_chain(llm, prompt)
            self.rag_chain = create_retrieval_chain(self.retriever, combine_chain)
            self.mode = "langchain_rag"
        except Exception as exc:  # pragma: no cover - runtime/network safety
            self.last_error = str(exc)
            self.mode = "keyword_fallback"
            self.vectorstore = None
            self.retriever = None
            self.rag_chain = None

    @staticmethod
    def _tokens(text: str) -> set[str]:
        raw = re.findall(r"[a-zA-Z0-9\-]+", text.lower())
        return {tok for tok in raw if tok not in STOPWORDS and len(tok) > 1}

    def _fallback_retrieve(self, question: str, top_k: int = 3) -> List[DocChunk]:
        q_tokens = self._tokens(question)
        scored: List[DocChunk] = []

        for chunk in self.chunks:
            c_tokens = self._tokens(chunk.heading + " " + chunk.text)
            overlap = len(q_tokens & c_tokens)
            score = overlap / max(len(q_tokens), 1)

            if question.lower().strip() in (chunk.heading + " " + chunk.text).lower():
                score += 0.2

            if score > 0:
                scored.append(DocChunk(chunk.source, chunk.heading, chunk.text, round(score, 3)))

        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:top_k]

    def retrieve(self, question: str, top_k: int = 3) -> List[DocChunk]:
        if self.mode == "langchain_rag" and self.retriever is not None:
            try:
                docs = self.retriever.invoke(question)
                chunks: list[DocChunk] = []
                for idx, doc in enumerate(docs[:top_k], start=1):
                    metadata = getattr(doc, "metadata", {}) or {}
                    chunks.append(
                        DocChunk(
                            source=str(metadata.get("source", "knowledge_base")),
                            heading=str(metadata.get("heading", f"Chunk {idx}")),
                            text=getattr(doc, "page_content", ""),
                            score=round(1.0 / idx, 3),
                        )
                    )
                return chunks
            except Exception as exc:  # pragma: no cover - runtime/network safety
                self.last_error = str(exc)
        return self._fallback_retrieve(question, top_k=top_k)

    def answer(self, question: str, top_k: int = 3) -> str:
        if self.mode == "langchain_rag" and self.rag_chain is not None:
            try:
                result = self.rag_chain.invoke({"input": question})
                answer = str(result.get("answer", "")).strip()
                context_docs = result.get("context", [])
                citations: list[str] = []
                for doc in context_docs[:top_k]:
                    metadata = getattr(doc, "metadata", {}) or {}
                    source = str(metadata.get("source", "knowledge_base"))
                    heading = str(metadata.get("heading", "section"))
                    label = f"{source} -> {heading}"
                    if label not in citations:
                        citations.append(label)
                if citations:
                    answer += "\n\nSources: " + "; ".join(citations)
                return answer
            except Exception as exc:  # pragma: no cover - runtime/network safety
                self.last_error = str(exc)

        hits = self._fallback_retrieve(question, top_k=top_k)
        if not hits:
            return (
                "I could not find a strong answer in the internal knowledge base. "
                "Please escalate this request to a human agent."
            )

        answer_lines = ["Here is the best information I found in the knowledge base:"]
        for idx, hit in enumerate(hits, start=1):
            snippet = hit.text.replace("\n", " ").strip()
            answer_lines.append(f"{idx}. [{hit.source} -> {hit.heading}] {snippet}")

        answer_lines.append(
            "If you need a formal commercial commitment, SDS, COA, or a special exception, route to a human rep."
        )
        return "\n".join(answer_lines)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "doc_count": len(self.documents),
            "persist_dir": str(self.persist_dir),
            "last_error": self.last_error,
        }


if __name__ == "__main__":
    kb = FAQKnowledgeBase(
        Path(__file__).resolve().parents[1] / "docs",
        persist_dir=Path(__file__).resolve().parents[1] / "vectorstore" / "faq_kb",
    )
    print(kb.diagnostics())
    print(kb.answer("Do you ship hazardous materials to Canada?"))
