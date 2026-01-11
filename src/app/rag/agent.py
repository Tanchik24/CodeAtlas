from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent
from langchain_mistralai import ChatMistralAI

try:
    from langgraph.checkpoint.memory import InMemorySaver as InMemoryCheckpointer
except Exception:
    from langgraph.checkpoint.memory import MemorySaver as InMemoryCheckpointer  
from src.app.config import get_config

cfg_llm = get_config().llm

SYSTEM_PROMPT_EN = """
You have a Neo4j code graph.

Node labels:
- Project {name, path}
- Package {full_name, name, path}
- Folder {path, name}
- Module {full_name, name, path}
- Class {full_name, name, path, decorators?, docstring?, start_line, end_line}
- Function {full_name, name, path, decorators?, docstring?, start_line, end_line}
- Method {full_name, name, path, decorators?, docstring?, start_line, end_line}

Relationship types (exact spelling):
- (Project)-[:CONTAINS_PACKAGE]->(Package)
- (Project|Package|Folder)-[:CONTAINS_FOLDER]->(Folder)
- (Project|Package|Folder)-[:CONTAINS_MODUL]->(Module)
- (Module)-[:DEFINES_CLASS]->(Class)
- (Module)-[:DEFINES_FUNCTION]->(Function)
- (Class)-[:DEFINES_METHOD]->(Method)
- (Module)-[:IMPORTS]->(Module)

Paths are repo-relative. start_line/end_line are 1-based inclusive.

You are a repo QA agent. Be grounded: do not guess.
Answer in the same language as the user.
Be VERY concise: max 3–6 bullet points, no fluff.

Tools:
1) semantic_search(question_text, top_k) -> hits with payload (often includes path + start_line/end_line)
2) graph_query_readonly(cypher_query, params) -> rows (read-only)
3) read_file_span(relative_path, start_line, end_line) -> exact snippet by lines

Policy / workflow:
- If the user needs exact code or exact behavior: semantic_search -> read_file_span.
- Use graph_query_readonly only if you must:
  - payload misses path/lines, OR
  - you need relations (imports/defines/contains).
- In the final answer, cite evidence as: `path: Lstart-Lend`.
- If evidence is missing, say exactly what is missing (one short sentence). Do NOT ask questions.
""".strip()


@dataclass(frozen=True)
class RepositoryAgentConfiguration:
    repository_root_directory: Path
    mistral_model_name: str = cfg_llm.mistral_model
    temperature: float = cfg_llm.temperature

    recursion_limit: int = 18
    semantic_top_k_default: int = 8
    max_cypher_rows: int = 50

    max_tool_output_characters: int = 12000
    max_file_snippet_characters: int = 9000


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if hasattr(value, "_properties"):
        try:
            return _to_jsonable(dict(getattr(value, "_properties")))
        except Exception:
            return str(value)
    return str(value)


def _dump_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _dump_json_limited(obj: Any, limit: int) -> str:
    text = _dump_json(obj)
    if len(text) <= int(limit):
        return text
    return _dump_json({"ok": False, "error": "tool output too large", "truncated": True})


def _is_read_only_cypher(cypher_query: str) -> bool:
    normalized_upper = " ".join((cypher_query or "").strip().split()).upper()
    forbidden_tokens = [
        "CREATE ", "MERGE ", "SET ", "DELETE ", "DETACH ", "DROP ", "REMOVE ",
        "CALL ", "LOAD CSV", "ADMIN",
    ]
    return not any(token in normalized_upper for token in forbidden_tokens)


def _enforce_repository_root(repository_root_directory: Path, relative_or_absolute_path: str) -> Path:
    repo_root = repository_root_directory.resolve()
    candidate = Path(relative_or_absolute_path)
    full_path = candidate if candidate.is_absolute() else (repo_root / candidate)
    full_path = full_path.resolve()
    if repo_root != full_path and repo_root not in full_path.parents:
        raise ValueError(f"Path escapes repository_root_directory: {relative_or_absolute_path}")
    return full_path


class GraphQueryReadOnlyToolBackend:
    def __init__(self, neo4j_ingestor: Any, max_cypher_rows: int, max_tool_output_characters: int) -> None:
        self._neo4j_ingestor = neo4j_ingestor
        self._max_cypher_rows = int(max_cypher_rows)
        self._max_tool_output_characters = int(max_tool_output_characters)

    def graph_query_readonly(self, cypher_query: str, params: Optional[Dict[str, Any]] = None) -> str:
        query_text = (cypher_query or "").strip()
        if not query_text:
            return _dump_json_limited({"ok": False, "error": "Empty cypher_query"}, self._max_tool_output_characters)

        if not _is_read_only_cypher(query_text):
            return _dump_json_limited(
                {"ok": False, "error": "Forbidden Cypher: only read-only MATCH/RETURN queries are allowed."},
                self._max_tool_output_characters,
            )

        if " LIMIT " not in (" " + query_text.upper() + " "):
            query_text = f"{query_text}\nLIMIT {self._max_cypher_rows}"

        try:
            records = self._neo4j_ingestor.fetch_all(query_text, params or {})
        except Exception as exc:
            return _dump_json_limited({"ok": False, "error": f"Neo4j error: {exc}"}, self._max_tool_output_characters)

        rows: List[dict] = []
        for record in (records or [])[: self._max_cypher_rows]:
            record_dict = dict(record) if not isinstance(record, dict) else record
            rows.append(_to_jsonable(record_dict))

        return _dump_json_limited({"ok": True, "rows": rows}, self._max_tool_output_characters)


class SemanticSearchToolBackend:
    def __init__(self, embedder: Any, semantic_top_k_default: int, max_tool_output_characters: int) -> None:
        self._embedder = embedder
        self._semantic_top_k_default = int(semantic_top_k_default)
        self._max_tool_output_characters = int(max_tool_output_characters)

    def semantic_search(self, question_text: str, top_k: int = 0) -> str:
        query_text = (question_text or "").strip()
        if not query_text:
            return _dump_json_limited({"ok": False, "error": "empty question_text", "hits": []}, self._max_tool_output_characters)

        effective_top_k = int(top_k) if int(top_k) > 0 else self._semantic_top_k_default

        try:
            hits = self._embedder.search_question(question_text=query_text, top_k=effective_top_k)
        except TypeError:
            hits = self._embedder.search_question(query_text, effective_top_k)
        except Exception as exc:
            return _dump_json_limited(
                {"ok": False, "error": f"semantic_search error: {exc}", "hits": []},
                self._max_tool_output_characters,
            )

        output_hits: List[dict] = []
        for node_id, score, payload in hits or []:
            output_hits.append(
                {"node_id": int(node_id), "score": float(score), "payload": _to_jsonable(payload or {})}
            )

        return _dump_json_limited({"ok": True, "hits": output_hits[:effective_top_k]}, self._max_tool_output_characters)


class FileSpanReaderToolBackend:

    def __init__(self, repository_root_directory: Path, max_tool_output_characters: int, max_file_snippet_characters: int) -> None:
        self._repository_root_directory = repository_root_directory
        self._max_tool_output_characters = int(max_tool_output_characters)
        self._max_file_snippet_characters = int(max_file_snippet_characters)

    def read_file_span(self, relative_path: str, start_line: int, end_line: int) -> str:
        rel = str(relative_path or "").strip()
        if not rel:
            return _dump_json_limited({"ok": False, "error": "empty relative_path"}, self._max_tool_output_characters)

        try:
            s = int(start_line)
            e = int(end_line)
        except Exception:
            return _dump_json_limited({"ok": False, "error": "start_line/end_line must be integers"}, self._max_tool_output_characters)

        if s <= 0 or e <= 0 or e < s:
            return _dump_json_limited({"ok": False, "error": "invalid line range"}, self._max_tool_output_characters)

        try:
            full_path = _enforce_repository_root(self._repository_root_directory, rel)
            if not full_path.is_file():
                return _dump_json_limited({"ok": False, "error": f"file not found: {rel}"}, self._max_tool_output_characters)
        except Exception as exc:
            return _dump_json_limited({"ok": False, "error": f"path error: {exc}"}, self._max_tool_output_characters)

        lines: List[str] = []
        try:
            with full_path.open("r", encoding="utf-8", errors="replace") as f:
                for ln, text in enumerate(f, start=1):
                    if ln < s:
                        continue
                    if ln > e:
                        break
                    lines.append(text)
        except Exception as exc:
            return _dump_json_limited({"ok": False, "error": f"file read error: {exc}"}, self._max_tool_output_characters)

        snippet = "".join(lines)
        truncated = False
        if len(snippet) > self._max_file_snippet_characters:
            snippet = snippet[: self._max_file_snippet_characters] + "\n# ... truncated ..."
            truncated = True

        payload = {
            "ok": True,
            "relative_path": rel,
            "start_line": s,
            "end_line": e,
            "truncated": truncated,
            "text": snippet,
        }
        return _dump_json_limited(payload, self._max_tool_output_characters)


class CodeRepositoryLangGraphAgent:
    def __init__(self, configuration: RepositoryAgentConfiguration, neo4j_ingestor: Any, embedder: Any) -> None:
        if create_react_agent is None:
            raise RuntimeError(
                "langgraph.prebuilt.create_react_agent is not available in your installed langgraph version."
            )

        self._configuration = configuration

        graph_backend = GraphQueryReadOnlyToolBackend(
            neo4j_ingestor=neo4j_ingestor,
            max_cypher_rows=configuration.max_cypher_rows,
            max_tool_output_characters=configuration.max_tool_output_characters,
        )
        semantic_backend = SemanticSearchToolBackend(
            embedder=embedder,
            semantic_top_k_default=configuration.semantic_top_k_default,
            max_tool_output_characters=configuration.max_tool_output_characters,
        )
        file_backend = FileSpanReaderToolBackend(
            repository_root_directory=configuration.repository_root_directory,
            max_tool_output_characters=configuration.max_tool_output_characters,
            max_file_snippet_characters=configuration.max_file_snippet_characters,
        )

        tools: List[StructuredTool] = [
            StructuredTool.from_function(
                func=graph_backend.graph_query_readonly,
                name="graph_query_readonly",
                description="Read-only Cypher query (MATCH/RETURN). Returns JSON rows.",
            ),
            StructuredTool.from_function(
                func=semantic_backend.semantic_search,
                name="semantic_search",
                description="Semantic search over code entities. Returns node_id + payload.",
            ),
            StructuredTool.from_function(
                func=file_backend.read_file_span,
                name="read_file_span",
                description="Read repository file by line range (1-based, inclusive). Returns JSON with exact text.",
            ),
        ]

        

        chat_model = ChatMistralAI(
            model=str(configuration.mistral_model_name),
            temperature=float(configuration.temperature),
            api_key=cfg_llm.mistral_api_key
        )

        checkpointer = InMemoryCheckpointer()

        self._compiled_graph = create_react_agent(
            model=chat_model,
            tools=tools,
            prompt=SYSTEM_PROMPT_EN,
            checkpointer=checkpointer,
        )

    def ask(self, question_text: str, thread_id: str) -> str:
        result_state = self._compiled_graph.invoke(
            {"messages": [{"role": "user", "content": str(question_text or "")}]},
            {
                "configurable": {"thread_id": str(thread_id)},
                "recursion_limit": int(self._configuration.recursion_limit),
            },
        )

        messages = result_state.get("messages") or []
        for message in reversed(messages):
            if getattr(message, "type", "") in {"ai", "assistant"}:
                content = getattr(message, "content", "")
                if isinstance(content, str) and content.strip():
                    return content.strip()
        return ""


class CodeRepoToolAgent:
    def __init__(
        self,
        project_root: Path,
        neo4j_ingestor: Any,
        store: Any,
        embedder: Any,
        top_k: int = 6,
        mistral_model: str | None = None,
    ) -> None:
        self.project_root_directory = Path(project_root).resolve()
        self.neo4j_ingestor = neo4j_ingestor
        self.store = store
        self.embedder = embedder

        self._thread_index: int = 0
        self._thread_id: str = self._make_thread_id()

        model_name = mistral_model or cfg_llm.mistral_model

        configuration = RepositoryAgentConfiguration(
            repository_root_directory=self.project_root_directory,
            mistral_model_name=str(model_name),
            semantic_top_k_default=int(top_k),
        )

        self._agent = CodeRepositoryLangGraphAgent(
            configuration=configuration,
            neo4j_ingestor=self.neo4j_ingestor,
            embedder=self.embedder,
        )

    def _make_thread_id(self) -> str:
        return f"repo-thread-{self._thread_index}"

    def new_thread(self) -> None:
        self._thread_index += 1
        self._thread_id = self._make_thread_id()

    def ask(self, question_text: str) -> str:
        return self._agent.ask(question_text=question_text, thread_id=self._thread_id)