from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.Logger import get_logger
from services import Neo4jConnection
from services.CodeEmbeddingsStore import CodeEmbeddingsStore

from tqdm import tqdm

logger = get_logger()


@dataclass(frozen=True)
class RepositoryEmbeddingConfig:
    dense_model_name: str = "jinaai/jina-embeddings-v2-base-code"
    sparse_model_name: str = ""
    use_cuda: bool = True
    indexing_batch_size: int = 4
    max_source_code_lines: int = 200
    max_source_code_characters: int = 4000
    max_docstring_characters: int = 1200
    bucket_lines_step: int = 50
    max_files_cache: int = 64


class CodeEmbeddingsGenerator:

    def __init__(
        self,
        repository_root_directory: Path,
        embeddings_store: CodeEmbeddingsStore | Any,
        neo4j_connection: Optional[Neo4jConnection] = None,
        config: RepositoryEmbeddingConfig = RepositoryEmbeddingConfig(),
        project_name: Optional[str] = None,
    ) -> None:
        self.repository_root_directory = Path(repository_root_directory).resolve()
        self.embeddings_store = embeddings_store
        self.neo4j_connection = neo4j_connection or Neo4jConnection()
        self._owns_neo4j = neo4j_connection is None
        self.config = config

        self.project_name = (project_name or "").strip() or None

        self.qdrant_client: QdrantClient = (embeddings_store._ensure_client())
        self.collection_name: str = getattr(embeddings_store, "collection_name", None) or "code_embeddings"

        self._using_vector_name: Optional[str] = None
        if hasattr(embeddings_store, "ensure_default_vector_name"):
            try:
                self._using_vector_name = embeddings_store.ensure_default_vector_name()
            except Exception:
                self._using_vector_name = None

        if hasattr(self.qdrant_client, "set_model"):
            self.qdrant_client.set_model(self.config.dense_model_name, cuda=bool(self.config.use_cuda))

        if getattr(self.config, "sparse_model_name", ""):
            if hasattr(self.qdrant_client, "set_sparse_model"):
                self.qdrant_client.set_sparse_model(self.config.sparse_model_name, cuda=bool(self.config.use_cuda))

        self._ensure_collection_fastembed()

        if not self._using_vector_name:
            self._using_vector_name = self._detect_collection_vector_name()

        if self._using_vector_name:
            logger.info(f"Using Qdrant vector name: {self._using_vector_name}")

        self._file_lines_cache: "OrderedDict[str, List[str]]" = OrderedDict()

    def close(self) -> None:
        if self._owns_neo4j:
            try:
                self.neo4j_connection.close()
            except Exception:
                pass


    def generate_all(self) -> Dict[str, int]:
        total = self._count_entities()
        if total <= 0:
            logger.info("No entities (Class/Function/Method) found in Neo4j.")
            return {"entities": 0, "indexed": 0}

        iterator = self._iter_entities()
        bar = tqdm(total=total, desc="index embeddings", unit="ent") if tqdm else None

        batch_documents: List[str] = []
        batch_payloads: List[Dict[str, Any]] = []
        batch_ids: List[int] = []

        indexed = 0
        current_bucket: Optional[int] = None
        current_path: Optional[str] = None

        for record in iterator:
            if bar:
                bar.update(1)

            node_id = record.get("node_id")
            if node_id is None:
                continue
            node_id_int = int(node_id)

            relative_path = str(record.get("path") or "").strip()
            if not relative_path:
                continue

            try:
                n_lines_int = int(record.get("n_lines") or 0)
            except Exception:
                n_lines_int = 0

            effective_lines = min(max(n_lines_int, 0), int(self.config.max_source_code_lines))
            bucket_step = max(1, int(self.config.bucket_lines_step))
            bucket = int(effective_lines // bucket_step)

            if batch_documents:
                if (current_path is not None and relative_path != current_path) or (
                    current_bucket is not None and bucket != current_bucket
                ):
                    indexed += self._flush(batch_documents, batch_payloads, batch_ids)
                    batch_documents.clear()
                    batch_payloads.clear()
                    batch_ids.clear()

            current_path = relative_path
            current_bucket = bucket

            code_text = self._read_code_snippet(
                relative_path=relative_path,
                start_line=record.get("start_line"),
                end_line=record.get("end_line"),
            )
            if not code_text:
                continue

            document_text = self._build_document(record, code_text)
            if not document_text:
                continue

            payload = self._build_payload(record)
            payload["node_id"] = node_id_int
            payload["n_lines"] = n_lines_int
            if self.project_name:
                payload["project_name"] = self.project_name

            batch_documents.append(document_text)
            batch_payloads.append(payload)
            batch_ids.append(node_id_int)

            if len(batch_documents) >= int(self.config.indexing_batch_size):
                indexed += self._flush(batch_documents, batch_payloads, batch_ids)
                batch_documents.clear()
                batch_payloads.clear()
                batch_ids.clear()

        if batch_documents:
            indexed += self._flush(batch_documents, batch_payloads, batch_ids)

        if bar:
            bar.close()

        logger.info(f"Indexing done: entities={total}, indexed={indexed}")
        return {"entities": int(total), "indexed": int(indexed)}

    def search_question(self, question_text: str, top_k: int = 8) -> List[Tuple[int, float, Dict[str, Any]]]:
        query_text = str(question_text or "").strip()
        if not query_text:
            return []

        if not self._using_vector_name:
            self._using_vector_name = self._detect_collection_vector_name()

        kwargs: Dict[str, Any] = dict(
            collection_name=self.collection_name,
            query=self._make_document(query_text),
            limit=int(top_k),
            with_payload=True,
        )
        if self._using_vector_name:
            kwargs["using"] = self._using_vector_name

        res = self.qdrant_client.query_points(**kwargs)
        points = getattr(res, "points", None) or []

        hits: List[Tuple[int, float, Dict[str, Any]]] = []
        for p in points:
            score = float(getattr(p, "score", 0.0))
            payload = dict(getattr(p, "payload", None) or {})
            pid = getattr(p, "id", None)

            node_id = payload.get("node_id")
            if node_id is None and pid is not None:
                node_id = pid
            if node_id is None:
                continue

            hits.append((int(node_id), score, payload))
        return hits


    def _ensure_collection_fastembed(self) -> None:
        try:
            if hasattr(self.qdrant_client, "collection_exists"):
                if self.qdrant_client.collection_exists(self.collection_name):
                    return
            else:
                self.qdrant_client.get_collection(self.collection_name)
                return
        except Exception:
            pass

        if not hasattr(self.qdrant_client, "get_fastembed_vector_params"):
            raise RuntimeError(
                "qdrant-client is missing fastembed integration: get_fastembed_vector_params() not available"
                "Install fastembed or use a qdrant-client build that includes it."
            )

        dense_params = self.qdrant_client.get_fastembed_vector_params()
        vectors_config: Any = {self._using_vector_name: dense_params} if self._using_vector_name else dense_params

        kwargs: Dict[str, Any] = dict(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
        )

        if getattr(self.config, "sparse_model_name", ""):
            if hasattr(self.qdrant_client, "get_fastembed_sparse_vector_params"):
                try:
                    kwargs["sparse_vectors_config"] = self.qdrant_client.get_fastembed_sparse_vector_params()
                except Exception:
                    pass

        self.qdrant_client.create_collection(**kwargs)
        logger.info(f"Created Qdrant collection (fastembed): {self.collection_name}")

        if not self._using_vector_name:
            self._using_vector_name = self._detect_collection_vector_name()

    def _detect_collection_vector_name(self) -> Optional[str]:
        try:
            info = self.qdrant_client.get_collection(self.collection_name)
            params = getattr(getattr(info, "config", None), "params", None)
            vectors = getattr(params, "vectors", None)
            if isinstance(vectors, dict) and vectors:
                return str(next(iter(vectors.keys())))
        except Exception:
            pass
        return None

    def _make_document(self, text: str) -> Any:
        try:
            return qm.Document(text=text, model=self.config.dense_model_name)
        except Exception:
            return qm.Document(text=text)


    def _count_entities(self) -> int:
        if self.project_name:
            params = {"project_name": self.project_name}

            q_fn = """
            MATCH (p:Project {name:$project_name})
              -[:CONTAINS_PACKAGE|CONTAINS_FOLDER|CONTAINS_MODUL*0..]->(m:Module)
              -[:DEFINES_FUNCTION]->(n:Function)
            RETURN count(n) AS c
            """
            q_cls = """
            MATCH (p:Project {name:$project_name})
              -[:CONTAINS_PACKAGE|CONTAINS_FOLDER|CONTAINS_MODUL*0..]->(m:Module)
              -[:DEFINES_CLASS]->(n:Class)
            RETURN count(n) AS c
            """
            q_m = """
            MATCH (p:Project {name:$project_name})
              -[:CONTAINS_PACKAGE|CONTAINS_FOLDER|CONTAINS_MODUL*0..]->(m:Module)
              -[:DEFINES_CLASS]->(c:Class)
              -[:DEFINES_METHOD]->(n:Method)
            RETURN count(n) AS c
            """

            def _one(q: str) -> int:
                rows = self.neo4j_connection.run(q, params)
                row = rows[0] if rows else {}
                row = row if isinstance(row, dict) else dict(row)
                return int(row.get("c") or 0)

            return _one(q_fn) + _one(q_cls) + _one(q_m)

        q = """
        MATCH (n)
        WHERE n:Function OR n:Class OR n:Method
        RETURN count(n) AS c
        """
        rows = self.neo4j_connection.run(q, {})
        if not rows:
            return 0
        row = rows[0]
        data = row if isinstance(row, dict) else dict(row)
        return int(data.get("c") or 0)

    def _iter_entities(self) -> Iterable[Dict[str, Any]]:
        if self.project_name:
            q = """
            MATCH (p:Project {name:$project_name})
            CALL {
              WITH p
              MATCH (p)-[:CONTAINS_PACKAGE|CONTAINS_FOLDER|CONTAINS_MODUL*0..]->(m:Module)
                    -[:DEFINES_FUNCTION]->(n:Function)
              RETURN n AS n, m AS m, NULL AS c
              UNION ALL
              WITH p
              MATCH (p)-[:CONTAINS_PACKAGE|CONTAINS_FOLDER|CONTAINS_MODUL*0..]->(m:Module)
                    -[:DEFINES_CLASS]->(n:Class)
              RETURN n AS n, m AS m, NULL AS c
              UNION ALL
              WITH p
              MATCH (p)-[:CONTAINS_PACKAGE|CONTAINS_FOLDER|CONTAINS_MODUL*0..]->(m:Module)
                    -[:DEFINES_CLASS]->(c:Class)
                    -[:DEFINES_METHOD]->(n:Method)
              RETURN n AS n, m AS m, c AS c
            }
            WITH n, m, c,
                 coalesce(n.path, m.path) AS path,
                 (coalesce(n.end_line,0) - coalesce(n.start_line,0) + 1) AS n_lines
            WHERE path IS NOT NULL
              AND n.start_line IS NOT NULL
              AND n.end_line IS NOT NULL
              AND n.end_line >= n.start_line
            RETURN
              id(n) AS node_id,
              labels(n)[0] AS node_label,
              n.full_name AS full_name,
              n.name AS name,
              n['decorators'] AS decorators,
              n['docstring'] AS docstring,
              n.start_line AS start_line,
              n.end_line AS end_line,
              path AS path,
              n_lines AS n_lines,
              m.full_name AS module_full_name,
              c.full_name AS class_full_name
            ORDER BY path, n_lines, n.start_line
            """
            for record in self.neo4j_connection.run(q, {"project_name": self.project_name}):
                yield record if isinstance(record, dict) else dict(record)
            return

        q = """
        MATCH (n)
        WHERE n:Function OR n:Class OR n:Method
        OPTIONAL MATCH (m:Module)-[:DEFINES_FUNCTION|DEFINES_CLASS]->(n)
        OPTIONAL MATCH (c:Class)-[:DEFINES_METHOD]->(n)
        OPTIONAL MATCH (m2:Module)-[:DEFINES_CLASS]->(c)
        WITH n, m, c, m2,
             coalesce(n.path, m.path, m2.path) AS path,
             (coalesce(n.end_line,0) - coalesce(n.start_line,0) + 1) AS n_lines
        WHERE path IS NOT NULL
          AND n.start_line IS NOT NULL
          AND n.end_line IS NOT NULL
          AND n.end_line >= n.start_line
        RETURN
          id(n) AS node_id,
          labels(n)[0] AS node_label,
          n.full_name AS full_name,
          n.name AS name,
          n['decorators'] AS decorators,
          n['docstring'] AS docstring,
          n.start_line AS start_line,
          n.end_line AS end_line,
          path AS path,
          n_lines AS n_lines,
          coalesce(m.full_name, m2.full_name) AS module_full_name,
          c.full_name AS class_full_name
        ORDER BY path, n_lines, n.start_line
        """
        for record in self.neo4j_connection.run(q, {}):
            yield record if isinstance(record, dict) else dict(record)


    def _flush(self, documents: List[str], payloads: List[Dict[str, Any]], ids: List[int]) -> int:
        if not documents:
            return 0

        if not self._using_vector_name:
            self._using_vector_name = self._detect_collection_vector_name()

        vn = self._using_vector_name

        points: List[qm.PointStruct] = []
        for point_id, document, payload in zip(ids, documents, payloads):
            doc_vec = self._make_document(document)
            vector: Any = {vn: doc_vec} if vn else doc_vec
            points.append(qm.PointStruct(id=int(point_id), vector=vector, payload=dict(payload)))

        self.qdrant_client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def _safe_repo_join(self, relative_path: str) -> Optional[Path]:
        rel = (relative_path or "").strip()
        if not rel:
            return None
        full = (self.repository_root_directory / Path(rel)).resolve()
        if full != self.repository_root_directory and self.repository_root_directory not in full.parents:
            return None
        return full

    def _get_file_lines_cached(self, relative_path: str) -> List[str]:
        if relative_path in self._file_lines_cache:
            self._file_lines_cache.move_to_end(relative_path)
            return self._file_lines_cache[relative_path]

        full_path = self._safe_repo_join(relative_path)
        if full_path is None or not full_path.is_file():
            return []

        try:
            with full_path.open("r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except Exception:
            return []

        self._file_lines_cache[relative_path] = lines
        self._file_lines_cache.move_to_end(relative_path)

        while len(self._file_lines_cache) > int(self.config.max_files_cache):
            self._file_lines_cache.popitem(last=False)

        return lines

    def _read_code_snippet(self, relative_path: str, start_line: Any, end_line: Any) -> str:
        try:
            start = int(start_line or 0)
            end = int(end_line or 0)
        except Exception:
            return ""

        if start <= 0 or end <= 0 or end < start:
            return ""

        lines = self._get_file_lines_cached(relative_path)
        if not lines:
            return ""

        sliced = lines[max(0, start - 1): min(len(lines), end)]

        max_lines = int(self.config.max_source_code_lines)
        truncated = False
        if max_lines > 0 and len(sliced) > max_lines:
            sliced = sliced[:max_lines]
            truncated = True

        snippet = "".join(sliced).strip()


        max_chars = int(self.config.max_source_code_characters)
        if max_chars > 0 and len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip()
            truncated = True

        if truncated:
            snippet += "\n# ... truncated ..."

        return snippet


    def _build_payload(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "node_label": record.get("node_label"),
            "full_name": record.get("full_name"),
            "name": record.get("name"),
            "module_full_name": record.get("module_full_name"),
            "class_full_name": record.get("class_full_name"),
            "path": record.get("path"),
            "start_line": record.get("start_line"),
            "end_line": record.get("end_line"),
        }

    def _build_document(self, record: Dict[str, Any], code_text: str) -> str:
        node_label = str(record.get("node_label") or "").strip()
        full_name = str(record.get("full_name") or "").strip()
        if not node_label or not full_name:
            return ""

        parts: List[str] = [f"{node_label}: {full_name}"]

        path = str(record.get("path") or "").strip()
        if path:
            parts.append(f"Path: {path}")

        module_full_name = str(record.get("module_full_name") or "").strip()
        if module_full_name:
            parts.append(f"Module: {module_full_name}")

        class_full_name = str(record.get("class_full_name") or "").strip()
        if class_full_name:
            parts.append(f"Class: {class_full_name}")

        try:
            sl = record.get("start_line")
            el = record.get("end_line")
            if sl is not None and el is not None:
                parts.append(f"Lines: {int(sl)}-{int(el)}")
        except Exception:
            pass

        decorators = record.get("decorators")
        if isinstance(decorators, list) and decorators:
            parts.append("Decorators: " + ", ".join(str(x) for x in decorators if x is not None))

        docstring = str(record.get("docstring") or "").strip()
        if docstring:
            max_len = int(self.config.max_docstring_characters)
            if max_len > 0 and len(docstring) > max_len:
                docstring = docstring[:max_len] + " ..."
            parts.append("Docstring:\n" + docstring)

        if code_text:
            parts.append("Source:\n" + code_text)

        return "\n".join(parts).strip()
