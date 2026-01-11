import json
import math
import time
import subprocess
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple
import pandas as pd


from src.app.config import get_config
from src.app.languages.LanguageRegistery import LanguageRegistry
from src.app.entities import Project
from src.services.CodeEmbeddingsStore import CodeEmbeddingsStore
from src.services.Neo4jIngestor import Neo4jIngestor
from src.app.code_indexer.CodebaseIndexer import CodebaseIndexer
from src.app.code_embedder.CodeEmbeddingsGenerator import RepositoryEmbeddingConfig

@dataclass
class Question:
    text: str
    gold_paths: list[str]
    gold_full_names: list[str]


@dataclass
class Repository:
    url: str
    name: str
    questions: list[Question]
    collection: Optional[str] = None
    path: Optional[Path] = None


class RetrieverTest:
    def __init__(self) -> None:
        self.config = get_config()
        self.test_data_dir = Path(self.config.gdb.retriever_test_dir).resolve()
        self.repos_root = Path(self.config.gdb.retriever_test_repos_dir).resolve()
        self.qdrant_path = Path(self.config.gdb.qdrant_path).resolve()

        self.language_registry = LanguageRegistry()
        self._languages_ready = False

        self.neo4j = Neo4jIngestor()
        self.top_k = 5

    def run(self) -> Tuple[dict[str, dict[str, dict[str, float]]], dict[str, dict[str, float]]]:
        self._ensure_dirs_exist()

        embedding_models = [
            "jinaai/jina-embeddings-v2-base-code",
            "BAAI/bge-small-en-v1.5",
            "sentence-transformers/all-MiniLM-L6-v2",
        ]

        repos = self._load_test_files()

        per_repo_by_model: dict[str, dict[str, dict[str, float]]] = {}
        overall_by_model: dict[str, dict[str, float]] = {}

        details_rows: list[dict[str, Any]] = []

        for dense_model_name in embedding_models:
            per_repo: dict[str, dict[str, float]] = {}

            overall_lists: dict[str, list[float]] = {
                "avg_recall@k_path": [],
                "avg_recall@k_full_name": [],
                "avg_mrr@k_path": [],
                "avg_mrr@k_full_name": [],
                "avg_ndcg@k_path": [],
                "avg_ndcg@k_full_name": [],
                "avg_search_latency_s": [],
                "index_latency_s": [],
                "n_questions": [],
            }

            for repo in repos:
                repo.path = self._ensure_repo_cloned(repo.url, repo.name)
                repo.collection = self._collection_name(repo.url)

                self._reset_repo_index(repo)
                t0_idx = time.perf_counter()
                embedder = self._index_repo(repo, dense_model_name=dense_model_name) 
                index_latency_s = time.perf_counter() - t0_idx

                recall_paths: list[float] = []
                recall_full_names: list[float] = []
                mrr_paths: list[float] = []
                mrr_full_names: list[float] = []
                ndcg_paths: list[float] = []
                ndcg_full_names: list[float] = []

                search_latencies_s: list[float] = []

                for question in repo.questions:
                    q_text = (question.text or "").strip()
                    if not q_text:
                        continue

                    t0 = time.perf_counter()
                    hits = embedder.search_question(question_text=q_text, top_k=self.top_k) or []
                    search_latencies_s.append(time.perf_counter() - t0)

                    paths: list[str] = []
                    full_names: list[str] = []

                    for _, _, payload in hits:
                        payload = self._to_jsonable(payload or {})
                        path = str(payload.get("path") or "").strip()
                        full_name = str(payload.get("full_name") or "").strip()
                        if path:
                            paths.append(path)
                        if full_name:
                            full_names.append(full_name)

                    details_rows.append({
                        "model": dense_model_name,
                        "repo": repo.name,
                        "question": q_text,
                        "gold_paths": json.dumps([str(x).strip() for x in (question.gold_paths or []) if str(x).strip()], ensure_ascii=False),
                        "retrieved_paths": json.dumps(paths, ensure_ascii=False),
                        "gold_full_names": json.dumps([str(x).strip() for x in (question.gold_full_names or []) if str(x).strip()], ensure_ascii=False),
                        "retrieved_full_names": json.dumps(full_names, ensure_ascii=False),
                    })

                    recall_paths.append(self._recall_at_k(question.gold_paths, paths))
                    recall_full_names.append(self._recall_at_k(question.gold_full_names, full_names))

                    mrr_paths.append(self._mrr_at_k(question.gold_paths, paths))
                    mrr_full_names.append(self._mrr_at_k(question.gold_full_names, full_names))

                    ndcg_paths.append(self._ndcg_at_k(question.gold_paths, paths, self.top_k))
                    ndcg_full_names.append(self._ndcg_at_k(question.gold_full_names, full_names, self.top_k))

                n_q = len(repo.questions)
                denom = max(1, n_q)

                avg_search_latency_s = float(sum(search_latencies_s) / len(search_latencies_s)) if search_latencies_s else 0.0

                metrics = {
                    "n_questions": float(n_q),

                    "avg_recall@k_path": float(sum(recall_paths) / denom) if recall_paths else 0.0,
                    "avg_recall@k_full_name": float(sum(recall_full_names) / denom) if recall_full_names else 0.0,

                    "avg_mrr@k_path": float(sum(mrr_paths) / denom) if mrr_paths else 0.0,
                    "avg_mrr@k_full_name": float(sum(mrr_full_names) / denom) if mrr_full_names else 0.0,

                    "avg_ndcg@k_path": float(sum(ndcg_paths) / denom) if ndcg_paths else 0.0,
                    "avg_ndcg@k_full_name": float(sum(ndcg_full_names) / denom) if ndcg_full_names else 0.0,

                    "avg_search_latency_s": avg_search_latency_s,
                    "index_latency_s": float(index_latency_s),
                }

                per_repo[repo.name] = metrics

                try:
                    self._reset_repo_index(repo)
                except Exception:
                    pass

                for k, v in metrics.items():
                    if k in overall_lists:
                        overall_lists[k].append(float(v))

            overall: dict[str, float] = {}
            for k, xs in overall_lists.items():
                overall[k] = float(sum(xs) / len(xs)) if xs else 0.0

            overall["n_repos"] = float(len(per_repo))

            per_repo_by_model[dense_model_name] = per_repo
            overall_by_model[dense_model_name] = overall

        try:
            import shutil
            shutil.rmtree(self.repos_root, ignore_errors=True)
        except Exception:
            pass

        self._save_metrics(per_repo_by_model, overall_by_model, details_rows)

        return per_repo_by_model, overall_by_model
    
    def _save_metrics(
        self,
        per_repo_by_model: dict[str, dict[str, dict[str, float]]],
        overall_by_model: dict[str, dict[str, float]],
        details_rows: list[dict[str, Any]],
    ) -> None:
        try:
            out_dir = self.test_data_dir.parent

            per_repo_rows: list[dict[str, Any]] = []
            for model_name, per_repo in (per_repo_by_model or {}).items():
                for repo_name, metrics in (per_repo or {}).items():
                    per_repo_rows.append({
                        "model": model_name,
                        "repo": repo_name,
                        **(metrics or {}),
                    })

            df_per_repo = pd.DataFrame(per_repo_rows)
            if not df_per_repo.empty:
                df_per_repo = df_per_repo.set_index(["model", "repo"]).sort_index()
            df_per_repo.to_csv(out_dir / "per_repo_metrics.csv", index=True)

            overall_rows: list[dict[str, Any]] = []
            for model_name, metrics in (overall_by_model or {}).items():
                overall_rows.append({
                    "model": model_name,
                    **(metrics or {}),
                })

            df_overall = pd.DataFrame(overall_rows)
            if not df_overall.empty:
                df_overall = df_overall.set_index(["model"]).sort_index()
            df_overall.to_csv(out_dir / "overall_metrics.csv", index=True)

            df_details = pd.DataFrame(details_rows)
            if not df_details.empty:
                df_details = df_details.set_index(["model", "repo", "question"]).sort_index()
            df_details.to_csv(out_dir / "per_question_details.csv", index=True)

        except Exception:
            pass

    @staticmethod
    def _mrr_at_k(golden: list[str], retrieved: list[str]) -> float:
        gold = {str(x).strip() for x in (golden or []) if str(x).strip()}
        if not gold:
            return 0.0
        for i, r in enumerate(retrieved or []):
            if str(r).strip() in gold:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def _ndcg_at_k(golden: list[str], retrieved: list[str], k: int) -> float:
        gold = {str(x).strip() for x in (golden or []) if str(x).strip()}
        if not gold:
            return 0.0

        retrieved = [str(x).strip() for x in (retrieved or []) if str(x).strip()]
        k = min(int(k), len(retrieved))
        if k <= 0:
            return 0.0

        dcg = 0.0
        for i in range(k):
            rel = 1.0 if retrieved[i] in gold else 0.0
            dcg += rel / math.log2(i + 2)

        ideal_hits = min(len(gold), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def _recall_at_k(golden: list[str], retrieved: list[str]) -> float:
        golden_set = {str(x).strip() for x in (golden or []) if str(x).strip()}
        if not golden_set:
            return 0.0
        retrieved_set = {str(x).strip() for x in (retrieved or []) if str(x).strip()}
        return len(golden_set & retrieved_set) / len(golden_set)

    def _index_repo(self, repo: Repository, dense_model_name: str):
        if not self._languages_ready:
            self.language_registry.auto_register()
            self._languages_ready = True

        project = Project(name=repo.name, path=repo.path)
        emb_store = CodeEmbeddingsStore(collection_name=repo.collection, path=self.qdrant_path)

        indexer: Optional[CodebaseIndexer] = None
        try:
            indexer = CodebaseIndexer(
                project=project,
                neo4j_ingestor=self.neo4j,
                language_registry=self.language_registry,
                emb_store=emb_store,
                emb_config=RepositoryEmbeddingConfig(
                    dense_model_name=dense_model_name,
                    use_cuda=True,
                    indexing_batch_size=32,
                    max_source_code_characters=5120,
                ),
            )
            indexer.index_codebase()
        finally:
            try:
                emb_store.close()
            except Exception:
                pass

        return indexer.embedder if indexer is not None else None

    def _ensure_dirs_exist(self) -> None:
        self.repos_root.mkdir(parents=True, exist_ok=True)

    def _ensure_repo_cloned(self, url: str, repo_name: str) -> Path:
        dest = (self.repos_root / repo_name).resolve()

        if dest.exists() and (dest / ".git").exists():
            return dest

        if dest.exists() and not (dest / ".git").exists():
            raise RuntimeError(f"Destination exists but is not a git repo: {dest}")

        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
            cwd=str(self.repos_root),
            check=True,
        )
        return dest

    def _load_test_files(self) -> list[Repository]:
        repos: list[Repository] = []
        json_files = sorted(self.test_data_dir.glob("*.json"))

        for fp in json_files:
            data = self._read_json(fp)
            if not isinstance(data, dict):
                continue

            url = str(data.get("url") or "").strip()
            if not url:
                continue

            name = self._derive_name_from_url(url)
            raw_questions = data.get("retrieval_questions") or []

            questions: list[Question] = []
            for q in raw_questions:
                if not isinstance(q, dict):
                    continue

                text = str(q.get("query") or "").strip()
                if not text:
                    continue

                gold_paths = q.get("gold_paths") or []
                gold_full_names = q.get("gold_full_names") or []

                questions.append(
                    Question(
                        text=text,
                        gold_paths=list(gold_paths),
                        gold_full_names=list(gold_full_names),
                    )
                )

            repos.append(Repository(url=url, name=name, questions=questions))

        return repos

    def _read_json(self, path: Path) -> Any:
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _derive_name_from_url(self, url: str) -> str:
        s = url.strip().rstrip("/").split("/")[-1].strip()
        if s.endswith(".git"):
            s = s[:-4]
        return s or "unknown_repo"

    def _collection_name(self, github_url: str) -> str:
        h = hashlib.sha1(github_url.strip().encode("utf-8")).hexdigest()[:10]
        return f"code_embeddings_{h}"

    def _reset_repo_index(self, repo: Repository) -> None:
        if not repo.collection:
            raise RuntimeError("repo.collection is None/empty")

        q = """
        MATCH (p:Project {name:$name})
        OPTIONAL MATCH (p)-[*0..]->(n)
        WITH collect(DISTINCT p) + collect(DISTINCT n) AS nodes
        UNWIND nodes AS x
        WITH DISTINCT x
        DETACH DELETE x
        """
        self.neo4j.fetch_all(q, {"name": repo.name})

        store = CodeEmbeddingsStore(collection_name=repo.collection, path=self.qdrant_path)
        try:
            store.clear()
        finally:
            store.close()

    def _to_jsonable(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self._to_jsonable(v) for k, v in value.items()}
        if hasattr(value, "_properties"):
            try:
                return self._to_jsonable(dict(getattr(value, "_properties")))
            except Exception:
                return str(value)
        return str(value)


if __name__ == "__main__":
    RetrieverTest().run()
