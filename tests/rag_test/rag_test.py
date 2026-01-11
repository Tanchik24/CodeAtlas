import json
import subprocess
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Any

from src.app.config import get_config
from src.app.languages.LanguageRegistery import LanguageRegistry
from src.services.Neo4jIngestor import Neo4jIngestor
from src.services.CodeEmbeddingsStore import CodeEmbeddingsStore
from src.app.entities import Project
from src.app.code_indexer.CodebaseIndexer import CodebaseIndexer


@dataclass
class Question:
    text: str
    response: str

@dataclass
class Repository:
    url: str
    name: str
    questions: list[Question]
    collection: Optional[str] = None
    path: Optional[Path] = None

class RagTester:
    def __init__(self) -> None:
        self.config = get_config()
        self.test_data_dir = Path(self.config.gdb.rag_test_dir).resolve()
        self.repos_root = Path(self.config.gdb.rag_test_repos_dir).resolve()
        self.qdrant_path = Path(self.config.gdb.qdrant_path).resolve()

        self.language_registry = LanguageRegistry()
        self._languages_ready = False

        self.neo4j = Neo4jIngestor()
        self.top_k = 5

    def run(self):
        self._ensure_dirs_exist()
        repos = self._load_test_files()

        per_repo_by_model: dict[str, dict[str, dict[str, float]]] = {}
        overall_by_model: dict[str, dict[str, float]] = {}

        details_rows: list[dict[str, Any]] = []

        for repo in repos:
            repo.path = self._ensure_repo_cloned(repo.url, repo.name)
            repo.collection = self._collection_name(repo.url)

            self._reset_repo_index(repo)

            embedder = self._index_repo(repo)

    def _ensure_dirs_exist(self) -> None:
        self.repos_root.mkdir(parents=True, exist_ok=True)

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
            raw_questions = data.get("questions") or []

            questions: list[Question] = []
            for q in raw_questions:
                if not isinstance(q, dict):
                    continue

                text = str(q.get("query") or "").strip()
                if not text:
                    continue

                response = q.get("response") or []

                questions.append(
                    Question(
                        text=text,
                        response=response
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

    def _index_repo(self, repo: Repository):
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
                emb_store=emb_store)
            indexer.index_codebase()
        finally:
            try:
                emb_store.close()
            except Exception:
                pass

        return indexer.embedder if indexer is not None else None