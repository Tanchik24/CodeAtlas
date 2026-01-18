from __future__ import annotations

import hashlib
import shutil
import re
import subprocess
from pathlib import Path
from typing import Dict, Tuple

from src.db import SQLiteStoreSA
from src.db import Repo, User

from src.services.Neo4jIngestor import Neo4jIngestor
from src.app.languages import LanguageRegistry
from src.app.entities.Project import Project
from src.app.code_indexer import CodebaseIndexer

from src.app.rag.agent import CodeRepoToolAgent
from src.services.CodeEmbeddingsStore import CodeEmbeddingsStore
from src.app.code_embedder.CodeEmbeddingsGenerator import CodeEmbeddingsGenerator, RepositoryEmbeddingConfig

from src.app.config import get_config


config = get_config().llm


def _repo_slug(github_url: str) -> str:
    m = re.match(r"^https://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", github_url.strip())
    if not m:
        raise ValueError("invalid github url")
    return f"{m.group(1)}__{m.group(2)}"


def _project_name(github_url: str) -> str:
    return _repo_slug(github_url)


def _collection_name(github_url: str) -> str:
    h = hashlib.sha1(github_url.encode("utf-8")).hexdigest()[:10]
    return f"code_embeddings_{h}"


class RepoQAService:
    def __init__(
        self,
        store: SQLiteStoreSA,
        repos_dir: Path,
        qdrant_path: Path,
        model_name: str = config.qwen_model_name
    ) -> None:
        self.store = store
        self.repos_dir = Path(repos_dir).resolve()
        self.qdrant_path = Path(qdrant_path).resolve()
        self.model_name = model_name

        self.repos_dir.mkdir(parents=True, exist_ok=True)
        self.qdrant_path.mkdir(parents=True, exist_ok=True)

        self.neo4j = Neo4jIngestor()
        self.language_registry = LanguageRegistry()
        self._languages_ready = False

        self._agent_cache: Dict[Tuple[str, int], CodeRepoToolAgent] = {}


    def ensure_repo(self, github_url: str) -> Repo:
        github_url = github_url.strip()
        slug = _repo_slug(github_url)

        repo_path = (self.repos_dir / slug).resolve()
        project_name = _project_name(github_url)
        collection = _collection_name(github_url)

        repo = self.store.get_repo_by_url(github_url)
        if repo is None:
            repo = self.store.upsert_repo(
                github_url=github_url,
                server_path=str(repo_path),
                project_name=project_name,
                collection_name=collection,
                is_indexed=False,
            )
        else:
            if repo.project_name != project_name or repo.collection_name != collection or repo.server_path != str(repo_path):
                repo = self.store.upsert_repo(
                    github_url=github_url,
                    server_path=str(repo_path),
                    project_name=project_name,
                    collection_name=collection,
                    is_indexed=bool(repo.is_indexed),
                )

        self._clone_if_needed(github_url, Path(repo.server_path))

        if not repo.is_indexed:
            self.reset_repo_index(repo)
            self.index_repo(repo)
            self.store.set_repo_indexed(repo.id, True)
            repo = self.store.get_repo_by_id(repo.id) or repo

        return repo

    def reset_repo_index(self, repo: Repo) -> None:
        q = """
        MATCH (p:Project {name:$name})
        OPTIONAL MATCH (p)-[*0..]->(n)
        WITH collect(DISTINCT p) + collect(DISTINCT n) AS nodes
        UNWIND nodes AS x
        WITH DISTINCT x
        DETACH DELETE x
        """
        self.neo4j.fetch_all(q, {"name": repo.project_name})

        store = CodeEmbeddingsStore(collection_name=repo.collection_name, path=self.qdrant_path)
        try:
            store.clear()
        finally:
            store.close()

        self._invalidate_repo_agents(int(repo.id))

    def index_repo(self, repo: Repo) -> None:
        if not self._languages_ready:
            self.language_registry.auto_register()
            self._languages_ready = True

        project_root = Path(repo.server_path).resolve()
        project = Project(name=repo.project_name, path=project_root)

        emb_store = CodeEmbeddingsStore(collection_name=repo.collection_name, path=self.qdrant_path)

        try:
            indexer = CodebaseIndexer(
                project=project,
                neo4j_ingestor=self.neo4j,
                language_registry=self.language_registry,
                emb_store=emb_store,
            )
            indexer.index_codebase()
        finally:
            try:
                emb_store.close()
            except Exception:
                pass


    def get_or_create_user(self, user_id: str, repo_id: int) -> User:
        return self.store.get_or_create_user(user_id=user_id, repo_id=repo_id)

    def ask(self, user_id: str, repo_id: int, message: str) -> str:
        repo = self.store.get_repo_by_id(int(repo_id))
        if repo is None:
            raise ValueError("repo not found")

        self._clone_if_needed(repo.github_url, Path(repo.server_path))

        if not repo.is_indexed:
            self.reset_repo_index(repo)
            self.index_repo(repo)
            self.store.set_repo_indexed(repo.id, True)
            repo = self.store.get_repo_by_id(int(repo_id)) or repo

        self.get_or_create_user(user_id, int(repo.id))
        self.store.append_message(user_id, "user", message)

        agent = self._get_agent(user_id=user_id, repo=repo)
        answer = agent.ask(message)

        self.store.append_message(user_id, "assistant", answer)
        return answer


    def _invalidate_repo_agents(self, repo_id: int) -> None:
        repo_id = int(repo_id)
        to_delete = [k for k in self._agent_cache.keys() if int(k[1]) == repo_id]
        for k in to_delete:
            agent = self._agent_cache.pop(k, None)
            if not agent:
                continue

            try:
                if getattr(agent, "store", None):
                    agent.store.close()
            except Exception:
                pass

            try:
                if getattr(agent, "embedder", None) and hasattr(agent.embedder, "close"):
                    agent.embedder.close()
            except Exception:
                pass

    def _get_agent(self, user_id: str, repo: Repo) -> CodeRepoToolAgent:
        key = (str(user_id), int(repo.id))
        cached = self._agent_cache.get(key)
        if cached is not None:
            return cached

        emb_store = CodeEmbeddingsStore(collection_name=repo.collection_name, path=self.qdrant_path)

        embedder = CodeEmbeddingsGenerator(
            repository_root_directory=Path(repo.server_path),
            embeddings_store=emb_store,
            neo4j_connection=self.neo4j._conn,
            project_name=repo.project_name,
            config=RepositoryEmbeddingConfig(
                sparse_model_name="",
                use_cuda=True,
                indexing_batch_size=4,
                max_source_code_lines=200,
                max_source_code_characters=4000,
                max_docstring_characters=1200,
                bucket_lines_step=50,
            ),
        )

        agent = CodeRepoToolAgent(
            project_root=Path(repo.server_path),
            neo4j_ingestor=self.neo4j,
            store=emb_store,
            embedder=embedder,
            qwen_model=self.model_name,
            top_k=5,
        )

        self._agent_cache[key] = agent
        return agent

    def _clone_if_needed(self, github_url: str, dest: Path) -> None:
        dest = Path(dest)

        if dest.exists() and (dest / ".git").exists():
            return

        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)

        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["git", "clone", "--depth", "1", github_url, str(dest)])