from __future__ import annotations

from typing import Optional

from src.app.entities.Project import Project
from src.services import Neo4jIngestor

from src.app.enums import NodeLabel
from src.app.languages import LanguageRegistry
from src.app.graph_builder.GraphBuilder import GraphBuilder

from src.services.CodeEmbeddingsStore import CodeEmbeddingsStore
from src.app.code_embedder.CodeEmbeddingsGenerator import CodeEmbeddingsGenerator, RepositoryEmbeddingConfig

from src.app.Logger import get_logger

logger = get_logger()


class CodebaseIndexer:
    def __init__(
        self,
        project: Project,
        neo4j_ingestor: Neo4jIngestor,
        language_registry: LanguageRegistry,
        emb_store: CodeEmbeddingsStore,
    ) -> None:
        self.project: Project = project
        self.neo4j_ingestor: Neo4jIngestor = neo4j_ingestor
        self.language_registry: LanguageRegistry = language_registry
        self.emb_store: CodeEmbeddingsStore = emb_store

        self.embedder: Optional[CodeEmbeddingsGenerator] = None
        self.graph_builder: GraphBuilder | None = None

        self.ignore_folders_and_files = {
            ".git",
            "venv",
            ".venv",
            "__pycache__",
            ".idea",
            ".vscode",
            ".mypy_cache",
            ".qdrant_code_embeddings",
        }

    def index_codebase(self) -> None:
        self.graph_builder = GraphBuilder(self.project, self.language_registry, self.neo4j_ingestor)
        self.upsert_project()

        files = 0
        code_files = 0

        for path in self.project.path.rglob("*"):
            if any(part in self.ignore_folders_and_files for part in path.parts):
                continue

            if path.is_dir():
                self.graph_builder.process_folders(path)
                continue

            if path.is_file():
                files += 1
                if self.language_registry.get_by_extension(path.suffix):
                    code_files += 1
                    self.graph_builder.process_files(path)

        logger.info(f"Scanned files={files}, code_files={code_files}")

        self.neo4j_ingestor.flush_all()

        try:
            try:
                self.emb_store.clear()
            except Exception:
                pass

            self.embedder = CodeEmbeddingsGenerator(
                repository_root_directory=self.project.path, 
                embeddings_store=self.emb_store,
                neo4j_connection=self.neo4j_ingestor._conn,
                project_name=self.project.name,
                config=RepositoryEmbeddingConfig(
                    use_cuda=True,
                    indexing_batch_size=32,
                    max_source_code_characters=5120,
                ),
            )

            stats = self.embedder.generate_all()
            logger.info(f"Embeddings indexed: {stats}")

        except Exception as e:
            logger.warning(f"Embeddings generation skipped/failed: {e}")
            self.embedder = None

    def upsert_project(self) -> None:
        if self.neo4j_ingestor.node_exists_by_unique(NodeLabel.PROJECT.value, self.project.name):
            return

        self.neo4j_ingestor.add_node_to_buffer(
            NodeLabel.PROJECT.value,
            name=self.project.name,
            path=str(self.project.path),
        )
