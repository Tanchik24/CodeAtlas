from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DBConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="DB_",
        populate_by_name=True,
        extra="ignore",
    )

    neo4j_host: str = Field("localhost")
    neo4j_port: int = Field(7687)
    neo4j_user: str = Field("neo4j")
    neo4j_password: str = Field(...)
    neo4j_database: str = Field("neo4j")

    sqlite_path: str = Field("app.sqlite")

    qdrant_path: str = Field("./.qdrant_code_embeddings")

    retriever_test_repos_dir: str = Field("./tests/retriever_test/repos")
    retriever_test_dir: str = Field("./tests/retriever_test/test_data")
    rag_test_dir: str = Field("./tests/rag_test/test_data")
    rag_test_repos_dir: str = Field("./tests/rag_test/repos")