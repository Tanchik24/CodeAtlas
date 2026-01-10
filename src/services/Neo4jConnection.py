from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from neo4j import GraphDatabase, Driver

from app.config import get_config
from app.Logger import get_logger

logger = get_logger()


class Neo4jConnection:
    def __init__(self) -> None:
        cfg = get_config().gdb 

        uri = f"bolt://{cfg.neo4j_host}:{cfg.neo4j_port}"
        self._database: str = cfg.neo4j_database

        logger.info(f"Connecting to Neo4j at {uri} (db={self._database})...")

        self._driver: Driver = GraphDatabase.driver(
            uri,
            auth=(cfg.neo4j_user, cfg.neo4j_password),
        )

        logger.info("Successfully connected to Neo4j")

    def close(self) -> None:
        self._driver.close()
        logger.info("Disconnected from Neo4j")

    def run(self, query: str, params: dict[str, Any] | None = None):
        params = params or {}
        with self._driver.session(database=self._database) as session:
            result = session.run(query, params)
            return list(result)

    @staticmethod
    def current_timestamp() -> str:
        return datetime.now(UTC).isoformat()
