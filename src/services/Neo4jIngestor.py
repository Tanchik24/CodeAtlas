from __future__ import annotations

from typing import Any

from app.Logger import get_logger
from services import Neo4jConnection

logger = get_logger()


class Neo4jIngestor:
    def __init__(self, batch_size: int = 50) -> None:
        self._conn = Neo4jConnection()
        self.batch_size = batch_size

        self.node_buffer: list[tuple[str, dict[str, Any]]] = []
        self.relationship_buffer: list[tuple[tuple, str, tuple, dict | None]] = []

        self.unique_constraints = {
            "Project": "name",
            "Package": "full_name",
            "Folder": "path",
            "Module": "full_name",
            "Class": "full_name",
            "Function": "full_name",
            "Method": "full_name",
            "File": "path",
            "ExtPackage": "name",
        }

    def close(self) -> None:
        self._conn.close()

    def flush_all(self) -> None:
        logger.debug(
            f"Flushing all buffers to Neo4j "
            f"(nodes={len(self.node_buffer)}, relationships={len(self.relationship_buffer)})"
        )
        self._flush_nodes()
        self._flush_relationships()


    def find_nodes(self, label: str, **props: Any) -> list[Any]:
        params: dict[str, Any] = dict(props)

        if props:
            props_str = ", ".join(f"{key}: ${key}" for key in props.keys())
            query = f"""
            MATCH (n:`{label}` {{{props_str}}})
            RETURN n
            """
        else:
            query = f"""
            MATCH (n:`{label}`)
            RETURN n
            """

        logger.debug(f"Finding nodes: label={label}, props={props}")

        records = self._conn.run(query, params)
        return records
    

    def find_nodes_by_unique(self, label: str, value: Any) -> list[Any]:
        prop = self.unique_constraints.get(label)
        if not prop:
            raise ValueError(f"No unique constraint configured for label {label}")

        return self.find_nodes(label, **{prop: value})
    

    def node_exists(self, label: str, **props: Any) -> bool:
        return bool(self.find_nodes(label, **props))


    def node_exists_by_unique(self, label: str, value: Any) -> bool:
        return bool(self.find_nodes_by_unique(label, value))
    

    def add_node_to_buffer(self, label: str, **props: Any) -> None:
        self.node_buffer.append((label, dict(props)))

        if len(self.node_buffer) >= self.batch_size:
            self._flush_nodes()
    

    def _flush_nodes(self) -> None:
        if not self.node_buffer:
            return

        logger.debug(f"Flushing {len(self.node_buffer)} nodes to Neo4j")

        for label, props in self.node_buffer:
            unique_prop = self.unique_constraints.get(label)

            if unique_prop:
                query = f"""
                MERGE (n:`{label}` {{{unique_prop}: $unique_value}})
                SET n += $props
                """
                params = {
                    "unique_value": props[unique_prop],
                    "props": props,
                }
            else:
                query = f"""
                CREATE (n:`{label}`)
                SET n += $props
                """
                params = {"props": props}

            self._conn.run(query, params)

        self.node_buffer.clear()


    def add_relationship_to_buffer(
        self,
        start_label: str,
        start_props: dict[str, Any],
        rel_type: str,
        end_label: str,
        end_props: dict[str, Any],
        rel_props: dict[str, Any] | None = None,
    ) -> None:

        self.relationship_buffer.append(
            ((start_label, dict(start_props)), rel_type, (end_label, dict(end_props)), rel_props)
        )

        if len(self.relationship_buffer) >= self.batch_size:
            self._flush_nodes()
            self._flush_relationships()


    def _flush_relationships(self) -> None:
        if not self.relationship_buffer:
            return

        logger.debug(f"Flushing {len(self.relationship_buffer)} relationships to Neo4j")

        for (start_label, start_props), rel_type, (end_label, end_props), rel_props in self.relationship_buffer:
            start_key = self.unique_constraints.get(start_label)
            end_key = self.unique_constraints.get(end_label)

            query = f"""
            MATCH (start:`{start_label}` {{{start_key}: $start_value}})
            MATCH (end:`{end_label}` {{{end_key}: $end_value}})
            MERGE (start)-[r:`{rel_type}`]->(end)
            SET r += $rel_props
            """

            params = {
                "start_value": start_props[start_key],
                "end_value": end_props[end_key],
                "rel_props": rel_props or {},
            }

            self._conn.run(query, params)

        self.relationship_buffer.clear()


    def fetch_all(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        params = params or {}
        logger.debug(f"Running fetch_all query:\n{query}\nparams={params}")
        records = self._conn.run(query, params)
        return records
    
    
    def clear_database(self) -> None:
        self._conn.run("MATCH (n) DETACH DELETE n", {})
        logger.info("Neo4j cleared: MATCH (n) DETACH DELETE n")