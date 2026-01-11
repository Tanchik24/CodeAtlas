from __future__ import annotations

from typing import Any, Dict, List, TypedDict, Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class CodeRAGState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]

    semantic_hits: List[Dict[str, Any]]
    graph_rows: List[Dict[str, Any]]
    context_snippets: List[str]
    answer: str
