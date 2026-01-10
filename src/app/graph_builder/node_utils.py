from typing import Optional, Iterable
from tree_sitter import Node
import ast

from app.Logger import get_logger

logger = get_logger()


def get_text(node: Node) -> str | None:
        if node is None or node.text is None:
            return None
        try:
            return node.text.decode("utf-8")
        except Exception as e:
            logger.debug(f"Failed to decode node text: {e}", exc_info=True)
            return None
        

def get_docstring(func_node: Node) -> str | None:
    body = func_node.child_by_field_name("body")
    if body is None:
        return None

    string_node: Node | None = None

    for child in body.children:
        if child.type == "expression_statement" and child.named_child_count == 1:
            inner = child.named_children[0]
            if inner.type == "string":
                string_node = inner
                break

        if child.type == "string":
            string_node = child
            break

    if string_node is None:
        return None

    raw = get_text(string_node)
    if not raw:
        return None

    try:
        evaluated = ast.literal_eval(raw)
        if isinstance(evaluated, str):
            return evaluated
    except Exception:
        return raw.strip().strip('\'"')

    return None


def extract_decorators(node: Node, language_grammar) -> list[str]:
        current = node.parent
        while current is not None and current.type not in language_grammar.decorators:
            current = current.parent

        if current is None or current.type != "decorated_definition":
            return []

        decorators: list[str] = []
        for child in current.children:
            if child.type == "decorator":
                name = read_decorator_name(child)
                if name:
                    decorators.append(name)

        if decorators:
            logger.debug(f"Found decorators on node at line {node.start_point[0] + 1}: {decorators}")

        return decorators


def read_decorator_name(decorator_node: Node) -> str | None:
    target: Node | None = None

    for child in decorator_node.children:
        if child.type in ("identifier", "attribute", "call"):
            target = child
            break

    if target is None:
        logger.debug("Decorator node has no identifier/attribute/call target")
        return None

    if target.type == "call":
        fn = target.child_by_field_name("function")
        if fn is None:
            logger.debug("Decorator call node has no function child")
            return None
        target = fn

    text = get_text(target)
    return text or None


def resolve_node_full_name(node: Node, language_grammar, module_full_name: str) -> Optional[str]:

    try:
        parts: list[str] = []

        node_name = language_grammar.get_name(node)
        if not node_name:
            return None

        parts.append(node_name)

        current = node.parent
        while current is not None:
            if current.type in language_grammar.scope_node_types:
                scope_name = language_grammar.get_name(current)
                if scope_name:
                    parts.append(scope_name)
            current = current.parent

        parts.reverse()

        full_name = f"{module_full_name}.{'.'.join(parts)}"
        return full_name

    except Exception as e:
        logger.debug(f"Failed to resolve full name for node: {e}", exc_info=True)
        return None
    


def iter_import_nodes(captures: dict[str, list[Node]]) -> Iterable[Node]:
    for key in ("import", "import_from"):
        for node in captures.get(key, []):
            yield node