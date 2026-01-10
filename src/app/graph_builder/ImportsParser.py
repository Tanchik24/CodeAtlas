from __future__ import annotations

from pathlib import Path
from typing import Optional, Set

from tree_sitter import Node

from app.languages import LanguageRegistry, LanguageGrammar
from app.enums import Language as LanguageEnum
from app.graph_builder.node_utils import iter_import_nodes, get_text
from app.Logger import get_logger
from services import Neo4jIngestor
from app.enums import NodeLabel, RelType
from . import CodeInfoRegestry

logger = get_logger()


class ImportsParser:
    def __init__(
        self,
        project_name: str,
        repo_path: Path,
        language_registry: LanguageRegistry,
        ingestor: Optional[Neo4jIngestor] = None,
        code_info_registry: Optional[CodeInfoRegestry] = None,
    ) -> None:
        self.code_info_registry = code_info_registry or CodeInfoRegestry()
        self.project_name = project_name
        self.repo_path = repo_path
        self.language_registry = language_registry
        self.ingestor = ingestor

    def parse_imports(self, file_path: Path) -> None:
        lang_grammar: LanguageGrammar | None = self.language_registry.get_by_extension(file_path.suffix)
        if lang_grammar is None:
            return
        try:
            rel = file_path.relative_to(self.repo_path)
        except Exception:
            rel = Path(file_path.name)

        module_qn = ".".join([self.project_name] + list(rel.with_suffix("").parts))

        try:
            tree = lang_grammar.parser.parse(file_path.read_bytes())
            root_node: Node = tree.root_node
        except Exception as e:
            logger.debug(f"Failed to parse file {file_path}: {e}", exc_info=True)
            return

        self.code_info_registry.clear_imports_for_module(module_qn)

        imports_query = getattr(lang_grammar.queries, "imports", None)
        if not imports_query:
            return

        captures: dict[str, list[Node]] = {}
        try:
            for node, capture_name in imports_query.captures(root_node):
                captures.setdefault(capture_name, []).append(node)
        except Exception as e:
            logger.debug(f"Failed to run imports query for {file_path}: {e}", exc_info=True)
            return

        if not captures:
            return

        lang = lang_grammar.get_lang_name()

        imported_modules: set[str] = set()
        if lang == LanguageEnum.PYTHON:
            imported_modules = self._parse_python_imports(captures, module_qn)
        elif lang == LanguageEnum.CPP:
            imported_modules = self._parse_cpp_imports(captures, module_qn, file_path)
        else:
            return

        self._link_imports_to_graph(module_qn=module_qn, imported_modules=imported_modules)


    def _parse_python_imports(
        self,
        captures: dict[str, list[Node]],
        module_qn: str,
    ) -> Set[str]:
        imported_modules: set[str] = set()

        for import_node in iter_import_nodes(captures):
            if import_node.type == "import_statement":
                imported_modules |= self._handle_python_import_statement(import_node, module_qn)
            elif import_node.type == "import_from_statement":
                imported_modules |= self._handle_python_import_from_statement(import_node, module_qn)

        return imported_modules

    def _handle_python_import_statement(self, import_node: Node, module_qn: str) -> Set[str]:
        imported_modules: set[str] = set()

        for child in import_node.named_children:
            if child.type == "dotted_name":
                raw = (get_text(child) or "").strip()
                if not raw:
                    continue

                qualified_mod = self._qualify_python_module(raw)
                imported_modules.add(qualified_mod)

                local_name = raw.split(".", 1)[0]
                self.code_info_registry.add_import_alias(module_qn, local_name, qualified_mod)

            elif child.type == "aliased_import":
                name_node = child.child_by_field_name("name")
                alias_node = child.child_by_field_name("alias")
                raw = (get_text(name_node) or "").strip() if name_node else ""
                alias = (get_text(alias_node) or "").strip() if alias_node else ""
                if not raw or not alias:
                    continue

                qualified_mod = self._qualify_python_module(raw)
                imported_modules.add(qualified_mod)
                self.code_info_registry.add_import_alias(module_qn, alias, qualified_mod)

        return imported_modules

    def _handle_python_import_from_statement(self, import_node: Node, module_qn: str) -> Set[str]:
        imported_modules: set[str] = set()

        module_name = self._extract_from_module_name(import_node, module_qn)
        if not module_name:
            return imported_modules

        base_module = self._qualify_python_module(module_name)
        imported_modules.add(base_module)

        imported_items, is_wildcard = self._collect_from_import_items(import_node)
        if is_wildcard:
            wildcard_key = f"*{base_module}"
            self.code_info_registry.add_import_alias(module_qn, wildcard_key, base_module)
            return imported_modules

        for local_name, original_name in imported_items:
            full_name = f"{base_module}.{original_name}"
            self.code_info_registry.add_import_alias(module_qn, local_name, full_name)

        return imported_modules

    def _qualify_python_module(self, raw_module: str) -> str:
        if raw_module.startswith(self.project_name + "."):
            return raw_module

        p = self.repo_path / Path(*raw_module.split("."))
        if p.is_dir() or (p.with_suffix(".py")).is_file():
            return f"{self.project_name}.{raw_module}"

        return raw_module

    def _extract_from_module_name(self, import_node: Node, module_qn: str) -> str | None:
        module_name_node = import_node.child_by_field_name("module_name")
        if not module_name_node:
            return None

        if module_name_node.type == "dotted_name":
            return (get_text(module_name_node) or "").strip() or None

        if module_name_node.type == "relative_import":
            return self._resolve_relative_import(module_name_node, module_qn)

        return None

    def _resolve_relative_import(self, relative_node: Node, module_qn: str) -> str:
        module_parts = module_qn.split(".")[1:] 

        dots = 0
        module_name = ""

        for child in relative_node.children:
            if child.type == "import_prefix":
                decoded = get_text(child) or ""
                dots = len(decoded)
            elif child.type == "dotted_name":
                module_name = (get_text(child) or "").strip()

        target_parts = module_parts[:-dots] if dots > 0 else module_parts
        if module_name:
            target_parts.extend(module_name.split("."))

        return ".".join(target_parts)

    def _collect_from_import_items(self, import_node: Node) -> tuple[list[tuple[str, str]], bool]:
        items: list[tuple[str, str]] = []
        is_wildcard = False

        for name_node in import_node.children_by_field_name("name"):
            if name_node.type == "dotted_name":
                decoded = (get_text(name_node) or "").strip()
                if decoded:
                    items.append((decoded, decoded))
            elif name_node.type == "aliased_import":
                orig_node = name_node.child_by_field_name("name")
                alias_node = name_node.child_by_field_name("alias")
                orig = (get_text(orig_node) or "").strip() if orig_node else ""
                alias = (get_text(alias_node) or "").strip() if alias_node else ""
                if orig and alias:
                    items.append((alias, orig))

        for child in import_node.children:
            if child.type == "wildcard_import":
                is_wildcard = True
                break

        return items, is_wildcard


    def _parse_cpp_imports(
        self,
        captures: dict[str, list[Node]],
        module_qn: str,
        file_path: Path,
    ) -> Set[str]:
        imported_modules: set[str] = set()

        for n in captures.get("import", []) + captures.get("import_from", []):
            if not isinstance(n, Node) or n.type != "preproc_include":
                continue

            inc = self._extract_cpp_include_path(n)
            if not inc:
                continue

            resolved = self._resolve_cpp_include_to_project_module(inc, file_path)
            if not resolved:
                continue

            imported_modules.add(resolved)

            local_name = Path(inc).stem or inc
            self.code_info_registry.add_import_alias(module_qn, local_name, resolved)

        return imported_modules

    def _extract_cpp_include_path(self, include_node: Node) -> str | None:
        path_node = include_node.child_by_field_name("path")
        candidate = get_text(path_node) if path_node is not None else None

        if not candidate:
            for ch in include_node.named_children:
                if ch.type in {"string_literal", "system_lib_string"}:
                    candidate = get_text(ch)
                    break

        if not candidate:
            return None

        s = candidate.strip()
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("<") and s.endswith(">")):
            s = s[1:-1].strip()

        s = s.replace("\\", "/").strip()
        return s or None

    def _resolve_cpp_include_to_project_module(self, include_path: str, src_file_path: Path) -> str | None:
        inc = Path(include_path)
        candidates: list[Path] = [
            (src_file_path.parent / inc).resolve(),
            (self.repo_path / inc).resolve(),
        ]

        if inc.suffix == "":
            for suf in (".h", ".hpp", ".hh", ".hxx", ".ixx"):
                candidates.append((src_file_path.parent / inc).with_suffix(suf).resolve())
                candidates.append((self.repo_path / inc).with_suffix(suf).resolve())

        real: Path | None = None
        for c in candidates:
            try:
                if c.is_file() and self.repo_path in c.parents:
                    real = c
                    break
            except Exception:
                continue

        if real is None:
            return None

        rel = real.relative_to(self.repo_path)
        return ".".join([self.project_name] + list(rel.with_suffix("").parts))


    def _link_imports_to_graph(self, module_qn: str, imported_modules: Set[str]) -> None:
        if self.ingestor is None or not imported_modules:
            return

        for target_module in sorted(imported_modules):
            if target_module == module_qn:
                continue

            self._ensure_module_node(target_module)

            self.ingestor.add_relationship_to_buffer(
                start_label=NodeLabel.MODULE.value,
                start_props={"full_name": module_qn},
                rel_type=RelType.IMPORTS.value,
                end_label=NodeLabel.MODULE.value,
                end_props={"full_name": target_module},
                rel_props={},
            )

    def _ensure_module_node(self, module_full_name: str) -> None:
        if self.ingestor is None:
            return

        name = module_full_name.split(".")[-1] if "." in module_full_name else module_full_name
        self.ingestor.add_node_to_buffer(
            NodeLabel.MODULE.value,
            full_name=module_full_name,
            name=name,
        )
