from __future__ import annotations

from pathlib import Path
from typing import Optional, List
from tree_sitter import Node, Query

from app.languages import LanguageRegistry, LanguageGrammar
from app.entities.Project import Project
from services import Neo4jIngestor
from app.enums import NodeLabel, RelType, Language as lang_name
from app.graph_builder.node_utils import get_text, get_docstring, extract_decorators, resolve_node_full_name
from app.Logger import get_logger
from . import CodeInfoRegestry

logger = get_logger()


class GraphBuilder:
    def __init__(
        self,
        project: Project,
        language_registry: LanguageRegistry,
        ingestor: Neo4jIngestor) -> None:
        
        self.project = project
        self.language_registry = language_registry
        self.ingestor = ingestor
        self.code_info_registry = CodeInfoRegestry()

        self.package_marker_files = self._get_package_marker_files()
        self.methods: List[tuple[int, int]] = []
        self._defined_classes: set[str] = set()

    
    def process_folders(self, folder_path: Path) -> None:
        relative_folder_path = folder_path.relative_to(self.project.path)

        parent_rel_path = relative_folder_path.parent
        parent_package = self.code_info_registry.get_package(parent_rel_path)

        if parent_rel_path == Path("."):
            parent_label, parent_props = (
                NodeLabel.PROJECT.value,
                {"name": self.project.name},
            )
        else:
            parent_label, parent_props = (
                (NodeLabel.PACKAGE.value, {"full_name": parent_package})
                if parent_package
                else (NodeLabel.FOLDER.value, {"path": str(parent_rel_path)})
            )

        def marker_exists(marker: str) -> bool:
            return any(folder_path.glob(marker))

        is_package = any(marker_exists(marker) for marker in self.package_marker_files)

        if is_package:
            package_full_name = ".".join(
                [self.project.name] + list(relative_folder_path.parts)
            )
            self.code_info_registry.add_to_packages(relative_folder_path, package_full_name)

            self.ingestor.add_node_to_buffer(
                NodeLabel.PACKAGE.value,
                full_name=package_full_name,
                name=folder_path.name,
                path=str(relative_folder_path),
            )

            self.ingestor.add_relationship_to_buffer(
                start_label=parent_label,
                start_props=parent_props,
                rel_type=RelType.CONTAINS_PACKAGE.value,
                end_label=NodeLabel.PACKAGE.value,
                end_props={"full_name": package_full_name},
            )
            return

        if folder_path == self.project.path:
            return

        self.ingestor.add_node_to_buffer(
            NodeLabel.FOLDER.value,
            path=str(relative_folder_path),
            name=folder_path.name,
        )

        self.ingestor.add_relationship_to_buffer(
            start_label=parent_label,
            start_props=parent_props,
            rel_type=RelType.CONTAINS_FOLDER.value,
            end_label=NodeLabel.FOLDER.value,
            end_props={"path": str(relative_folder_path)},
        )


    def process_files(self, file_path: Path) -> None:

        if file_path.name in self.package_marker_files:
            return

        relative_path = file_path.relative_to(self.project.path)
        file_extension = file_path.suffix
        lang_grammar = self.language_registry.get_by_extension(file_extension)

        if lang_grammar is None:
            return

        tree = lang_grammar.parser.parse(file_path.read_bytes())
        root = tree.root_node

        module_full_name = ".".join(
            [self.project.name] + list(relative_path.with_suffix("").parts)
        )

        self._process_module(file_path, module_full_name, relative_path)
        self._process_classes_methods(root, lang_grammar, module_full_name, relative_path)
        self._process_functions(lang_grammar, root, module_full_name, relative_path)


    def _process_module(
        self,
        file_path: Path,
        module_full_name: str,
        relative_file_path: Path,
    ) -> None:
        self.ingestor.add_node_to_buffer(
            NodeLabel.MODULE.value,
            full_name=module_full_name,
            name=file_path.name,
            path=str(relative_file_path),
        )

        parent_rel_path = relative_file_path.parent
        parent_package = self.code_info_registry.get_package(parent_rel_path)

        if parent_rel_path == Path("."):
            parent_label, parent_props = (
                NodeLabel.PROJECT.value,
                {"name": self.project.name},
            )
        else:
            parent_label, parent_props = (
                (NodeLabel.PACKAGE.value, {"full_name": parent_package})
                if parent_package
                else (NodeLabel.FOLDER.value, {"path": str(parent_rel_path)})
            )

        self.code_info_registry.add_symbol(module_full_name, NodeLabel.MODULE)

        self.ingestor.add_relationship_to_buffer(
            start_label=parent_label,
            start_props=parent_props,
            rel_type=RelType.CONTAINS_MODUL.value,
            end_label=NodeLabel.MODULE.value,
            end_props={"full_name": module_full_name},
        )


    def _process_functions(self, lang_grammar: LanguageGrammar, root: Node, module_full_name: str, relative_file_path: Path) -> None:
        query: Query | None = getattr(lang_grammar.queries, "functions", None)
        if query is None:
            return

        methods_list = getattr(self, "methods", []) or []
        methods_set: set[tuple[int, int]] = set(methods_list)
        methods_starts: set[int] = {s for s, _ in methods_set}

        is_cpp = (lang_grammar.get_lang_name() == lang_name.CPP)

        def _find_first_descendant(n: Node, types: set[str]) -> Optional[Node]:
            stack = [n]
            while stack:
                cur = stack.pop()
                if cur.type in types:
                    return cur
                stack.extend(reversed(cur.named_children))
            return None

        def _find_existing_class_by_suffix(class_suffix: str) -> Optional[str]:
            for qn, kind in self.code_info_registry.registry.items():
                if kind == NodeLabel.CLASS or kind == NodeLabel.CLASS.value:
                    if qn.endswith(class_suffix):
                        return qn
            return None

        def _extract_cpp_qname_parts(func_node: Node) -> Optional[list[str]]:
            decl = func_node.child_by_field_name("declarator") or _find_first_descendant(func_node, {"function_declarator"})
            if decl is None:
                decl = func_node

            qi = _find_first_descendant(decl, {"qualified_identifier", "scoped_identifier"})
            if qi is None:
                qi = _find_first_descendant(decl, {"identifier", "field_identifier", "operator_name", "destructor_name"})

            raw = (get_text(qi) or "").strip() if qi else ""
            if not raw:
                return None

            raw = raw.split("(")[0].strip()
            parts = [p.strip() for p in raw.split("::") if p.strip()]
            return parts or None

        for node, cap in query.captures(root):
            if cap != "function" or not isinstance(node, Node):
                continue

            span = (node.start_byte, node.end_byte)
            if span in methods_set or node.start_byte in methods_starts:
                continue

            if not is_cpp:
                name_node = node.child_by_field_name("name")
                func_name = get_text(name_node) if name_node else None
                if not func_name:
                    continue
                qn_parts = [func_name]
            else:
                qn_parts = _extract_cpp_qname_parts(node)
                if not qn_parts:
                    continue

                if len(qn_parts) >= 2:
                    qualifier = qn_parts[:-1] 
                    class_suffix = "." + ".".join(qualifier)
                    if _find_existing_class_by_suffix(class_suffix) is not None:
                        continue 

            func_full_name = f"{module_full_name}." + ".".join(qn_parts)

            self.code_info_registry.add_symbol(full_name=func_full_name, kind=NodeLabel.FUNCTION)

            self.ingestor.add_node_to_buffer(
                NodeLabel.FUNCTION.value,
                full_name=func_full_name,
                name=qn_parts[-1],
                path=str(relative_file_path),
                decorators=extract_decorators(node, lang_grammar),
                docstring=get_docstring(node),
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
            )

            self.ingestor.add_relationship_to_buffer(
                start_label=NodeLabel.MODULE.value,
                start_props={"full_name": module_full_name},
                rel_type=RelType.DEFINES_FUNCTION.value,
                end_label=NodeLabel.FUNCTION.value,
                end_props={"full_name": func_full_name},
            )


    def _process_classes_methods(
        self,
        root: Node,
        lang_grammar: LanguageGrammar,
        module_full_name: str,
        relative_file_path: Path,
    ) -> None:
        if not hasattr(self, "methods") or not isinstance(self.methods, list):
            self.methods: List[tuple[int, int]] = []
        else:
            self.methods.clear()
        methods_set: set[tuple[int, int]] = set()

        class_query: Query | None = getattr(lang_grammar.queries, "classes", None)
        func_query: Query | None = getattr(lang_grammar.queries, "functions", None)
        if class_query is None:
            return

        is_cpp = (lang_grammar.get_lang_name() == lang_name.CPP)

        if not hasattr(self, "_pending_cpp_methods") or not isinstance(self._pending_cpp_methods, list):
            self._pending_cpp_methods: list[dict] = []

        def _find_first_descendant(n: Node, types: set[str]) -> Optional[Node]:
            stack = [n]
            while stack:
                cur = stack.pop()
                if cur.type in types:
                    return cur
                for ch in reversed(cur.named_children):
                    stack.append(ch)
            return None

        def _cpp_namespace_parts(n: Node) -> list[str]:
            parts: list[str] = []
            cur = n.parent
            while cur is not None:
                if cur.type == "namespace_definition":
                    nm = get_text(cur.child_by_field_name("name"))
                    if nm:
                        parts.append(nm)
                cur = cur.parent
            parts.reverse()
            return parts

        def _is_cpp_func_like(n: Node) -> bool:
            if n.type in {"function_definition", "lambda_expression"}:
                return True
            return _find_first_descendant(n, {"function_declarator", "function_definition"}) is not None

        def _extract_cpp_method_name(func_like: Node) -> Optional[str]:
            fd = _find_first_descendant(func_like, {"function_declarator"})
            if fd is not None:
                dn = fd.child_by_field_name("declarator") or fd.child_by_field_name("name")
                txt = (get_text(dn) or "").strip() if dn else ""
                if txt:
                    if "::" in txt:
                        txt = txt.split("::")[-1].strip()
                    return txt or None

            name_node = func_like.child_by_field_name("name")
            txt = (get_text(name_node) or "").strip() if name_node else ""
            if txt:
                if "::" in txt:
                    txt = txt.split("::")[-1].strip()
                return txt or None

            decl = func_like.child_by_field_name("declarator") or func_like
            id_node = _find_first_descendant(
                decl,
                {"identifier", "field_identifier", "destructor_name", "operator_name", "qualified_identifier", "scoped_identifier"},
            )
            txt = (get_text(id_node) or "").strip() if id_node else ""
            if "::" in txt:
                txt = txt.split("::")[-1].strip()
            return txt or None

        def _extract_cpp_out_of_class_target(fn_node: Node) -> Optional[tuple[list[str], str]]:
            decl = fn_node.child_by_field_name("declarator") or _find_first_descendant(fn_node, {"function_declarator"})
            raw = (get_text(decl) or "").strip() if decl else ""
            if "::" not in raw:
                qi = _find_first_descendant(fn_node, {"qualified_identifier", "scoped_identifier"})
                raw = (get_text(qi) or "").strip() if qi else ""
            if "::" not in raw:
                return None

            raw = raw.split("(")[0].strip()
            parts = [p.strip() for p in raw.split("::") if p.strip()]
            if len(parts) < 2:
                return None
            return parts[:-1], parts[-1]

        def _find_existing_class_qn(ns_plus_class: list[str]) -> Optional[str]:
            suffix = "." + ".".join(ns_plus_class)
            best: Optional[str] = None
            for qn, kind in self.code_info_registry.registry.items():
                if kind == NodeLabel.CLASS or kind == NodeLabel.CLASS.value:
                    if qn.endswith(suffix):
                        if best is None or len(qn) > len(best):
                            best = qn
            return best

        for node, capture_name in class_query.captures(root):
            if capture_name != "class":
                continue
            if not isinstance(node, Node):
                continue

            class_node = node

            class_full_name = resolve_node_full_name(class_node, lang_grammar, module_full_name)
            if not class_full_name:
                continue

            class_name = lang_grammar.get_name(class_node)
            if not class_name:
                continue

            decorators = extract_decorators(class_node, lang_grammar)

            self._defined_classes.add(class_full_name)
            self.code_info_registry.add_symbol(full_name=class_full_name, kind=NodeLabel.CLASS)

            self.ingestor.add_node_to_buffer(
                NodeLabel.CLASS.value,
                full_name=class_full_name,
                name=class_name,
                path=str(relative_file_path),
                decorators=decorators,
                start_line=class_node.start_point[0] + 1,
                end_line=class_node.end_point[0] + 1,
                docstring=get_docstring(class_node),
            )

            self.ingestor.add_relationship_to_buffer(
                start_label=NodeLabel.MODULE.value,
                start_props={"full_name": module_full_name},
                rel_type=RelType.DEFINES_CLASS.value,
                end_label=NodeLabel.CLASS.value,
                end_props={"full_name": class_full_name},
            )

            if func_query is None:
                continue

            if not is_cpp:
                for child_node, child_cap in func_query.captures(class_node):
                    if child_cap != "function" or not isinstance(child_node, Node):
                        continue

                    name_node = child_node.child_by_field_name("name")
                    method_name = get_text(name_node)
                    if not method_name:
                        continue

                    method_full_name = f"{class_full_name}.{method_name}"
                    span = (child_node.start_byte, child_node.end_byte)
                    if span not in methods_set:
                        self.methods.append(span)
                        methods_set.add(span)

                    self.code_info_registry.add_symbol(full_name=method_full_name, kind=NodeLabel.METHOD)

                    self.ingestor.add_node_to_buffer(
                        NodeLabel.METHOD.value,
                        full_name=method_full_name,
                        name=method_name,
                        path=str(relative_file_path),
                        decorators=extract_decorators(child_node, lang_grammar),
                        docstring=get_docstring(child_node),
                        start_line=child_node.start_point[0] + 1,
                        end_line=child_node.end_point[0] + 1,
                    )

                    self.ingestor.add_relationship_to_buffer(
                        start_label=NodeLabel.CLASS.value,
                        start_props={"full_name": class_full_name},
                        rel_type=RelType.DEFINES_METHOD.value,
                        end_label=NodeLabel.METHOD.value,
                        end_props={"full_name": method_full_name},
                    )

                continue

            body = class_node.child_by_field_name("body") or _find_first_descendant(class_node, {"field_declaration_list"})
            search_node = body if body is not None else class_node

            for child_node, child_cap in func_query.captures(search_node):
                if child_cap != "function" or not isinstance(child_node, Node):
                    continue

                if not _is_cpp_func_like(child_node):
                    continue

                method_name = _extract_cpp_method_name(child_node)
                if not method_name:
                    continue

                method_full_name = f"{class_full_name}.{method_name}"
                span = (child_node.start_byte, child_node.end_byte)
                if span not in methods_set:
                    self.methods.append(span)
                    methods_set.add(span)

                self.code_info_registry.add_symbol(full_name=method_full_name, kind=NodeLabel.METHOD)

                self.ingestor.add_node_to_buffer(
                    NodeLabel.METHOD.value,
                    full_name=method_full_name,
                    name=method_name,
                    path=str(relative_file_path),
                    decorators=[],
                    docstring=None,
                    start_line=child_node.start_point[0] + 1,
                    end_line=child_node.end_point[0] + 1,
                )

                self.ingestor.add_relationship_to_buffer(
                    start_label=NodeLabel.CLASS.value,
                    start_props={"full_name": class_full_name},
                    rel_type=RelType.DEFINES_METHOD.value,
                    end_label=NodeLabel.METHOD.value,
                    end_props={"full_name": method_full_name},
                )

        if is_cpp and func_query is not None:
            for fn_node, cap in func_query.captures(root):
                if cap != "function" or not isinstance(fn_node, Node):
                    continue
                if not _is_cpp_func_like(fn_node):
                    continue

                target = _extract_cpp_out_of_class_target(fn_node)
                if not target:
                    continue

                class_path, method_name = target
                ns_parts = _cpp_namespace_parts(fn_node)
                ns_plus_class = ns_parts + class_path

                span = (fn_node.start_byte, fn_node.end_byte)
                if span not in methods_set:
                    self.methods.append(span)
                    methods_set.add(span)

                class_qn = _find_existing_class_qn(ns_plus_class)
                if class_qn is None:
                    self._pending_cpp_methods.append(
                        {
                            "ns_plus_class": ns_plus_class,
                            "method_name": method_name,
                            "start_line": fn_node.start_point[0] + 1,
                            "end_line": fn_node.end_point[0] + 1,
                            "path": str(relative_file_path),
                        }
                    )
                    continue

                method_full_name = f"{class_qn}.{method_name}"
                kind = self.code_info_registry.get_symbol_kind(method_full_name)
                if kind == NodeLabel.METHOD or kind == NodeLabel.METHOD.value:
                    continue

                self.code_info_registry.add_symbol(full_name=method_full_name, kind=NodeLabel.METHOD)

                self.ingestor.add_node_to_buffer(
                    NodeLabel.METHOD.value,
                    full_name=method_full_name,
                    name=method_name,
                    path=str(relative_file_path),
                    decorators=[],
                    docstring=None,
                    start_line=fn_node.start_point[0] + 1,
                    end_line=fn_node.end_point[0] + 1,
                )

                self.ingestor.add_relationship_to_buffer(
                    start_label=NodeLabel.CLASS.value,
                    start_props={"full_name": class_qn},
                    rel_type=RelType.DEFINES_METHOD.value,
                    end_label=NodeLabel.METHOD.value,
                    end_props={"full_name": method_full_name},
                )

        if is_cpp and self._pending_cpp_methods:
            still: list[dict] = []
            for item in self._pending_cpp_methods:
                class_qn = _find_existing_class_qn(item["ns_plus_class"])
                if class_qn is None:
                    still.append(item)
                    continue

                method_full_name = f"{class_qn}.{item['method_name']}"
                kind = self.code_info_registry.get_symbol_kind(method_full_name)
                if kind == NodeLabel.METHOD or kind == NodeLabel.METHOD.value:
                    continue

                self.code_info_registry.add_symbol(full_name=method_full_name, kind=NodeLabel.METHOD)

                self.ingestor.add_node_to_buffer(
                    NodeLabel.METHOD.value,
                    full_name=method_full_name,
                    name=item["method_name"],
                    path=item.get("path"),
                    decorators=[],
                    docstring=None,
                    start_line=item["start_line"],
                    end_line=item["end_line"],
                )

                self.ingestor.add_relationship_to_buffer(
                    start_label=NodeLabel.CLASS.value,
                    start_props={"full_name": class_qn},
                    rel_type=RelType.DEFINES.value,
                    end_label=NodeLabel.METHOD.value,
                    end_props={"full_name": method_full_name},
                )

            self._pending_cpp_methods = still

    
    def _process_python_class_inheritance(
        self, root: Node, lang_grammar: LanguageGrammar, module_full_name: str
    ) -> None:
        query: Query | None = getattr(lang_grammar.queries, "classes", None)
        if query is None:
            return

        for node, capture_name in query.captures(root):
            if capture_name != "class":
                continue

            class_node = node
            if not isinstance(class_node, Node):
                continue

            class_full_name = resolve_node_full_name(
                class_node,
                lang_grammar,
                module_full_name,
            )
            if not class_full_name:
                continue

            self.code_info_registry.add_symbol(
                full_name=class_full_name,
                kind=NodeLabel.CLASS,
            )

            parent_classes = self._extract_python_parent_classes(
                class_node, module_full_name
            )
            if not parent_classes:
                continue

            for parent_full_name in parent_classes:
                self.code_info_registry.add_symbol(
                    full_name=parent_full_name,
                    kind=NodeLabel.CLASS,
                )

                if parent_full_name not in self._defined_classes:
                    parent_simple_name = parent_full_name.split(".")[-1]
                    self.ingestor.add_node_to_buffer(
                        NodeLabel.CLASS.value,
                        full_name=parent_full_name,
                        name=parent_simple_name,
                    )

                self.ingestor.add_relationship_to_buffer(
                    start_label=NodeLabel.CLASS.value,
                    start_props={"full_name": class_full_name},
                    rel_type=RelType.INHERITS.value,
                    end_label=NodeLabel.CLASS.value,
                    end_props={"full_name": parent_full_name},
                )

                logger.info(
                    f"Class inheritance: {class_full_name} INHERITS {parent_full_name}"
                )


    def _extract_python_parent_classes(
        self, class_node: Node, module_full_name: str
    ) -> list[str]:
        parents: list[str] = []

        super_node = class_node.child_by_field_name("superclasses")
        if not super_node:
            return parents

        for child in super_node.named_children:
            if child.type == "keyword_argument":
                continue

            parent_fqn = self._resolve_python_parent_from_expr(
                child, module_full_name
            )
            if parent_fqn and parent_fqn not in parents:
                parents.append(parent_fqn)

        return parents


    def _resolve_python_parent_from_expr(
        self, expr_node: Node, module_full_name: str
    ) -> str | None:
        if expr_node.type == "identifier":
            name = get_text(expr_node)
            if not name:
                return None
            return self._resolve_python_qualified_name_from_name(name, module_full_name)

        if expr_node.type == "attribute":
            text = get_text(expr_node)
            if not text:
                return None

            parts = text.split(".")
            if not parts:
                return None

            head = parts[0]
            tail = parts[-1]

            alias_map = self.code_info_registry.get_import_aliases(module_full_name)

            if head in alias_map:
                base_module_qn = alias_map[head]
                if len(parts) == 2:
                    return f"{base_module_qn}.{tail}"
                else:
                    return ".".join([base_module_qn] + parts[1:])

            return self._resolve_python_qualified_name_from_name(tail, module_full_name)

        if expr_node.type == "keyword_argument":
            value = expr_node.child_by_field_name("value")
            if value is not None:
                return self._resolve_python_parent_from_expr(value, module_full_name)
            return None

        for child in expr_node.children:
            if child.type in ("identifier", "attribute"):
                return self._resolve_python_parent_from_expr(child, module_full_name)

        return None


    def _resolve_python_qualified_name_from_name(self, simple_name: str, module_full_name: str) -> str:

        alias_map = self.code_info_registry.get_import_aliases(module_full_name)
        if simple_name in alias_map:
            return alias_map[simple_name]

        same_module = f"{module_full_name}.{simple_name}"
        if self.code_info_registry.get_symbol_kind(same_module) == NodeLabel.CLASS:
            return same_module

        for full_name, kind in self.code_info_registry.registry.items():
            if kind == NodeLabel.CLASS and full_name.endswith(f".{simple_name}"):
                return full_name

        return same_module


    def _get_package_marker_files(self) -> set[str]:
        package_marker_files: set[str] = set()
        for lang in self.language_registry.languages.values():
            package_marker_files.update(lang().package_markers)
        return package_marker_files
