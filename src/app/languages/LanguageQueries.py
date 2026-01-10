from __future__ import annotations
from app.languages import LanguageGrammar


class LanguageQueries:
    def __init__(self, grammar: LanguageGrammar) -> None:
        self._grammar = grammar
        self.imports = self._build_import_query()
        self.functions = self._build_function_query()
        self.classes = self._build_class_query()

    def _build_import_query(self):
        parts: list[str] = []

        parts.extend(f"({import_type}) @import" for import_type in self._grammar.import_nodes)
        parts.extend(f"({import_type}) @import_from" for import_type in self._grammar.import_from_nodes)

        pattern = " ".join(parts)
        return self._grammar.language.query(pattern)
    
    def _build_function_query(self):
        pattern = " ".join(f"({node_type}) @function" for node_type in self._grammar.function_nodes).strip()
        return self._grammar.language.query(pattern)
    
    def _build_class_query(self):
        pattern = " ".join(f"({node_type}) @class" for node_type in self._grammar.class_nodes).strip()
        return self._grammar.language.query(pattern)
