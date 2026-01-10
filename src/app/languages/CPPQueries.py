from __future__ import annotations
from .LanguageGrammar import LanguageGrammar


class CPPQueries:
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
        pattern = """
    (function_definition) @function
    (template_declaration (function_definition)) @function
    (lambda_expression) @function
    (field_declaration) @function
    (declaration) @function
    """
        return self._grammar.language.query(pattern)
    
    def _build_class_query(self):
        pattern = """
    (class_specifier) @class
    (struct_specifier) @class
    (union_specifier) @class
    (enum_specifier) @class
    (template_declaration (class_specifier)) @class
    (template_declaration (struct_specifier)) @class
    (template_declaration (union_specifier)) @class
    (template_declaration (enum_specifier)) @class
    """
        return self._grammar.language.query(pattern)
