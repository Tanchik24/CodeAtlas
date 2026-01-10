from __future__ import annotations
from typing import List
from tree_sitter import Language
from tree_sitter_languages import get_language

from .LanguageGrammar import LanguageGrammar
from .CPPQueries import CPPQueries
from app.enums import Language as lang_name

class CGrammar(LanguageGrammar):
    file_exts: List[str] = [
        ".c", ".h",
        ".cpp", ".hpp", ".cc", ".cxx", ".hxx", ".hh",
        ".ixx", ".cppm", ".ccm",
    ]

    module_nodes: List[str] = [
        "translation_unit",
        "namespace_definition",
        "linkage_specification",
        "declaration",
    ]

    class_nodes: List[str] = [
        "class_specifier",
        "struct_specifier",
        "union_specifier",
        "enum_specifier",
    ]

    function_nodes: List[str] = [
        "function_definition",
        "declaration",
        "field_declaration",
        "template_declaration",
        "lambda_expression",
    ]

    call_nodes: List[str] = [
        "call_expression",
        "field_expression",
        "subscript_expression",
        "new_expression",
        "delete_expression",
        "binary_expression",
        "unary_expression",
        "update_expression",
    ]

    import_nodes: List[str] = [
        "preproc_include",
        "template_function",
        "declaration",
    ]
    import_from_nodes: List[str] = [
        "preproc_include",
        "template_function",
        "declaration",
    ]

    package_markers: List[str] = [
        "CMakeLists.txt",
        "Makefile",
        "*.vcxproj",
        "conanfile.txt",
    ]

    scope_node_types: List[str] = [
        "class_specifier",
        "struct_specifier",
        "namespace_definition",
        "translation_unit",
    ]

    def init_grammar_function(self) -> Language:
        return get_language(lang_name.CPP.value)
    
    def init_queries(self) -> CPPQueries:
        if self._queries is None:
            self._queries = CPPQueries(self)
        return self._queries
    
    def get_lang_name(self) -> lang_name:
        return lang_name.CPP
