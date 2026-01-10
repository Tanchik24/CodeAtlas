from __future__ import annotations
from typing import List
from tree_sitter import Language
from tree_sitter_languages import get_language

from .LanguageGrammar import LanguageGrammar
from app.enums import Language as lang_name


class PythonGrammar(LanguageGrammar):
    file_exts: List[str]  = [".py"]
    module_nodes: List[str]  = ["module"]
    class_nodes: List[str]  = ["class_definition"]
    function_nodes: List[str]  = ["function_definition"]
    call_nodes: List[str]  = ["call", "with_statement"]
    decorators: List[str]  = ["decorated_definition"]
    

    import_nodes: List[str]  = ["import_statement"]
    import_from_nodes: List[str]  = ["import_from_statement"]
    package_markers: List[str]  = ["__init__.py"]

    scope_node_types: List[str] = ["module", "class_definition", "function_definition"]


    def init_grammar_function(self) -> Language:
        return get_language(lang_name.PYTHON.value)
    
    def get_lang_name(self) -> lang_name:
        return lang_name.PYTHON
    