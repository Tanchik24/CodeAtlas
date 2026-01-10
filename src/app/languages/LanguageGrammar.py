from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, PrivateAttr
from tree_sitter import Language, Parser, Node
from .LanguageQueries import LanguageQueries

from app.graph_builder.node_utils import get_text
from app.enums import Language as lang_name


class LanguageGrammar(BaseModel):
    file_exts: List[str]
    module_nodes: List[str]
    class_nodes: List[str]
    function_nodes: List[str]
    call_nodes: List[str]
    decorators: List[str] = Field(default_factory=list)

    import_nodes: List[str] = Field(default_factory=list)
    import_from_nodes: List[str] = Field(default_factory=list)
    package_markers: List[str] = Field(default_factory=list)
    scope_node_types: List[str] = Field(default_factory=list)

    _language: Optional[Language] = PrivateAttr(default=None)
    _parser: Optional[Parser] = PrivateAttr(default=None)
    _queries: Optional[LanguageQueries] = PrivateAttr(default=None)

    class Config:
        arbitrary_types_allowed = True

    def init_grammar_function(self) -> Language:
        raise NotImplementedError

    def init_language(self) -> Language:
        if self._language is None:
            self._language = self.init_grammar_function()
        return self._language

    def init_parser(self) -> Parser:
        if self._parser is None:
            lang = self.init_language()
            parser = Parser()
            parser.set_language(lang)
            self._parser = parser
        return self._parser
    
    def init_queries(self) -> LanguageQueries:
        if self._queries is None:
            self._queries = LanguageQueries(self)
        return self._queries

    @property
    def parser(self) -> Parser:
        return self.init_parser()

    @property
    def language(self) -> Language:
        return self.init_language()
    
    @property
    def queries(self) -> LanguageQueries:
        return self.init_queries()
    
    def get_name(self, node: Node) -> str:
        name_node = node.child_by_field_name("name")
        return get_text(name_node) or ""
    
    def get_lang_name(self) -> lang_name:
        raise NotImplementedError
