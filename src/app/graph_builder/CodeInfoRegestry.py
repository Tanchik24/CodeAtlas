from __future__ import annotations
from typing import Dict
from pathlib import Path

from app.enums import NodeLabel


class CodeInfoRegestry:

    def __init__(self) -> None:
        self.imports_alias_full_name_mapper: dict[str, dict[str, str]] = {}
        self.registry: dict[str, str] = {}
        self.packages: Dict[Path, str] = {}

    def add_import_alias(
        self,
        module_qn: str,
        local_name: str,
        qualified_name: str,
    ) -> None:
        bucket = self.imports_alias_full_name_mapper.setdefault(module_qn, {})
        bucket[local_name] = qualified_name

    def clear_imports_for_module(self, module_qn: str) -> None:
        self.imports_alias_full_name_mapper[module_qn] = {}

    def get_import_aliases(self, module_qn: str) -> dict[str, str]:
        return self.imports_alias_full_name_mapper.get(module_qn, {})

    def add_symbol(self, full_name: str, kind: str = NodeLabel.FUNCTION) -> None:
        self.registry[full_name] = kind

    def get_symbol_kind(self, qualified_name: str) -> NodeLabel | None:
        return self.registry.get(qualified_name)
    
    def add_to_packages(self, relative_folder_path: str, package_full_name: str):
        self.packages[relative_folder_path] = package_full_name

    def get_package(self, relative_folder_path: str) -> str | None:
        return self.packages.get(relative_folder_path)