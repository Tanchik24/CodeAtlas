from __future__ import annotations

import importlib
import inspect
import pkgutil
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Type

from .LanguageGrammar import LanguageGrammar
from app.Logger import get_logger


logger = get_logger()


class LanguageRegistry:

    def __init__(self) -> None:
        self._ext_to_cls: Dict[str, Type[LanguageGrammar]] = {}
        self._package_name: str = __package__ or "languages"
        logger.info(f"LanguageRegistry initialized with package '{self._package_name}'")

    @staticmethod
    def _normalize_ext(ext: str) -> str:
        if not ext.startswith("."):
            ext = "." + ext
        return ext.lower()

    def register_class(self, grammar_cls: Type[LanguageGrammar]) -> None:
        tmp = grammar_cls()
        for ext in tmp.file_exts:
            norm = self._normalize_ext(ext)
            self._ext_to_cls[norm] = grammar_cls

        logger.info(f"Registered grammar '{grammar_cls.__name__}' for extensions: {tmp.file_exts}")

    def auto_register(self) -> None:

        pkg = importlib.import_module(self._package_name)

        for module_info in pkgutil.iter_modules(pkg.__path__):
            module_name = module_info.name

            if module_name in {"__init__", "LanguageGrammar", "LanguageRegistry"}:
                continue

            full_name = f"{self._package_name}.{module_name}"
            module = importlib.import_module(full_name)

            for _, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, LanguageGrammar) and obj is not LanguageGrammar:
                    self.register_class(obj)

        logger.info(f"Auto registration completed. Total extensions registered: {len(self._ext_to_cls)}")

    @lru_cache(maxsize=64)
    def get_by_extension(self, ext: str) -> Optional[LanguageGrammar]:
        norm = self._normalize_ext(ext)
        cls = self._ext_to_cls.get(norm)
        if cls is None:
            logger.warning(f"No language grammar registered for extension '{norm}'")
            return None
        return cls()

    def get_for_path(self, path: Path | str) -> Optional[LanguageGrammar]:
        p = Path(path)
        return self.get_by_extension(p.suffix)
    
    @property
    def languages(self) -> Dict[str, Type[LanguageGrammar]]:
        return self._ext_to_cls



@lru_cache(maxsize=1)
def get_language_registry() -> LanguageRegistry:
    registry = LanguageRegistry()
    registry.auto_register()
    return registry