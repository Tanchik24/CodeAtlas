from functools import lru_cache
from .DBConfig import DBConfig
from .LLMConfig import LLMConfig
from .APIConfig import APIConfig


class Config:
    def __init__(self):
        self.gdb = DBConfig()
        self.llm = LLMConfig()
        self.api = APIConfig()


@lru_cache()
def get_config():
    return Config()