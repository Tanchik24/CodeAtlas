from functools import lru_cache
from .DBConfig import DBConfig
from .LLMConfig import LLMConfig
from .APIConfig import APIConfig
from .TEST import TEST


class Config:
    def __init__(self):
        self.gdb = DBConfig()
        self.llm = LLMConfig()
        self.api = APIConfig()
        self.test = TEST()


@lru_cache()
def get_config():
    return Config()