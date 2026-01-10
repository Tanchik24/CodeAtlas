from pathlib import Path
from pydantic import BaseModel


class Project(BaseModel):
    name: str 
    path: Path