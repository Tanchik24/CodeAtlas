from __future__ import annotations

from typing import Any, Dict, List

from sqlalchemy import String, Integer, Boolean, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    pass


class Repo(Base):
    __tablename__ = "repo"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    github_url: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    server_path: Mapped[str] = mapped_column(String, nullable=False)
    is_indexed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    project_name: Mapped[str] = mapped_column(String, nullable=False)
    collection_name: Mapped[str] = mapped_column(String, nullable=False)

    users: Mapped[List["User"]] = relationship(
        back_populates="repo",
        cascade="all, delete-orphan",
    )


class User(Base):
    __tablename__ = "user"

    id: Mapped[str] = mapped_column(String, primary_key=True)  
    repo_id: Mapped[int] = mapped_column(Integer, ForeignKey("repo.id", ondelete="CASCADE"), nullable=False)

    messages_history: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)

    repo: Mapped[Repo] = relationship(back_populates="users")
