from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import sessionmaker

from .models import Base, Repo, User


class SQLiteStoreSA:
    def __init__(self, sqlite_file: str = "app.sqlite") -> None:
        self.engine = create_engine(f"sqlite:///{sqlite_file}", future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(self.engine, expire_on_commit=False, future=True)


    def upsert_repo(
        self,
        github_url: str,
        server_path: str,
        project_name: str,
        collection_name: str,
        is_indexed: bool = False,
    ) -> Repo:
        with self.Session() as s:
            repo = s.scalar(select(Repo).where(Repo.github_url == github_url))
            if repo is None:
                repo = Repo(
                    github_url=github_url,
                    server_path=server_path,
                    is_indexed=is_indexed,
                    project_name=project_name,
                    collection_name=collection_name,
                )
                s.add(repo)
            else:
                repo.server_path = server_path
                repo.project_name = project_name
                repo.collection_name = collection_name
                repo.is_indexed = is_indexed

            s.commit()
            return repo

    def get_repo_by_id(self, repo_id: int) -> Optional[Repo]:
        with self.Session() as s:
            return s.get(Repo, int(repo_id))

    def get_repo_by_url(self, github_url: str) -> Optional[Repo]:
        with self.Session() as s:
            return s.scalar(select(Repo).where(Repo.github_url == github_url))

    def set_repo_indexed(self, repo_id: int, is_indexed: bool) -> None:
        with self.Session() as s:
            s.execute(update(Repo).where(Repo.id == int(repo_id)).values(is_indexed=bool(is_indexed)))
            s.commit()

    def get_or_create_user(self, user_id: str, repo_id: int) -> User:
        with self.Session() as s:
            user = s.get(User, user_id)
            if user is None:
                user = User(id=user_id, repo_id=int(repo_id), messages_history=[])
                s.add(user)
                s.commit()
                return user

            if user.repo_id != int(repo_id):
                user.repo_id = int(repo_id)
                s.commit()
            return user

    def get_user(self, user_id: str) -> Optional[User]:
        with self.Session() as s:
            return s.get(User, user_id)

    def append_message(self, user_id: str, role: str, content: str) -> User:
        with self.Session() as s:
            user = s.get(User, user_id)
            if user is None:
                raise ValueError(f"user not found: {user_id}")

            history: List[Dict[str, Any]] = list(user.messages_history or [])
            history.append({"role": role, "content": content})
            user.messages_history = history

            s.commit()
            return user

    def clear_history(self, user_id: str) -> None:
        with self.Session() as s:
            user = s.get(User, user_id)
            if user is None:
                return
            user.messages_history = []
            s.commit()
