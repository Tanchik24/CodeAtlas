from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from db import SQLiteStoreSA
from db import Repo, User

from app.RepoQAService import RepoQAService 

from app.config import get_config

llm_config = get_config().llm
db_config = get_config().gdb

class EnsureRepoRequest(BaseModel):
    github_url: str = Field(..., examples=["https://github.com/huggingface/pytorch-image-models"])


class RepoResponse(BaseModel):
    id: int
    github_url: str
    server_path: str
    is_indexed: bool
    project_name: str
    collection_name: str


class ChatRequest(BaseModel):
    repo_id: int
    message: str


class ChatResponse(BaseModel):
    answer: str


class HistoryResponse(BaseModel):
    user_id: str
    repo_id: int
    messages_history: List[Dict[str, Any]]


class App:

    def __init__(
        self,
        sqlite_file: str = db_config.sqlite_path,
        repos_dir: str | Path = db_config.repos_dir,
        qdrant_path: str | Path = db_config.qdrant_path,
        model_name: str = llm_config.qwen_model_name,
        cookie_name: str = "uid",
    ) -> None:
        self.cookie_name = cookie_name

        self.store = SQLiteStoreSA(sqlite_file=sqlite_file)
        self.service = RepoQAService(
            store=self.store,
            repos_dir=Path(repos_dir),
            qdrant_path=Path(qdrant_path),
            model_name=model_name,
        )

        self.api = FastAPI(title="RepoQA API")
        self._mount_middlewares()
        self._mount_routes()
        self._mount_lifecycle()


    def fastapi(self) -> FastAPI:
        return self.api


    def _mount_middlewares(self) -> None:
        @self.api.middleware("http")
        async def ensure_user_cookie(request: Request, call_next):
            uid = request.cookies.get(self.cookie_name)
            if not uid:
                uid = uuid.uuid4().hex
                request.state.user_id = uid
                response = await call_next(request)
                response.set_cookie(
                    key=self.cookie_name,
                    value=uid,
                    httponly=True,
                    samesite="lax",
                    max_age=60 * 60 * 24 * 365,
                )
                return response

            request.state.user_id = uid
            return await call_next(request)

    def _user_id(self, request: Request) -> str:
        uid = getattr(request.state, "user_id", None)
        if not uid:
            uid = uuid.uuid4().hex
            request.state.user_id = uid
        return str(uid)

    def _repo_to_response(self, repo: Repo) -> RepoResponse:
        return RepoResponse(
            id=int(repo.id),
            github_url=str(repo.github_url),
            server_path=str(repo.server_path),
            is_indexed=bool(repo.is_indexed),
            project_name=str(repo.project_name),
            collection_name=str(repo.collection_name),
        )

    def _mount_routes(self) -> None:
        @self.api.get("/health")
        def health() -> Dict[str, Any]:
            return {"ok": True}

        @self.api.post("/repos/ensure", response_model=RepoResponse)
        def ensure_repo(req: EnsureRepoRequest, request: Request) -> RepoResponse:
            user_id = self._user_id(request)
            try:
                repo = self.service.ensure_repo(req.github_url)
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

            self.service.get_or_create_user(user_id=user_id, repo_id=int(repo.id))
            return self._repo_to_response(repo)

        @self.api.get("/repos/{repo_id}", response_model=RepoResponse)
        def get_repo(repo_id: int) -> RepoResponse:
            repo = self.store.get_repo_by_id(int(repo_id))
            if repo is None:
                raise HTTPException(status_code=404, detail="repo not found")
            return self._repo_to_response(repo)

        @self.api.post("/repos/{repo_id}/reindex", response_model=RepoResponse)
        def reindex_repo(repo_id: int) -> RepoResponse:
            repo = self.store.get_repo_by_id(int(repo_id))
            if repo is None:
                raise HTTPException(status_code=404, detail="repo not found")

            try:
                self.service.reset_repo_index(repo)
                self.service.index_repo(repo)
                self.store.set_repo_indexed(repo.id, True)
                repo = self.store.get_repo_by_id(int(repo.id)) or repo
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

            return self._repo_to_response(repo)

        @self.api.post("/chat", response_model=ChatResponse)
        def chat(req: ChatRequest, request: Request) -> ChatResponse:
            user_id = self._user_id(request)
            try:
                answer = self.service.ask(
                    user_id=user_id,
                    repo_id=int(req.repo_id),
                    message=str(req.message),
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

            return ChatResponse(answer=answer)

        @self.api.get("/chat/history", response_model=HistoryResponse)
        def chat_history(repo_id: int, request: Request) -> HistoryResponse:
            user_id = self._user_id(request)
            user = self.store.get_or_create_user(user_id=user_id, repo_id=int(repo_id))
            return HistoryResponse(
                user_id=str(user.id),
                repo_id=int(user.repo_id),
                messages_history=list(user.messages_history or []),
            )

        @self.api.post("/chat/clear", response_model=HistoryResponse)
        def chat_clear(repo_id: int, request: Request) -> HistoryResponse:
            user_id = self._user_id(request)
            self.store.get_or_create_user(user_id=user_id, repo_id=int(repo_id))
            self.store.clear_history(user_id=user_id)
            user = self.store.get_user(user_id=user_id)
            if user is None:
                raise HTTPException(status_code=404, detail="user not found")
            return HistoryResponse(
                user_id=str(user.id),
                repo_id=int(user.repo_id),
                messages_history=list(user.messages_history or []),
            )

    def _mount_lifecycle(self) -> None:
        @self.api.on_event("shutdown")
        def _shutdown() -> None:
            try:
                self.service.neo4j.close()
            except Exception:
                pass


def create_app() -> FastAPI:
    return App().fastapi()

app = create_app()