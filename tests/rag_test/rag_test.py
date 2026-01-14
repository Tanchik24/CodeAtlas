import json
import subprocess
import hashlib
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Any, Tuple

import pandas as pd
from pydantic import BaseModel, conint
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

from src.app.config import get_config
from src.app.languages.LanguageRegistery import LanguageRegistry
from src.services.Neo4jIngestor import Neo4jIngestor
from src.services.CodeEmbeddingsStore import CodeEmbeddingsStore
from src.app.entities import Project
from src.app.code_indexer.CodebaseIndexer import CodebaseIndexer
from src.app.rag.agent import CodeRepoToolAgent


load_dotenv()


@dataclass
class Question:
    text: str
    response: str


@dataclass
class Repository:
    url: str
    name: str
    questions: list[Question]
    collection: Optional[str] = None
    path: Optional[Path] = None


class RagTester:
    def __init__(self) -> None:
        self.config = get_config()
        self.test_data_dir = Path(self.config.gdb.rag_test_dir).resolve()
        self.repos_root = Path(self.config.gdb.rag_test_repos_dir).resolve()
        self.qdrant_path = Path(self.config.gdb.qdrant_path).resolve()

        self.language_registry = LanguageRegistry()
        self._languages_ready = False

        self.neo4j = Neo4jIngestor()
        self.top_k = 5

    def run(self) -> Tuple[dict[str, dict[str, float]], dict[str, float]]:
        self._ensure_dirs_exist()
        repos = self._load_test_files()

        per_repo: dict[str, dict[str, float]] = {}
        overall_lists: dict[str, list[float]] = {
            "avg_correctness": [],
            "avg_completeness": [],
            "avg_precision": [],
            "avg_refusal_appropriateness": [],
            "avg_actionability": [],
            "avg_clarity": [],
            "avg_answer_latency_s": [],
            "avg_judge_latency_s": [],
            "n_questions": [],
        }

        details_rows: list[dict[str, Any]] = []

        for repo in repos:
            repo.path = self._ensure_repo_cloned(repo.url, repo.name)
            repo.collection = self._collection_name(repo.url)

            self._reset_repo_index(repo)

            embedder, emb_store = self._index_repo(repo)
            agent = self._get_agent(repo, embedder, emb_store)

            scores_correctness: list[float] = []
            scores_completeness: list[float] = []
            scores_precision: list[float] = []
            scores_refusal: list[float] = []
            scores_actionability: list[float] = []
            scores_clarity: list[float] = []
            answer_latencies: list[float] = []
            judge_latencies: list[float] = []

            for question in repo.questions:
                q_text = (question.text or "").strip()
                gold = (question.response or "").strip()
                if not q_text:
                    continue

                agent.new_thread()

                t0 = time.perf_counter()
                model_answer = agent.ask(q_text)
                answer_latencies.append(time.perf_counter() - t0)

                t1 = time.perf_counter()
                judge_scores = self._llm_as_judge(q_text, gold, model_answer)
                judge_latencies.append(time.perf_counter() - t1)

                details_rows.append(
                    {
                        "repo": repo.name,
                        "question": q_text,
                        "golden_answer": gold,
                        "model_answer": model_answer,
                        **(judge_scores or {}),
                    }
                )

                scores_correctness.append(float(judge_scores.get("correctness", 0.0) or 0.0))
                scores_completeness.append(float(judge_scores.get("completeness", 0.0) or 0.0))
                scores_precision.append(float(judge_scores.get("precision", 0.0) or 0.0))
                scores_refusal.append(float(judge_scores.get("refusal_appropriateness", 0.0) or 0.0))
                scores_actionability.append(float(judge_scores.get("actionability", 0.0) or 0.0))
                scores_clarity.append(float(judge_scores.get("clarity", 0.0) or 0.0))

            n_q = len(repo.questions)
            denom = max(1, n_q)

            metrics = {
                "n_questions": float(n_q),

                "avg_correctness": float(sum(scores_correctness) / denom) if scores_correctness else 0.0,
                "avg_completeness": float(sum(scores_completeness) / denom) if scores_completeness else 0.0,
                "avg_precision": float(sum(scores_precision) / denom) if scores_precision else 0.0,
                "avg_refusal_appropriateness": float(sum(scores_refusal) / denom) if scores_refusal else 0.0,
                "avg_actionability": float(sum(scores_actionability) / denom) if scores_actionability else 0.0,
                "avg_clarity": float(sum(scores_clarity) / denom) if scores_clarity else 0.0,

                "avg_answer_latency_s": float(sum(answer_latencies) / len(answer_latencies)) if answer_latencies else 0.0,
                "avg_judge_latency_s": float(sum(judge_latencies) / len(judge_latencies)) if judge_latencies else 0.0,
            }

            per_repo[repo.name] = metrics

            for k, v in metrics.items():
                if k in overall_lists:
                    overall_lists[k].append(float(v))

            try:
                self._reset_repo_index(repo)
            except Exception:
                pass

        overall: dict[str, float] = {}
        for k, xs in overall_lists.items():
            overall[k] = float(sum(xs) / len(xs)) if xs else 0.0
        overall["n_repos"] = float(len(per_repo))
        try:
            import shutil
            shutil.rmtree(self.repos_root, ignore_errors=True)
        except Exception:
            pass

        self._save_reports(per_repo, overall, details_rows)
        return per_repo, overall

    def _save_reports(
        self,
        per_repo: dict[str, dict[str, float]],
        overall: dict[str, float],
        details_rows: list[dict[str, Any]],
    ) -> None:
        try:
            out_dir = self.test_data_dir.parent 

            df_per_repo = pd.DataFrame.from_dict(per_repo, orient="index").reset_index()
            df_per_repo = df_per_repo.rename(columns={"index": "repo"})
            df_per_repo.to_csv(out_dir / "per_repo_metrics.csv", index=False)

            df_overall = pd.DataFrame([overall])
            df_overall.to_csv(out_dir / "overall_metrics.csv", index=False)

            df_details = pd.DataFrame(details_rows)
            if not df_details.empty:
                df_details = df_details.set_index(["repo", "question"])
            df_details.to_csv(out_dir / "per_question_details.csv", index=True)
        except Exception:
            pass

    def _llm_as_judge(self, question: str, golden_answer: str, model_answer: str) -> dict[str, Any]:
        Score = conint(ge=1, le=5)

        class JudgeResult(BaseModel):
            correctness: Score
            completeness: Score
            precision: Score
            refusal_appropriateness: Score
            actionability: Score
            clarity: Score

        system_prompt = (
        "You are an evaluator (LLM-as-a-judge). "
        "Return ONLY valid JSON with EXACT keys: "
        "correctness, completeness, precision, refusal_appropriateness, actionability, clarity. "
        "Each value must be an integer 1..5.\n"
        "Rubric:\n"
        "- correctness - 5: fully correct, 1: incorrect\n"
        "- completeness - 5: covers the entire golden answer, 1: covers nothing from the golden answer\n"
        "- precision - 5: strictly to the point, 1: goes off-topic\n"
        "- refusal_appropriateness - 5: refuses only when it should, 1: refuses when an answer exists\n"
        "- actionability - 5: can be applied immediately, 1: useless\n"
        "- clarity - 5: very clear, 1: chaotic\n"
        "No extra keys. No text outside JSON.\n"
        'Output format example: {"correctness":1,"completeness":1,"precision":1,'
        '"refusal_appropriateness":1,"actionability":1,"clarity":1}'
    )

        llm = ChatMistralAI(
            model=self.config.llm.mistral_model,
            temperature=self.config.llm.temperature,
            api_key=self.config.llm.mistral_api_key,
            max_tokens=80,
        )

        judge = llm.with_structured_output(JudgeResult, method="json_mode")

        user_msg = (
            "Question:\n"
            f"{question}\n\n"
            "Golden answer:\n"
            f"{golden_answer}\n\n"
            "Model answer:\n"
            f"{model_answer}\n"
        )

        result = judge.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ]
        )
        return result.model_dump() if hasattr(result, "model_dump") else dict(result)

    def _ensure_dirs_exist(self) -> None:
        self.repos_root.mkdir(parents=True, exist_ok=True)

    def _load_test_files(self) -> list[Repository]:
        repos: list[Repository] = []
        json_files = sorted(self.test_data_dir.glob("*.json"))

        for fp in json_files:
            data = self._read_json(fp)
            if not isinstance(data, dict):
                continue

            url = str(data.get("url") or "").strip()
            if not url:
                continue

            name = self._derive_name_from_url(url)
            raw_questions = data.get("questions") or []

            questions: list[Question] = []
            for q in raw_questions:
                if not isinstance(q, dict):
                    continue

                text = str(q.get("query") or "").strip()
                if not text:
                    continue

                response = str(q.get("response") or "").strip()

                questions.append(Question(text=text, response=response))

            repos.append(Repository(url=url, name=name, questions=questions))

        return repos

    def _read_json(self, path: Path) -> Any:
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _derive_name_from_url(self, url: str) -> str:
        s = url.strip().rstrip("/").split("/")[-1].strip()
        if s.endswith(".git"):
            s = s[:-4]
        return s or "unknown_repo"

    def _ensure_repo_cloned(self, url: str, repo_name: str) -> Path:
        dest = (self.repos_root / repo_name).resolve()

        if dest.exists() and (dest / ".git").exists():
            return dest

        if dest.exists() and not (dest / ".git").exists():
            raise RuntimeError(f"Destination exists but is not a git repo: {dest}")

        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
            cwd=str(self.repos_root),
            check=True,
        )
        return dest

    def _collection_name(self, github_url: str) -> str:
        h = hashlib.sha1(github_url.strip().encode("utf-8")).hexdigest()[:10]
        return f"code_embeddings_{h}"

    def _reset_repo_index(self, repo: Repository) -> None:
        if not repo.collection:
            raise RuntimeError("repo.collection is None/empty")

        q = """
        MATCH (p:Project {name:$name})
        OPTIONAL MATCH (p)-[*0..]->(n)
        WITH collect(DISTINCT p) + collect(DISTINCT n) AS nodes
        UNWIND nodes AS x
        WITH DISTINCT x
        DETACH DELETE x
        """
        self.neo4j.fetch_all(q, {"name": repo.name})

        store = CodeEmbeddingsStore(collection_name=repo.collection, path=self.qdrant_path)
        try:
            store.clear()
        finally:
            store.close()

    def _get_agent(self, repo: Repository, embedder: Any, emb_store: CodeEmbeddingsStore) -> CodeRepoToolAgent:
        return CodeRepoToolAgent(
                project_root=Path(repo.path),
                neo4j_ingestor=self.neo4j,
                store=emb_store,
                embedder=embedder,
                qwen_model=self.config.llm.qwen_model_name,
                top_k=8,
            )

    def _index_repo(self, repo: Repository) -> Tuple[Any, CodeEmbeddingsStore]:
        if not self._languages_ready:
            self.language_registry.auto_register()
            self._languages_ready = True

        project = Project(name=repo.name, path=repo.path)
        emb_store = CodeEmbeddingsStore(collection_name=repo.collection, path=self.qdrant_path)

        indexer: Optional[CodebaseIndexer] = None
        indexer = CodebaseIndexer(
            project=project,
            neo4j_ingestor=self.neo4j,
            language_registry=self.language_registry,
            emb_store=emb_store,
        )
        indexer.index_codebase()

        if indexer.embedder is None:
            raise RuntimeError(f"Embedder is None after indexing repo={repo.name}")

        return indexer.embedder, emb_store


if __name__ == "__main__":
    RagTester().run()
