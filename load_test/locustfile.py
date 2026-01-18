import json
import random
from locust import HttpUser, task, between

from src.app.config import get_config

config = get_config().test

with open(config.questions_path, "r", encoding="utf-8") as f:
    data = json.load(f)

QUESTIONS = [q.get("query", "").strip() for q in data.get("questions", []) if (q.get("query") or "").strip()]

if not QUESTIONS:
    raise RuntimeError(f"No questions loaded from questions_path={config.questions_path}")

TIMEOUT_S = int(getattr(config, "timeout_s", 120))


class ChatOnlyUser(HttpUser):
    wait_time = between(float(config.think_min), float(config.think_max))

    @task
    def chat(self):
        msg = random.choice(QUESTIONS)

        with self.client.post(
            "/chat",
            json={"repo_id": int(config.repo_id), "message": msg},
            name="/chat",
            timeout=TIMEOUT_S,
            catch_response=True,
            headers={"Accept": "application/json"},
        ) as result:
            if result.status_code != 200:
                result.failure(f"HTTP {result.status_code}: {result.text[:200]}")
                return

            try:
                j = result.json()
                if not isinstance(j, dict) or not str(j.get("answer", "")).strip():
                    result.failure("bad response: missing/empty 'answer'")
            except Exception as e:
                result.failure(f"bad json: {e}")