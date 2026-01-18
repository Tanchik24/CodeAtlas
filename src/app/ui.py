import os
import requests
import streamlit as st
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_config

from langchain.messages import HumanMessage, AIMessage

config = get_config().api


def render_answer_feedback(answer_id: str) -> None:
    if answer_id in st.session_state.feedback:
        fb = st.session_state.feedback[answer_id]
        st.caption(f"✅ Оценка сохранена: {fb['rating']}/5 ({fb['bucket']})")
        return

    with st.form(key=f"feedback_{answer_id}", clear_on_submit=True):
        rating = st.select_slider(
            "Оценка ответа (1 — плохо, 5 — отлично)",
            options=[1, 2, 3, 4, 5],
            value=4,
        )

        bucket = "очень полезно" if rating == 5 else ("полезно" if rating == 4 else "не полезно")
        st.caption(f"Категория: **{bucket}**")

        comment = st.text_input("Комментарий (опционально)")

        if st.form_submit_button("Отправить оценку"):
            st.session_state.feedback[answer_id] = {
                "rating": int(rating),
                "bucket": bucket,
                "comment": comment,
            }
            st.success("Спасибо! Оценка сохранена.")
            st.rerun()


def _get_api_session() -> requests.Session:
    if "api_session" not in st.session_state:
        st.session_state.api_session = requests.Session()
    return st.session_state.api_session


def _api_base_url() -> str:
    return str(
        st.session_state.get("api_base_url")
        or f"http://{config.host}:{config.port}"
    ).rstrip("/")


def api_get(path: str, params: dict | None = None, timeout: int = 300) -> dict:
    session = _get_api_session()
    url = f"{_api_base_url()}{path}"
    r = session.get(url, params=params or {}, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"GET {path} -> {r.status_code}: {r.text}")
    return r.json()


def api_post(path: str, json: dict | None = None, timeout: int = 300) -> dict:
    s = _get_api_session()
    url = f"{_api_base_url()}{path}"
    if json is None:
        r = s.post(url, timeout=timeout)
    else:
        r = s.post(url, json=json, timeout=timeout)
    if not r.ok:
        raise RuntimeError(f"POST {path} -> {r.status_code}: {r.text}")
    return r.json()


def _history_dicts_to_messages(hist: list[dict]) -> list:
    out = []
    for m in hist or []:
        role = str(m.get("role") or "").strip().lower()
        content = str(m.get("content") or  "")
        if not content.strip():
            continue
        if role == "user":
            out.append(HumanMessage(content))
        else:
            out.append(AIMessage(content))
    return out


def _load_history(repo_id: int) -> None:
    data = api_get("/chat/history", params={"repo_id": int(repo_id)}, timeout=120)
    hist = data.get("messages_history") or []
    st.session_state.chat_history = _history_dicts_to_messages(hist)


def render_main_page() -> None:
    st.title("💻 GitHub Repository Q&A Assistant")

    with st.expander("⚙️ Settings", expanded=False):
        st.session_state.api_base_url = st.text_input(
            "API Base URL",
            value=st.session_state.get("api_base_url", _api_base_url()),
        )
        if st.button("✅ Health check"):
            try:
                st.success(api_get("/health", timeout=30))
            except Exception as e:
                st.error(f"Health failed: {e}")

    st.markdown("""
    Welcome! This tool allows you to:
    1. Ensure (clone) a public GitHub repository on the server
    2. Index it on the server
    3. Ask questions about the repository
    """)

    with st.form("github_url_form"):
        github_url = st.text_input(
            "GitHub Repository URL",
            placeholder="https://github.com/username/repository",
        )
        submitted = st.form_submit_button("Download Repository")

        repo_path = getattr(getattr(st.session_state, "app_config", None), "save_dir", None)

        if submitted and github_url:
            if not github_url.startswith("https://github.com/"):
                st.error("Please enter a valid GitHub URL starting with https://github.com/")
            else:
                with st.spinner(f"Ensuring repository on server: {github_url}..."):
                    try:
                        repo = api_post("/repos/ensure", json={"github_url": github_url}, timeout=1800)
                        st.session_state.repo = repo
                        st.session_state.repo_id = int(repo["id"])
                        st.session_state.repo_is_indexed = bool(repo.get("is_indexed", True))

                        try:
                            _load_history(st.session_state.repo_id)
                        except Exception:
                            st.session_state.chat_history = []

                        st.success("✅ Repository ensured and processed successfully!")

                        with st.expander("Repository Info"):
                            st.code(
                                f"id: {repo.get('id')}\n"
                                f"url: {repo.get('github_url')}\n"
                                f"server_path: {repo.get('server_path')}\n"
                                f"is_indexed: {repo.get('is_indexed')}\n"
                                f"project_name: {repo.get('project_name')}\n"
                                f"collection_name: {repo.get('collection_name')}\n",
                                language="text",
                            )

                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    if st.session_state.repo_is_indexed and st.button("🚀 Proceed to Chat"):
        st.session_state.current_page = "chat"
        st.rerun()


def render_chat_page() -> None:
    st.title("💬 Repository Chat")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Ask questions about the repository")
    with col2:
        if st.button("⬅️ Back to Main"):
            st.session_state.current_page = "main"
            st.session_state.app_is_inited = False
            st.rerun()

    repo_id = st.session_state.get("repo_id")
    if not repo_id:
        st.warning("No repository selected yet. Go back and enter a GitHub URL.")
        return

    col_a, col_b, _spacer = st.columns([1, 1, 8], gap="small")
    with col_a:
        if st.button("🔄 Refresh history"):
            try:
                _load_history(int(repo_id))
                st.rerun()
            except Exception as e:
                st.error(f"History refresh failed: {e}")

    with col_b:
        if st.button("🧹 Clear history"):
            try:
                api_post(f"/chat/clear?repo_id={int(repo_id)}", json=None, timeout=120)
                st.session_state.chat_history = []
                st.rerun()
            except Exception as e:
                st.error(f"Clear failed: {e}")

    for i, message in enumerate(st.session_state.chat_history):
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

        if role == "assistant" and str(message.content).strip():
            answer_id = f"{int(repo_id)}:{i}" 
            render_answer_feedback(answer_id)


    if "pending_retry" in st.session_state:
        st.error(st.session_state.pending_retry["error_display"])
        col_retry, col_cancel = st.columns(2)

        with col_retry:
            if st.button("🔄 Retry Last Question"):
                prompt = st.session_state.pending_retry.get("prompt")
                del st.session_state.pending_retry
                if prompt:
                    process_chat_message(prompt)
                st.rerun()

        with col_cancel:
            if st.button("❌ Cancel Retry"):
                del st.session_state.pending_retry
                st.rerun()

    if prompt := st.chat_input("Ask a question about the repository..."):
        process_chat_message(prompt)


def process_chat_message(prompt: str) -> None:
    repo_id = st.session_state.get("repo_id")
    if not repo_id:
        st.error("No repo_id. Go back to Main and ensure a repo first.")
        return

    st.session_state.chat_history.append(HumanMessage(prompt))

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = api_post("/chat", json={"repo_id": int(repo_id), "message": str(prompt)}, timeout=1800)
                response = str(resp.get("answer", "")).strip()
                st.markdown(response or "_(empty answer)_")

                st.session_state.chat_history.append(AIMessage(response))

                try:
                    _load_history(int(repo_id))
                except Exception:
                    pass

                if "pending_retry" in st.session_state:
                    del st.session_state.pending_retry

            except Exception as e:
                error_display = f"""
                **I encountered an issue processing your request:**

                `{str(e)}`

                You can try:
                - Rephrasing your question
                - Breaking it down into simpler parts
                - Retrying
                """
                st.session_state.pending_retry = {
                    "prompt": prompt,
                    "error": str(e),
                    "error_display": error_display,
                }

            st.rerun()


def init_ui(app_config=None, llm_config=None, secrets=None) -> None:
    if st.session_state.get("app_is_inited"):
        return

    st.set_page_config(
        page_title="GitHub Repository Q&A",
        page_icon="💻",
        layout="wide",
    )

    st.session_state.app_config = app_config
    st.session_state.model_config = llm_config

    st.session_state.api_base_url = st.session_state.get("api_base_url") or f"http://{config.host}:{config.port}"
    st.session_state.repo_is_indexed = False
    st.session_state.chat_history = []
    st.session_state.current_page = "main"

    st.session_state.app_is_inited = True
    st.session_state.feedback = {}


def run_app() -> None:
    init_ui()

    if st.session_state.current_page == "main":
        render_main_page()
    elif st.session_state.current_page == "chat":
        render_chat_page()


run_app()
