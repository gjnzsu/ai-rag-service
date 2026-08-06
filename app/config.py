from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str
    jira_url: str = ""
    jira_email: str = ""
    jira_api_token: str = ""
    confluence_url: str = ""
    chroma_persist_dir: str = "./chroma_db"
    lexical_db_path: str = "./lexical.db"
    lexical_issue_key_weight: float = 10.0
    lexical_title_weight: float = 5.0
    lexical_content_weight: float = 1.0
    jira_key_pattern: str = r"\b[A-Z][A-Z0-9]+-\d+\b"
    retrieval_mode: str = "hybrid"
    retrieval_candidate_top_k: int = 30
    retrieval_final_top_k: int = 10
    retrieval_rrf_k: int = 60
    retrieval_score_threshold: float | None = None
    reranker_provider: Literal["none", "openai", "qwen_local"] = "none"
    reranker_openai_model: str = "gpt-5"
    reranker_openai_timeout_seconds: float = 5.0
    reranker_qwen_model: str = "Qwen/Qwen3-Reranker-0.6B"
    reranker_qwen_revision: str = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
    reranker_qwen_max_candidates: int = 20
    reranker_qwen_max_length: int = 512
    reranker_qwen_batch_size: int = 4
    reranker_qwen_timeout_seconds: float = 5.0
    reranker_qwen_circuit_breaker_seconds: float = 30.0
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    log_level: str = "INFO"


settings = Settings()
