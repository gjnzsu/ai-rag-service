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
    retrieval_final_top_k: int = 20
    retrieval_rrf_k: int = 60
    retrieval_score_threshold: float | None = None
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    log_level: str = "INFO"


settings = Settings()
