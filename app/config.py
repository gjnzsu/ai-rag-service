from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str
    jira_url: str = ""
    jira_email: str = ""
    jira_api_token: str = ""
    confluence_url: str = ""
    chroma_persist_dir: str = "./chroma_db"
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    log_level: str = "INFO"


settings = Settings()
