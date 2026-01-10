from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class APIConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="API_",
        populate_by_name=True,
        extra="ignore",
    )

    host: str = Field("0.0.0.0")
    port: int = Field(8000)
    ui_port: int = Field(8501)