
from __future__ import annotations

from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="LLM_",
        populate_by_name=True,
        extra="ignore",
    )

    mistral_api_key: str = Field(
        ...,
        validation_alias=AliasChoices("MISTRAL_API_KEY", "LLM_MISTRAL_API_KEY"),
    )
    mistral_model: str = Field(
        "mistral-large-latest",
        validation_alias=AliasChoices("LLM_MISTRAL_MODEL", "MISTRAL_MODEL"),
    )
    temperature: float = Field(0.0)