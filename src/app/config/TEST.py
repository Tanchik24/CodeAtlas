from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TEST(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="TEST_",
        populate_by_name=True,
        extra="ignore",
    )

    questions_path: str = Field("load_test/questions.json")
    repo_id: int = Field(0)
    think_min: float = Field(0.2)
    think_max: float = Field(0.6)