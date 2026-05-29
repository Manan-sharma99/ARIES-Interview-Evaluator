from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "ARIES Interview Evaluation API"
    app_version: str = "0.1.0"
    api_prefix: str = "/api"


settings = Settings()
