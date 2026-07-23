"""请求数据模型

定义 API 请求的 Pydantic 模型
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import config


SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]+$"


class ChatRequest(BaseModel):
    """对话请求"""

    id: str = Field(
        ...,
        description="会话 ID",
        alias="Id",
        min_length=1,
        max_length=config.max_session_id_length,
        pattern=SESSION_ID_PATTERN,
    )
    question: str = Field(
        ...,
        description="用户问题",
        alias="Question",
        min_length=1,
        max_length=config.max_question_length,
    )

    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "Id": "session-123",
                "Question": "什么是向量数据库？"
            }
        },
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("问题不能为空")
        return value


class ClearRequest(BaseModel):
    """清空会话请求"""

    session_id: str = Field(
        ...,
        description="会话 ID",
        alias="sessionId",
        min_length=1,
        max_length=config.max_session_id_length,
        pattern=SESSION_ID_PATTERN,
    )

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)
