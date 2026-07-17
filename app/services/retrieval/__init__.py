"""检索子包 — 超额召回、Rerank、来源多样性和上下文构建

对外入口：
    from app.services.retrieval import retrieval_service
    context, artifact = await retrieval_service.retrieve(query="...")
"""

from app.services.retrieval.retrieval_service import retrieval_service

__all__ = ["retrieval_service"]
