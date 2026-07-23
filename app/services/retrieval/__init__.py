"""检索子包 — Top-20召回、单次Rerank、索引邻居扩展和上下文构建

对外入口：
    from app.services.retrieval import retrieval_service
    context, artifact = await retrieval_service.retrieve(query="...")
"""

from app.services.retrieval.retrieval_service import retrieval_service

__all__ = ["retrieval_service"]
