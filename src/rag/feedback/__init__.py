"""Feedback collection, analytics and (future) training-data preparation.

The feedback loop NEVER modifies model weights. It feeds analytics, retrieval
optimization, query expansion and semantic caching, and exports clean rows for
a future LoRA / DPO / reranker training pipeline.
"""
