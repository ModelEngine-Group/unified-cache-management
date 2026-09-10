"""Version-tolerant access to SGLang model inspection APIs."""

from .inspector import collect_model_facts, environment_info, refine_facts_from_model

__all__ = ["collect_model_facts", "environment_info", "refine_facts_from_model"]
