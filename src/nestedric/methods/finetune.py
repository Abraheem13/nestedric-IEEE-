"""Naive sequential fine-tuning: the lower bound and the forgetting reference.

Deliberately does nothing to protect old environments. Its |BWT| is the phenomenon the
paper is about -- if this method does not forget on a stream, there is nothing there to
improve on, which is exactly what the Day 4 gate checks.
"""

from __future__ import annotations

from nestedric.methods import register
from nestedric.methods.base import SgdMethod


@register("finetune")
class Finetune(SgdMethod):
    """Naive sequential fine-tuning (lower bound / forgetting reference)."""
