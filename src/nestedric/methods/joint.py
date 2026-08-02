"""Joint training on all environments at once: the oracle upper bound.

Not a continual-learning method -- it is handed the union of every environment, which no
deployed xApp could have. It bounds what the backbone can express, so a gap between
`joint` and the best continual method is a capacity ceiling rather than a learning-rule
failure. The trainer recognises `wants_joint_data` and feeds it accordingly.
"""

from __future__ import annotations

from nestedric.methods import register
from nestedric.methods.base import SgdMethod


@register("joint")
class Joint(SgdMethod):
    """Joint training on all environments (upper bound / oracle)."""

    #: Read by the trainer, which then supplies batches drawn from every environment.
    wants_joint_data = True
