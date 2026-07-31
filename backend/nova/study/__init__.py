"""Studies you can run with NOVA, as opposed to features NOVA has.

These exist because parametric robots make certain questions cheap to ask.
`cross_body` is the first: train on one machine, measure on machines the policy
has never seen.
"""

from __future__ import annotations

__all__ = ["cross_body"]
