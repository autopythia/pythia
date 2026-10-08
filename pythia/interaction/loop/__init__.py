"""The shared turn loop and the supervision protocol, as reusable parts.

The package imports only lower layers of ``pythia.interaction`` (never
``auto``, ``cli`` or ``demo``), writes no files and prints nothing; hosts do.
"""

from .supervision import Fault
from .supervision import SupervisedHandle
from .supervision import YIELD_KINDS
from .supervision import Yield
from .supervision import YieldChannel
from .supervision import YieldTool
from .supervision import run_supervised_task
from .supervision import supervise
from .supervision import supervised_turn
from .kernel import Interrupt
from .kernel import MissingFinalText
from .kernel import SampleLimitExceeded
from .kernel import Steer
from .kernel import TurnHost
from .kernel import TurnResult
from .kernel import run_turn
from .preemption import Preemption
from .preemption import Urgency
from .preemption import command_urgency
from .preemption import escalated_stop
from .tail import TurnTail
from .tail import turn_tail
from .tail import unfinished_turn

__all__ = [
    "Fault",
    "Interrupt",
    "MissingFinalText",
    "Preemption",
    "SampleLimitExceeded",
    "Steer",
    "SupervisedHandle",
    "TurnHost",
    "TurnResult",
    "TurnTail",
    "Urgency",
    "YIELD_KINDS",
    "Yield",
    "YieldChannel",
    "YieldTool",
    "command_urgency",
    "escalated_stop",
    "run_supervised_task",
    "run_turn",
    "supervise",
    "supervised_turn",
    "turn_tail",
    "unfinished_turn",
]
