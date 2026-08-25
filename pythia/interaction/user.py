from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Tuple

from .items import InteractionItem
from .items import Message
from .items import UserInteractionBoundary

if TYPE_CHECKING:
    from .display import DisplayItem


@dataclass(frozen=True)
class UserInteraction:
    items: Tuple[InteractionItem, ...]

    def __post_init__(self) -> None:
        items = tuple(self.items)
        if not items:
            raise ValueError("user interaction must contain at least one item")
        for index, item in enumerate(items):
            if not isinstance(item, Message):
                raise ValueError(
                    "user interaction items must be user messages; "
                    f"item {index} is {type(item).__name__}"
                )
            if item.role != "user":
                raise ValueError(
                    "user interaction messages must have role 'user'; "
                    f"item {index} has role {item.role!r}"
                )
        object.__setattr__(self, "items", items)

    def context_items(self) -> Tuple[InteractionItem, ...]:
        """Return this interaction's items followed by its durable boundary."""
        return (*self.items, UserInteractionBoundary())

    def display_items(self) -> Tuple["DisplayItem", ...]:
        from .display import render_interaction_items

        return render_interaction_items(self.items)


__all__ = [
    "UserInteraction",
]
