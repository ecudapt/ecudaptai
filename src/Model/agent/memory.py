# src/Model/agent/memory.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CarContext:
    year: Optional[int] = None
    make: Optional[str] = None
    model: Optional[str] = None
    engine_code: Optional[str] = None
    fuel_type: Optional[str] = None  # e.g. "93", "E30", "E50"


@dataclass
class SessionMemory:
    """
    In-memory representation of one user's ECU context.
    Later, you can persist this to Supabase keyed by user/session id.
    """
    car: CarContext = field(default_factory=CarContext)
    mods: List[str] = field(default_factory=list)
    goal: Optional[str] = None          # e.g. "reliable street", "max power drag"
    risk_tolerance: str = "normal"      # "low", "normal", "high"
    past_issues: List[str] = field(default_factory=list)  # plain-text notes

    def update_from_message(self, message: str) -> None:
        """
        Very simple placeholder.

        Later:
        - You can parse lines like 'Car: 2008 335i N54, 6MT on 93 octane'
        - Or call a small LLM with a 'extract_car_context' prompt.

        For now, we just keep this stub so the agent code is wired.
        """
        # TODO: implement heuristics / NLP-based extraction.
        # Example heuristic: if '335i' or 'N54' appears, set engine_code / model.
        lowered = message.lower()
        if "n54" in lowered and not self.car.engine_code:
            self.car.engine_code = "N54"
        if "335i" in lowered and not self.car.model:
            self.car.model = "335i"

        # You can also append to past_issues here based on keywords like 'misfire', 'knock', etc.

    def to_summary(self) -> str:
        """
        Serialize memory into a text block for the prompt.
        """
        lines: List[str] = []

        # Car
        car_bits = []
        if self.car.year:
            car_bits.append(str(self.car.year))
        if self.car.make:
            car_bits.append(self.car.make)
        if self.car.model:
            car_bits.append(self.car.model)
        if self.car.engine_code:
            car_bits.append(f"({self.car.engine_code})")
        if self.car.fuel_type:
            car_bits.append(f"fuel: {self.car.fuel_type}")

        if car_bits:
            lines.append("Car: " + " ".join(car_bits))
        else:
            lines.append("Car: unknown (user has not specified yet)")

        # Mods
        if self.mods:
            lines.append("Mods: " + ", ".join(self.mods))
        else:
            lines.append("Mods: not explicitly specified yet")

        # Goal
        lines.append(f"Goal: {self.goal or 'not specified'}")

        # Risk tolerance
        lines.append(f"Risk tolerance: {self.risk_tolerance}")

        # Past issues
        if self.past_issues:
            lines.append("Past issues noted:")
            for issue in self.past_issues[-5:]:
                lines.append(f"  - {issue}")
        else:
            lines.append("Past issues: none recorded")

        return "\n".join(lines)
