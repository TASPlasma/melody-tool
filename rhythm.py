from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Optional


@dataclass(frozen=True, slots=True)
class Rhythm:
    """
    Duration representation with conversions between:
    - beats (quarter-note beats)
    - seconds (requires tempo_bpm)
    - MIDI ticks (requires ppq)
    - symbols (requires a mapping, default supports common note values)

    Convention:
    - 1 beat = 1 quarter note.
    - tempo_bpm is quarter-notes per minute.
    - ppq is ticks per quarter note (a.k.a. TPQN).

    Implementation notes:
    - We store `beats` as a `fractions.Fraction` so dotted/triplet values can be exact.
    - Conversions that must be integer (ticks) round to the nearest tick.
    """

    beats: Fraction

    # -------- constructors --------
    @staticmethod
    def from_beats(beats: float | int | Fraction) -> "Rhythm":
        # Accept floats/ints for convenience but store as a Fraction for stable conversions.
        if isinstance(beats, Fraction):
            b = beats
        else:
            b = Fraction(beats).limit_denominator(1920)
        if b <= 0:
            raise ValueError("beats must be > 0")
        return Rhythm(b)

    @staticmethod
    def from_seconds(seconds: float, *, tempo_bpm: float) -> "Rhythm":
        # Convert real time to musical time using tempo.
        if seconds <= 0:
            raise ValueError("seconds must be > 0")
        bpm = float(tempo_bpm)
        if bpm <= 0:
            raise ValueError("tempo_bpm must be > 0")
        beats = Fraction(seconds * bpm / 60.0).limit_denominator(1920)
        return Rhythm.from_beats(beats)

    @staticmethod
    def from_ticks(ticks: int, *, ppq: int) -> "Rhythm":
        # Convert MIDI ticks to beats using PPQ (ticks per quarter note).
        if ticks <= 0:
            raise ValueError("ticks must be > 0")
        if ppq <= 0:
            raise ValueError("ppq must be > 0")
        beats = Fraction(int(ticks), int(ppq))
        return Rhythm.from_beats(beats)

    @staticmethod
    def from_symbol(symbol: str) -> "Rhythm":
        """
        Supported symbols:
        - Fractional strings: "1/1" (whole), "1/2" (half), "1/4" (quarter), "1/8", ...
          with optional dotted "." and triplet "t" suffix: "1/8.", "1/8t"
        - Shorthands: "w" (whole), "h" (half), "q" (quarter), "e" (eighth), "s" (sixteenth)
          also dotted/triplet variants: "q.", "et"
        """
        s = symbol.strip().lower()
        if not s:
            raise ValueError("symbol cannot be empty")

        dotted = s.endswith(".")
        triplet = s.endswith("t")
        # We treat dotted and triplet as mutually exclusive modifiers.
        if dotted and triplet:
            raise ValueError("symbol cannot be both dotted and triplet")
        if dotted or triplet:
            s = s[:-1]

        base = _symbol_to_beats(s)
        if dotted:
            base *= Fraction(3, 2)
        if triplet:
            base *= Fraction(2, 3)
        return Rhythm.from_beats(base)

    # -------- converters --------
    def to_beats(self) -> Fraction:
        return self.beats

    def to_seconds(self, *, tempo_bpm: float) -> float:
        # 1 beat (quarter) = 60 / BPM seconds
        bpm = float(tempo_bpm)
        if bpm <= 0:
            raise ValueError("tempo_bpm must be > 0")
        return float(self.beats) * 60.0 / bpm

    def to_ticks(self, *, ppq: int) -> int:
        if ppq <= 0:
            raise ValueError("ppq must be > 0")
        # MIDI time is discrete; we round to nearest tick.
        return int(round(float(self.beats * int(ppq))))

    def to_symbol(self, *, allowed: Optional[list[str]] = None) -> str:
        """
        Attempt to represent this duration as a symbol.

        - If `allowed` is provided, returns the first matching symbol in that list.
        - Otherwise returns a canonical fractional string like "1/8", "1/8.", "1/8t" when exact.
        - If it can't be expressed exactly using the default symbol set, returns "beats:<n>".
        """
        beats = self.beats

        symbols = allowed or _DEFAULT_SYMBOLS
        for sym in symbols:
            if Rhythm.from_symbol(sym).beats == beats:
                return sym

        # Try to generate canonical from fractional (including dotted/triplet) if possible.
        for base_sym in _FRACTIONAL_BASES:
            base_beats = Rhythm.from_symbol(base_sym).beats
            if base_beats == beats:
                return base_sym
            if base_beats * Fraction(3, 2) == beats:
                return base_sym + "."
            if base_beats * Fraction(2, 3) == beats:
                return base_sym + "t"

        return f"beats:{beats}"

    def __str__(self) -> str:
        return f"Rhythm(beats={self.beats})"


def _symbol_to_beats(base_symbol: str) -> Fraction:
    # Shorthands (common “note value” letters).
    shorthand = {
        "w": Fraction(4, 1),  # whole = 4 beats
        "h": Fraction(2, 1),  # half = 2 beats
        "q": Fraction(1, 1),  # quarter = 1 beat
        "e": Fraction(1, 2),  # eighth = 1/2 beat
        "s": Fraction(1, 4),  # sixteenth = 1/4 beat
        "t": Fraction(1, 8),  # thirty-second (mnemonic)
    }
    if base_symbol in shorthand:
        return shorthand[base_symbol]

    # Fractional: "1/8" etc, interpreted as a fraction of a whole note.
    if "/" in base_symbol:
        num_s, den_s = base_symbol.split("/", 1)
        num = int(num_s)
        den = int(den_s)
        if num <= 0 or den <= 0:
            raise ValueError("fractional symbol must be positive")
        # whole note = 4 beats, so (num/den) of a whole note = 4*(num/den) beats
        return Fraction(4 * num, den)

    raise ValueError(f"Unsupported rhythm symbol: {base_symbol!r}")


_FRACTIONAL_BASES = ["1/1", "1/2", "1/4", "1/8", "1/16", "1/32", "1/64"]
_DEFAULT_SYMBOLS = (
    ["w", "h", "q", "e", "s"]
    + _FRACTIONAL_BASES
    + [b + "." for b in _FRACTIONAL_BASES]
    + [b + "t" for b in _FRACTIONAL_BASES]
)

