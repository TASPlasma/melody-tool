from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import mido

from rhythm import Rhythm


@dataclass(frozen=True, slots=True)
class MidiExportSettings:
    # These defaults make the exported file easy to audition in any DAW.
    tempo_bpm: float = 120.0
    ppq: int = 480
    channel: int = 0
    velocity: int = 96
    program: Optional[int] = None  # 0-127, General MIDI program number


def export_midi_clip(
    events: Iterable[dict],
    out_path: str,
    *,
    settings: MidiExportSettings = MidiExportSettings(),
    name: str = "Melody",
) -> str:
    """
    Write a single-track MIDI clip.

    Expected event fields:
    - midi: int (required)
    - rhythm: Rhythm | None (optional)
      If rhythm is None, we default to 1/8.

    Timing:
    - Events are written sequentially.
    - Each event is a note with note_on at time=0, note_off after its duration.

    Design choice:
    - We use a fixed velocity and no legato/ties. This makes it easier to
      validate pitch/rhythm logic before adding performance nuances.
    """
    mid = mido.MidiFile(type=1, ticks_per_beat=int(settings.ppq))
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.name = name
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(float(settings.tempo_bpm)), time=0))
    if settings.program is not None:
        # Optional instrument selection (General MIDI program number).
        track.append(
            mido.Message(
                "program_change",
                program=int(settings.program),
                channel=int(settings.channel),
                time=0,
            )
        )

    for ev in events:
        note = int(ev["midi"])
        rhythm = ev.get("rhythm")
        if rhythm is None:
            # Generator can emit None (or you might omit rhythm entirely). For auditioning,
            # defaulting to 1/8 keeps the file playable.
            rhythm = Rhythm.from_symbol("1/8")
        if not isinstance(rhythm, Rhythm):
            raise TypeError(f"event['rhythm'] must be Rhythm or None, got {type(rhythm)}")

        dur_ticks = rhythm.to_ticks(ppq=int(settings.ppq))
        if dur_ticks <= 0:
            dur_ticks = 1

        track.append(
            mido.Message(
                "note_on",
                note=note,
                velocity=int(settings.velocity),
                channel=int(settings.channel),
                time=0,
            )
        )
        track.append(
            mido.Message(
                "note_off",
                note=note,
                velocity=0,
                channel=int(settings.channel),
                time=int(dur_ticks),
            )
        )

    mid.save(out_path)
    return out_path

