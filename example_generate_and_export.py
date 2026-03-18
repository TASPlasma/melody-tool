from generators import Distribution, MelodyGenerator
from midi_export import MidiExportSettings, export_midi_clip
from melody_viz import melody_bases_from_events, save_melody_video


class DummyChord:
    def get_notes(self):
        # Pitch classes for a simple triad; the generator decides octaves later.
        return [0, 3, 7]


def main():
    tempo_bpm = 120
    ppq = 480

    # Example palette: sixteenth, eighth, quarter notes
    # (These are symbols understood by `Rhythm.from_symbol()`.)
    rhythm_palette = ["1/16", "1/8", "1/4"]

    mg = MelodyGenerator(
        chord=DummyChord(),
        scale=[0, 2, 3, 5, 7, 9, 11],  # C melodic minor
        note_distribution=Distribution([1] * 12),
        octave_distribution=Distribution(
            [0, 0, 2, 4, 3, 1]),  # favor middle octaves (2-4)
        # Weight the palette; larger weight => more likely duration.
        rhythm_distribution=Distribution([2, 6, 2]),
        rhythm_palette=rhythm_palette,
        tempo_bpm=tempo_bpm,
        ppq=ppq,
    )

    # Any technique method returns a list of event dicts (pitch + rhythm).
    # Here we also request a probability trace so we can render a “brightness over time” video.
    events, note_prob_history = mg.run_with_trace(
        mg.side_step_line, max_length=24)

    out = export_midi_clip(
        events,
        "melody_test.mid",
        settings=MidiExportSettings(tempo_bpm=tempo_bpm, ppq=ppq, program=0),
        name="Melody Test",
    )
    print(f"Wrote: {out}")

    melody_bases = melody_bases_from_events(events)
    viz_out = save_melody_video(
        note_prob_history=note_prob_history,
        melody_bases=melody_bases,
        out_path="melody_test.mp4",
        fps=8,
        hold_frames=24,
    )
    print(f"Wrote: {viz_out}")


if __name__ == "__main__":
    main()
