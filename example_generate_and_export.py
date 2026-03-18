from generators import Distribution, MelodyGenerator
from midi_export import MidiExportSettings, export_midi_clip


class DummyChord:
    def get_notes(self):
        # C major triad pitch classes; generator will pick octaves
        return [0, 4, 7]


def main():
    tempo_bpm = 120
    ppq = 480

    # Example palette: sixteenth, eighth, quarter notes
    rhythm_palette = ["1/16", "1/8", "1/4"]

    mg = MelodyGenerator(
        chord=DummyChord(),
        scale=[0, 2, 3, 5, 7, 9, 11],  # C melodic minor
        note_distribution=Distribution([1] * 12),
        octave_distribution=Distribution(
            [0, 0, 2, 4, 3, 1]),  # favor middle octaves (2-4)
        rhythm_distribution=Distribution([2, 6, 2]),
        rhythm_palette=rhythm_palette,
        tempo_bpm=tempo_bpm,
        ppq=ppq,
    )

    events = mg.side_step_line(max_length=8)

    out = export_midi_clip(
        events,
        "melody_test.mid",
        settings=MidiExportSettings(tempo_bpm=tempo_bpm, ppq=ppq, program=0),
        name="Melody Test",
    )
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
