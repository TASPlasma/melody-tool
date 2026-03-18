import numpy as np

from rhythm import Rhythm


class Distribution:
    """
    A wrapper for a probability distribution, which could be used for
    generating a sequence of notes, octaves, or rhythms.
    distribution: an array of weights
    """

    def __init__(self, distribution, *, epsilon=1e-9):
        self.distribution = np.asarray(distribution, dtype=float)
        self.epsilon = float(epsilon)
        self._normalize_in_place()

    def _normalize_in_place(self):
        self.distribution = np.maximum(self.distribution, self.epsilon)
        s = float(np.sum(self.distribution))
        if s <= 0:
            self.distribution = np.full_like(
                self.distribution, 1.0 / len(self.distribution))
        else:
            self.distribution = self.distribution / s

    def sample_index(self, rng=None):
        rng = rng or np.random
        self._normalize_in_place()
        return int(rng.choice(len(self.distribution), p=self.distribution))

    def reinforce(self, index, *, lr=0.15):
        """
        Simple online update: move probability mass toward `index`.
        lr in (0, 1]: higher means more "sticky" repeats.
        """
        lr = float(lr)
        if not (0 < lr <= 1):
            raise ValueError("lr must be in (0, 1].")
        self._normalize_in_place()
        self.distribution *= (1.0 - lr)
        self.distribution[int(index)] += lr
        self._normalize_in_place()


class MelodyGenerator:
    """
    A class to generate a melody, i.e. a sequence of notes based on a chord or a scale.
    chord: a chord object from the chord class
    scale: a list of notes ranging from 0 to 12
    note_distribution: a probability distribution for the notes
    octave_distribution: a probability distribution for the octaves 
    associated with the notes
    """

    def __init__(
        self,
        chord,
        scale,
        note_distribution,
        octave_distribution,
        rhythm_distribution=None,
        rhythm_palette=None,
        *,
        tempo_bpm=None,
        ppq=None,
        rng=None,
        update_lr_note=0.12,
        update_lr_octave=0.10,
        update_lr_rhythm=0.08,
    ):
        self.chord = chord
        self.scale = list(scale)
        self.note_distribution = note_distribution
        self.octave_distribution = octave_distribution
        self.rhythm_distribution = rhythm_distribution
        self.rhythm_palette = rhythm_palette
        self.tempo_bpm = tempo_bpm
        self.ppq = ppq
        self.rng = rng or np.random
        self.update_lr_note = float(update_lr_note)
        self.update_lr_octave = float(update_lr_octave)
        self.update_lr_rhythm = float(update_lr_rhythm)

    @staticmethod
    def _coerce_note_base(note):
        # Accept int 0-11 or MIDI int; reduce to pitch class.
        if isinstance(note, int):
            return int(note) % 12
        # Lazy import avoidance; supports note.Note if used.
        if hasattr(note, "get_note_base"):
            return int(note.get_note_base())
        if hasattr(note, "note_base"):
            return int(note.note_base)
        raise TypeError(f"Unsupported note type: {type(note)}")

    @staticmethod
    def _coerce_midi_int(note):
        if isinstance(note, int):
            return int(note)
        if hasattr(note, "get_note"):
            return int(note.get_note())
        if hasattr(note, "note"):
            return int(note.note)
        raise TypeError(f"Unsupported note type: {type(note)}")

    def _pick_note_base(self, allowed_bases=None):
        """
        Sample a pitch class (0-11). If allowed_bases is provided, renormalize to that subset.
        Updates `note_distribution` after the pick.
        """
        dist = np.asarray(self.note_distribution.distribution, dtype=float)
        if allowed_bases is None:
            idx = int(self.rng.choice(12, p=dist / np.sum(dist)))
        else:
            allowed = [int(x) % 12 for x in allowed_bases]
            mask = np.zeros(12, dtype=bool)
            mask[allowed] = True
            sub = dist.copy()
            sub[~mask] = 0.0
            if np.sum(sub) <= 0:
                # fallback uniform over allowed
                sub = np.zeros(12, dtype=float)
                sub[allowed] = 1.0 / len(allowed)
            else:
                sub = sub / np.sum(sub)
            idx = int(self.rng.choice(12, p=sub))
        self.note_distribution.reinforce(idx, lr=self.update_lr_note)
        return idx

    def _pick_octave(self, note_base):
        """
        Octave distribution can be:
        - Distribution (shared for all notes)
        - dict[int pitchclass -> Distribution]
        - list/tuple length 12 of Distribution
        Updates the chosen octave distribution after the pick.
        """
        nb = int(note_base) % 12
        od = self.octave_distribution
        if isinstance(od, Distribution):
            idx = od.sample_index(self.rng)
            od.reinforce(idx, lr=self.update_lr_octave)
            return idx
        if isinstance(od, dict):
            d = od[nb]
            idx = d.sample_index(self.rng)
            d.reinforce(idx, lr=self.update_lr_octave)
            return idx
        if isinstance(od, (list, tuple)) and len(od) == 12:
            d = od[nb]
            idx = d.sample_index(self.rng)
            d.reinforce(idx, lr=self.update_lr_octave)
            return idx
        raise TypeError(
            "octave_distribution must be Distribution, dict, or length-12 list/tuple of Distribution.")

    def _pick_rhythm(self):
        """
        Returns a Rhythm (if rhythm_distribution is provided), otherwise None.

        If `rhythm_palette` is provided, it defines what indices mean:
        - list[str]: symbols like "1/8", "q.", "1/8t"
        - list[Rhythm]: explicit durations
        """
        if self.rhythm_distribution is None:
            return None
        idx = self.rhythm_distribution.sample_index(self.rng)
        self.rhythm_distribution.reinforce(idx, lr=self.update_lr_rhythm)

        if self.rhythm_palette is None:
            # Default: constant eighth-notes unless you provide a palette.
            return Rhythm.from_symbol("1/8")

        choice = self.rhythm_palette[int(idx)]
        if isinstance(choice, Rhythm):
            return choice
        if isinstance(choice, str):
            return Rhythm.from_symbol(choice)
        raise TypeError(
            "rhythm_palette entries must be Rhythm or str symbols.")

    def _event_from_components(self, *, note_base: int, octave: int, rhythm: Rhythm | None):
        midi = int(note_base + 12 * octave)
        event = {"note_base": int(note_base),
                 "octave": int(octave), "midi": midi}
        if rhythm is None:
            event["rhythm"] = None
            return event

        event["rhythm"] = rhythm
        event["beats"] = float(rhythm.to_beats())
        event["symbol"] = rhythm.to_symbol(allowed=[x for x in self.rhythm_palette if isinstance(
            x, str)]) if self.rhythm_palette else rhythm.to_symbol()
        if self.tempo_bpm is not None:
            event["seconds"] = rhythm.to_seconds(
                tempo_bpm=float(self.tempo_bpm))
        if self.ppq is not None:
            event["ticks"] = rhythm.to_ticks(ppq=int(self.ppq))
        return event

    def generate(self):
        """
        Generate a sequence of notes based on the chord or scale
        """
        raise NotImplementedError(
            "Call a technique method directly (chromatic_approach/arpeggio_line/side_step_line/chromatic_nonsense), "
            "or extend generate() with your preferred technique selection policy."
        )

    def chromatic_approach(self, target_note, max_length=10):
        """
        Generates a sequence that ends on the target note, using chromatic neighboring notes.
        """
        if max_length < 2:
            raise ValueError("max_length must be >= 2")

        target_midi = self._coerce_midi_int(target_note)
        target_base = target_midi % 12

        # Choose a short approach (typical: 1-3 notes) then land on target.
        approach_len = int(min(max_length - 1, self.rng.choice([1, 2, 3])))
        direction = int(self.rng.choice([-1, 1]))

        # Build approach as semitone steps toward target_base from a neighbor.
        start_base = (target_base + direction * self.rng.choice([1, 2])) % 12
        approach_bases = [start_base]
        while len(approach_bases) < approach_len:
            prev = approach_bases[-1]
            step = -direction  # move toward target
            nxt = (prev + step) % 12
            approach_bases.append(nxt)
            if nxt == target_base:
                break

        # Ensure last is NOT target (we append target explicitly).
        approach_bases = [b for b in approach_bases if b != target_base]

        out = []
        for b in approach_bases + [target_base]:
            # For chromatic approach we *force* the pitch class, but still update distributions.
            self.note_distribution.reinforce(int(b), lr=self.update_lr_note)
            octave = self._pick_octave(b)
            dur = self._pick_rhythm()
            out.append(self._event_from_components(
                note_base=int(b), octave=int(octave), rhythm=dur))
        return out

    def arpeggio_line(self, chord):
        """
        Generates a sequence of notes that form an arpeggio line.
        """
        chord_notes = chord.get_notes()
        if not chord_notes:
            return []

        # Coerce to pitch classes; allow repeats across octaves via octave_distribution.
        bases = [self._coerce_note_base(n) for n in chord_notes]
        # Choose an up/down pattern (simple but musical).
        direction = int(self.rng.choice([-1, 1]))
        seq = sorted(bases)
        if direction < 0:
            seq = list(reversed(seq))

        out = []
        for b in seq:
            self.note_distribution.reinforce(int(b), lr=self.update_lr_note)
            octave = self._pick_octave(b)
            dur = self._pick_rhythm()
            out.append(self._event_from_components(
                note_base=int(b), octave=int(octave), rhythm=dur))
        return out

    def side_step_line(self, max_length=16):
        """
        Generates a sequence of notes that form a side step line.
        """
        if max_length <= 0:
            return []

        on_scale = [int(n) % 12 for n in self.scale]
        off_scale = [n for n in range(12) if n not in set(on_scale)]

        # Alternate blocks: on-scale phrase then off-scale phrase (brief).
        # Example: 6 on, 2 off, repeat...
        out = []
        remaining = int(max_length)
        while remaining > 0:
            on_len = int(min(remaining, self.rng.choice([4, 6, 8])))
            for _ in range(on_len):
                b = self._pick_note_base(on_scale)
                octave = self._pick_octave(b)
                dur = self._pick_rhythm()
                out.append(self._event_from_components(
                    note_base=int(b), octave=int(octave), rhythm=dur))
            remaining -= on_len
            if remaining <= 0:
                break

            off_len = int(min(remaining, self.rng.choice([1, 2, 2, 3])))
            for _ in range(off_len):
                b = self._pick_note_base(off_scale)
                octave = self._pick_octave(b)
                dur = self._pick_rhythm()
                out.append(self._event_from_components(
                    note_base=int(b), octave=int(octave), rhythm=dur))
            remaining -= off_len

        return out

    def chromatic_nonsense(self, base_phrase):
        """
        Starts with a base phrase, then repeats that phrase but shifted 
        chromatically until a stopping point.
        """
        if not base_phrase:
            return []

        # base_phrase: list of notes (pitch class ints, midi ints, or Note-like)
        phrase_bases = [self._coerce_note_base(n) for n in base_phrase]

        # Choose how many transpositions and direction.
        repeats = int(self.rng.choice([2, 3, 4]))
        direction = int(self.rng.choice([-1, 1]))

        out = []
        for r in range(repeats):
            shift = direction * r  # chromatic shift by semitones
            for b0 in phrase_bases:
                b = (int(b0) + shift) % 12
                self.note_distribution.reinforce(
                    int(b), lr=self.update_lr_note)
                octave = self._pick_octave(b)
                dur = self._pick_rhythm()
                out.append(self._event_from_components(
                    note_base=int(b), octave=int(octave), rhythm=dur))
        return out
