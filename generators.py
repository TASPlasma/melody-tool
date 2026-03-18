import numpy as np

from rhythm import Rhythm


class Distribution:
    """
    A tiny “adaptive” categorical distribution.

    This is the probability model behind:
    - pitch-class selection (C..B mapped to 0..11)
    - octave selection (index => octave number)
    - rhythm selection (index => symbol or `Rhythm` via `rhythm_palette`)

    The key behavior is `reinforce()`: after you select an index, the distribution
    is updated online so future samples are biased toward recent choices.

    Notes:
    - `distribution` is treated as non-negative weights (not necessarily normalized).
    - We keep an `epsilon` floor so probabilities never hit exactly 0.
    """

    def __init__(self, distribution, *, epsilon=1e-9):
        self.distribution = np.asarray(distribution, dtype=float)
        self.epsilon = float(epsilon)
        self._normalize_in_place()

    def _normalize_in_place(self):
        # Guardrails: keep weights positive and normalize to sum=1.
        self.distribution = np.maximum(self.distribution, self.epsilon)
        s = float(np.sum(self.distribution))
        if s <= 0:
            self.distribution = np.full_like(
                self.distribution, 1.0 / len(self.distribution))
        else:
            self.distribution = self.distribution / s

    def sample_index(self, rng=None):
        # Sample using the current probabilities.
        rng = rng or np.random
        self._normalize_in_place()
        return int(rng.choice(len(self.distribution), p=self.distribution))

    def reinforce(self, index, *, lr=0.15):
        """
        Simple online update: move probability mass toward `index`.

        Conceptually:
        - shrink all probabilities by (1 - lr)
        - add lr to the chosen index
        - renormalize

        Larger lr makes the generator more “self-reinforcing”.
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
    Probabilistic melody generator with several common melodic techniques.

    Core idea (your outline):
    - choose a pitch-class (0..11) using `note_distribution`
    - choose an octave using `octave_distribution` (optionally conditioned on pitch-class)
    - choose a rhythmic duration using `rhythm_distribution` (+ `rhythm_palette`)
    - after each selection, update (“reinforce”) the corresponding distribution

    Inputs:
    - `chord`: any object with `get_notes()` returning chord tones (pitch classes, MIDI ints, or Note-like)
    - `scale`: list of pitch classes (0..11). Used for on-scale/off-scale decisions (side stepping).

    Optional export context:
    - `tempo_bpm` lets events include seconds
    - `ppq` lets events include MIDI ticks
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
        # Starting notes should prefer chord tones.
        start_chord_tone_window=3,
        start_chord_tone_multiplier=2.2,
        # Enforce: the number of selected 1/16 rhythms in a melody is even.
        enforce_even_sixteenth=True,
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

        # Optional tracing for visualization/debugging.
        # When enabled, we store a copy of the 12-note probabilities after each
        # pitch-class update. This makes it easy to render “brightness over time”.
        self._trace_note_probs = False
        self._note_prob_history = None

        # Repeat-avoidance state.
        self._last_note_base = None
        self._last_beats = None
        self._last_was_chord_tone = None
        self._chord_tones = self._compute_chord_tones(self.chord)

        self.start_chord_tone_window = int(start_chord_tone_window)
        self.start_chord_tone_multiplier = float(start_chord_tone_multiplier)
        self.enforce_even_sixteenth = bool(enforce_even_sixteenth)
        self._note_pick_index = 0
        self._sixteenth_beats = Rhythm.from_symbol("1/16").to_beats()

    def _compute_chord_tones(self, chord):
        """
        Return a set of pitch classes considered "chord tones" for repeat-avoidance weighting.
        If chord is missing or doesn't support get_notes(), we return an empty set.
        """
        if chord is None or not hasattr(chord, "get_notes"):
            return set()
        try:
            return {int(self._coerce_note_base(n)) % 12 for n in chord.get_notes()}
        except Exception:
            return set()

    def _repeat_aversion_strength(self):
        """
        Compute how strongly to suppress repeating the previous pitch class.

        Design:
        - Fast durations => stronger suppression (prevents "machine-gun" repeats)
        - Non-chord-tones => stronger suppression (repeats of tensions feel more accidental)
        """
        beats = self._last_beats
        if beats is None:
            base = 0.35
        else:
            b = float(beats)
            if b <= 0.25:      # 1/16
                base = 0.70
            elif b <= 0.5:     # 1/8
                base = 0.55
            elif b <= 1.0:     # 1/4
                base = 0.40
            else:
                base = 0.25

        if self._last_was_chord_tone is False:
            base *= 1.35

        # Clamp to avoid driving probability to ~0.
        return max(0.0, min(0.85, base))

    def _apply_repeat_aversion(self):
        """
        Slightly reduce the probability mass of the previously played pitch class.
        This is applied BEFORE sampling the next note, so it affects the next choice.
        """
        if self._last_note_base is None:
            return

        strength = self._repeat_aversion_strength()
        if strength <= 0:
            return

        idx = int(self._last_note_base) % 12
        # Multiply the previous note's weight by a factor < 1, then renormalize.
        # Example: strength=0.55 => factor=0.45
        factor = 1.0 - strength
        self.note_distribution.distribution[idx] *= factor
        self.note_distribution._normalize_in_place()

    def _apply_start_chord_tone_bias(self):
        """
        Bias early pitch-class sampling toward chord tones.

        Only applies during the first `start_chord_tone_window` note picks.
        """
        if self._note_pick_index >= self.start_chord_tone_window:
            return
        if not self._chord_tones:
            return
        if self.start_chord_tone_multiplier <= 1.0:
            return

        tones = list(self._chord_tones)
        self.note_distribution.distribution[tones] *= self.start_chord_tone_multiplier
        self.note_distribution._normalize_in_place()

    def _reset_generation_state(self):
        """
        Reset state that should not carry over between separate melodies.
        """
        self._last_note_base = None
        self._last_beats = None
        self._last_was_chord_tone = None
        self._note_pick_index = 0

    def _trace_start(self):
        self._reset_generation_state()
        self._trace_note_probs = True
        self._note_prob_history = []
        # Snapshot initial distribution (before any notes are selected).
        self.note_distribution._normalize_in_place()
        self._note_prob_history.append(self.note_distribution.distribution.copy())

    def _trace_stop(self):
        self._trace_note_probs = False
        hist = self._note_prob_history or []
        self._note_prob_history = None
        return hist

    def _trace_after_note_update(self):
        if not self._trace_note_probs:
            return
        self.note_distribution._normalize_in_place()
        self._note_prob_history.append(self.note_distribution.distribution.copy())

    @staticmethod
    def _coerce_note_base(note):
        # Reduce anything note-like to a pitch class (0..11).
        # Accepts:
        # - int (treated as pitch class or MIDI; we mod 12 either way)
        # - Note-like objects (duck-typed)
        if isinstance(note, int):
            return int(note) % 12
        # Note-like object support without importing `Note` here.
        if hasattr(note, "get_note_base"):
            return int(note.get_note_base())
        if hasattr(note, "note_base"):
            return int(note.note_base)
        raise TypeError(f"Unsupported note type: {type(note)}")

    @staticmethod
    def _coerce_midi_int(note):
        # Reduce anything note-like to a MIDI note number.
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
        # Starting chord-tone bias should affect early picks.
        self._apply_start_chord_tone_bias()

        # Reduce immediate repeats before sampling (see user request).
        self._apply_repeat_aversion()

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
                # If the model assigns zero mass to the allowed set, fall back to uniform
                # rather than failing or producing NaNs.
                sub = np.zeros(12, dtype=float)
                sub[allowed] = 1.0 / len(allowed)
            else:
                sub = sub / np.sum(sub)
            idx = int(self.rng.choice(12, p=sub))
        self.note_distribution.reinforce(idx, lr=self.update_lr_note)
        self._trace_after_note_update()
        self._note_pick_index += 1
        return idx

    def run_with_trace(self, technique_callable, *args, **kwargs):
        """
        Run any technique method and return (events, note_prob_history).

        `note_prob_history` is a list of length (len(events)+1) where:
        - history[0] is the initial note probability distribution
        - history[i] is the distribution after selecting the i-th note
        """
        self._trace_start()
        try:
            events = technique_callable(*args, **kwargs)
        finally:
            history = self._trace_stop()
        events = self._enforce_even_sixteenth(events)
        return events, history

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

        Default behavior:
        - If you provide a rhythm distribution but no palette, we emit constant "1/8"
          (and still reinforce the distribution index that was sampled).
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
        # Standardize technique outputs into a consistent event dictionary.
        midi = int(note_base + 12 * octave)
        event = {"note_base": int(note_base),
                 "octave": int(octave), "midi": midi}
        if rhythm is None:
            event["rhythm"] = None
            return event

        event["rhythm"] = rhythm
        # Convenience fields for debugging / exporting.
        event["beats"] = float(rhythm.to_beats())
        event["symbol"] = rhythm.to_symbol(allowed=[x for x in self.rhythm_palette if isinstance(
            x, str)]) if self.rhythm_palette else rhythm.to_symbol()
        if self.tempo_bpm is not None:
            event["seconds"] = rhythm.to_seconds(
                tempo_bpm=float(self.tempo_bpm))
        if self.ppq is not None:
            event["ticks"] = rhythm.to_ticks(ppq=int(self.ppq))

        # Update "previous note" state for repeat-avoidance on the NEXT pick.
        nb = int(note_base) % 12
        self._last_note_base = nb
        self._last_beats = rhythm.to_beats()
        chord_tones = self._chord_tones
        self._last_was_chord_tone = (nb in chord_tones) if chord_tones else None
        return event

    def _rewrite_event_rhythm(self, event: dict, new_rhythm: Rhythm):
        """
        Replace an event's rhythm (used for enforcing constraints post-generation).
        """
        event["rhythm"] = new_rhythm
        event["beats"] = float(new_rhythm.to_beats())
        allowed = [x for x in self.rhythm_palette if isinstance(x, str)] if self.rhythm_palette else None
        event["symbol"] = new_rhythm.to_symbol(allowed=allowed)
        if self.tempo_bpm is not None:
            event["seconds"] = new_rhythm.to_seconds(tempo_bpm=float(self.tempo_bpm))
        if self.ppq is not None:
            event["ticks"] = new_rhythm.to_ticks(ppq=int(self.ppq))

    def _enforce_even_sixteenth(self, events: list[dict]) -> list[dict]:
        """
        Enforce: number of events with rhythm == 1/16 must be even.

        If it's odd, we replace the last 1/16 rhythm event with a non-1/16 rhythm.
        """
        if not self.enforce_even_sixteenth or not events:
            return events

        sixteen_idxs: list[int] = []
        for i, ev in enumerate(events):
            r = ev.get("rhythm")
            if isinstance(r, Rhythm) and r.to_beats() == self._sixteenth_beats:
                sixteen_idxs.append(i)

        if not sixteen_idxs:
            return events
        if len(sixteen_idxs) % 2 == 0:
            return events

        chosen_pos = sixteen_idxs[-1]

        # Choose a replacement rhythm (prefer using the palette if present).
        if self.rhythm_palette is None or self.rhythm_distribution is None:
            replacement = Rhythm.from_symbol("1/8")
        else:
            non16_palette_idxs = []
            non16_rhythms = []
            for pi, choice in enumerate(self.rhythm_palette):
                rr = choice if isinstance(choice, Rhythm) else Rhythm.from_symbol(choice)
                if rr.to_beats() != self._sixteenth_beats:
                    non16_palette_idxs.append(pi)
                    non16_rhythms.append(rr)

            if not non16_palette_idxs:
                replacement = Rhythm.from_symbol("1/8")
            else:
                weights = np.asarray(self.rhythm_distribution.distribution, dtype=float)[non16_palette_idxs]
                weights = np.maximum(weights, 1e-12)
                weights = weights / np.sum(weights)
                sel = int(self.rng.choice(len(non16_palette_idxs), p=weights))
                replacement = non16_rhythms[sel]

        self._rewrite_event_rhythm(events[chosen_pos], replacement)
        return events

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
        Technique 1: choose a target note and approach it chromatically.

        We generate a short lead-in (typically 1-3 notes) using semitone motion
        around the target pitch-class, then land on the target.
        """
        self._reset_generation_state()
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
            # This technique *forces* the pitch-class contour, but still reinforces
            # those pitch-classes so the model learns what it “played”.
            self.note_distribution.reinforce(int(b), lr=self.update_lr_note)
            self._trace_after_note_update()
            octave = self._pick_octave(b)
            dur = self._pick_rhythm()
            out.append(self._event_from_components(
                note_base=int(b), octave=int(octave), rhythm=dur))
        out = self._enforce_even_sixteenth(out)
        return out

    def arpeggio_line(self, chord):
        """
        Technique 2: arpeggiate chord tones.

        Uses `chord.get_notes()` and walks them ascending or descending.
        Octaves are sampled separately, which produces “broken” arpeggios
        across registers instead of a single fixed voicing.
        """
        self._reset_generation_state()
        # Treat the passed chord as chord-tone context for this technique.
        prev_chord_tones = self._chord_tones
        self._chord_tones = self._compute_chord_tones(chord)

        chord_notes = chord.get_notes()
        if not chord_notes:
            self._chord_tones = prev_chord_tones
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
        out = self._enforce_even_sixteenth(out)
        self._chord_tones = prev_chord_tones
        return out

    def side_step_line(self, max_length=16):
        """
        Technique 3: side stepping.

        Alternates between:
        - on-scale phrases (restricted to `self.scale`)
        - brief off-scale phrases (restricted to the chromatic complement)
        """
        self._reset_generation_state()
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

        out = self._enforce_even_sixteenth(out)
        return out

    def chromatic_nonsense(self, base_phrase):
        """
        Technique 4: repeat a phrase chromatically.

        Takes a base phrase and repeats it transposed by semitones:
        0, +1, +2, ... (or downward), which gives the “chromatic sequence” effect.
        """
        self._reset_generation_state()
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
        out = self._enforce_even_sixteenth(out)
        return out
