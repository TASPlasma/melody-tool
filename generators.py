import numpy as np


class Distribution:
    """
    A wrapper for a probability distribution, which could be used for
    generating a sequence of notes, octaves, or rhythms.
    """

    def __init__(self, distribution):
        self.distribution = distribution


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
    ):
        self.chord = chord
        self.scale = scale
        self.note_distribution = note_distribution
        self.octave_distribution = octave_distribution

    def generate(self):
        """
        Generate a sequence of notes based on the chord or scale
        """

    def chromatic_approach(self, target_note, max_length=10):
        """
        Generates a sequence that ends on the target note, using chromatic neighboring notes.
        """
        notes = [target_note]
        neighboring_notes = [
            target_note - i for i in range(1, 3)] + [target_note + i for i in range(1, 3)]
        while len(notes) < max_length:
            next_note = np.random.choice(neighboring_notes)
            notes.append(next_note)
        return notes

    def arpeggio_line(self, chord):
        """
        Generates a sequence of notes that form an arpeggio line.
        """
        notes = chord.get_notes()
        return notes

    def side_step_line(self, max_length=16):
        """
        Generates a sequence of notes that form a side step line.
        """
        complementary__base_notes = [
            note for note in range(12) if note not in self.scale]
        pass
