note_dict = {
    0: "C",
    1: "C#",
    2: "D",
    3: "D#",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "G#",
    9: "A",
    10: "A#",
    11: "B"
}

name_dict = {
    "C": 0,
    "C#": 1,
    "D": 2,
    "D#": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "G": 7,
    "G#": 8,
    "A": 9,
    "A#": 10,
    "B": 11
}


class Note:
    """
    A class to represent a note in the chromatic scale.
    A note is a number between 0 and 127
    """

    def __init__(self, note=0):
        self.note = note
        self.name = self.get_note_name()
        self.octave = note // 12
        self.note_base = note % 12

    def get_note(self):
        return self.note

    def get_octave(self):
        return self.octave

    def get_note_base(self):
        return self.note_base

    def get_note_name(self):
        return note_dict[self.get_note_base()]

    def __str__(self):
        note_name = self.get_note_name()
        octave = self.get_octave()
        note = self.get_note()
        return note_name + str(octave) + "(" + str(note) + ")"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.get_note() == other.get_note()

    def __ne__(self, other):
        return self.get_note() != other.get_note()

    def __lt__(self, other):
        return self.get_note() < other.get_note()

    def __le__(self, other):
        return self.get_note() <= other.get_note()

    def __gt__(self, other):
        return self.get_note() > other.get_note()

    def __ge__(self, other):
        return self.get_note() >= other.get_note()

    @staticmethod
    def _coerce_other_to_int(other):
        if isinstance(other, Note):
            return other.get_note()
        if isinstance(other, int):
            return other
        raise TypeError(f"Unsupported operand type: {type(other)}")

    def __add__(self, other):
        return Note(self.get_note() + self._coerce_other_to_int(other))

    def __sub__(self, other):
        return Note(self.get_note() - self._coerce_other_to_int(other))

    def __mul__(self, other):
        return Note(self.get_note() * self._coerce_other_to_int(other))


class Chord:
    """
    A chord is a collection of notes
    """
    def __init__(kill_thyself, name, choices, notes):
        pass
