from __future__ import annotations

import math
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@dataclass(frozen=True, slots=True)
class VizStyle:
    # Use 896 so MP4 export doesn't need macroblock resizing (divisible by 16).
    width: int = 896
    height: int = 896
    bg: tuple[float, float, float] = (0.05, 0.06, 0.08)
    ring_radius: float = 0.36  # relative to min(width,height)
    node_radius: float = 0.055
    edge_alpha: float = 0.85
    base_node_alpha: float = 0.18
    max_node_alpha: float = 0.98


def _node_positions(style: VizStyle):
    size = min(style.width, style.height)
    cx, cy = style.width / 2.0, style.height / 2.0
    r = style.ring_radius * size
    pos = []
    for i in range(12):
        # Put C at the top, clockwise.
        ang = -math.pi / 2 + i * (2 * math.pi / 12)
        pos.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    return pos


def _prob_to_alpha(p: float, style: VizStyle):
    p = max(0.0, float(p))
    # Use a gentle curve so small probabilities still show.
    x = p ** 0.6
    return style.base_node_alpha + (style.max_node_alpha - style.base_node_alpha) * x


def render_frame(
    note_probs: Sequence[float],
    melody_bases: Sequence[int],
    *,
    step: int,
    style: VizStyle = VizStyle(),
):
    """
    Render a single RGB frame as a numpy array.

    - `note_probs`: length-12 probabilities for node brightness.
    - `melody_bases`: pitch classes (0..11) for the melody (ignores rhythm/octave).
    - `step`: how many melody notes to show as “played so far”.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch

    note_probs = np.asarray(note_probs, dtype=float)
    if note_probs.shape != (12,):
        raise ValueError("note_probs must be length 12.")

    fig = plt.figure(figsize=(style.width / 100, style.height / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, style.width)
    ax.set_ylim(style.height, 0)  # invert Y so it feels screen-like
    ax.axis("off")
    fig.patch.set_facecolor(style.bg)
    ax.set_facecolor(style.bg)

    pos = _node_positions(style)

    # Draw edges of the directed “melody so far”.
    played = list(melody_bases[: max(0, min(step, len(melody_bases)))])
    for a, b in zip(played, played[1:]):
        x1, y1 = pos[int(a) % 12]
        x2, y2 = pos[int(b) % 12]
        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.6,
            color=(0.92, 0.34, 0.23, style.edge_alpha),
            shrinkA=22,
            shrinkB=22,
        )
        ax.add_patch(arrow)

    # Melody text at the bottom (grows as notes are played).
    # We wrap across multiple lines so longer melodies stay readable in the video.
    if played:
        names = [NOTE_NAMES[int(n) % 12] for n in played]
        max_per_line = 18
        lines = []
        for i in range(0, len(names), max_per_line):
            lines.append(", ".join(names[i:i + max_per_line]))

        ax.text(
            style.width / 2,
            style.height - 64,
            "Melody:",
            ha="center",
            va="center",
            fontsize=16,
            color=(0.92, 0.94, 0.98, 0.9),
            fontweight="bold",
        )
        # Draw lines from bottom upward.
        base_y = style.height - 34
        for li, line in enumerate(reversed(lines[-3:])):  # show last ~3 lines if very long
            ax.text(
                style.width / 2,
                base_y - li * 22,
                line,
                ha="center",
                va="center",
                fontsize=14,
                color=(0.92, 0.94, 0.98, 0.82),
            )

    # Draw nodes (brightness encodes probability).
    for i, (x, y) in enumerate(pos):
        alpha = _prob_to_alpha(note_probs[i], style)
        node = Circle(
            (x, y),
            radius=style.node_radius * min(style.width, style.height),
            facecolor=(0.78, 0.86, 0.96, alpha),
            edgecolor=(0.85, 0.9, 0.98, min(1.0, alpha + 0.12)),
            linewidth=2.0,
        )
        ax.add_patch(node)
        ax.text(
            x,
            y,
            NOTE_NAMES[i],
            ha="center",
            va="center",
            fontsize=18,
            color=(0.03, 0.04, 0.06, 0.95),
            fontweight="bold",
        )

    # Headline and current note indicator.
    ax.text(
        style.width / 2,
        70,
        "Melody Generator Probability Graph",
        ha="center",
        va="center",
        fontsize=20,
        color=(0.92, 0.94, 0.98, 0.95),
        fontweight="bold",
    )
    if played:
        cur = int(played[-1]) % 12
        ax.text(
            style.width / 2,
            110,
            f"Step {len(played)}/{len(melody_bases)}  •  Current: {NOTE_NAMES[cur]}",
            ha="center",
            va="center",
            fontsize=16,
            color=(0.92, 0.94, 0.98, 0.85),
        )

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return frame


def save_melody_video(
    *,
    note_prob_history: Sequence[Sequence[float]],
    melody_bases: Sequence[int],
    out_path: str,
    fps: int = 30,
    hold_frames: int = 12,
    style: VizStyle = VizStyle(),
    # Optional audio support.
    events: Optional[Iterable[dict]] = None,
    tempo_bpm: Optional[float] = None,
    rhodes_samples_dir: Optional[str] = None,
    embed_audio: bool = False,
    audio_sample_rate: int = 44100,
):
    """Export an animation.

    - If out_path ends with ".gif": writes a GIF (no ffmpeg required).
    - If out_path ends with ".mp4": writes an MP4 (requires ffmpeg available to imageio).

    If `embed_audio=True`, an audio track will be rendered from `events` and merged into
    the MP4 using ffmpeg. This requires `events` (with rhythm/tempo info) and a
    Rhodes sample folder accessible via `rhodes_samples_dir`.
    """
    import imageio.v3 as iio

    if len(note_prob_history) < 2:
        raise ValueError("note_prob_history should include an initial snapshot + updates.")

    # If we have events, sync the video playback to their durations.
    # This keeps the “graph traversal” speed aligned with the rendered audio.
    durations: list[float] | None = None
    if events is not None:
        from rhythm import Rhythm

        tempo = float(tempo_bpm) if tempo_bpm is not None else 120.0
        durations = []
        for ev in events:
            r = ev.get("rhythm")
            if r is None:
                r = Rhythm.from_symbol("1/8")
            if not hasattr(r, "to_seconds"):
                raise TypeError("event['rhythm'] must be a Rhythm instance or None")
            durations.append(float(r.to_seconds(tempo_bpm=tempo)))

    frames = []
    # We expect history length to be len(melody)+1, but we tolerate mismatch.
    steps = min(len(melody_bases), len(note_prob_history) - 1)

    if durations is None or len(durations) == 0:
        # Fallback: animate 1 frame per step (legacy behavior).
        for i in range(steps + 1):
            probs = note_prob_history[min(i, len(note_prob_history) - 1)]
            frame = render_frame(probs, melody_bases, step=i, style=style)
            frames.append(frame)
    else:
        # Use note durations to determine how many frames each step should span.
        # Longer notes hold the same visual state longer, so the graph traverses
        # at the same pace as the audio.
        # Start with a single frame for the initial state (step=0).
        frames.append(render_frame(note_prob_history[0], melody_bases, step=0, style=style))

        for i in range(steps):
            dur = durations[i] if i < len(durations) else 1.0 / fps
            frame = render_frame(
                note_prob_history[min(i + 1, len(note_prob_history) - 1)],
                melody_bases,
                step=i + 1,
                style=style,
            )
            count = max(1, math.ceil(dur * fps))
            frames.extend([frame] * count)

    # Hold the final frame a bit so the viewer can read the result.
    frames.extend([frames[-1]] * int(hold_frames))

    ext = out_path.lower().split(".")[-1]
    if ext == "gif":
        iio.imwrite(out_path, frames, duration=1.0 / fps)
        return out_path

    if ext != "mp4":
        raise ValueError("out_path must end with .gif or .mp4")

    if embed_audio:
        if events is None:
            raise ValueError("embed_audio=True requires `events` to render audio.")

        # Render video to a temporary file while we also render audio.
        tmp_video = Path(tempfile.mktemp(suffix=".mp4"))
        audio_path: Optional[str] = None
        try:
            iio.imwrite(tmp_video, frames, fps=fps)

            # If we have event durations, use them to keep video/audio aligned.
            target_duration = None
            if events is not None:
                from rhythm import Rhythm

                tempo = float(tempo_bpm) if tempo_bpm is not None else 120.0
                total_audio = 0.0
                for ev in events:
                    r = ev.get("rhythm")
                    if r is None:
                        r = Rhythm.from_symbol("1/8")
                    total_audio += float(r.to_seconds(tempo_bpm=tempo))
                target_duration = total_audio + (hold_frames / fps)

            audio_path = render_melody_audio(
                events,
                out_path=None,
                tempo_bpm=tempo_bpm,
                sample_rate=audio_sample_rate,
                rhodes_samples_dir=rhodes_samples_dir,
                target_duration=target_duration,
            )

            _merge_video_audio(tmp_video, audio_path, Path(out_path))
            return out_path
        finally:
            try:
                tmp_video.unlink(missing_ok=True)
            except Exception:
                pass
            if audio_path:
                try:
                    Path(audio_path).unlink(missing_ok=True)
                except Exception:
                    pass
    else:
        iio.imwrite(out_path, frames, fps=fps)
        return out_path


def melody_bases_from_events(events: Iterable[dict]) -> list[int]:
    """
    Utility: extract pitch classes from MelodyGenerator events.
    """
    bases = []
    for ev in events:
        if "note_base" in ev:
            bases.append(int(ev["note_base"]) % 12)
        else:
            bases.append(int(ev["midi"]) % 12)
    return bases


# --- Audio helpers -----------------------------------------------------------

_DEFAULT_RHODES_DIR = Path(__file__).resolve().parent / "Rhodes_Notes"


def _estimate_midi_from_wav(path: Path) -> Optional[int]:
    """Estimate the MIDI note number for a single mono WAV sample."""
    try:
        with wave.open(str(path), "rb") as w:
            sr = w.getframerate()
            n = w.getnframes()
            data = w.readframes(n)
    except Exception:
        return None

    if sr <= 0 or n <= 0:
        return None

    audio = np.frombuffer(data, dtype=np.int16).astype(float)
    if audio.size == 0:
        return None

    # Use a short window for speed/stability.
    length = min(audio.size, int(sr * 0.05))
    audio = audio[:length]

    # Autocorrelation-based pitch detection.
    corr = np.correlate(audio, audio, mode="full")[len(audio) - 1 :]
    deriv = np.diff(corr)
    peaks = np.where(deriv > 0)[0]
    if len(peaks) == 0:
        return None
    start = int(peaks[0])
    peak = int(np.argmax(corr[start:]) + start)
    if peak <= 0:
        return None

    freq = sr / peak
    midi = int(round(69 + 12 * math.log2(freq / 440.0)))
    return midi


@lru_cache(maxsize=4)
def _build_rhodes_sample_map(sample_dir: Optional[str] = None) -> list[tuple[int, Path]]:
    """Return a list of (midi_note, path) for Rhodes note samples."""
    root = Path(sample_dir) if sample_dir else _DEFAULT_RHODES_DIR
    root = root.expanduser().resolve()
    paths = sorted(root.glob("RhodesNote-*.wav"))
    samples: list[tuple[int, Path]] = []
    for p in paths:
        midi = _estimate_midi_from_wav(p)
        if midi is None:
            continue
        samples.append((midi, p))
    samples.sort(key=lambda x: x[0])
    return samples


def _find_closest_rhodes_sample(midi: int, samples: list[tuple[int, Path]]) -> Path:
    if not samples:
        raise FileNotFoundError("No Rhodes sample files found (RhodesNote-*.wav).")
    best = min(samples, key=lambda mp: abs(mp[0] - midi))
    return best[1]


def _load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    """Load a WAV file as a mono float32 numpy array in range [-1, 1]."""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        frames = w.readframes(w.getnframes())
        channels = w.getnchannels()

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)
    return audio, sr


def _resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    duration = len(audio) / float(orig_sr)
    if duration <= 0:
        return np.zeros(0, dtype=audio.dtype)
    new_len = int(round(duration * target_sr))
    if new_len <= 0:
        return np.zeros(0, dtype=audio.dtype)
    old_times = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    new_times = np.linspace(0.0, duration, num=new_len, endpoint=False)
    return np.interp(new_times, old_times, audio).astype(audio.dtype)


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    audio = np.clip(audio, -1.0, 1.0)
    data = (audio * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(data.tobytes())


def render_melody_audio(
    events: Iterable[dict],
    *,
    out_path: Optional[str] = None,
    tempo_bpm: Optional[float] = None,
    sample_rate: int = 44100,
    rhodes_samples_dir: Optional[str] = None,
    target_duration: Optional[float] = None,
) -> str:
    """Render a melody event sequence as a WAV file using Rhodes note samples."""
    from rhythm import Rhythm

    samples = _build_rhodes_sample_map(rhodes_samples_dir)
    if not samples:
        raise FileNotFoundError(
            "No Rhodes note samples found. Place `RhodesNote-*.wav` in the Rhodes_Notes folder."
        )

    segments = []
    total_seconds = 0.0

    # Choose tempo to use if event doesn't provide seconds.
    tempo = float(tempo_bpm) if tempo_bpm is not None else 120.0

    for ev in events:
        midi = int(ev.get("midi", 0))
        rhythm = ev.get("rhythm")
        if rhythm is None:
            rhythm = Rhythm.from_symbol("1/8")
        if not hasattr(rhythm, "to_seconds"):
            raise TypeError("event['rhythm'] must be a Rhythm instance or None")

        duration = float(rhythm.to_seconds(tempo_bpm=tempo))
        if duration <= 0:
            continue

        sample_path = _find_closest_rhodes_sample(midi, samples)
        audio, sr = _load_wav_mono(sample_path)
        audio = _resample_audio(audio, sr, sample_rate)

        n_samples = int(round(duration * sample_rate))
        if n_samples <= 0:
            continue

        if len(audio) >= n_samples:
            segment = audio[:n_samples]
        else:
            segment = np.zeros(n_samples, dtype=audio.dtype)
            segment[: len(audio)] = audio

        segments.append(segment)
        total_seconds += duration

    if segments:
        audio_all = np.concatenate(segments)
    else:
        audio_all = np.zeros(0, dtype=np.float32)

    # Pad or trim to match requested duration (helpful when embedding in video).
    if target_duration is not None and target_duration > 0:
        target_samples = int(round(target_duration * sample_rate))
        if len(audio_all) < target_samples:
            pad = np.zeros(target_samples - len(audio_all), dtype=audio_all.dtype)
            audio_all = np.concatenate([audio_all, pad])
        else:
            audio_all = audio_all[:target_samples]

    if out_path is None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name

    _write_wav(Path(out_path), audio_all, sample_rate)
    return str(out_path)


def _merge_video_audio(video_path: Path, audio_path: str, out_path: Path) -> None:
    """Merge an MP4 video and WAV audio into a single MP4 using ffmpeg."""
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)

