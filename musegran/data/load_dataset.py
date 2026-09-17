"""Custom metadata loaders for MuseGran training datasets.

Transforms raw JSONL/CSV metadata entries into the conditioning dict format
expected by the MuseGran training pipeline. Each loader returns a dict with
keys: prompt, style, structure, music_key, chord, beat, motif, seconds_start, seconds_total.
"""

import json
import os
import random

import numpy as np


def _parse_json_list(value):
    """Parse a value that may be a JSON string or already a list."""
    if not value:
        return []
    return json.loads(value) if isinstance(value, str) else value


def build_conditioning(info, audio=None, bpm_simu_prob=0, pmt_keyword_prob=0.5):
    """Build conditioning dict from a single sample's metadata.

    Args:
        info: Dict of raw metadata fields from JSONL/CSV.
        audio: Audio tensor (passed through for style conditioning reference).
        bpm_simu_prob: Probability of using synthetic BPM instead of beat file.
        pmt_keyword_prob: Probability of overriding description prompt with keywords.

    Returns:
        Dict with conditioning fields ready for the training collator.
    """
    melody_path = info.get('melody_path')
    chord_file = info.get('chord_path')
    beat_file = info.get('beat_path')
    bpm = float(info.get('bpm')) if info.get('bpm') else 0
    music_key = info.get('key')
    seconds_start = int(info.get('seconds_start', 0))
    seconds_total = int(info.get('seconds_total', 0) or info.get('duration_seconds', 0))
    crop_start = float(info.get('crop_start', seconds_start))
    crop_end = float(info.get('crop_end', seconds_total))

    prompt = (info.get('prompt') or "").strip()
    detailed_prompt = (info.get('detailed_prompt') or "").strip()

    genre = _parse_json_list(info.get('genre'))
    instruments = [item for item in _parse_json_list(info.get('instruments'))
                   if isinstance(item, str) and "vocal" not in item.lower()]
    mood = _parse_json_list(info.get('mood'))


    structure_infos = _parse_json_list(info.get('structure_segments_json', []))

    result = {
        'prompt': "",
        'style': None,
        'structure': None,
        'music_key': music_key,
        'chord': None,
        'beat': None,
        'motif': None,
        'seconds_start': seconds_start,
        'seconds_total': seconds_total,
    }

    # --- Style conditioning (audio reference for timbre) ---
    if audio is not None:
        result['style'] = {
            "audio": audio,
            "seconds_start": crop_start,
            "seconds_end": crop_end,
        }

    # --- Chord conditioning ---
    if chord_file and os.path.exists(chord_file):
        result['chord'] = {
            "chord_path": chord_file,
            "seconds_start": crop_start,
            "seconds_end": crop_end,
        }

    # --- Motif melody conditioning ---
    if melody_path and os.path.exists(melody_path):
        result['motif'] = {
            "midi_path": melody_path,
            "structure_infos": structure_infos,
        }

    # --- Structure conditioning ---
    if structure_infos:
        result['structure'] = {
            "structure_infos": structure_infos,
            "seconds_start": crop_start,
            "seconds_end": crop_end,
        }

    # --- Prompt ---
    result['prompt'] = _build_prompt(
        prompt, detailed_prompt, structure_infos, genre, instruments,
        mood, pmt_keyword_prob,
    )

    # --- Beat conditioning ---
    result['beat'] = _build_beat(beat_file, bpm, crop_start, crop_end, bpm_simu_prob)

    return result


def _build_prompt(prompt, detailed_prompt, structure_infos, genre, instruments,
                  mood, pmt_keyword_prob):
    """Construct text prompt for training.

    Randomly selects between: prompt, detailed_prompt (optional), or keywords (optional).
    Falls back to prompt if nothing else is available.
    """
    # Randomly pick between prompt / detailed_prompt / keywords
    if structure_infos and random.random() < 0.5:
        candidates = [p for p in [prompt, detailed_prompt] if p]
        selected = random.choice(candidates) if candidates else prompt
    else:
        selected = ""

    field_mappings = {
        "genre": genre,
        "instruments": instruments,
        "mood": mood,
    }

    valid_fields = []
    valid_keywords = []
    for field_name, field_value in field_mappings.items():
        if isinstance(field_value, list) and field_value:
            parts = [part.strip() for part in field_value if str(part).strip()]
            if parts:
                random.shuffle(parts)
                valid_fields.append(f"{field_name}: {', '.join(parts)}")
                valid_keywords.extend(parts)

    # Randomly choose between structured ("genre: X; instruments: Y") or flat ("X, Y, Z")
    if len(valid_fields) >= 2:
        use_keywords = (not selected) or (random.random() < pmt_keyword_prob)
        if use_keywords:
            if random.random() < 0.5:
                random.shuffle(valid_fields)
                selected = "; ".join(valid_fields)
            else:
                random.shuffle(valid_keywords)
                selected = ", ".join(valid_keywords)

    # Fallback: always return at least the prompt
    if not selected:
        if valid_keywords:
            random.shuffle(valid_keywords)
            selected = ", ".join(valid_keywords)
        else:
            selected = prompt

    return selected.rstrip(". ")


def _build_beat(beat_file, bpm, seconds_start, seconds_end, bpm_simu_prob):
    """Build beat conditioning dict.

    Two modes controlled by bpm_simu_prob:
      - File mode (default): pass the .npy beat annotation file directly.
        The conditioner reads per-beat timestamps from it.
      - BPM simulation mode: extract meter (time signature numerator) from
        the .npy file, then generate a uniform beat grid from BPM + meter.
        This teaches the model to accept BPM-only input at inference time.

    Falls back to file mode if BPM simulation fails (e.g. invalid meter or bpm=0).
    """
    if not beat_file or not os.path.exists(beat_file):
        return None

    # With probability (1 - bpm_simu_prob), use the real beat file directly
    if random.random() > bpm_simu_prob:
        return {"beat_path": beat_file, "seconds_start": seconds_start, "seconds_end": seconds_end}

    # Otherwise, simulate: read meter from file, pair with BPM for a synthetic grid
    try:
        beats_np = np.load(beat_file, allow_pickle=True)
        meter = int(np.max(beats_np[:, 1]))
        assert meter > 1 and bpm > 0
        return {"bpm": bpm, "meter": meter, "seconds_start": seconds_start, "seconds_total": seconds_end}
    except (ValueError, IndexError, AssertionError):
        return {"beat_path": beat_file, "seconds_start": seconds_start, "seconds_end": seconds_end}


def load_dataset(info, audio, **kwargs):
    """Entry point referenced by dataset config JSON files."""
    return build_conditioning(
        info, audio,
        bpm_simu_prob=kwargs.get("bpm_simu_prob", 0),        # prob of simulating BPM from beat intervals
        pmt_keyword_prob=kwargs.get("pmt_keyword_prob", 0.3),  # prob of appending keyword tags to prompt
    )
