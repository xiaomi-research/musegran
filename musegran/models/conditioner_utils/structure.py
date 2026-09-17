"""Structure annotation parsing for segment-level conditioning.

Converts JSON-based segment analysis annotations into frame-level structure
tensors with per-segment text descriptions, boundaries, and metadata.
"""

import json
import random

import torch


def _shuffle_str(s):
    """Shuffle comma-separated items in a string or list."""
    if isinstance(s, str):
        if not s:
            return s
        items = [item.strip() for item in s.split(',')]
    elif isinstance(s, list):
        if not s:
            return ''
        items = list(s)
    else:
        return ''
    random.shuffle(items)
    return ', '.join(items)


def _dot(s):
    """Ensure a non-empty string ends with a period."""
    if s and not s.endswith('.'):
        return f"{s}."
    return s


def _load_segments(structure):
    """Load segment list from a JSON path or pass through a list directly."""
    if isinstance(structure, str):
        if not structure.endswith('.json'):
            structure += '.json'
        with open(structure, 'r') as f:
            return list(json.load(f).values())[0]['segment analysis']
    elif isinstance(structure, list):
        return structure
    return []


# Text fields to extract from each segment, with optional label prefix
_TEXT_FIELDS = [
    ('style', 'Style'),
    ('emotions', 'Emotions'),
    ('rhythm', 'Rhythm'),
    ('instruments', 'Instruments'),
    ('mood', 'Moods'),
    ('dynamics_energy', 'Dynamics_energy'),
]


def _build_combined_text(fields: dict, seg_drop: float, text_only: bool, rel_start: float, rel_end: float):
    """Assemble combined text from segment fields with per-field random dropout.

    Returns the combined string, or None if all fields were dropped/empty.
    """
    if not any(fields.values()):
        return ""

    # Apply dropout and format each field
    parts = []
    for key, label in _TEXT_FIELDS:
        val = fields.get(key, '')
        if not val or random.random() < seg_drop:
            continue
        parts.append(f"{label}: {_dot(val)}")

    if not parts:
        return ""

    if text_only:
        header = f"The music section from {rel_start:.1f}s to {rel_end:.1f}s,"
        return " ".join([header] + parts).strip()

    random.shuffle(parts)
    return " ".join(parts).strip()


def get_structure_tensor_from_json(structure_infos, sample_size=None, compress_ratio=1, text_only=False, seg_drop=0.1, time_frame=None):
    """Parse segment structure annotations into frame-level conditioning data.

    Args:
        structure_infos: dict with 'structure_infos'/'seconds_start'/'seconds_end',
            or legacy list [structure, (start, end)].
        sample_size: latent frame count (before compress_ratio division).
        compress_ratio: additional temporal compression factor.
        text_only: if True, only return combined_text with timestamps.
        seg_drop: probability of dropping individual segment fields.
        time_frame: latent frames per second.

    Returns:
        dict with keys: latent_boundaries, audio_boundaries,
        mood, instrument, rms, combined_text.
        Returns None on parse failure.
    """
    from . import DEFAULT_TIME_FRAME
    if time_frame is None:
        time_frame = DEFAULT_TIME_FRAME

    time_frame /= compress_ratio
    sample_size //= int(compress_ratio)

    try:
        if isinstance(structure_infos, dict):
            structure = structure_infos['structure_infos']
            seconds_start = float(structure_infos['seconds_start'])
            seconds_end = float(structure_infos['seconds_end'])
        else:
            structure, timestamp = structure_infos[0], structure_infos[1]
            seconds_start = float(timestamp[0])
            seconds_end = float(timestamp[1])

        structures = _load_segments(structure)
        if not structures:
            return None

        offset_st_frame = int(seconds_start * time_frame)

        output = {
            'latent_boundaries': [],
            'audio_boundaries': [],
            'mood': [],
            'instrument': [],
            'rms': [],
            'combined_text': [],
        }

        for seg in structures:
            st_sec, ed_sec = float(seg['start']), float(seg['end'])

            if ed_sec <= seconds_start:
                continue
            if st_sec >= seconds_end:
                break

            st_sec = max(st_sec, seconds_start)
            ed_sec = min(ed_sec, seconds_end)

            st_frame = min(int(st_sec * time_frame) - offset_st_frame, sample_size)
            ed_frame = min(int(ed_sec * time_frame) - offset_st_frame, sample_size)

            # Extract and normalize text fields
            fields = {
                'style': (seg.get('style') or '').strip(),
                'emotions': (seg.get('emotions') or '').strip(),
                'rhythm': (seg.get('rhythm') or '').strip(),
                'instruments': _shuffle_str(seg.get('instruments') or seg.get('instrument') or ''),
                'mood': (seg.get('mood') or '').strip(),
                'dynamics_energy': ', '.join(_shuffle_list(seg.get('dynamics_energy') or [])),
            }

            combined_text = _build_combined_text(
                fields, seg_drop, text_only,
                st_sec - seconds_start, ed_sec - seconds_start,
            )

            if text_only:
                output['combined_text'].append(combined_text)
                continue

            output['combined_text'].append(combined_text)
            output['latent_boundaries'].append([st_frame, ed_frame])
            output['audio_boundaries'].append([
                max(st_sec - seconds_start, 0),
                min(ed_sec - seconds_start, sample_size / time_frame),
            ])
            output['mood'].append(fields['mood'])
            output['instrument'].append(fields['instruments'])
            output['rms'].append(torch.tensor(round(seg['rms_dB'])) if seg.get('rms_dB') is not None else None)

        return output
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
        return None


def _shuffle_list(lst):
    """Return a shuffled copy of the list."""
    lst = list(lst)
    random.shuffle(lst)
    return lst


def get_motif_boundary(structure_file, required_labels=('chorus', 'verse')):
    """Find the time boundary of the first matching segment label.

    Searches for segments with labels in required_labels, returning [start, end].
    Falls back to the first non-intro/outro segment if no exact match found.
    """
    _SKIP_LABELS = {'intro', 'outro', 'silence', 'end', ''}

    try:
        structures = _load_segments(structure_file)

        default_output = None
        for seg in structures:
            st_sec, ed_sec = float(seg['start']), float(seg['end'])
            label = (seg.get("label") or "").lower()

            if label in _SKIP_LABELS:
                continue
            if label in required_labels:
                return [st_sec, ed_sec]
            if default_output is None:
                default_output = [st_sec, ed_sec]

        return default_output
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
        return None
