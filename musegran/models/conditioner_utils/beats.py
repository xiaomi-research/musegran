# Copyright (c) 2024 Cyan
# SPDX-License-Identifier: MIT
# Adapted from https://github.com/YatingMusic/MusiConGen

"""Beat and bar position encoding for rhythmic conditioning.

Provides:
- get_beat_tensor: smoothed beat/bar onset features for BeatConditioner
- get_beat_pos: per-frame beat position IDs for BeatPositions
- get_bar_pos: per-frame bar position IDs (+ bar frame list) for BarPositions
"""

import numpy as np
import torch

from . import (
    DEFAULT_TIME_FRAME,
    compute_frame_range,
    parse_beat_infos,
    get_beat_and_bar_frames,
)


def _frames_to_position_ids(frames, total_length):
    """Assign incremental position IDs to segments between frame boundaries.

    Each segment between consecutive frames gets a unique ascending ID.
    Returns array of shape [total_length + 1] with a leading zero (for padding offset).
    """
    pos = np.zeros(total_length)
    pos_id = 1

    if len(frames) == 0:
        pos[:] = pos_id
    else:
        if frames[0] > 0:
            pos[:frames[0]] = pos_id
            pos_id += 1
        for i in range(len(frames)):
            st = frames[i]
            ed = frames[i + 1] if i < len(frames) - 1 else total_length
            pos[st:ed] = pos_id
            pos_id += 1

    output = np.zeros(total_length + 1)
    output[1:] = pos
    return torch.tensor(output)


def _parse_and_get_frames(beat_infos, sample_size, time_frame):
    """Common setup: parse beat_infos → compute frames. Returns None on failure."""
    parsed = parse_beat_infos(beat_infos)
    if parsed is None:
        return None

    beat, meter, seconds_start, seconds_end = parsed
    offset_st_frame, offset_ed_frame = compute_frame_range(seconds_start, seconds_end, sample_size, time_frame)

    result = get_beat_and_bar_frames(beat, meter, offset_st_frame, offset_ed_frame, time_frame)
    if result is None:
        return None

    beat_frame, bar_frame = result
    return beat_frame, bar_frame, sample_size


def get_beat_tensor(beat_infos, sample_size, time_frame=DEFAULT_TIME_FRAME):
    """Generate smoothed beat event features [sample_size, 1] for BeatConditioner."""
    result = _parse_and_get_frames(beat_infos, sample_size, time_frame)
    if result is None:
        return torch.zeros((sample_size, 1))

    beat_frame, bar_frame, _ = result

    # Soft beat impulses smoothed with a gaussian-like kernel
    feat_beats = np.zeros((2, sample_size))
    feat_beats[0, beat_frame] = 1
    feat_beats[1, bar_frame] = 1
    kernel = np.array([0.05, 0.1, 0.3, 0.9, 0.3, 0.1, 0.05])
    feat_beats[0] = np.convolve(feat_beats[0], kernel, 'same')

    beat_events = feat_beats[0] + feat_beats[1]
    return torch.tensor(beat_events).unsqueeze(-1)


def get_beat_pos(beat_infos, sample_size, time_frame=DEFAULT_TIME_FRAME):
    """Generate per-frame beat position IDs [sample_size + 1] for BeatPositions."""
    result = _parse_and_get_frames(beat_infos, sample_size, time_frame)
    if result is None:
        return None

    beat_frame, _, adjusted_ed_frame = result
    return _frames_to_position_ids(beat_frame, adjusted_ed_frame)


def get_bar_pos(beat_infos, sample_size, time_frame=DEFAULT_TIME_FRAME):
    """Generate per-frame bar position IDs [sample_size + 1] + bar frame list for BarPositions."""
    result = _parse_and_get_frames(beat_infos, sample_size, time_frame)
    if result is None:
        return None

    _, bar_frame, adjusted_ed_frame = result
    bar_pos = _frames_to_position_ids(bar_frame, adjusted_ed_frame)

    # Append end frame for downstream bar boundary usage
    if len(bar_frame) > 0 and bar_frame[-1] != adjusted_ed_frame:
        bar_frame = bar_frame + [adjusted_ed_frame]
    elif len(bar_frame) == 0:
        bar_frame = []

    return bar_pos, bar_frame
