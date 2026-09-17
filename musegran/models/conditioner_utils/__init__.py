"""Shared utilities for conditioner data preprocessing.

Provides constants, frame computation, and beat/bar parsing used across
chord, beat, structure, and style conditioners.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 44100
DEFAULT_DOWNSAMPLING_RATIO = 2048
DEFAULT_TIME_FRAME = DEFAULT_SAMPLE_RATE / DEFAULT_DOWNSAMPLING_RATIO


def compute_frame_range(seconds_start, seconds_end, sample_size, time_frame=DEFAULT_TIME_FRAME):
    """Convert time range to frame indices, clamped to sample_size."""
    offset_st_frame = int(seconds_start * time_frame)
    offset_ed_frame = min(int(seconds_end * time_frame), sample_size + offset_st_frame)
    return offset_st_frame, offset_ed_frame


def parse_beat_infos(beat_infos):
    """Parse beat_infos into (beat, meter, seconds_start, seconds_end) or None.

    Accepted formats:
      - dict with 'beat_path': file-based beat annotations
      - dict with 'bpm': synthetic beat from BPM + meter
      - list [path_or_bpm, (start, end), ...]: legacy positional format
    """
    if beat_infos is None:
        return None
    if isinstance(beat_infos, dict):
        if 'beat_path' in beat_infos:
            beat = beat_infos['beat_path']
            meter = beat_infos.get('meter', 4)
            seconds_start = float(beat_infos['seconds_start'])
            seconds_end = float(beat_infos['seconds_end'])
        else:
            beat = beat_infos.get('bpm', 120)
            meter = beat_infos.get('meter', 4)
            seconds_start = float(beat_infos.get('seconds_start', 0))
            seconds_end = float(beat_infos.get('seconds_total', 60))
    elif isinstance(beat_infos, list) and len(beat_infos) > 0:
        beat, timestamp = beat_infos[0], beat_infos[1]
        meter = beat_infos[-1] if isinstance(beat, (int, float)) else 4
        seconds_start = float(timestamp[0])
        seconds_end = float(timestamp[1])
    else:
        return None
    return beat, meter, seconds_start, seconds_end


def get_beat_and_bar_frames(beat, meter, offset_st_frame, offset_ed_frame, time_frame=DEFAULT_TIME_FRAME):
    """Get beat_frame and bar_frame lists from a beat source (.npy file or BPM value).
    Returns (beat_frame, bar_frame) or None if no valid frames."""
    if isinstance(beat, str) and beat.endswith('.npy'):
        beats_np = np.load(beat, allow_pickle=True)
        beat_time = beats_np[:, 0]
        bar_time = beats_np[np.where(beats_np[:, 1] == 1)[0], 0]

        beat_frame = [int(t * time_frame) for t in beat_time]
        beat_frame = [x - offset_st_frame for x in beat_frame if offset_st_frame <= x < offset_ed_frame]

        bar_frame = [int(t * time_frame) for t in bar_time]
        bar_frame = [x - offset_st_frame for x in bar_frame if offset_st_frame <= x < offset_ed_frame]

    elif isinstance(beat, (int, float)):
        beat_gap = 60 / beat * time_frame
        beat_frame = np.round(np.arange(offset_st_frame, offset_ed_frame, beat_gap)).astype(int)
        if len(beat_frame) == 0:
            return None
        if beat_frame[-1] == offset_ed_frame:
            beat_frame = beat_frame[:-1]
        beat_frame = [x - offset_st_frame for x in beat_frame if x < offset_ed_frame]
        bar_frame = list(beat_frame[::meter]) if meter > 0 else []
    else:
        return None

    return beat_frame, bar_frame


