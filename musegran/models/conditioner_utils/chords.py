# Copyright (c) 2024 Cyan
# SPDX-License-Identifier: MIT
# Adapted from https://github.com/YatingMusic/MusiConGen

"""Chord parsing and chroma tensor generation for chord conditioning.

Provides a Chords class that parses chord labels (e.g. "C:maj7", "Eb:min")
into 12-dimensional chroma vectors, and a get_chord2chroma_tensor function
that converts chord annotation files or symbolic chord sequences into
frame-level chroma features for the diffusion model.
"""

import numpy as np
import torch

NO_CHORD = (-1, -1, np.zeros(12, dtype=np.int_), False)
UNKNOWN_CHORD = (-1, -1, np.ones(12, dtype=np.int_) * -1, False)


class Chords:
    """Parses chord labels into numeric (root, bass, intervals, is_major) tuples."""

    def __init__(self):
        self._shorthands = {
            'maj': self.interval_list('(1,3,5)'),
            'min': self.interval_list('(1,b3,5)'),
            'dim': self.interval_list('(1,b3,b5)'),
            'hdim7': self.interval_list('(1,b3,b5,b7)'),
            'dim7': self.interval_list('(1,b3,b5,bb7)'),
            'dim9': self.interval_list('(1,b3,b5,b7,b9)'),
            'aug': self.interval_list('(1,3,#5)'),
            'maj6': self.interval_list('(1,3,5,6)'),
            'maj7': self.interval_list('(1,3,5,7)'),
            'maj9': self.interval_list('(1,3,5,7,9)'),
            'maj11': self.interval_list('(1,3,5,7,9)'),
            'maj13': self.interval_list('(1,3,5,7,13)'),
            '13': self.interval_list('(1,3,5,b7,13)'),
            '11': self.interval_list('(1,3,5,b7,9,11)'),
            '9': self.interval_list('(1,3,5,b7,9)'),
            '7': self.interval_list('(1,3,5,b7)'),
            '6': self.interval_list('(1,6)'),
            '5': self.interval_list('(1,5)'),
            '4': self.interval_list('(1,4)'),
            '1': self.interval_list('(1)'),
            'min6': self.interval_list('(1,b3,5,6)'),
            'min7': self.interval_list('(1,b3,5,b7)'),
            'min9': self.interval_list('(1,b3,5,b7,9)'),
            'min11': self.interval_list('(1,b3,5,b7,9,11)'),
            'min13': self.interval_list('(1,b3,5,b7,13)'),
            'minmaj7': self.interval_list('(1,b3,5,7)'),
            'minmaj9': self.interval_list('(1,b3,5,7,9)'),
            'minmaj11': self.interval_list('(1,b3,5,7,9,11)'),
            'add9': self.interval_list('(1,3,5,9)'),
            'add11': self.interval_list('(1,3,5,11)'),
            'add13': self.interval_list('(1,3,5,13)'),
            'sus2': self.interval_list('(1,2,5)'),
            'sus4': self.interval_list('(1,4,5)'),
            '7sus2': self.interval_list('(1,2,5,b7)'),
            '7sus4': self.interval_list('(1,4,5,b7)'),
            'maj7sus2': self.interval_list('(1,2,5,7)'),
            '7b9': self.interval_list('(1,3,5,b7,b9)'),
        }

    # Semitone offsets for diatonic scale: 0=whole step, 1=half step (B-C, E-F)
    _l = [0, 1, 1, 0, 1, 1, 1]
    # Maps interval numbers (1-14) to chromatic pitch classes (0-11)
    _chroma_id = (np.arange(len(_l) * 2) + 1) + np.array(_l + _l).cumsum() - 1

    def chord(self, label):
        """Parse a chord label string into (root, bass, intervals, is_major)."""
        if label == 'N':
            return NO_CHORD
        if label == 'X':
            return UNKNOWN_CHORD

        label = self._fix_label(label)

        c_idx = label.find(':')
        s_idx = label.find('/')

        if c_idx == -1:
            quality_str = 'maj'
            if s_idx == -1:
                root_str = label
                bass_str = ''
            else:
                root_str = label[:s_idx]
                bass_str = label[s_idx + 1:]
        else:
            root_str = label[:c_idx]
            if s_idx == -1:
                quality_str = label[c_idx + 1:]
                bass_str = ''
            else:
                quality_str = label[c_idx + 1:s_idx]
                bass_str = label[s_idx + 1:]

        root = self._pitch(root_str)
        bass = self._interval(bass_str) if bass_str else 0
        ivs = self._chord_intervals(quality_str)
        ivs[bass] = 1
        is_major = 'min' not in quality_str

        return root, bass, ivs, is_major

    _LABEL_FIXES = {
        'Emin/4': 'E:min/4',
        'A7/3': 'A:7/3',
        'Bb7/3': 'Bb:7/3',
        'Bb7/5': 'Bb:7/5',
    }

    def _fix_label(self, label):
        """Fix common chord label formatting errors."""
        if label in self._LABEL_FIXES:
            return self._LABEL_FIXES[label]
        if ':' not in label and 'min' in label:
            idx = label.find('min')
            return label[:idx] + ':' + label[idx:]
        return label

    def _modify(self, base_pitch, modifier):
        for m in modifier:
            if m == 'b':
                base_pitch -= 1
            elif m == '#':
                base_pitch += 1
            else:
                raise ValueError(f'Unknown modifier: {m}')
        return base_pitch

    def _pitch(self, pitch_str):
        return self._modify(self._chroma_id[(ord(pitch_str[0]) - ord('C')) % 7], pitch_str[1:]) % 12

    def _interval(self, interval_str):
        for i, c in enumerate(interval_str):
            if c.isdigit():
                return self._modify(self._chroma_id[int(interval_str[i:]) - 1], interval_str[:i]) % 12

    def interval_list(self, intervals_str, given_pitch_classes=None):
        """Convert interval list string to binary chroma array."""
        if given_pitch_classes is None:
            given_pitch_classes = np.zeros(12, dtype=np.int_)
        for int_def in intervals_str[1:-1].split(','):
            int_def = int_def.strip()
            if int_def[0] == '*':
                given_pitch_classes[self._interval(int_def[1:])] = 0
            else:
                given_pitch_classes[self._interval(int_def)] = 1
        return given_pitch_classes

    def _chord_intervals(self, quality_str):
        list_idx = quality_str.find('(')
        if list_idx == -1:
            return self._shorthands[quality_str].copy()
        if list_idx != 0:
            ivs = self._shorthands[quality_str[:list_idx]].copy()
        else:
            ivs = np.zeros(12, dtype=np.int_)
        return self.interval_list(quality_str[list_idx:], ivs)


_chords = Chords()


def get_chord2chroma_tensor(chord_infos, sample_size, drop=False, time_frame=None):
    """Convert chord annotations to a frame-level chroma tensor [T, 12].

    Args:
        chord_infos: dict with 'chord_path' or 'chord', or legacy list format, or None.
        sample_size: number of latent frames.
        drop: if True, emphasize root note at chord onset frames.
        time_frame: latent frames per second (default: 44100/2048).

    Returns:
        Tensor [sample_size, 12]. All -1 indicates unknown/missing chords.
    """
    from . import DEFAULT_TIME_FRAME
    if time_frame is None:
        time_frame = DEFAULT_TIME_FRAME

    unknown = torch.ones((sample_size, 12)) * (-1)

    if isinstance(chord_infos, dict):
        if 'chord_path' in chord_infos:
            chord = chord_infos['chord_path']
            seconds_start = float(chord_infos['seconds_start'])
            seconds_end = float(chord_infos['seconds_end'])
        else:
            chord = chord_infos['chord']
            seconds_start = float(chord_infos.get('seconds_start', 0))
            seconds_end = float(chord_infos.get('seconds_total', 60))
    elif isinstance(chord_infos, list) and len(chord_infos) > 0:
        chord, timestamp = chord_infos
        seconds_start = float(timestamp[0])
        seconds_end = float(timestamp[1])
    else:
        return unknown

    offset_st_frame = int(seconds_start * time_frame)
    offset_ed_frame = int(seconds_end * time_frame)
    feat_chord = np.zeros((sample_size, 12))

    if isinstance(chord, str) and chord.endswith('.lab'):
        _fill_from_lab_file(chord, feat_chord, seconds_start, seconds_end, offset_st_frame, sample_size, time_frame, drop)
    elif isinstance(chord, str):
        bpm = chord_infos.get('bpm', 128) if isinstance(chord_infos, dict) else 128
        beats_per_chord = chord_infos.get('beats_per_chord', 4) if isinstance(chord_infos, dict) else 4
        _fill_from_chord_sequence(chord, feat_chord, bpm, beats_per_chord, offset_st_frame, offset_ed_frame, time_frame, drop)
    else:
        return unknown

    return torch.tensor(feat_chord, dtype=torch.float)


def _fill_from_lab_file(chord_file, feat_chord, seconds_start, seconds_end, offset_st_frame, sample_size, time_frame, drop):
    """Fill chroma features in-place from a .lab chord annotation file."""
    with open(chord_file, 'r') as f:
        for line in f:
            splits = line.split()
            if len(splits) != 3:
                continue
            st_sec, ed_sec, ctag = float(splits[0]), float(splits[1]), splits[2]

            if ed_sec <= seconds_start:
                continue
            if st_sec >= seconds_end:
                break

            st_sec = max(st_sec, seconds_start)
            ed_sec = min(ed_sec, seconds_end)

            st_frame = min(int(st_sec * time_frame) - offset_st_frame, sample_size)
            ed_frame = min(int(ed_sec * time_frame) - offset_st_frame, sample_size)
            if st_frame == ed_frame:
                continue

            root, _, intervals, _ = _chords.chord(ctag)
            feat_chord[st_frame:ed_frame, :] = np.roll(intervals, root)
            if drop and root != -1:
                feat_chord[st_frame, root] += 1


def _fill_from_chord_sequence(chord_str, feat_chord, bpm, beats_per_chord, offset_st_frame, offset_ed_frame, time_frame, drop):
    """Fill chroma features in-place from a symbolic chord sequence string."""
    frames_per_chord = int(60.0 / bpm * beats_per_chord * time_frame)
    chord_sequence = chord_str.split(" ")
    frame_idx = 0
    drift = offset_st_frame

    while frame_idx < offset_ed_frame:
        for token_group in chord_sequence:
            if frame_idx >= offset_ed_frame:
                break
            sub_tokens = token_group.split(',')
            n_sub = len(sub_tokens)

            for token in sub_tokens:
                if frame_idx >= offset_ed_frame:
                    break
                raw_frames = frames_per_chord + drift
                rounded_frames = round(raw_frames)
                drift = (raw_frames - rounded_frames) / n_sub
                sub_frames = rounded_frames // n_sub

                root, _, intervals, _ = _chords.chord(token)
                chroma = np.roll(intervals, root)
                end_frame = min(frame_idx + sub_frames, offset_ed_frame)
                feat_chord[frame_idx:end_frame] = chroma
                if drop and root != -1:
                    feat_chord[frame_idx, root] += 1
                frame_idx = end_frame
