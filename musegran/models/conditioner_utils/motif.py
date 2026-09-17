"""MIDI motif extraction: parse MIDI → piano-roll tensor for MotifMelodyConditioner."""

import numpy as np
import pretty_midi
import torch
from torch.nn.utils.rnn import pad_sequence

from .structure import get_motif_boundary

MIN_NOTE_DURATION = 0.05
MIN_NOTE_VELOCITY = 30


def midi_to_pianoroll(input_info, fs=10, max_len=256, pitch_start=0, pitch_end=128):
    """Convert MIDI to piano-roll tensor [T, n_pitch]. Returns empty (0, n_pitch) on failure."""
    n_pitch = pitch_end - pitch_start
    empty = torch.zeros((0, n_pitch))

    if input_info is None:
        return empty

    try:
        if isinstance(input_info, dict):
            midi_path = input_info['midi_path']
            structure_file = input_info['structure_infos']
        else:
            midi_path, structure_file = input_info
        if midi_path is None or structure_file is None:
            return empty

        motif_boundary = get_motif_boundary(structure_file)
        if motif_boundary is None or len(motif_boundary) != 2:
            return empty

        start_sec, end_sec = motif_boundary
        pm = pretty_midi.PrettyMIDI(midi_path)
        if not pm.instruments:
            return empty

        instrument = max(pm.instruments, key=lambda i: len(i.notes))
        instrument.notes = [
            n for n in instrument.notes
            if (n.end - n.start > MIN_NOTE_DURATION) and (n.velocity > MIN_NOTE_VELOCITY)
        ]

        roll = instrument.get_piano_roll(fs=fs)[pitch_start:pitch_end, :]

        start_idx = int(round(start_sec * fs))
        end_idx = int(round(end_sec * fs))
        if start_idx >= roll.shape[1]:
            return empty
        roll = roll[:, start_idx:min(end_idx, roll.shape[1])]

        non_zero_cols = np.any(roll, axis=0)
        if not np.any(non_zero_cols):
            return empty
        first = np.argmax(non_zero_cols)
        last = len(non_zero_cols) - 1 - np.argmax(non_zero_cols[::-1])
        roll = roll[:, first:last + 1]

        if max_len is not None and roll.shape[1] > max_len:
            roll = roll[:, :max_len]

        roll = torch.tensor(roll.T.astype(np.float32))
        return roll if roll.shape[0] > 0 else empty

    except (OSError, ValueError, IndexError, KeyError, TypeError):
        return empty


def batch_pad_pianorolls(sequences, n_pitch, device, max_len=None):
    """Pad/tile piano-roll sequences to uniform length. Returns (padded [B,T,D], lengths [B])."""
    lengths = torch.tensor([s.shape[0] for s in sequences], dtype=torch.long, device=device)

    if not sequences:
        return torch.zeros((0, max_len or 0, n_pitch), device=device), lengths

    if max_len is None:
        max_len = lengths.max().item()
        padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    else:
        aligned = []
        for s in sequences:
            cur_len = s.shape[0]
            if cur_len >= max_len:
                aligned.append(s[:max_len])
            elif cur_len == 0:
                aligned.append(torch.zeros((max_len, n_pitch)))
            else:
                repeats = [1] * s.dim()
                repeats[0] = (max_len // cur_len) + 1
                aligned.append(s.repeat(*repeats)[:max_len])
        padded = torch.stack(aligned).to(device)

    if padded.shape[1] < max_len:
        padded = torch.nn.functional.pad(padded, (0, 0, 0, max_len - padded.shape[1]))

    return padded, lengths
