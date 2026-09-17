"""StructPlanner: Qwen3-14B + LoRA for converting free-form text
into structured music generation specifications.

Wraps the fine-tuned LLM with utilities for inference, schema validation,
and conversion to MuseGran diffusion model conditioning format.
"""

from __future__ import annotations

import json
import os
import random
import typing as tp

from dataclasses import dataclass

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Constants ──────────────────────────────────────────────────────────

VALID_KEYS = [
    "C major", "C minor", "C# major", "C# minor", "Db major", "Db minor",
    "D major", "D minor", "D# major", "D# minor", "Eb major", "Eb minor",
    "E major", "E minor", "F major", "F minor", "F# major", "F# minor",
    "Gb major", "Gb minor", "G major", "G minor", "G# major", "G# minor",
    "Ab major", "Ab minor", "A major", "A minor", "A# major", "A# minor",
    "Bb major", "Bb minor", "B major", "B minor",
]

VALID_DYNAMICS = ["pp", "p", "mp", "mf", "f", "ff"]

BPM_RANGE = (50, 210)
METER_VALUES = [2, 3, 4]
BARS_RANGE = (1, 16)

DYNAMICS_TO_RMS: tp.Dict[str, float] = {
    "pp": -27, "p": -21, "mp": -18, "mf": -15, "f": -12, "ff": -6,
}

SYSTEM_PROMPT = """\
You are a music specification assistant. Convert the user's music request \
into a structured JSON that an AI music generation system can understand.

Output ONLY a valid JSON object with this schema:
{
  "key": "string",
  "bpm": number,
  "meter": number,
  "summary_description": "string",
  "detailed_description": "string",
  "outro_bars": number,
  "segment_analysis": [
    {
      "bars": number,
      "dynamics": "string (pp/p/mp/mf/f/ff)",
      "style": "string",
      "rhythm": "string",
      "emotions": "string",
      "instruments": "string"
    }
  ]
}

No markdown, no extra text, only JSON."""


# ── Helpers ────────────────────────────────────────────────────────────

def parse_json_response(response: str) -> tp.Tuple[tp.Optional[dict], tp.Optional[str]]:
    """Try to parse JSON from model response. Returns (parsed_dict, error_msg)."""
    text = response.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, f"JSONDecodeError: {str(e)[:100]}"


def find_latest_checkpoint(lora_dir: str) -> str:
    """Find the latest checkpoint subdirectory inside a LoRA directory."""
    ckpts = sorted(
        [d for d in os.listdir(lora_dir) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[1]),
    )
    if not ckpts:
        return lora_dir
    return os.path.join(lora_dir, ckpts[-1])




# ── Spec → Conditioning Conversion ────────────────────────────────────

def spec_to_conditioning(
    spec: dict,
    duration: tp.Optional[float] = None,
    prompt_mode: str = "summary",
) -> tp.Tuple[tp.List[dict], tp.List[dict], float]:
    """Convert a StructPlanner specification to MuseGran conditioning format.

    Args:
        spec: The parsed JSON spec from StructPlanner (V4 format).
        duration: Override duration in seconds. If None, computed from
                  total_bars + outro_bars × meter × 60 / bpm.
        prompt_mode: Which description to use for the text prompt:
                     'summary', 'detail', or 'random'.

    Returns:
        (conditioning, negative_conditioning, duration_seconds)
    """
    bpm = float(spec.get("bpm", 120))
    meter = int(spec.get("meter", 4))
    key = spec.get("key", "") or ""
    segments = spec.get("segment_analysis", [])
    outro_bars = spec.get("outro_bars", 0) or 0

    if duration is None:
        total_bars = sum(s.get("bars", 0) for s in segments) + outro_bars
        if total_bars > 0 and bpm > 0 and meter > 0:
            duration = round(total_bars * meter * 60.0 / bpm)
        else:
            duration = 60.0
    duration = min(duration, 180.0)

    # V4 flattens descriptions to top level
    summary = spec.get("summary_description", "") or ""
    detailed = spec.get("detailed_description", "") or ""

    if prompt_mode == "summary":
        prompt_text = summary or detailed
    elif prompt_mode == "detail":
        prompt_text = detailed or summary
    else:
        candidates = [p for p in [summary, detailed] if p]
        prompt_text = random.choice(candidates) if candidates else ""

    structure_infos = None
    if segments and bpm > 0 and meter > 0:
        bar_duration = meter * 60.0 / bpm
        t = 0.0
        structure_list = []
        for seg in segments:
            bars = seg.get("bars", 8)
            seg_duration = bars * bar_duration
            dynamics = seg.get("dynamics", "mf")
            rms_db = DYNAMICS_TO_RMS.get(dynamics, -15)
            structure_list.append({
                "label": "inst",
                "start": round(t, 2),
                "end": round(t + seg_duration, 2),
                "rms_dB": rms_db,
                "is_silent": False,
                "style": seg.get("style", ""),
                "rhythm": seg.get("rhythm", ""),
                "emotions": seg.get("emotions", ""),
                "instruments": seg.get("instruments", ""),
            })
            t += seg_duration
        structure_infos = [structure_list, (0, duration), None]

    beat = {"bpm": bpm, "seconds_start": 0, "seconds_total": int(duration), "meter": meter}

    conditioning = [{
        "prompt": prompt_text,
        "beat": beat,
        "seconds_start": 0,
        "seconds_total": int(duration),
        "music_key": key if key else None,
        "structure": structure_infos,
    }]

    negative_conditioning = [{
        "prompt": "sampled vocals, noise, distorted harmonics, artifacts, low quality",
        "chord": None,
        "beat": None,
        "seconds_start": 0,
        "seconds_total": int(duration),
        "music_key": None,
        "structure": None,
    }]

    return conditioning, negative_conditioning, duration


# ── StructPlanner Model ────────────────────────────────────────────────

_DEFAULT_LORA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "pretrained_models", "structplanner_lora",
)


@dataclass
class StructPlannerConfig:
    """Configuration for StructPlanner model loading."""

    base_model: str = "Qwen/Qwen3-14B"
    lora_dir: str = _DEFAULT_LORA
    checkpoint: tp.Optional[str] = None
    device: str = "cuda"
    torch_dtype: torch.dtype = torch.bfloat16
    max_new_tokens: int = 4096
    temperature: float = 0.3
    top_p: float = 0.9
    seed: int = -1


class StructPlanner:
    """Qwen3-14B + LoRA model for music structure planning.

    Converts free-form text prompts into structured JSON specifications
    compatible with MuseGran's diffusion model conditioning pipeline.
    """

    def __init__(self, config: StructPlannerConfig):
        self.config = config
        self.model: tp.Optional[torch.nn.Module] = None
        self.tokenizer: tp.Optional[AutoTokenizer] = None

        # Resolve checkpoint path
        if config.checkpoint:
            lora_path = os.path.join(config.lora_dir, config.checkpoint)
        else:
            lora_path = find_latest_checkpoint(config.lora_dir)

        print(f"StructPlanner loading base model: {config.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.base_model, trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=config.torch_dtype,
            device_map=config.device,
            trust_remote_code=True,
        )
        print(f"StructPlanner loading LoRA: {lora_path}")
        self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.model.eval()

        if config.seed >= 0:
            torch.manual_seed(config.seed)
            random.seed(config.seed)

    @classmethod
    def from_pretrained(
        cls,
        lora_dir: str,
        base_model: str = "Qwen/Qwen3-14B",
        checkpoint: tp.Optional[str] = None,
        device: str = "cuda",
        **kwargs,
    ) -> "StructPlanner":
        """Convenience constructor with inline paths."""
        cfg = StructPlannerConfig(
            base_model=base_model,
            lora_dir=lora_dir,
            checkpoint=checkpoint,
            device=device,
            **{k: v for k, v in kwargs.items() if hasattr(StructPlannerConfig, k)},
        )
        return cls(cfg)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a raw JSON response string from a text prompt."""
        assert self.model is not None and self.tokenizer is not None

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        temperature = kwargs.pop("temperature", self.config.temperature)
        max_new_tokens = kwargs.pop("max_new_tokens", self.config.max_new_tokens)

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def generate_spec(self, prompt: str, **kwargs) -> tp.Tuple[tp.Optional[dict], tp.Optional[str]]:
        """Generate and parse a structured specification.

        Returns:
            (parsed_dict, error_message)
        """
        response = self.generate(prompt, **kwargs)
        spec, error = parse_json_response(response)
        if spec is not None:
            return spec, None
        return None, error

    def print_spec(self, spec: dict, prompt: str):
        """Pretty-print a generated specification."""
        segments = spec.get("segment_analysis", [])
        outro = spec.get("outro_bars", 0) or 0
        total_bars = sum(s.get("bars", 0) for s in segments) + outro
        bpm = spec.get("bpm", 120)
        meter = spec.get("meter", 4)
        duration = round(total_bars * meter * 60.0 / bpm) if bpm > 0 else 0
        print(f"key={spec.get('key')}, bpm={bpm}, meter={meter}, "
              f"total_bars={total_bars}, duration≈{duration}s")
        print(f"summary: {(spec.get('summary_description') or '')[:80]}")
        print(f"segments ({len(segments)}):")
        for s in segments:
            print(f"  {int(s.get('bars', 0)):2d} bars | "
                  f"{(s.get('dynamics') or ''):3s} | {(s.get('instruments') or '')[:50]}")
