"""MuseGran inference.

Automatically selects model based on input JSONL content:
- If samples contain structure_segments_json → MuseGran-Co (control-oriented, with structure conditioning)
- Otherwise → MuseGran-Qo (quality-oriented, text + beat + key only)

When --structplanner is set, the text prompt is first run through StructPlanner
(Qwen3-14B + LoRA) to generate a full structured spec, which is then used as
conditioning for the diffusion model (always uses MuseGran-Co in this mode).

Usage:
    python inference.py --test_jsonl prompts_qo.jsonl
    python inference.py --prompt "A vibrant folk tune" --structplanner
    python inference.py --prompt "安静的钢琴曲" --structplanner --duration 90
"""

import argparse
import json
import os
import re
import time

import torch
import torchaudio
from einops import rearrange

from musegran import StructPlanner, get_pretrained_model, spec_to_conditioning
from musegran.inference.generation import generate_diffusion_cond

_MODEL_DIR = "./pretrained_models/diffusion"
_CO_CKPT = "musegran_co.ckpt"
_CO_CONFIG = "configs/model/musegran_co.yaml"
_QO_CKPT = "musegran_qo.ckpt"
_QO_CONFIG = "configs/model/musegran_qo.yaml"
_VAE_CKPT = "./pretrained_models/vae/vae_sr.ckpt"
_SP_LORA = "./pretrained_models/structplanner_lora"


def parse_args():
    parser = argparse.ArgumentParser(description="MuseGran Inference")
    parser.add_argument("--test_jsonl", default=None,
                        help="JSONL file with test samples (batch mode)")
    parser.add_argument("--structure_json", default=None,
                        help="Single-sample JSON file (same schema as one JSONL line); "
                             "overrides --prompt/--test_jsonl")
    parser.add_argument("--output_dir", default="./inference_samples/generated",
                        help="Output directory for generated audio")
    parser.add_argument("--cfg_scale", type=float, default=5.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--steps", type=int, default=100,
                        help="Diffusion sampling steps")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: random)")
    # Quick mode: generate a single sample from command-line args
    parser.add_argument("--prompt", default=None,
                        help="Text prompt (quick mode, single sample)")
    parser.add_argument("--bpm", type=float, default=120,
                        help="BPM (quick mode)")
    parser.add_argument("--key", default=None,
                        help="Musical key, e.g. 'C major' (quick mode)")
    parser.add_argument("--duration", type=float, default=120,
                        help="Duration in seconds (quick mode)")
    # StructPlanner mode
    parser.add_argument("--structplanner", action="store_true",
                        help="Enable StructPlanner: use Qwen3-14B + LoRA to convert "
                             "prompt into structured spec before generation")
    return parser.parse_args()


def _has_structure(samples):
    """Check if samples contain non-empty structure conditioning."""
    first = samples[0]
    raw = first.get('structure_segments_json', '[]')
    segs = json.loads(raw) if isinstance(raw, str) else raw
    return bool(segs)


def build_prompt_description(sample):
    return sample.get('prompt', '') or sample.get('detailed_prompt', '')


def build_structure(sample, duration):
    seg_raw = sample.get('structure_segments_json', '[]')
    structure_infos = json.loads(seg_raw) if isinstance(seg_raw, str) else seg_raw
    if not structure_infos:
        return None
    return {"structure_infos": structure_infos, "seconds_start": 0, "seconds_end": duration}


def _sanitize_prompt(text: str, max_len: int = 30) -> str:
    s = text.strip().lower()
    s = re.sub(r'[^a-z0-9\u4e00-\u9fff]+', '_', s)
    s = s.strip('_')
    return s[:max_len]


def _next_gen_id(output_dir: str) -> int:
    max_id = 0
    if not os.path.isdir(output_dir):
        return 1
    for fname in os.listdir(output_dir):
        m = re.match(r'gen(\d+)_.*\.(wav|json)$', fname)
        if m:
            max_id = max(max_id, int(m.group(1)))
    return max_id + 1


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    next_id = _next_gen_id(args.output_dir)
    print(f"Starting gen ID: {next_id}")

    # Single sample: from a JSON file or command-line args
    if args.structure_json:
        with open(args.structure_json, 'r', encoding='utf-8') as f:
            samples = [json.load(f)]
        print(f"Single sample loaded from {args.structure_json}")
    elif args.prompt:
        samples = [{
            "prompt": args.prompt,
            "bpm": args.bpm,
            "key": args.key,
            "duration_seconds": args.duration,
        }]
        print(f"Single sample: \"{args.prompt[:80]}...\"")
    elif args.test_jsonl:
        with open(args.test_jsonl, 'r', encoding='utf-8') as f:
            samples = [json.loads(line.strip()) for line in f if line.strip()]
        print(f"Loaded {len(samples)} samples from {args.test_jsonl}")
    else:
        print("Error: provide --prompt, --structure_json or --test_jsonl")
        return

    # StructPlanner mode: pre-process all prompts through LLM
    structplanner = None
    if args.structplanner:
        print("StructPlanner mode enabled — loading Qwen3-14B + LoRA...")
        structplanner = StructPlanner.from_pretrained(
            lora_dir=_SP_LORA,
            device=device,
        )
        print("StructPlanner loaded.\n")

    # Auto-detect model
    if args.structplanner:
        use_structure = True
    else:
        use_structure = _has_structure(samples)

    ckpt = os.path.join(_MODEL_DIR, _CO_CKPT if use_structure else _QO_CKPT)
    model_config_name = _CO_CONFIG if use_structure else _QO_CONFIG

    mode_name = "MuseGran-Co (control-oriented)" if use_structure else "MuseGran-Qo (quality-oriented)"
    print(f"Mode: {mode_name}")
    print(f"Loading model: {ckpt}")

    model, model_config = get_pretrained_model(
        _MODEL_DIR, ckpt,
        model_config_name=model_config_name,
        vae_ckpt=_VAE_CKPT,
    )
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    model = model.to(device).eval()
    print(f"Model loaded. sample_rate={sample_rate}, sample_size={sample_size}\n")

    for idx, sample in enumerate(samples, start=1):
        duration = sample.get('duration_seconds', 60.0)
        prompt = sample.get('prompt') or build_prompt_description(sample)
        sp_spec = None

        # StructPlanner path: prompt → spec → conditioning
        if structplanner is not None:
            print(f"[{idx}/{len(samples)}] StructPlanner: \"{prompt[:80]}...\"")
            sp_spec, error = structplanner.generate_spec(prompt)
            if error is not None or sp_spec is None:
                print(f"  [WARNING] StructPlanner failed: {error}, falling back to raw prompt")
                music_key = sample.get('key')
                structure = None
                bpm = sample.get('bpm', 120.0)
                beat = {"bpm": float(bpm), "seconds_start": 0, "seconds_total": int(duration), "meter": 4}
                base_prompt = prompt
            else:
                structplanner.print_spec(sp_spec, prompt)
                cond, neg_cond, dur = spec_to_conditioning(
                    sp_spec, None, "summary",
                )
                duration = dur
                music_key = sp_spec.get("key") or None
                structure = cond[0].get("structure")
                beat = cond[0].get("beat")
                base_prompt = cond[0].get("prompt")
        else:
            music_key = sample.get('key')
            if music_key and music_key.lower() == "none":
                music_key = None
            structure = build_structure(sample, duration) if use_structure else None
            base_prompt = prompt

            beat_path = sample.get('beat_path')
            if beat_path and os.path.exists(beat_path):
                beat = {"beat_path": beat_path, "seconds_start": 0, "seconds_end": duration}
            else:
                bpm = sample.get('bpm', 120.0)
                beat = {"bpm": float(bpm), "seconds_start": 0, "seconds_total": int(duration), "meter": 4}

        print(f"  duration={duration}s | key={music_key} | beat={beat.get('bpm', '?')}bpm")
        if structure:
            n_segs = len(structure[0]) if isinstance(structure, list) and structure[0] else 0
            print(f"  structure: {n_segs} segments")

        conditioning = [{
            "prompt": base_prompt,
            "seconds_start": 0,
            "seconds_total": int(duration),
            "beat": beat,
            "music_key": music_key,
            "structure": structure,
        }]

        gen_id = next_id + idx - 1
        prompt_slug = _sanitize_prompt(prompt)
        mode = "structplanner" if structplanner is not None else "standard"
        mode_tag = "_sp" if sp_spec is not None else ""

        json_data = {
            "mode": mode,
            "user_prompt": prompt,
            "conditioning": conditioning,
        }
        if sp_spec is not None:
            json_data["spec"] = sp_spec
        json_path = os.path.join(args.output_dir, f"gen{gen_id}{mode_tag}_{prompt_slug}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"  metadata saved: {json_path}")

        negative_conditioning = [{
            "prompt": "sampled vocals, noise, distorted harmonics, artifacts, low quality",
            "chord": None,
            "beat": None,
            "seconds_start": 0,
            "seconds_total": int(duration),
            "music_key": None,
            "structure": None,
        }]

        t0 = time.time()
        with torch.inference_mode(), torch.cuda.amp.autocast():
            output, seed_used = generate_diffusion_cond(
                model,
                steps=args.steps,
                cfg_scale=args.cfg_scale,
                conditioning=conditioning,
                negative_conditioning=negative_conditioning,
                sample_size=sample_size,
                sigma_min=0.3,
                sigma_max=500,
                sampler_type="dpmpp-3m-sde",
                seed=(args.seed + idx - 1) if args.seed is not None else -1,
                device=device,
            )
        elapsed = time.time() - t0

        output = rearrange(output, "b d n -> d (b n)")
        output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1)
        output = output.mul(32767).to(torch.int16).cpu()
        output = output[:, :round(sample_rate * duration)]

        out_path = os.path.join(args.output_dir, f"gen{gen_id}{mode_tag}_{prompt_slug}.wav")
        torchaudio.save(out_path, output, sample_rate)
        print(f"  -> {out_path} ({elapsed:.1f}s, seed={seed_used})\n")

    print(f"Done. {len(samples)} files saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
