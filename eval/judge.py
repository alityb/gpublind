from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from corpus import CorpusEntry, load_corpus

JUDGE_SYSTEM = "You are evaluating an LLM's GPU bottleneck diagnosis. Score each stage of the diagnostic funnel."

JUDGE_PROMPT = """True bottleneck: {true_bottleneck}
Misleading signal: {misleading_signal}
Must cite (at least one): {must_cite_one_of}
Must NOT cite as primary: {must_not_cite_as_primary}

Raw profiling metrics:
  Arithmetic Intensity: {ai} FLOP/byte
  DRAM BW Utilization: {dram}%
  Stall Long Scoreboard: {stall_long}%
  Achieved Occupancy: {occ}%

LLM response:
{response}

Score each stage:

STAGE_1_DRR: 1 if BOTTLENECK label matches true bottleneck, else 0

STAGE_2_RVR: (only scored if STAGE_1=1)
  1 if response cites at least one specific metric value
  (e.g. "57% long scoreboard stalls") or specific code pattern
  (e.g. "strided access with stride=97") as evidence.
  0 if response only makes generic GPU statements.
  NULL if STAGE_1=0

STAGE_3_MPR: (only scored if STAGE_2=1)
  1 if response does NOT primarily blame the misleading signal.
  0 if response cites misleading signal as primary cause.
  NULL if STAGE_2 is 0 or NULL

CONFIDENCE_CALIBRATED:
  1 if (HIGH confidence AND correct) OR
       (LOW/MEDIUM confidence AND incorrect)
  0 if HIGH confidence AND incorrect (overconfident error)

Respond in exactly this format:
STAGE_1_DRR: <0 or 1>
STAGE_2_RVR: <0, 1, or NULL>
STAGE_3_MPR: <0, 1, or NULL>
CONFIDENCE_CALIBRATED: <0 or 1>
EXPLANATION: <one sentence explaining the scoring>"""

LABEL_RE = re.compile(r"BOTTLENECK:\s*([A-Za-z\-]+)", re.IGNORECASE)
CONF_RE = re.compile(r"CONFIDENCE:\s*(HIGH|MEDIUM|LOW)", re.IGNORECASE)


def judge_cache_key(entry: CorpusEntry, response: str, judge_model: str) -> str:
    digest = hashlib.sha256()
    digest.update(entry.id.encode("utf-8"))
    digest.update(judge_model.encode("utf-8"))
    digest.update(response.encode("utf-8"))
    return digest.hexdigest()


def parse_label(response: str) -> str:
    match = LABEL_RE.search(response)
    return match.group(1).strip().lower() if match else "parse_error"


def parse_confidence(response: str) -> str:
    match = CONF_RE.search(response)
    return match.group(1).strip().upper() if match else "LOW"


def mock_judge(entry: CorpusEntry, response: str) -> dict[str, Any]:
    label = parse_label(response)
    confidence = parse_confidence(response)
    text = response.lower()
    rubric = entry.reasoning_rubric
    must_cite = [str(token).lower() for token in rubric.get("must_cite_one_of", [])]
    must_not = [str(token).lower() for token in rubric.get("must_not_cite_as_primary", [])]
    stage_1 = int(label == entry.true_bottleneck)
    cites_number = bool(re.search(r"\d+(\.\d+)?%?", text))
    cites_rubric = any(token in text for token in must_cite)
    stage_2: int | None = None
    if stage_1 == 1:
        stage_2 = 1 if (cites_number or cites_rubric) else 0
    stage_3: int | None = None
    if stage_2 == 1:
        first_sentence = text.split(".", 1)[0]
        stage_3 = 0 if any(token in first_sentence for token in must_not) else 1
    confidence_calibrated = int((confidence == "HIGH" and stage_1 == 1) or (confidence != "HIGH" and stage_1 == 0))
    return {
        "stage_1_drr": stage_1,
        "stage_2_rvr": stage_2,
        "stage_3_mpr": stage_3,
        "confidence_calibrated": confidence_calibrated,
        "explanation": "Mock judge applied the funnel rubric heuristically.",
        "judge_model": "mock-judge",
    }


def parse_judge_response(text: str) -> dict[str, Any]:
    values: dict[str, Any] = {"stage_1_drr": 0, "stage_2_rvr": None, "stage_3_mpr": None, "confidence_calibrated": 0, "explanation": ""}
    mapping = {
        "STAGE_1_DRR": "stage_1_drr",
        "STAGE_2_RVR": "stage_2_rvr",
        "STAGE_3_MPR": "stage_3_mpr",
        "CONFIDENCE_CALIBRATED": "confidence_calibrated",
    }
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        key = key.strip().upper()
        raw = raw.strip()
        if key == "EXPLANATION":
            values["explanation"] = raw
        elif key in mapping:
            values[mapping[key]] = None if raw.upper() == "NULL" else int(raw)
    return values


def call_judge(prompt: str, judge_model: str) -> str:
    from litellm import completion
    from litellm.exceptions import InternalServerError, RateLimitError

    attempt = 0
    while True:
        try:
            response = completion(
                model=judge_model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            )
            return str(response.choices[0].message.content)
        except RateLimitError:
            time.sleep(min(2**attempt, 60))
            attempt += 1
        except InternalServerError:
            time.sleep(min(2**attempt, 60))
            attempt += 1


def judge_response(
    entry: CorpusEntry,
    response: str,
    judge_model: str = "anthropic/claude-sonnet-4-6",
    *,
    mock: bool = False,
    cache_dir: Path = Path("results/v2/judge_cache"),
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{judge_cache_key(entry, response, judge_model)}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    if mock:
        payload = mock_judge(entry, response)
    else:
        prompt = JUDGE_PROMPT.format(
            true_bottleneck=entry.true_bottleneck,
            misleading_signal=entry.misleading_signal,
            must_cite_one_of=", ".join(entry.reasoning_rubric.get("must_cite_one_of", [])),
            must_not_cite_as_primary=", ".join(entry.reasoning_rubric.get("must_not_cite_as_primary", [])),
            ai=entry.profile["arithmetic_intensity_flop_per_byte"],
            dram=entry.profile["dram_bw_utilization_pct"],
            stall_long=entry.profile["stall_long_scoreboard_pct"],
            occ=entry.profile["achieved_occupancy_pct"],
            response=response,
        )
        payload = parse_judge_response(call_judge(prompt, judge_model))
        payload["judge_model"] = judge_model
    cache_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge a sample response with the diagnostic funnel")
    parser.add_argument("--kernels", type=Path, default=Path("corpus/kernels"))
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    entry = load_corpus(args.kernels, min_confidence="low")[0]
    fake_response = (
        f"BOTTLENECK: {entry.true_bottleneck}\n"
        "CONFIDENCE: MEDIUM\n"
        f"REASONING: The kernel shows {entry.profile['stall_long_scoreboard_pct']:.1f}% long scoreboard stalls, "
        f"with only {entry.profile['dram_bw_utilization_pct']:.1f}% DRAM utilization."
    )
    print(json.dumps(judge_response(entry, fake_response, mock=args.mock), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
