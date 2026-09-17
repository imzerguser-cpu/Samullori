# 가락보 학습 도구 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the offline alignment pipeline (score ↔ reference audio) and the static web player described in `docs/superpowers/specs/2026-09-17-jangdan-score-trainer-design.md`, using synthetic audio/score fixtures so the whole system is buildable and testable before real hwp/audio files arrive.

**Architecture:** A Python "prep" pipeline (score schema → onset detection → alignment → timeline JSON, plus a multi-track time-stretch step) produces per-instrument audio + timeline JSON files. A dependency-free static web page plays 1-4 of those tracks in sync via the Web Audio API and highlights the current cell on each instrument's score panel, with per-instrument show/mute toggles.

**Tech Stack:** Python 3.11, librosa, soundfile, numpy, pytest, pyrubberband (+ Rubber Band CLI binary) for the ensemble time-stretch step; vanilla HTML/CSS/JS (Web Audio API) for the player, no build tooling or server.

## Global Constraints

- Python 3.11+ (matches the installed interpreter; verified via `python --version`).
- No servers or build tooling for the web player — it must run by opening `web/index.html` directly or via a plain static file server, per the approved design's "방식 A" (offline prep + static web player).
- All user-facing text in the web player UI is Korean, matching the rest of the project.
- Real audio/score files are not available yet — every module must be independently testable with synthetic fixtures (see Task 3) before real files exist.
- Do not commit real audio files or `.venv/` to git; `data/audio/*` and `data/output/*` are gitignored except `.gitkeep`.

---

### Task 1: Project scaffolding and dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `prep/__init__.py`
- Create: `tests/__init__.py`
- Create: `data/scores/.gitkeep`
- Create: `data/audio/.gitkeep`
- Create: `data/output/.gitkeep`

**Interfaces:**
- Produces: a `prep` package importable from repo root in both scripts and pytest (via `pythonpath = ["."]`), and a working `pytest` command.

- [ ] **Step 1: Create `requirements.txt`**

```
numpy
librosa
soundfile
pytest
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

- [ ] **Step 3: Create `.gitignore`**

```
__pycache__/
*.pyc
.pytest_cache/
.venv/
data/audio/*
!data/audio/.gitkeep
data/output/*
!data/output/.gitkeep
```

- [ ] **Step 4: Create empty package markers and data placeholders**

Create `prep/__init__.py` (empty file), `tests/__init__.py` (empty file), `data/scores/.gitkeep`, `data/audio/.gitkeep`, `data/output/.gitkeep` (all empty files).

- [ ] **Step 5: Create a virtual environment and install dependencies**

Run:
```bash
cd /d/samulnori-trainer
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```
Expected: dependencies install with no errors.

- [ ] **Step 6: Verify pytest runs with zero tests**

Run: `source .venv/Scripts/activate && pytest`
Expected: `no tests ran` (pytest exits with status 5 for "no tests collected" — that is expected and fine at this point).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml requirements.txt .gitignore prep/__init__.py tests/__init__.py data/scores/.gitkeep data/audio/.gitkeep data/output/.gitkeep
git commit -m "chore: scaffold project structure and dependencies"
```

---

### Task 2: Score schema module

**Files:**
- Create: `prep/score_schema.py`
- Test: `tests/test_score_schema.py`

**Interfaces:**
- Consumes: nothing (first domain module).
- Produces: `Cell` (dataclass: `id: str, label: str, strike_count: int, measure: int, bbox: BBox, mnemonic: str | None = None`), `Score` (dataclass: `instrument: str, source_image: str, cells: list[Cell]`), `load_score(path) -> Score`, `score_from_dict(data: dict) -> Score`, `expected_onset_sequence(score: Score) -> list[str]`. Later tasks (onset detection, alignment, timeline) consume `Score`, `expected_onset_sequence`, and `Cell.measure`/`Cell.bbox`. The web player (Task 8) reads `Cell.mnemonic` for display only — it never affects alignment.

**Note on `mnemonic`:** real gakbo tables sometimes pair a rhythm cell with a memorization word (e.g. "짜장면", "스파게티") written at the same beat position, purely so students can memorize the rhythm — it has no bearing on `strike_count` or timing. See `data/scores/가락보_표기_규칙.md` for the full transcription rules discovered from the teacher's actual gakbo material, and `data/scores/구음_기법표.md` for what each syllable (구음) technically means per instrument.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_score_schema.py`:

```python
import pytest
from prep.score_schema import score_from_dict, expected_onset_sequence


def sample_data():
    return {
        "instrument": "janggu",
        "source_image": "scores/janggu_table.png",
        "cells": [
            {"id": "c1", "label": "덩", "strike_count": 1, "measure": 1, "bbox": {"x": 0, "y": 0, "w": 10, "h": 10}, "mnemonic": "짜"},
            {"id": "c2", "label": "쉼", "strike_count": 0, "measure": 1, "bbox": {"x": 10, "y": 0, "w": 10, "h": 10}},
            {"id": "c3", "label": "더러러", "strike_count": 3, "measure": 1, "bbox": {"x": 20, "y": 0, "w": 10, "h": 10}},
        ],
    }


def test_score_from_dict_parses_cells():
    score = score_from_dict(sample_data())
    assert score.instrument == "janggu"
    assert len(score.cells) == 3
    assert score.cells[0].label == "덩"
    assert score.cells[0].bbox.w == 10


def test_score_from_dict_parses_optional_mnemonic():
    score = score_from_dict(sample_data())
    assert score.cells[0].mnemonic == "짜"
    assert score.cells[1].mnemonic is None  # not every cell has one


def test_score_from_dict_rejects_negative_strike_count():
    data = sample_data()
    data["cells"][0]["strike_count"] = -1
    with pytest.raises(ValueError):
        score_from_dict(data)


def test_expected_onset_sequence_flattens_by_strike_count():
    score = score_from_dict(sample_data())
    assert expected_onset_sequence(score) == ["c1", "c3", "c3", "c3"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source .venv/Scripts/activate && pytest tests/test_score_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prep.score_schema'`

- [ ] **Step 3: Write the implementation**

Create `prep/score_schema.py`:

```python
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BBox:
    x: float
    y: float
    w: float
    h: float


@dataclass
class Cell:
    id: str
    label: str
    strike_count: int
    measure: int
    bbox: BBox
    mnemonic: str | None = None


@dataclass
class Score:
    instrument: str
    source_image: str
    cells: list[Cell]


def load_score(path: str | Path) -> Score:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return score_from_dict(data)


def score_from_dict(data: dict) -> Score:
    cells = []
    for raw_cell in data["cells"]:
        if raw_cell["strike_count"] < 0:
            raise ValueError(
                f"strike_count must be >= 0, got {raw_cell['strike_count']} for cell {raw_cell['id']}"
            )
        cells.append(Cell(
            id=raw_cell["id"],
            label=raw_cell["label"],
            strike_count=raw_cell["strike_count"],
            measure=raw_cell["measure"],
            bbox=BBox(**raw_cell["bbox"]),
            mnemonic=raw_cell.get("mnemonic"),
        ))
    return Score(instrument=data["instrument"], source_image=data["source_image"], cells=cells)


def expected_onset_sequence(score: Score) -> list[str]:
    """Flatten cells into one entry per expected strike: a cell with
    strike_count=2 contributes its id twice; a rest (strike_count=0)
    contributes nothing."""
    sequence: list[str] = []
    for cell in score.cells:
        sequence.extend([cell.id] * cell.strike_count)
    return sequence
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source .venv/Scripts/activate && pytest tests/test_score_schema.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add prep/score_schema.py tests/test_score_schema.py
git commit -m "feat: add score schema parsing and expected onset sequence"
```

---

### Task 3: Synthetic audio test helper

**Files:**
- Create: `tests/helpers.py`
- Test: `tests/test_helpers.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `make_click_track(onset_times: list[float], sr: int = 22050, duration: float | None = None, click_freq: float = 2000.0, click_len: float = 0.03) -> tuple[np.ndarray, int]`. Tasks 4, 7, 9, 10, 11 use this to generate audio with known onset times instead of needing real recordings.

- [ ] **Step 1: Write the failing test**

Create `tests/test_helpers.py`:

```python
import numpy as np
from tests.helpers import make_click_track


def test_make_click_track_places_energy_at_onsets():
    audio, sr = make_click_track([0.1, 0.5], sr=22050)
    idx_a = int(0.1 * sr)
    idx_b = int(0.5 * sr)
    window = int(0.01 * sr)
    assert np.max(np.abs(audio[idx_a:idx_a + window])) > 0.1
    assert np.max(np.abs(audio[idx_b:idx_b + window])) > 0.1
    assert np.max(np.abs(audio[: idx_a - window])) < 0.05
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `source .venv/Scripts/activate && pytest tests/test_helpers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tests.helpers'`

- [ ] **Step 3: Write the implementation**

Create `tests/helpers.py`:

```python
from __future__ import annotations
import numpy as np


def make_click_track(onset_times: list[float], sr: int = 22050, duration: float | None = None,
                      click_freq: float = 2000.0, click_len: float = 0.03) -> tuple[np.ndarray, int]:
    """Generate a mono signal with a short decaying tone burst at each time
    in onset_times, for testing onset detection/alignment without real
    audio. Returns (samples, sample_rate)."""
    if duration is None:
        duration = (max(onset_times) if onset_times else 0.0) + 0.5
    n_samples = int(duration * sr)
    audio = np.zeros(n_samples, dtype=np.float32)
    click_n = int(click_len * sr)
    t = np.arange(click_n) / sr
    envelope = np.exp(-t / (click_len / 5))
    click = (np.sin(2 * np.pi * click_freq * t) * envelope).astype(np.float32)
    for onset in onset_times:
        start = int(onset * sr)
        end = min(start + click_n, n_samples)
        audio[start:end] += click[: end - start]
    peak = np.max(np.abs(audio)) if n_samples else 0.0
    if peak > 0:
        audio = audio / peak * 0.9
    return audio, sr
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `source .venv/Scripts/activate && pytest tests/test_helpers.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add tests/helpers.py tests/test_helpers.py
git commit -m "test: add synthetic click-track helper for audio fixtures"
```

---

### Task 4: Onset detection module

**Files:**
- Create: `prep/onset_detect.py`
- Test: `tests/test_onset_detect.py`

**Interfaces:**
- Consumes: `tests/helpers.make_click_track` (test only).
- Produces: `detect_onsets(audio: np.ndarray, sr: int, backtrack: bool = True) -> list[float]`, `detect_onsets_from_file(path: str) -> tuple[list[float], int]`. Tasks 7, 12 consume `detect_onsets`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_onset_detect.py`:

```python
from prep.onset_detect import detect_onsets
from tests.helpers import make_click_track


def test_detect_onsets_finds_synthetic_clicks_within_tolerance():
    expected = [0.2, 0.6, 1.1, 1.4]
    audio, sr = make_click_track(expected, sr=22050)
    detected = detect_onsets(audio, sr)
    assert len(detected) == len(expected)
    for exp, det in zip(expected, detected):
        assert abs(exp - det) < 0.05
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `source .venv/Scripts/activate && pytest tests/test_onset_detect.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prep.onset_detect'`

- [ ] **Step 3: Write the implementation**

Create `prep/onset_detect.py`:

```python
from __future__ import annotations
import librosa
import numpy as np


def detect_onsets(audio: np.ndarray, sr: int, backtrack: bool = True) -> list[float]:
    """Detect percussive onset times (seconds) in a mono audio signal."""
    onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, backtrack=backtrack, units="frames")
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    return [float(t) for t in onset_times]


def detect_onsets_from_file(path: str) -> tuple[list[float], int]:
    audio, sr = librosa.load(path, sr=None, mono=True)
    return detect_onsets(audio, sr), sr
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `source .venv/Scripts/activate && pytest tests/test_onset_detect.py -v`
Expected: 1 passed. (If it fails because timing tolerance is too tight, this is expected to be tuned later against real instrument audio — see Task 12's notes.)

- [ ] **Step 5: Commit**

```bash
git add prep/onset_detect.py tests/test_onset_detect.py
git commit -m "feat: add onset detection wrapper around librosa"
```

---

### Task 5: Sequence alignment module

**Files:**
- Create: `prep/align.py`
- Test: `tests/test_align.py`

**Design decision (refines the spec):** the design doc describes a DP-based alignment that auto-resolves any mismatch between expected and detected onset counts. Building that heuristic concretely during planning showed it needs real tempo data to disambiguate which onsets are noise vs. which cells were missed — without that data it can silently produce a wrong alignment with no signal that anything went wrong. Since the reference audio is expected to be a clean take that matches the score exactly, this plan instead: (a) does a direct in-order mapping when counts match (the expected common case), and (b) raises a clear, actionable error when they don't, deferring to the manual correction tool described in the spec rather than guessing. Mention this to the user when this task is reviewed.

**Interfaces:**
- Consumes: `prep.onset_detect.detect_onsets` output (list of floats), `prep.score_schema.expected_onset_sequence` output (list of cell ids).
- Produces: `align_onsets(expected_ids: list[str], detected_times: list[float]) -> list[dict]` (each dict: `{"cell_id": str, "time": float}`), `AlignmentMismatchError` (attributes: `expected_count: int`, `detected_count: int`, `detected_times: list[float]`). Tasks 7, 12 consume both.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_align.py`:

```python
import pytest
from prep.align import align_onsets, AlignmentMismatchError


def test_align_onsets_maps_in_order_when_counts_match():
    expected = ["c1", "c1", "c2"]  # c1 has strike_count=2
    detected = [0.5, 0.1, 0.9]  # deliberately out of chronological order
    result = align_onsets(expected, detected)
    assert [r["time"] for r in result] == [0.1, 0.5, 0.9]
    assert [r["cell_id"] for r in result] == expected


def test_align_onsets_raises_on_count_mismatch():
    with pytest.raises(AlignmentMismatchError) as exc_info:
        align_onsets(["c1", "c2", "c3"], [0.1, 0.9])
    err = exc_info.value
    assert err.expected_count == 3
    assert err.detected_count == 2
    assert err.detected_times == [0.1, 0.9]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source .venv/Scripts/activate && pytest tests/test_align.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prep.align'`

- [ ] **Step 3: Write the implementation**

Create `prep/align.py`:

```python
from __future__ import annotations


class AlignmentMismatchError(Exception):
    """Raised when the number of detected onsets doesn't match the number
    of expected strikes, so a direct in-order mapping isn't reliable."""

    def __init__(self, expected_count: int, detected_count: int, detected_times: list[float]):
        self.expected_count = expected_count
        self.detected_count = detected_count
        self.detected_times = detected_times
        super().__init__(
            f"expected {expected_count} strikes but detected {detected_count} onsets "
            f"(diff={detected_count - expected_count}); use the manual correction tool"
        )


def align_onsets(expected_ids: list[str], detected_times: list[float]) -> list[dict]:
    """Map each expected strike (in score order) to a detected onset time.

    Requires len(detected_times) == len(expected_ids). A clean reference
    recording should produce exactly one onset per expected strike; when
    it doesn't, raise so the mismatch is fixed explicitly rather than
    guessed at.
    """
    if len(detected_times) != len(expected_ids):
        raise AlignmentMismatchError(len(expected_ids), len(detected_times), detected_times)
    return [{"cell_id": cid, "time": t} for cid, t in zip(expected_ids, sorted(detected_times))]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source .venv/Scripts/activate && pytest tests/test_align.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add prep/align.py tests/test_align.py
git commit -m "feat: add in-order onset alignment with explicit mismatch error"
```

---

### Task 6: Timeline builder module

**Files:**
- Create: `prep/timeline.py`
- Test: `tests/test_timeline.py`

**Interfaces:**
- Consumes: `prep.score_schema.Score`, `prep.align.align_onsets` output.
- Produces: `build_timeline(score: Score, aligned: list[dict]) -> list[dict]` (each dict: `{"cell_id": str, "start": float, "end": float}`, one per score cell including rests), `write_timeline(windows: list[dict], path) -> None`. Tasks 7, 12, and the web player consume this JSON shape (`{"cells": [...]}`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_timeline.py`:

```python
import json
import pytest
from prep.score_schema import score_from_dict
from prep.timeline import build_timeline, write_timeline


def sample_score_with_rest():
    return score_from_dict({
        "instrument": "janggu",
        "source_image": "x.png",
        "cells": [
            {"id": "c1", "label": "덩", "strike_count": 1, "measure": 1, "bbox": {"x": 0, "y": 0, "w": 1, "h": 1}},
            {"id": "c2", "label": "쉼", "strike_count": 0, "measure": 1, "bbox": {"x": 1, "y": 0, "w": 1, "h": 1}},
            {"id": "c3", "label": "쿵", "strike_count": 1, "measure": 1, "bbox": {"x": 2, "y": 0, "w": 1, "h": 1}},
        ],
    })


def test_build_timeline_struck_cells_use_aligned_time():
    score = sample_score_with_rest()
    aligned = [{"cell_id": "c1", "time": 1.0}, {"cell_id": "c3", "time": 2.0}]
    windows = build_timeline(score, aligned)
    assert windows[0] == {"cell_id": "c1", "start": 1.0, "end": 1.5}
    assert windows[2]["start"] == 2.0


def test_build_timeline_rest_cell_interpolated_between_neighbors():
    score = sample_score_with_rest()
    aligned = [{"cell_id": "c1", "time": 1.0}, {"cell_id": "c3", "time": 2.0}]
    windows = build_timeline(score, aligned)
    assert windows[1]["cell_id"] == "c2"
    assert windows[1]["start"] == 1.5
    assert windows[1]["end"] == 2.0


def test_build_timeline_last_cell_gets_one_second_window():
    score = sample_score_with_rest()
    aligned = [{"cell_id": "c1", "time": 1.0}, {"cell_id": "c3", "time": 2.0}]
    windows = build_timeline(score, aligned)
    assert windows[2]["end"] == 3.0


def test_build_timeline_raises_when_nothing_aligned():
    score = sample_score_with_rest()
    with pytest.raises(ValueError):
        build_timeline(score, [])


def test_write_timeline_serializes_to_json(tmp_path):
    windows = [{"cell_id": "c1", "start": 0.0, "end": 1.0}]
    out_path = tmp_path / "timeline.json"
    write_timeline(windows, out_path)
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data == {"cells": windows}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source .venv/Scripts/activate && pytest tests/test_timeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prep.timeline'`

- [ ] **Step 3: Write the implementation**

Create `prep/timeline.py`:

```python
from __future__ import annotations
import json
from pathlib import Path
from prep.score_schema import Score


def build_timeline(score: Score, aligned: list[dict]) -> list[dict]:
    """Build a highlight window {cell_id, start, end} for every cell in
    the score, in cell order. Struck cells use their aligned strike time
    as `start`; rest cells (no strikes) get a `start` linearly
    interpolated between the nearest struck cells before and after them.
    The last cell's window extends 1 second past its start."""
    first_strike_time: dict[str, float] = {}
    for entry in aligned:
        first_strike_time.setdefault(entry["cell_id"], entry["time"])

    all_cell_ids = [cell.id for cell in score.cells]
    if not any(cid in first_strike_time for cid in all_cell_ids):
        raise ValueError("no strikes were aligned; cannot build a timeline")

    starts: list[float | None] = [first_strike_time.get(cid) for cid in all_cell_ids]
    _interpolate_gaps(starts)

    windows = []
    for i, cid in enumerate(all_cell_ids):
        start = starts[i]
        end = starts[i + 1] if i + 1 < len(starts) else start + 1.0
        windows.append({"cell_id": cid, "start": start, "end": end})
    return windows


def _interpolate_gaps(starts: list[float | None]) -> None:
    n = len(starts)
    known = [i for i, s in enumerate(starts) if s is not None]
    for i in range(n):
        if starts[i] is not None:
            continue
        left = max([k for k in known if k < i], default=None)
        right = min([k for k in known if k > i], default=None)
        if left is None:
            starts[i] = starts[right]
        elif right is None:
            starts[i] = starts[left]
        else:
            frac = (i - left) / (right - left)
            starts[i] = starts[left] + frac * (starts[right] - starts[left])


def write_timeline(windows: list[dict], path: str | Path) -> None:
    Path(path).write_text(json.dumps({"cells": windows}, ensure_ascii=False, indent=2), encoding="utf-8")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source .venv/Scripts/activate && pytest tests/test_timeline.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add prep/timeline.py tests/test_timeline.py
git commit -m "feat: add timeline builder with rest-cell interpolation"
```

---

### Task 7: Single-track prepare CLI

**Files:**
- Create: `prep/prepare.py`
- Test: `tests/test_prepare_single.py`

**Interfaces:**
- Consumes: `prep.score_schema.load_score`/`expected_onset_sequence`, `prep.onset_detect.detect_onsets`, `prep.align.align_onsets`/`AlignmentMismatchError`, `prep.timeline.build_timeline`/`write_timeline`.
- Produces: `prepare_single_track(audio_path: str, score_path: str, output_path: str) -> None`, a `main(argv=None)` CLI with a `single` subcommand (`--audio`, `--score`, `--output`). Task 12's ensemble command reuses the same per-track logic.

- [ ] **Step 1: Write the failing test**

Create `tests/test_prepare_single.py`:

```python
import json
import soundfile as sf
from prep.prepare import prepare_single_track
from tests.helpers import make_click_track


def test_prepare_single_track_end_to_end(tmp_path):
    score_data = {
        "instrument": "janggu",
        "source_image": "x.png",
        "cells": [
            {"id": "c1", "label": "덩", "strike_count": 1, "measure": 1, "bbox": {"x": 0, "y": 0, "w": 1, "h": 1}},
            {"id": "c2", "label": "쉼", "strike_count": 0, "measure": 1, "bbox": {"x": 1, "y": 0, "w": 1, "h": 1}},
            {"id": "c3", "label": "쿵", "strike_count": 1, "measure": 1, "bbox": {"x": 2, "y": 0, "w": 1, "h": 1}},
        ],
    }
    score_path = tmp_path / "score.json"
    score_path.write_text(json.dumps(score_data), encoding="utf-8")

    audio, sr = make_click_track([0.3, 0.9], sr=22050)
    audio_path = tmp_path / "audio.wav"
    sf.write(str(audio_path), audio, sr)

    output_path = tmp_path / "timeline.json"
    prepare_single_track(str(audio_path), str(score_path), str(output_path))

    result = json.loads(output_path.read_text(encoding="utf-8"))
    cells = {c["cell_id"]: c for c in result["cells"]}
    assert abs(cells["c1"]["start"] - 0.3) < 0.05
    assert abs(cells["c3"]["start"] - 0.9) < 0.05
    assert cells["c1"]["start"] < cells["c2"]["start"] < cells["c3"]["start"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `source .venv/Scripts/activate && pytest tests/test_prepare_single.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prep.prepare'`

- [ ] **Step 3: Write the implementation**

Create `prep/prepare.py`:

```python
from __future__ import annotations
import argparse
import sys
import librosa

from prep.score_schema import load_score, expected_onset_sequence
from prep.onset_detect import detect_onsets
from prep.align import align_onsets, AlignmentMismatchError
from prep.timeline import build_timeline, write_timeline


def prepare_single_track(audio_path: str, score_path: str, output_path: str) -> None:
    score = load_score(score_path)
    expected_ids = expected_onset_sequence(score)

    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    detected_times = detect_onsets(audio, sr)

    try:
        aligned = align_onsets(expected_ids, detected_times)
    except AlignmentMismatchError as exc:
        print(f"[prepare] alignment failed for {audio_path}: {exc}", file=sys.stderr)
        print(f"[prepare] detected onset times: {exc.detected_times}", file=sys.stderr)
        raise SystemExit(1) from exc

    windows = build_timeline(score, aligned)
    write_timeline(windows, output_path)
    print(f"[prepare] wrote {len(windows)} cell windows to {output_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Align one instrument's reference audio to its gakbo score.")
    sub = parser.add_subparsers(dest="command", required=True)

    single = sub.add_parser("single", help="Align one instrument's audio to its score")
    single.add_argument("--audio", required=True, help="Path to the reference audio file")
    single.add_argument("--score", required=True, help="Path to the score JSON file")
    single.add_argument("--output", required=True, help="Path to write the timeline JSON to")

    args = parser.parse_args(argv)
    if args.command == "single":
        prepare_single_track(args.audio, args.score, args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `source .venv/Scripts/activate && pytest tests/test_prepare_single.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add prep/prepare.py tests/test_prepare_single.py
git commit -m "feat: add single-track prepare CLI wiring the alignment pipeline together"
```

---

### Task 8: Web player MVP (single track)

**Files:**
- Create: `web/player.js`
- Create: `web/app.js`
- Create: `web/style.css`
- Create: `web/index.html`
- Create: `web/fixtures/dummy_score.json`
- Create: `web/fixtures/dummy_timeline.json`
- Create: `web/fixtures/dummy_audio.wav`

**Interfaces:**
- Consumes: nothing from Python directly (loads JSON/audio fixtures over `fetch`, same shape `prep/timeline.write_timeline` produces: `{"cells": [{"cell_id", "start", "end"}]}`, and the score JSON shape from `prep/score_schema.py`).
- Produces: `ScorePlayer` class (`loadTrack(config)`, `play()`, `pause()`, `setRate(rate)`, `setMuted(trackId, muted)`, `currentCellFor(trackId)`, `currentTime` getter) — Task 11's multi-track UI work reuses this unchanged; it already supports N tracks.

There's no Python test harness for this task; verification is a manual browser check with exact steps below.

- [ ] **Step 1: Generate fixture audio and score/timeline JSON**

Create `web/fixtures/dummy_score.json`:

```json
{
  "instrument": "janggu",
  "source_image": null,
  "cells": [
    {"id": "c1", "label": "덩", "strike_count": 1, "measure": 1, "bbox": {"x": 10, "y": 10, "w": 80, "h": 60}, "mnemonic": "짜"},
    {"id": "c2", "label": "쉼", "strike_count": 0, "measure": 1, "bbox": {"x": 100, "y": 10, "w": 80, "h": 60}},
    {"id": "c3", "label": "기덕", "strike_count": 1, "measure": 1, "bbox": {"x": 190, "y": 10, "w": 80, "h": 60}, "mnemonic": "면"},
    {"id": "c4", "label": "쿵", "strike_count": 1, "measure": 2, "bbox": {"x": 280, "y": 10, "w": 80, "h": 60}}
  ]
}
```

Run this from the repo root to generate matching audio and timeline fixtures from real project code (so the fixtures are guaranteed consistent with the pipeline):

```bash
source .venv/Scripts/activate
python - <<'EOF'
import soundfile as sf
from tests.helpers import make_click_track
from prep.score_schema import load_score, expected_onset_sequence
from prep.align import align_onsets
from prep.timeline import build_timeline, write_timeline

onset_times = [0.5, 2.0]  # c1 at 0.5s, c3 at 2.0s; c2 (rest) and c4 interpolate/extrapolate
audio, sr = make_click_track(onset_times, sr=22050, duration=3.5)
sf.write("web/fixtures/dummy_audio.wav", audio, sr)

score = load_score("web/fixtures/dummy_score.json")
aligned = align_onsets(expected_onset_sequence(score), onset_times)
windows = build_timeline(score, aligned)
write_timeline(windows, "web/fixtures/dummy_timeline.json")
print(windows)
EOF
```

Expected: prints 4 windows (c1, c2, c3, c4) with increasing `start` times, and creates `web/fixtures/dummy_audio.wav` + `web/fixtures/dummy_timeline.json`.

- [ ] **Step 2: Write the playback engine**

Create `web/player.js`:

```javascript
class ScorePlayer {
  constructor(audioContext) {
    this.ctx = audioContext;
    this.tracks = [];
    this.startedAt = null;
    this.startOffset = 0;
    this.playing = false;
    this.rate = 1.0;
  }

  async loadTrack(config) {
    const buf = await fetch(config.audioUrl).then(r => r.arrayBuffer());
    const audioBuf = await this.ctx.decodeAudioData(buf);
    const timelineRes = await fetch(config.timelineUrl).then(r => r.json());
    const gainNode = this.ctx.createGain();
    gainNode.connect(this.ctx.destination);
    const track = {
      id: config.id,
      label: config.label,
      buffer: audioBuf,
      gainNode,
      timeline: timelineRes.cells,
      sourceNode: null,
    };
    this.tracks.push(track);
    return track;
  }

  play() {
    if (this.playing) return;
    const when = this.ctx.currentTime;
    for (const track of this.tracks) {
      const src = this.ctx.createBufferSource();
      src.buffer = track.buffer;
      src.playbackRate.value = this.rate;
      src.connect(track.gainNode);
      src.start(when, this.startOffset);
      track.sourceNode = src;
    }
    this.startedAt = when;
    this.playing = true;
  }

  pause() {
    if (!this.playing) return;
    this.startOffset = this.currentTime;
    for (const track of this.tracks) {
      if (track.sourceNode) track.sourceNode.stop();
      track.sourceNode = null;
    }
    this.playing = false;
  }

  get currentTime() {
    if (!this.playing) return this.startOffset;
    return this.startOffset + (this.ctx.currentTime - this.startedAt) * this.rate;
  }

  setRate(rate) {
    const wasPlaying = this.playing;
    const pos = this.currentTime;
    if (wasPlaying) this.pause();
    this.rate = rate;
    this.startOffset = pos;
    if (wasPlaying) this.play();
  }

  setMuted(trackId, muted) {
    const track = this.tracks.find(t => t.id === trackId);
    if (!track) return;
    track.gainNode.gain.value = muted ? 0 : 1;
  }

  currentCellFor(trackId) {
    const track = this.tracks.find(t => t.id === trackId);
    if (!track) return null;
    const t = this.currentTime;
    return track.timeline.find(c => t >= c.start && t < c.end) || null;
  }
}

window.ScorePlayer = ScorePlayer;
```

- [ ] **Step 3: Write the UI wiring**

Create `web/app.js`:

```javascript
const TRACK_DEFS = [
  { id: 'janggu', label: '장구', audioUrl: 'fixtures/dummy_audio.wav', timelineUrl: 'fixtures/dummy_timeline.json', scoreUrl: 'fixtures/dummy_score.json' },
];

const ctx = new (window.AudioContext || window.webkitAudioContext)();
const player = new ScorePlayer(ctx);

async function init() {
  const togglesEl = document.getElementById('instrument-toggles');
  const panelsEl = document.getElementById('panels');

  for (const def of TRACK_DEFS) {
    def.score = await fetch(def.scoreUrl).then(r => r.json());
    await player.loadTrack(def);

    const panel = document.createElement('div');
    panel.className = 'panel';
    panel.dataset.trackId = def.id;
    if (def.score.source_image) {
      const img = document.createElement('img');
      img.src = def.score.source_image;
      panel.appendChild(img);
    }
    for (const cell of def.score.cells) {
      const box = document.createElement('div');
      box.className = 'cell-box';
      box.dataset.cellId = cell.id;
      box.innerHTML = cell.mnemonic
        ? `<span class="cell-label">${cell.label}</span><span class="cell-mnemonic">${cell.mnemonic}</span>`
        : `<span class="cell-label">${cell.label}</span>`;
      box.style.left = cell.bbox.x + 'px';
      box.style.top = cell.bbox.y + 'px';
      box.style.width = cell.bbox.w + 'px';
      box.style.height = cell.bbox.h + 'px';
      panel.appendChild(box);
    }
    panelsEl.appendChild(panel);

    const toggle = document.createElement('div');
    toggle.className = 'instrument-toggle';
    toggle.innerHTML = `
      <label><input type="checkbox" class="show-toggle" checked data-track-id="${def.id}"> ${def.label}</label>
      <label class="mute-only-label"><input type="checkbox" class="mute-only-toggle" data-track-id="${def.id}"> 소리만 끄기</label>
    `;
    togglesEl.appendChild(toggle);
  }

  togglesEl.addEventListener('change', onToggleChange);
  document.getElementById('play-btn').addEventListener('click', () => player.play());
  document.getElementById('pause-btn').addEventListener('click', () => player.pause());
  document.getElementById('rate-select').addEventListener('change', e => player.setRate(parseFloat(e.target.value)));

  requestAnimationFrame(renderLoop);
}

function onToggleChange(e) {
  const trackId = e.target.dataset.trackId;
  const panel = document.querySelector(`.panel[data-track-id="${trackId}"]`);
  if (e.target.classList.contains('show-toggle')) {
    panel.style.display = e.target.checked ? '' : 'none';
    player.setMuted(trackId, !e.target.checked);
    document.querySelector(`.mute-only-toggle[data-track-id="${trackId}"]`).checked = false;
  } else if (e.target.classList.contains('mute-only-toggle')) {
    player.setMuted(trackId, e.target.checked);
  }
}

function renderLoop() {
  for (const def of TRACK_DEFS) {
    const cell = player.currentCellFor(def.id);
    const panel = document.querySelector(`.panel[data-track-id="${def.id}"]`);
    if (!panel) continue;
    panel.querySelectorAll('.cell-box').forEach(box => {
      box.classList.toggle('active', !!cell && box.dataset.cellId === cell.id);
    });
  }
  requestAnimationFrame(renderLoop);
}

init();
```

- [ ] **Step 4: Write the stylesheet**

Create `web/style.css`:

```css
body { font-family: sans-serif; margin: 0; padding: 16px; background: #fff; color: #222; }
#controls { display: flex; gap: 16px; align-items: center; margin-bottom: 16px; flex-wrap: wrap; }
#instrument-toggles { display: flex; gap: 12px; flex-wrap: wrap; }
.instrument-toggle { display: flex; gap: 8px; align-items: center; font-size: 14px; }
.mute-only-label { color: #888; }
#panels { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
.panel { position: relative; border: 1px solid #ccc; min-height: 100px; background: #fafafa; }
.panel img { display: block; max-width: 100%; }
.cell-box {
  position: absolute;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  border: 1px solid #999;
  font-size: 14px;
  box-sizing: border-box;
  background: rgba(255, 255, 255, 0.7);
}
.cell-box.active {
  background: rgba(255, 200, 0, 0.65);
  border-color: rgb(230, 140, 0);
  border-width: 2px;
}
.cell-mnemonic {
  font-size: 11px;
  color: #666;
}
```

- [ ] **Step 5: Write the HTML shell**

Create `web/index.html`:

```html
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>가락보 연습</title>
<link rel="stylesheet" href="style.css">
</head>
<body>
  <div id="controls">
    <div id="instrument-toggles"></div>
    <button id="play-btn">재생</button>
    <button id="pause-btn">일시정지</button>
    <label>배속
      <select id="rate-select">
        <option value="0.5">0.5x</option>
        <option value="0.75">0.75x</option>
        <option value="1" selected>1x</option>
      </select>
    </label>
  </div>
  <div id="panels"></div>
  <script src="player.js"></script>
  <script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 6: Manually verify in a browser**

`fetch()` requires the page to be served over http(s), not opened as a `file://` URL, so start a static server:

```bash
cd /d/samulnori-trainer/web
python -m http.server 8000
```

Open `http://localhost:8000/` in a browser and check:
- The 장구 panel shows 4 boxes labeled 덩/쉼/기덕/쿵, with 덩 and 기덕 also showing a small "짜"/"면" mnemonic word underneath (쉼 and 쿵 have none, since not every cell has a mnemonic).
- Clicking 재생 starts audio and the boxes light up (yellow background) in order as playback passes each cell's time window; 쉼 lights up between 덩 and 기덕 even though it's silent.
- Clicking 일시정지 stops audio and freezes the highlight.
- Changing 배속 to 0.5x noticeably slows playback and the highlight timing slows with it.
- Unchecking the 장구 checkbox hides the panel and silences the audio; re-checking it restores both.
- Checking "소리만 끄기" mutes audio while the panel stays visible; unchecking it restores sound.

Stop the server with Ctrl+C when done.

- [ ] **Step 7: Commit**

```bash
git add web/
git commit -m "feat: add single-track web player with show/mute toggles"
```

---

### Task 9: Rubber Band dependency setup and stretch-plan module

**Files:**
- Create: `prep/stretch_plan.py`
- Test: `tests/test_stretch_plan.py`

**Interfaces:**
- Consumes: `prep.score_schema.Score`.
- Produces: `measure_start_times(score: Score, windows: list[dict]) -> dict[int, float]`, `build_stretch_segments(source_measure_times: dict[int, float], target_measure_times: dict[int, float]) -> list[dict]` (each dict: `{"measure", "source_start", "source_end", "target_start", "target_end", "rate"}`). Tasks 10, 12 consume both.

This task's logic doesn't need real audio (pure arithmetic on timestamps), so it's fully testable now. The Rubber Band CLI binary is only needed starting in Task 10.

- [ ] **Step 1: Add pyrubberband to dependencies and install the Rubber Band CLI**

Add to `requirements.txt`:
```
pyrubberband
```

Run: `source .venv/Scripts/activate && pip install -r requirements.txt`

The `pyrubberband` Python package calls out to a separate `rubberband` command-line binary that isn't installed by pip. Check whether it's already on PATH:

```bash
rubberband -V
```

- If that prints a version, skip to Step 2.
- If not found, download the Windows build from https://breakfastquay.com/rubberband/ (the "command-line utility" zip), extract it, and add the extracted folder (containing `rubberband.exe`) to your PATH for the current shell:

```bash
export PATH="/c/path/to/extracted/rubberband:$PATH"
rubberband -V
```

Expected: version output. This task's own tests don't require the binary (they're pure timestamp math); Task 10's tests use `pytest.importorskip`/try-except so they skip cleanly if the binary still isn't set up when you get there — install it now or defer to Task 10, whichever is convenient.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_stretch_plan.py`:

```python
import pytest
from prep.stretch_plan import build_stretch_segments


def test_build_stretch_segments_computes_rate_per_measure():
    source = {1: 0.0, 2: 1.0, 3: 2.5}
    target = {1: 0.0, 2: 1.2, 3: 2.4}
    segments = build_stretch_segments(source, target)
    assert len(segments) == 3
    assert segments[0]["rate"] == pytest.approx(1.0 / 1.2)
    assert segments[1]["rate"] == pytest.approx(1.5 / 1.2)


def test_build_stretch_segments_infers_last_segment_duration():
    source = {1: 0.0, 2: 1.0}
    target = {1: 0.0, 2: 1.5}
    segments = build_stretch_segments(source, target)
    assert segments[1]["source_end"] == pytest.approx(2.0)
    assert segments[1]["target_end"] == pytest.approx(3.0)


def test_build_stretch_segments_rejects_mismatched_measures():
    with pytest.raises(ValueError):
        build_stretch_segments({1: 0.0, 2: 1.0}, {1: 0.0, 3: 1.0})


def test_build_stretch_segments_rejects_single_measure():
    with pytest.raises(ValueError):
        build_stretch_segments({1: 0.0}, {1: 0.0})
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `source .venv/Scripts/activate && pytest tests/test_stretch_plan.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prep.stretch_plan'`

- [ ] **Step 4: Write the implementation**

Create `prep/stretch_plan.py`:

```python
from __future__ import annotations
from prep.score_schema import Score


def measure_start_times(score: Score, windows: list[dict]) -> dict[int, float]:
    """Return {measure_number: start_time}, using each measure's first
    cell's start time from a timeline built by timeline.build_timeline."""
    window_by_cell = {w["cell_id"]: w["start"] for w in windows}
    starts: dict[int, float] = {}
    for cell in score.cells:
        if cell.measure not in starts:
            starts[cell.measure] = window_by_cell[cell.id]
    return starts


def build_stretch_segments(source_measure_times: dict[int, float],
                            target_measure_times: dict[int, float]) -> list[dict]:
    """Compute per-measure time-stretch segments mapping one track's own
    timeline onto a shared canonical timeline. Both inputs must cover the
    same measure numbers. `rate` follows pyrubberband.time_stretch's
    convention: rate > 1 shortens audio, rate < 1 lengthens it
    (rate = source_duration / target_duration)."""
    measures = sorted(source_measure_times.keys())
    if measures != sorted(target_measure_times.keys()):
        raise ValueError("source and target must cover the same measure numbers")
    if len(measures) < 2:
        raise ValueError("need at least two measures to infer segment durations")

    segments = []
    for i, measure in enumerate(measures):
        source_start = source_measure_times[measure]
        target_start = target_measure_times[measure]
        if i + 1 < len(measures):
            next_measure = measures[i + 1]
            source_end = source_measure_times[next_measure]
            target_end = target_measure_times[next_measure]
        else:
            prev_measure = measures[i - 1]
            source_end = source_start + (source_start - source_measure_times[prev_measure])
            target_end = target_start + (target_start - target_measure_times[prev_measure])

        source_duration = source_end - source_start
        target_duration = target_end - target_start
        if source_duration <= 0 or target_duration <= 0:
            raise ValueError(f"non-positive duration for measure {measure}")

        segments.append({
            "measure": measure,
            "source_start": source_start,
            "source_end": source_end,
            "target_start": target_start,
            "target_end": target_end,
            "rate": source_duration / target_duration,
        })
    return segments
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `source .venv/Scripts/activate && pytest tests/test_stretch_plan.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add requirements.txt prep/stretch_plan.py tests/test_stretch_plan.py
git commit -m "feat: add per-measure stretch segment planning"
```

---

### Task 10: Timestretch/mix module

**Files:**
- Create: `prep/timestretch_mix.py`
- Test: `tests/test_timestretch_mix.py`

**Interfaces:**
- Consumes: `prep.stretch_plan.build_stretch_segments` output, `prep.timeline.build_timeline` output.
- Produces: `apply_stretch_segments(audio: np.ndarray, sr: int, segments: list[dict]) -> np.ndarray`, `retime_windows(windows: list[dict], segments: list[dict]) -> list[dict]`. Task 12 consumes both.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_timestretch_mix.py`:

```python
import pytest
import numpy as np
from prep.timestretch_mix import retime_windows, _crossfade_concat


def test_retime_windows_maps_source_time_into_target_time():
    segments = [
        {"source_start": 0.0, "source_end": 1.0, "target_start": 0.0, "target_end": 1.2, "rate": 1.0 / 1.2, "measure": 1},
        {"source_start": 1.0, "source_end": 2.0, "target_start": 1.2, "target_end": 2.4, "rate": 1.0 / 1.2, "measure": 2},
    ]
    windows = [{"cell_id": "c1", "start": 0.5, "end": 1.5}]
    result = retime_windows(windows, segments)
    assert result[0]["start"] == pytest.approx(0.6)
    assert result[0]["end"] == pytest.approx(1.8)


def test_crossfade_concat_preserves_total_sample_count_minus_overlap():
    a = np.ones(100, dtype=np.float32)
    b = np.ones(100, dtype=np.float32) * 2
    result = _crossfade_concat([a, b], fade_n=10)
    assert len(result) == 190


def test_apply_stretch_segments_produces_target_duration():
    pytest.importorskip("pyrubberband")
    from prep.timestretch_mix import apply_stretch_segments
    from tests.helpers import make_click_track

    audio, sr = make_click_track([0.5, 1.5], sr=22050, duration=2.0)
    segments = [
        {"source_start": 0.0, "source_end": 1.0, "target_start": 0.0, "target_end": 1.2, "rate": 1.0 / 1.2, "measure": 1},
        {"source_start": 1.0, "source_end": 2.0, "target_start": 1.2, "target_end": 2.4, "rate": 1.0 / 1.2, "measure": 2},
    ]
    try:
        result = apply_stretch_segments(audio, sr, segments)
    except (FileNotFoundError, RuntimeError) as exc:
        pytest.skip(f"rubberband CLI not available: {exc}")
    expected_len = int(2.4 * sr)
    assert abs(len(result) - expected_len) < sr * 0.05
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source .venv/Scripts/activate && pytest tests/test_timestretch_mix.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prep.timestretch_mix'`

- [ ] **Step 3: Write the implementation**

Create `prep/timestretch_mix.py`:

```python
from __future__ import annotations
import numpy as np

try:
    import pyrubberband as pyrb
except ImportError:  # Rubber Band CLI / pyrubberband not installed yet
    pyrb = None

CROSSFADE_SECONDS = 0.01
SAFE_RATE_RANGE = (0.7, 1.3)  # beyond ±30% speed change, percussive audio starts sounding artificial


def require_pyrubberband() -> None:
    if pyrb is None:
        raise RuntimeError(
            "pyrubberband is not installed, or the 'rubberband' command-line "
            "tool isn't on PATH. See docs/superpowers/plans/2026-09-17-jangdan-score-trainer.md Task 9."
        )


def apply_stretch_segments(audio: np.ndarray, sr: int, segments: list[dict]) -> np.ndarray:
    """Time-stretch each measure segment of `audio` per `segments` (see
    stretch_plan.build_stretch_segments) and concatenate with a short
    crossfade at each boundary to avoid clicks."""
    require_pyrubberband()
    fade_n = int(CROSSFADE_SECONDS * sr)
    pieces = []
    for seg in segments:
        _warn_if_rate_extreme(seg)
        start_sample = int(seg["source_start"] * sr)
        end_sample = int(seg["source_end"] * sr)
        chunk = audio[start_sample:end_sample]
        if len(chunk) < end_sample - start_sample:
            print(
                f"[timestretch] warning: measure {seg['measure']} wanted "
                f"{end_sample - start_sample} samples but only {len(chunk)} were available "
                f"in the source audio (recording likely ends too early) — output for this "
                f"measure will be shorter than the target duration"
            )
        stretched = pyrb.time_stretch(chunk, sr, seg["rate"])
        pieces.append(stretched.astype(np.float32))
    return _crossfade_concat(pieces, fade_n)


def _warn_if_rate_extreme(seg: dict) -> None:
    low, high = SAFE_RATE_RANGE
    if not (low <= seg["rate"] <= high):
        print(
            f"[timestretch] warning: measure {seg['measure']} needs a {seg['rate']:.2f}x "
            f"stretch, outside the {low}-{high}x range where percussive audio still sounds "
            f"natural. Consider re-recording that measure closer to the reference tempo."
        )


def _crossfade_concat(pieces: list[np.ndarray], fade_n: int) -> np.ndarray:
    if not pieces:
        return np.array([], dtype=np.float32)
    result = pieces[0]
    for piece in pieces[1:]:
        n = min(fade_n, len(result), len(piece))
        if n == 0:
            result = np.concatenate([result, piece])
            continue
        fade_out = np.linspace(1.0, 0.0, n, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, n, dtype=np.float32)
        overlapped = result[-n:] * fade_out + piece[:n] * fade_in
        result = np.concatenate([result[:-n], overlapped, piece[n:]])
    return result


def retime_windows(windows: list[dict], segments: list[dict]) -> list[dict]:
    """Recompute a track's cell windows from its own (source) time into
    the shared canonical (target) time, using the same per-measure
    segments used to stretch its audio."""
    def to_target(t: float) -> float:
        for seg in segments:
            if seg["source_start"] <= t <= seg["source_end"]:
                frac = (t - seg["source_start"]) / (seg["source_end"] - seg["source_start"])
                return seg["target_start"] + frac * (seg["target_end"] - seg["target_start"])
        last = segments[-1]
        return last["target_end"] + (t - last["source_end"])

    return [
        {"cell_id": w["cell_id"], "start": to_target(w["start"]), "end": to_target(w["end"])}
        for w in windows
    ]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source .venv/Scripts/activate && pytest tests/test_timestretch_mix.py -v`
Expected: 3 passed (or 2 passed + 1 skipped if the `rubberband` binary isn't installed yet — that's fine, install it per Task 9 Step 1 when ready to validate real stretching quality).

- [ ] **Step 5: Commit**

```bash
git add prep/timestretch_mix.py tests/test_timestretch_mix.py
git commit -m "feat: add time-stretching and window-retiming for multi-track sync"
```

---

### Task 11: Ensemble prepare CLI

**Files:**
- Modify: `prep/prepare.py`
- Test: `tests/test_prepare_ensemble.py`

**Interfaces:**
- Consumes: everything from Tasks 2, 4, 5, 6, 9, 10.
- Produces: `prepare_ensemble(track_specs: list[dict], reference_id: str, output_dir: str) -> None` (`track_specs` entries: `{"id": str, "audio": str, "score": str}`), and an `ensemble` CLI subcommand (`--manifest`, `--reference`, `--output-dir`). Writes `<output_dir>/<id>.wav` and `<output_dir>/<id>_timeline.json` per track, all sharing the reference track's canonical timeline.

- [ ] **Step 1: Write the failing test**

Create `tests/test_prepare_ensemble.py`:

```python
import json
import pytest
import soundfile as sf
from prep.prepare import prepare_ensemble
from tests.helpers import make_click_track


def two_measure_score(instrument, cell_ids_by_measure):
    cells = []
    x = 0
    for measure, ids in enumerate(cell_ids_by_measure, start=1):
        for cid in ids:
            cells.append({"id": cid, "label": cid, "strike_count": 1, "measure": measure,
                          "bbox": {"x": x, "y": 0, "w": 10, "h": 10}})
            x += 10
    return {"instrument": instrument, "source_image": "x.png", "cells": cells}


def test_prepare_ensemble_aligns_second_track_to_reference_duration(tmp_path):
    pytest.importorskip("pyrubberband")

    janggu_score = two_measure_score("janggu", [["j1"], ["j2"]])
    buk_score = two_measure_score("buk", [["b1"], ["b2"]])
    janggu_score_path = tmp_path / "janggu.json"
    buk_score_path = tmp_path / "buk.json"
    janggu_score_path.write_text(json.dumps(janggu_score), encoding="utf-8")
    buk_score_path.write_text(json.dumps(buk_score), encoding="utf-8")

    # janggu (reference) plays its two measures at t=0.5 and t=1.5 (1.0s/measure)
    janggu_audio, sr = make_click_track([0.5, 1.5], sr=22050, duration=2.0)
    janggu_audio_path = tmp_path / "janggu.wav"
    sf.write(str(janggu_audio_path), janggu_audio, sr)

    # buk plays the same two measures slower: at t=0.6 and t=2.0 (1.4s/measure).
    # Duration must cover the extrapolated end of the last measure segment
    # (2.0 + (2.0-0.6) = 3.4s), plus a little headroom, or apply_stretch_segments
    # will silently read a truncated last chunk.
    buk_audio, sr = make_click_track([0.6, 2.0], sr=22050, duration=3.6)
    buk_audio_path = tmp_path / "buk.wav"
    sf.write(str(buk_audio_path), buk_audio, sr)

    output_dir = tmp_path / "output"
    track_specs = [
        {"id": "janggu", "audio": str(janggu_audio_path), "score": str(janggu_score_path)},
        {"id": "buk", "audio": str(buk_audio_path), "score": str(buk_score_path)},
    ]

    try:
        prepare_ensemble(track_specs, reference_id="janggu", output_dir=str(output_dir))
    except (FileNotFoundError, RuntimeError) as exc:
        pytest.skip(f"rubberband CLI not available: {exc}")

    assert (output_dir / "janggu.wav").exists()
    assert (output_dir / "buk.wav").exists()

    janggu_out, sr = sf.read(str(output_dir / "janggu.wav"))
    buk_out, _ = sf.read(str(output_dir / "buk.wav"))
    # buk was stretched to match janggu's measure timing, so both tracks
    # should now run roughly the same total length
    assert abs(len(janggu_out) - len(buk_out)) < sr * 0.1

    buk_timeline = json.loads((output_dir / "buk_timeline.json").read_text(encoding="utf-8"))
    b1_start = next(c["start"] for c in buk_timeline["cells"] if c["cell_id"] == "b1")
    assert abs(b1_start - 0.5) < 0.05  # retimed onto janggu's measure-1 start
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `source .venv/Scripts/activate && pytest tests/test_prepare_ensemble.py -v`
Expected: FAIL with `ImportError: cannot import name 'prepare_ensemble' from 'prep.prepare'`

- [ ] **Step 3: Add `prepare_ensemble` to `prep/prepare.py`**

Add these imports to the top of `prep/prepare.py`:

```python
import json
from pathlib import Path
import soundfile as sf
from prep.stretch_plan import measure_start_times, build_stretch_segments
from prep.timestretch_mix import apply_stretch_segments, retime_windows
```

Add this function after `prepare_single_track`:

```python
def prepare_ensemble(track_specs: list[dict], reference_id: str, output_dir: str) -> None:
    """track_specs: [{"id", "audio", "score"}, ...]. The entry matching
    reference_id defines the canonical measure timeline; every other
    track is time-stretched to match it. Writes <output_dir>/<id>.wav and
    <output_dir>/<id>_timeline.json for every track."""
    prepared = {}
    for spec in track_specs:
        score = load_score(spec["score"])
        expected_ids = expected_onset_sequence(score)
        audio, sr = librosa.load(spec["audio"], sr=None, mono=True)
        detected_times = detect_onsets(audio, sr)
        try:
            aligned = align_onsets(expected_ids, detected_times)
        except AlignmentMismatchError as exc:
            print(f"[prepare] alignment failed for track '{spec['id']}': {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        windows = build_timeline(score, aligned)
        prepared[spec["id"]] = {"score": score, "audio": audio, "sr": sr, "windows": windows}

    reference = prepared[reference_id]
    target_measure_times = measure_start_times(reference["score"], reference["windows"])

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for track_id, data in prepared.items():
        if track_id == reference_id:
            final_audio, final_windows = data["audio"], data["windows"]
        else:
            source_measure_times = measure_start_times(data["score"], data["windows"])
            segments = build_stretch_segments(source_measure_times, target_measure_times)
            final_audio = apply_stretch_segments(data["audio"], data["sr"], segments)
            final_windows = retime_windows(data["windows"], segments)

        sf.write(str(Path(output_dir) / f"{track_id}.wav"), final_audio, data["sr"])
        write_timeline(final_windows, Path(output_dir) / f"{track_id}_timeline.json")
        print(f"[prepare] wrote {track_id}.wav and {track_id}_timeline.json")
```

Add the `ensemble` subcommand inside `main()`, after the `single` subparser block and before `args = parser.parse_args(argv)`:

```python
    ensemble = sub.add_parser("ensemble", help="Align and time-sync multiple instruments into one set")
    ensemble.add_argument("--manifest", required=True, help="JSON file: [{id, audio, score}, ...]")
    ensemble.add_argument("--reference", required=True, help="Track id whose timing the others are stretched to match")
    ensemble.add_argument("--output-dir", required=True)
```

And extend the dispatch at the bottom of `main()`:

```python
    if args.command == "single":
        prepare_single_track(args.audio, args.score, args.output)
    elif args.command == "ensemble":
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        prepare_ensemble(manifest, args.reference, args.output_dir)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `source .venv/Scripts/activate && pytest tests/test_prepare_ensemble.py -v`
Expected: 1 passed (or 1 skipped if the `rubberband` binary isn't installed yet — install it per Task 9 Step 1 to actually exercise this).

- [ ] **Step 5: Run the full test suite**

Run: `source .venv/Scripts/activate && pytest -v`
Expected: all tests pass (or skip only where `rubberband` is unavailable).

- [ ] **Step 6: Commit**

```bash
git add prep/prepare.py tests/test_prepare_ensemble.py
git commit -m "feat: add ensemble prepare CLI syncing multiple tracks to one timeline"
```

---

### Task 12: Multi-track web player verification

**Files:**
- Modify: `web/fixtures/dummy_score.json` (rename usage — create a second instrument's fixtures alongside it)
- Create: `web/fixtures/dummy_score_2.json`
- Create: `web/fixtures/dummy_timeline_2.json`
- Create: `web/fixtures/dummy_audio_2.wav`
- Modify: `web/app.js`

**Interfaces:**
- Consumes: `ScorePlayer` from Task 8 unchanged (it already supports N tracks).
- Produces: nothing new consumed by later tasks — this is the final integration check for the design's core requirement (1-4 tracks, independent show/mute).

- [ ] **Step 1: Generate a second instrument's fixtures**

Create `web/fixtures/dummy_score_2.json`:

```json
{
  "instrument": "buk",
  "source_image": null,
  "cells": [
    {"id": "b1", "label": "쿵", "strike_count": 1, "measure": 1, "bbox": {"x": 10, "y": 10, "w": 80, "h": 60}},
    {"id": "b2", "label": "쿵", "strike_count": 1, "measure": 2, "bbox": {"x": 100, "y": 10, "w": 80, "h": 60}}
  ]
}
```

Run:

```bash
source .venv/Scripts/activate
python - <<'EOF'
import soundfile as sf
from tests.helpers import make_click_track
from prep.score_schema import load_score, expected_onset_sequence
from prep.align import align_onsets
from prep.timeline import build_timeline, write_timeline

onset_times = [0.5, 2.0]
audio, sr = make_click_track(onset_times, sr=22050, duration=2.5)
sf.write("web/fixtures/dummy_audio_2.wav", audio, sr)

score = load_score("web/fixtures/dummy_score_2.json")
aligned = align_onsets(expected_onset_sequence(score), onset_times)
windows = build_timeline(score, aligned)
write_timeline(windows, "web/fixtures/dummy_timeline_2.json")
EOF
```

Expected: creates `web/fixtures/dummy_audio_2.wav` and `web/fixtures/dummy_timeline_2.json`.

- [ ] **Step 2: Add the second track to the player's track list**

In `web/app.js`, replace the `TRACK_DEFS` array:

```javascript
const TRACK_DEFS = [
  { id: 'janggu', label: '장구', audioUrl: 'fixtures/dummy_audio.wav', timelineUrl: 'fixtures/dummy_timeline.json', scoreUrl: 'fixtures/dummy_score.json' },
  { id: 'buk', label: '북', audioUrl: 'fixtures/dummy_audio_2.wav', timelineUrl: 'fixtures/dummy_timeline_2.json', scoreUrl: 'fixtures/dummy_score_2.json' },
];
```

- [ ] **Step 3: Manually verify multi-track behavior in a browser**

```bash
cd /d/samulnori-trainer/web
python -m http.server 8000
```

Open `http://localhost:8000/` and check:
- Both 장구 and 북 panels appear side by side (or stacked on a narrow window).
- Clicking 재생 starts both tracks at the same instant; both highlight their first cell together at the start.
- Unchecking 북's "표시" checkbox hides only the 북 panel and mutes only 북's audio; 장구 keeps playing and highlighting normally, with no audible gap or restart.
- Re-checking 북 resumes its sound in sync with 장구 (not restarted from the beginning) — confirms the shared `AudioContext` clock kept them aligned while muted.
- Using "소리만 끄기" on 장구 keeps its panel visible and highlighting while silencing only its audio.

Stop the server with Ctrl+C when done.

- [ ] **Step 4: Commit**

```bash
git add web/
git commit -m "test: verify multi-track show/mute and sync with a second fixture instrument"
```

---

## Deferred from the spec (not in this plan's scope)

- **검수 도구 (manual correction UI):** the design spec calls for a waveform-based tool to manually fix bad alignments. This plan's `align.py` (Task 5) instead fails loudly with `AlignmentMismatchError` on any count mismatch rather than guessing — see that task's "Design decision" note for why. Whether a manual-correction UI is still worth building depends on how often real recordings actually trigger that error; decide after running Task 7/11's CLI against real audio a few times. If it comes up often, that's a follow-up plan: a small web page that plots the waveform (Web Audio API `getChannelData`) with detected onsets marked, letting you drag-add/remove onset markers and re-run `align_onsets` with a hand-edited detected-times list.
- **정렬 오차 정량 측정 (accuracy measurement against real audio):** the spec's validation plan calls for measuring alignment error in milliseconds against real recordings. Not doable without real audio; do this manually once Task 7's real-file run is possible, by ear/eye comparison against the waveform, and note here if it needs to become an automated regression check.

## Not yet in this plan: encoding the real gakbo patterns

The teacher has already provided the real gakbo material (마동초 사물놀이1.pdf — 쩍쩍이굿, 타령장단, 칠채 기본/변형 가락, 벙어리 칠채, 육채, 별달거리, 휘모리, 짝쇠, etc.) and clarified three transcription rules now captured in `data/scores/가락보_표기_규칙.md`. Turning that PDF into actual `data/scores/*.json` files per instrument per jangdan is real work but doesn't need audio — it could be its own follow-up task/plan once:
- the "털기읏" ambiguity in 타령장단 is resolved with the teacher (see the open question in `가락보_표기_규칙.md`), and
- bbox coordinates are extracted from the real table image (needs the PDF converted to per-page images, then coordinates picked per cell — Task 8's dummy fixtures show the JSON shape but used made-up coordinates).

## Notes for whoever picks this up with real files

- Real hwp gakbo files should be converted to PDF or image first (Claude can't parse `.hwp` directly) — see the design spec's "가락보 데이터 모델" section for the JSON shape to fill in from the converted table, including real `bbox` coordinates per cell.
- Once a real reference recording exists, run `prepare.py single` first and sanity-check the output timeline against the audio by ear/eye before trusting it; if `AlignmentMismatchError` fires, the onset detector's parameters (in `prep/onset_detect.py`) likely need tuning for that instrument's real timbre (e.g. 꽹과리's sharp high-frequency transients vs 징's long low-frequency decay may need different `backtrack`/threshold settings) — this is expected and not a bug in the pipeline logic itself.
- Real recordings used as non-reference tracks in `prepare_ensemble` (Task 11) should include a few seconds of trailing audio after the last measure's last hit. `build_stretch_segments` extrapolates the last measure's duration from the previous measure, which can ask for more source audio than exists if the recording cuts off right at the final hit; `apply_stretch_segments` now warns (doesn't fail) when that happens, but the output for that measure will be shorter than intended.
