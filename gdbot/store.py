"""Run directory: logs, tapes, policies and resumable state.

Training runs for hours.  Everything worth not losing goes in one directory, in
formats that can be inspected without this package: JSONL for the log, plain JSON
for tapes and config, ``.npy`` for policy weights.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from gdbot.sim.level import _rle_decode, _rle_encode


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in obj]
    return obj


class RunStore:
    """Everything one training run writes to disk."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.log_path = self.root / "log.jsonl"
        self._t0 = time.time()

    # ------------------------------------------------------------------- log
    def log(self, event: str, **fields: Any) -> None:
        record = {"t": round(time.time() - self._t0, 3), "event": event}
        record.update(_jsonable(fields))
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def read_log(self) -> Iterator[dict]:
        if not self.log_path.exists():
            return iter(())
        return (json.loads(line) for line in self.log_path.read_text(encoding="utf-8").splitlines() if line)

    # ----------------------------------------------------------------- tapes
    def save_tape(self, name: str, tape: np.ndarray, **meta: Any) -> Path:
        """Store a tape run-length encoded, with whatever metadata describes it."""
        arr = np.asarray(tape, dtype=np.uint8).reshape(-1)
        path = self.root / f"{name}.tape.json"
        payload = {
            "ticks": int(arr.size),
            "presses": int(np.count_nonzero(np.diff(np.concatenate(([0], arr))) > 0)),
            "rle": _rle_encode(arr.tolist()),
            **_jsonable(meta),
        }
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        return path

    def load_tape(self, name: str) -> np.ndarray:
        path = self.root / f"{name}.tape.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        return np.asarray(_rle_decode(data["rle"]), dtype=np.uint8)

    def has_tape(self, name: str) -> bool:
        return (self.root / f"{name}.tape.json").exists()

    # -------------------------------------------------------------- policies
    def save_policy(self, name: str, params: np.ndarray) -> Path:
        path = self.root / f"{name}.policy.npy"
        np.save(path, np.asarray(params, dtype=np.float32))
        return path

    def load_policy(self, name: str) -> np.ndarray:
        return np.load(self.root / f"{name}.policy.npy")

    def has_policy(self, name: str) -> bool:
        return (self.root / f"{name}.policy.npy").exists()

    # ------------------------------------------------------------------ misc
    def save_json(self, name: str, obj: Any) -> Path:
        path = self.root / f"{name}.json"
        path.write_text(json.dumps(_jsonable(obj), indent=2), encoding="utf-8")
        return path

    def load_json(self, name: str) -> Any:
        return json.loads((self.root / f"{name}.json").read_text(encoding="utf-8"))
