from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PrefillRecordWriter:
    output_path: Path

    @classmethod
    def from_env(
        cls,
        *,
        tp_rank: int,
        attn_tp_rank: int,
        attn_tp_size: int,
        attn_dp_rank: int,
        attn_cp_rank: int,
        moe_ep_rank: int,
        dp_rank: int,
    ) -> "PrefillRecordWriter | None":
        record_dir = os.environ.get("SGLANG_PREFILL_RECORD_DIR")
        if not record_dir:
            return None

        base_dir = Path(record_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        file_name = (
            "prefill_"
            f"tp{tp_rank}_attn-group{attn_tp_rank}-of-{attn_tp_size}_"
            f"attn-dp{attn_dp_rank}_attn-cp{attn_cp_rank}_"
            f"moe-ep{moe_ep_rank}_dp{dp_rank}.jsonl"
        )
        return cls(output_path=base_dir / file_name)

    def write(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"
        with self.output_path.open("a", encoding="utf-8") as fh:
            fh.write(line)
