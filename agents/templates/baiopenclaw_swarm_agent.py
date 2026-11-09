"""BAIopenclaw Key-Gated VLIW Swarm Agent for ARC-AGI-3 (Zero External API).

Native non-LLM runtime controller implementing INTERFACE CONTRACT v1.0 (`CONTRACT_v1.0.md`):
1. Agent 001 `ModelAdapter` (`D:\\MuonAnomalyStudy\\000AgentInbox\\coordination\\experiments\\model_adapter.py`):
   - 2-step causal patch encoding `[c(t), c(t+1)]` at Goldilocks amplitude `A = 0.25`, DynamicFlux `S = 0.0`.
   - Emits `feature_vector` (`[B, 256]`), `causal_transition_score` (`float` in `[0.0, 1.0]`), and
     `specialist_logits` (`{"col_-3".."col_+3": [B, 2]}`).
2. Agent 002 `GateController` (`D:\\MuonAnomalyStudy\\000AgentInbox\\coordination\\experiments\\gate_controller.py`):
   - Native `baiopenclaw` `VLIW5DRouter`, `VesselConsentEngine`, `StateEntropyMonitor`,
     `CriticalPathSieve`, and `SystemSynchronizer` BLAKE2b hash-chained ledger.
   - Enforces Lean 4 Theorem 1 (`df/dr = ~q` on `col_+0`), Theorem 14 (`rule30(0,1,1) = 1`
     Corner-3 self-ejection), and 3-state deterministic gating (`+1` Unlock, `0` Quarantine, `-1` Revoke).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from arcengine import FrameData, GameAction, GameState

from ..agent import Agent

# Wire in canonical `baiopenclaw/tools/hashlocker` and `coordination/experiments` (zero fallback stubs)
_BAIOPENCLAW_HASHLOCKER = Path(r"C:\Users\walla\source\repos\baiopenclaw\tools\hashlocker")
if str(_BAIOPENCLAW_HASHLOCKER) not in sys.path:
    sys.path.insert(0, str(_BAIOPENCLAW_HASHLOCKER))

_COORDINATION_EXPERIMENTS = Path(r"D:\MuonAnomalyStudy\000AgentInbox\coordination\experiments")
if str(_COORDINATION_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_COORDINATION_EXPERIMENTS))

from gateway.consent_engine import ConsentFactor, ExcitationLevel, VesselConsentEngine
from gateway.critical_path_sieve import CriticalPathSieve
from gateway.state_entropy_monitor import StateEntropyMonitor
from gateway.state_synchronizer import SystemSynchronizer
from gateway.vliw_5d_router import VLIW5DRouter
from gate_controller import GateController
from model_adapter import ArcGridPatchTransformer, ModelAdapter


class BAIopenclawSwarmAgent(Agent):
    """
    Deterministic non-LLM controller for ARC-AGI-3 that streams real game frames
    through `build_contract_observation()` -> `ModelAdapter.forward()` -> `GateController.evaluate()`
    inside `BAIopenclawSwarmAgent.step()`.
    """

    MAX_ACTIONS: int = 12
    CERTIFIED_729_KEYS = (364, 378, 405, 406, 416, 504, 567, 648)
    CAUSAL_REVOKE_THRESHOLD: float = 0.50
    TEMPORAL_WINDOW_SIZE: int = 4
    SPECIALIST_COLUMNS: Tuple[str, ...] = (
        "col_-3",
        "col_-2",
        "col_-1",
        "col_+0",
        "col_+1",
        "col_+2",
        "col_+3",
    )

    SPECIALIST_COUNCIL: Dict[str, Dict[str, Any]] = {
        "col_-3": {"ckpt": "tph_model_k8s_20250805_015941.pth", "invert": False, "acc_35": 0.68571, "pt_modes": (218, 229), "aab": (447, 447, 759)},
        "col_-2": {"ckpt": "tph_model_k8s_20250804_202443.pth", "invert": True,  "acc_35": 0.65714, "pt_modes": (251, 254), "aab": (850, 850, 571)},
        "col_-1": {"ckpt": "tph_model_k8s_20250805_015837.pth", "invert": True,  "acc_35": 0.62857, "pt_modes": (218, 229), "aab": (411, 411, 539)},
        "col_+0": {"ckpt": "tph_model_k8s_20250804_202444.pth", "invert": True,  "acc_35": 0.74286, "pt_modes": (251, 254), "aab": (850, 850, 571)},
        "col_+1": {"ckpt": "tph_model_k8s_20250805_014525.pth", "invert": False, "acc_35": 0.65714, "pt_modes": (218, 229), "aab": (300, 300, 699)},
        "col_+2": {"ckpt": "tph_model_k8s_20250804_202444.pth", "invert": False, "acc_35": 0.71429, "pt_modes": (251, 254), "aab": (850, 850, 571)},
        "col_+3": {"ckpt": "tph_model_k8s_20250805_020406.pth", "invert": True,  "acc_35": 0.65714, "pt_modes": (218, 229), "aab": (411, 411, 806)},
    }

    def __init__(
        self,
        *args: Any,
        model_adapter: Optional[ModelAdapter] = None,
        gate_controller: Optional[GateController] = None,
        **kwargs: Any,
    ) -> None:
        self.model = "baiopenclaw-vliw-7col-swarm"
        super().__init__(*args, **kwargs)
        self.model_adapter: ModelAdapter = (
            model_adapter
            if model_adapter is not None
            else ModelAdapter(window_size=self.TEMPORAL_WINDOW_SIZE, calibrated_std=0.25)
        )
        self.gate_controller: GateController = (
            gate_controller
            if gate_controller is not None
            else GateController(enable_theorem14_scrub=True)
        )
        self.arc_grid_transformer: ArcGridPatchTransformer = ArcGridPatchTransformer(
            mode="bbox_strip",
            width=7,
            quarantine_on_noop=True,
            quarantine_on_cycle=True,
        )
        self.consent_engine = self.gate_controller.consent_engine
        self.vliw_router = self.gate_controller.vliw_router
        self.entropy_monitor = self.gate_controller.entropy_monitor
        self.synchronizer = self.gate_controller.synchronizer
        self.sieve = self.gate_controller.sieve

        self.controller_state: Optional[int] = None
        self.step_index: int = 0
        self.quarantine_pivot_offset: int = 0
        self.reset_flag: bool = True
        self.temporal_grid_buffer: List[List[List[int]]] = []
        self.temporal_strip_buffer: List[List[int]] = []
        self.temporal_row256_buffer: List[List[int]] = []
        self.last_arc_grid_meta: Optional[Dict[str, Any]] = None
        self.last_gate_result: Optional[Dict[str, Any]] = None

    def reset(self) -> None:
        """
        Explicit episode reset per CONTRACT_v1.0.md:
        Clears temporal buffers, resets `ModelAdapter` and `GateController`, and sets
        `controller_state = None` (uninitialized) with `reset_flag = True`.
        """
        self.controller_state = None
        self.step_index = 0
        self.quarantine_pivot_offset = 0
        self.reset_flag = True
        self.temporal_grid_buffer.clear()
        self.temporal_strip_buffer.clear()
        self.temporal_row256_buffer.clear()
        self.last_arc_grid_meta = None
        self.last_gate_result = None
        self.arc_grid_transformer.reset()
        self.model_adapter.reset()
        self.gate_controller.reset()
        self.vliw_router = self.gate_controller.vliw_router

    @property
    def name(self) -> str:
        return f"{super().name}.{self.model}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    @staticmethod
    def _rule30(p: int, q: int, r: int) -> int:
        return (p ^ (q | r)) & 1

    @classmethod
    def _step_row_rule30(cls, row: List[int]) -> List[int]:
        n = len(row)
        if n == 0:
            return []
        out = [0] * n
        for i in range(n):
            p = row[(i - 1) % n]
            q = row[i]
            r = row[(i + 1) % n]
            out[i] = cls._rule30(p, q, r)
        return out

    def _extract_grid_2d(self, frame_3d: Optional[list[list[list[int]]]]) -> list[list[int]]:
        if not frame_3d or not frame_3d[0]:
            return [[0] * 64 for _ in range(64)]
        plane = frame_3d[0]
        rows = len(plane)
        cols = len(plane[0]) if rows > 0 else 0
        out = [[0] * 64 for _ in range(64)]
        for r in range(min(64, rows)):
            for c in range(min(64, cols)):
                out[r][c] = int(plane[r][c])
        return out

    def _encode_frame_to_rule30_and_729(self, grid: list[list[int]]) -> Dict[str, Any]:
        """
        Maps a 64x64 ARC grid into:
        - Active binary matrix (foreground non-background cells),
        - 3-way Bit-Overlap Collapse & Surviving Transfer counts,
        - 7-column central strip `[7]` around `c = center_c + dx` (`dx in -3..+3`),
        - 256-cell row vector `[256]`,
        - 6-trit `729`-space index and 3-bit Chair octant (`0..7`).
        """
        counts: Dict[int, int] = {}
        for r in range(64):
            for c in range(64):
                v = grid[r][c]
                counts[v] = counts.get(v, 0) + 1
        bg_color = max(counts, key=counts.get) if counts else 0

        bin_grid = [[1 if grid[r][c] != bg_color else 0 for c in range(64)] for r in range(64)]

        shielded = 0
        xor_cancelled = 0
        surviving_transfer = 0
        corner3_fractures = 0
        active_centroids_x: List[int] = []
        active_centroids_y: List[int] = []

        for r in range(1, 63):
            for c in range(1, 63):
                p = bin_grid[r][c - 1]
                q = bin_grid[r][c]
                rv = bin_grid[r][c + 1]
                if q == 1:
                    shielded += 1
                    active_centroids_x.append(c)
                    active_centroids_y.append(r)
                    if p == 0 and rv == 1:
                        corner3_fractures += 1
                else:
                    if p == rv:
                        xor_cancelled += 1
                    else:
                        surviving_transfer += 1

        center_c = int(sum(active_centroids_x) / len(active_centroids_x)) if active_centroids_x else 32
        center_r = int(sum(active_centroids_y) / len(active_centroids_y)) if active_centroids_y else 32
        center_c = max(4, min(59, center_c))
        center_r = max(2, min(61, center_r))

        seam_row = center_r if any(bin_grid[center_r]) else (31 if any(bin_grid[31]) else 17)
        strip_row = [bin_grid[seam_row][center_c + dx] for dx in range(-3, 4)]

        row_256: List[int] = []
        for dr in (-1, 0, 1, 2):
            rr = max(0, min(63, seam_row + dr))
            row_256.extend(bin_grid[rr])

        trits_6: List[int] = []
        idx_729 = 0
        for i, dx in enumerate(range(-2, 4)):
            q = bin_grid[seam_row][center_c + dx]
            rv = bin_grid[seam_row][center_c + dx + 1]
            trit = 1 if q == 1 else (2 if rv == 1 else 0)
            trits_6.append(trit)
            idx_729 += trit * (3 ** i)

        p0, q0, r0 = strip_row[2], strip_row[3], strip_row[4]
        octant = (p0 << 2) | (q0 << 1) | r0

        self.temporal_grid_buffer.append([list(r) for r in grid])
        self.temporal_strip_buffer.append(list(strip_row))
        self.temporal_row256_buffer.append(list(row_256))
        if len(self.temporal_grid_buffer) > self.TEMPORAL_WINDOW_SIZE:
            self.temporal_grid_buffer = self.temporal_grid_buffer[-self.TEMPORAL_WINDOW_SIZE :]
        if len(self.temporal_strip_buffer) > self.TEMPORAL_WINDOW_SIZE:
            self.temporal_strip_buffer = self.temporal_strip_buffer[-self.TEMPORAL_WINDOW_SIZE :]
        if len(self.temporal_row256_buffer) > self.TEMPORAL_WINDOW_SIZE:
            self.temporal_row256_buffer = self.temporal_row256_buffer[-self.TEMPORAL_WINDOW_SIZE :]

        return {
            "bin_grid": bin_grid,
            "strip_7col": strip_row,
            "row_256": row_256,
            "shielded_by_1": shielded,
            "xor_seam_cancelled": xor_cancelled,
            "surviving_transfer": surviving_transfer,
            "corner3_fractures": corner3_fractures,
            "idx_729": idx_729,
            "trits_6": trits_6,
            "octant": octant,
            "target_x": center_c,
            "target_y": center_r,
        }

    def build_contract_observation(
        self,
        width: int = 7,
        mode: Optional[str] = None,
    ) -> Tuple[np.ndarray, bool]:
        """
        Phase 2 Grid-to-Causal Patch Transformer (`CONTRACT_v1.0.md`, `DECISION-001`, `MSG-005`):
        Calls `ArcGridPatchTransformer(mode="bbox_strip" | "scanline_256").transform(grids_4d)`
        across the sliding 2D ARC grid window `[1, T, 64, 64]` when `T >= 2`, or rolls out
        a 4-step causal carrier patch `[1, 4, W]` on the initial frame (`T = 1`).
        """
        effective_mode = mode or ("scanline_256" if width == 256 else "bbox_strip")
        if effective_mode == "bbox_strip" and width == 7:
            transformer = self.arc_grid_transformer
        else:
            transformer = ArcGridPatchTransformer(
                mode=effective_mode,
                width=width,
                quarantine_on_noop=True,
                quarantine_on_cycle=True,
            )

        if len(self.temporal_grid_buffer) >= 2:
            grids_list = list(self.temporal_grid_buffer)
            grids_4d = np.asarray([grids_list], dtype=np.int64)
            obs, arc_info = transformer.transform(grids_4d)
            self.last_arc_grid_meta = arc_info["batch_metadata"][0]
            if obs.shape[1] < self.TEMPORAL_WINDOW_SIZE:
                rows_ext = [ [int(b) for b in r] for r in obs[0].tolist() ]
                while len(rows_ext) < self.TEMPORAL_WINDOW_SIZE:
                    rows_ext.append(self._step_row_rule30(rows_ext[-1]))
                obs = np.asarray([rows_ext], dtype=np.float32)
        else:
            self.last_arc_grid_meta = None
            if width == 256:
                seed_row = (
                    list(self.temporal_row256_buffer[-1])
                    if self.temporal_row256_buffer
                    else [0] * 256
                )
                if not any(seed_row):
                    seed_row[128] = 1
            else:
                seed_row = (
                    list(self.temporal_strip_buffer[-1])
                    if self.temporal_strip_buffer
                    else [0, 0, 0, 1, 0, 0, 0]
                )
                if not any(seed_row):
                    seed_row[3] = 1

            window_rows: List[List[int]] = [seed_row]
            for _ in range(self.TEMPORAL_WINDOW_SIZE - 1):
                window_rows.append(self._step_row_rule30(window_rows[-1]))
            obs = np.asarray([window_rows], dtype=np.float32)  # [1, 4, W]

        flag = self.reset_flag
        self.reset_flag = False
        return obs, flag

    @classmethod
    def decode_specialist_logits(
        cls,
        specialist_logits: Optional[Dict[str, Any]],
        fallback_strip: List[int],
    ) -> Dict[str, Any]:
        """
        Decodes Agent 001's `specialist_logits` dictionary (`CONTRACT_v1.0.md`) and
        verifies Theorem 1 (`df/dr = ~q`) on `col_+0`.
        """
        if not specialist_logits:
            return {
                "provided": False,
                "predicted_strip": list(fallback_strip),
                "raw_bits": {},
                "calibrated_bits": {
                    col: int(fallback_strip[idx]) for idx, col in enumerate(cls.SPECIALIST_COLUMNS)
                },
                "col0_df_dr_bit": 1 - int(fallback_strip[3]),
                "col0_theorem1_verified": True,
                "strip_agreement": 1.0,
            }

        raw_bits: Dict[str, int] = {}
        calibrated_bits: Dict[str, int] = {}
        predicted_strip: List[int] = []

        for idx, col in enumerate(cls.SPECIALIST_COLUMNS):
            val = specialist_logits.get(col)
            if val is None:
                raw_bit = int(fallback_strip[idx]) ^ (1 if cls.SPECIALIST_COUNCIL[col]["invert"] else 0)
            else:
                if hasattr(val, "detach"):
                    arr = val.detach().cpu().numpy()
                else:
                    arr = np.asarray(val, dtype=np.float32)
                arr_flat = arr.reshape(-1)
                if arr_flat.size >= 2:
                    raw_bit = 1 if float(arr_flat[1]) > float(arr_flat[0]) else 0
                elif arr_flat.size == 1:
                    raw_bit = 1 if float(arr_flat[0]) > 0.0 else 0
                else:
                    raw_bit = int(fallback_strip[idx])

            invert = bool(cls.SPECIALIST_COUNCIL[col]["invert"])
            cal_bit = (1 - raw_bit) if invert else raw_bit
            raw_bits[col] = raw_bit
            calibrated_bits[col] = cal_bit
            predicted_strip.append(cal_bit)

        q0_cal = int(calibrated_bits["col_+0"])
        col0_raw = int(raw_bits["col_+0"])
        col0_theorem1_verified = (col0_raw == (1 - q0_cal)) and (
            q0_cal == int(fallback_strip[3])
        )

        matches = sum(1 for a, b in zip(predicted_strip, fallback_strip) if int(a) == int(b))
        strip_agreement = float(matches) / float(len(fallback_strip))

        return {
            "provided": True,
            "predicted_strip": predicted_strip,
            "raw_bits": raw_bits,
            "calibrated_bits": calibrated_bits,
            "col0_df_dr_bit": col0_raw,
            "col0_theorem1_verified": col0_theorem1_verified,
            "strip_agreement": round(strip_agreement, 4),
        }

    def step(
        self,
        latest_frame: FrameData,
        causal_transition_score: Optional[float] = None,
        specialist_logits: Optional[Dict[str, Any]] = None,
        feature_vector: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Phase 2 Core Step Pipeline (`CONTRACT_v1.0.md`):
        1. Extracts the 64x64 grid from `latest_frame` and encodes Rule 30 / 729 features.
        2. Calls `build_contract_observation(width=7)` to construct `[1, 4, 7]` (`float32`).
        3. Streams `raw_obs` through `self.model_adapter.forward(raw_obs, reset_flag=True)`.
        4. Evaluates the 3-state gate (`+1` Unlock, `0` Quarantine, `-1` Revoke) via
           `self.gate_controller.evaluate(feature_vector, causal_transition_score, specialist_logits)`.
        """
        self.step_index += 1
        grid = self._extract_grid_2d(latest_frame.frame)
        encoded = self._encode_frame_to_rule30_and_729(grid)
        raw_obs, r_flag = self.build_contract_observation(width=7)

        # Reference strip aligned with ModelAdapter's strip7[0] = window_grid[-2]
        ref_strip = [int(x) for x in raw_obs[0, -2].tolist()]

        if causal_transition_score is None and specialist_logits is None and feature_vector is None:
            adapter_out = self.model_adapter.forward(raw_obs, reset_flag=True)
            feature_vector = adapter_out["feature_vector"]
            causal_transition_score = float(adapter_out["causal_transition_score"])
            specialist_logits = adapter_out["specialist_logits"]
        else:
            if feature_vector is None:
                adapter_out = self.model_adapter.forward(raw_obs, reset_flag=True)
                feature_vector = adapter_out["feature_vector"]
                if causal_transition_score is None:
                    causal_transition_score = float(adapter_out["causal_transition_score"])
                if specialist_logits is None:
                    specialist_logits = adapter_out["specialist_logits"]

        council_eval = self.decode_specialist_logits(specialist_logits, ref_strip)

        # If caller explicitly provided specialist_logits that violate Theorem 1 against ref_strip,
        # or if ArcGridPatchTransformer detected a 1-step no-op or k-step cycle trap (G(t) in {G(0)..G(t-1)}),
        # place score in Quarantine [0.50, 0.65) band
        effective_score = float(causal_transition_score if causal_transition_score is not None else 0.0)
        is_noop_2d = bool(self.last_arc_grid_meta and self.last_arc_grid_meta.get("is_noop"))
        is_cycle_trap_2d = bool(
            self.last_arc_grid_meta
            and (self.last_arc_grid_meta.get("is_noop") or self.last_arc_grid_meta.get("is_cycle_trap"))
        )
        if (not council_eval["col0_theorem1_verified"] or is_cycle_trap_2d) and effective_score >= 0.50:
            effective_score = 0.58
            if causal_transition_score is None or is_cycle_trap_2d:
                causal_transition_score = 0.58

        gate = self.gate_controller.evaluate(
            feature_vector=feature_vector,
            causal_transition_score=effective_score,
            specialist_logits=specialist_logits,
        )
        eval_meta = dict(self.gate_controller.last_evaluation or {})

        pt_mode = 220 if (self.step_index % 2 == 0) else 249
        aab_triplet = self.SPECIALIST_COUNCIL["col_+0"]["aab"]
        ent_out = self.entropy_monitor.encode_telemetry_vector(
            [
                float(ref_strip[1]),
                float(ref_strip[3]),
                float(ref_strip[5]),
                float(eval_meta.get("chair_octant", encoded["octant"])) / 7.0,
            ]
        )

        self.controller_state = gate
        result = {
            "f_value": int(eval_meta.get("f_value", 0 if gate == 1 else 1)),
            "ternary_state": int(gate),
            "gate_status": str(eval_meta.get("status", "UNLOCK" if gate == 1 else ("REVOKE" if gate == -1 else "QUARANTINE"))),
            "causal_transition_score": round(float(causal_transition_score or 0.0), 4),
            "col0_theorem1_verified": bool(council_eval["col0_theorem1_verified"]),
            "is_noop": is_noop_2d,
            "is_cycle_trap": is_cycle_trap_2d,
            "euclid_mode": str(eval_meta.get("euclid_mode", "INIT_STEP")),
            "euclid_mirror_consistent": bool(eval_meta.get("euclid_mirror_consistent", True)),
            "specialist_strip_agreement": float(council_eval["strip_agreement"]),
            "corner3_hits": int(eval_meta.get("corner3_hits", 0)),
            "turns_to_unlock": 1,
            "chair_octant": int(eval_meta.get("chair_octant", encoded["octant"])),
            "key_729": int(eval_meta.get("key_729", encoded["idx_729"])),
            "pt_outlier_mode": pt_mode,
            "aab_tokens": list(aab_triplet),
            "synthesized_key": str(eval_meta.get("synthesized_key", "")),
            "spectral_gap_lambda1": float(eval_meta.get("spectral_gap_lambda1", 0.0)),
            "lyapunov_energy": float(self.vliw_router.prev_lyapunov_energy),
            "delta_lyapunov": float(eval_meta.get("delta_lyapunov", 0.0)),
            "shannon_entropy": float(ent_out.get("shannon_entropy", 0.0)),
            "blake2b_anchor": str(eval_meta.get("blake2b_anchor", "")),
            "strip_7col": list(council_eval["predicted_strip"]),
            "raw_obs_shape": list(raw_obs.shape),
            "reset_flag_consumed": bool(r_flag),
            "encoded": encoded,
        }
        self.last_gate_result = result
        if hasattr(self, "recorder") and not self.is_playback:
            self.recorder.record(
                {
                    "step": self.step_index,
                    "gate_status": result["gate_status"],
                    "ternary_state": result["ternary_state"],
                    "f_value": result["f_value"],
                    "causal_transition_score": result["causal_transition_score"],
                    "col0_theorem1_verified": result["col0_theorem1_verified"],
                    "is_cycle_trap": result["is_cycle_trap"],
                    "euclid_mode": result["euclid_mode"],
                    "euclid_mirror_consistent": result["euclid_mirror_consistent"],
                    "synthesized_key": result["synthesized_key"],
                    "raw_obs_shape": result["raw_obs_shape"],
                    "blake2b_anchor": result["blake2b_anchor"],
                }
            )
        return result

    def choose_action(
        self,
        frames: list[FrameData],
        latest_frame: FrameData,
        causal_transition_score: Optional[float] = None,
        specialist_logits: Optional[Dict[str, Any]] = None,
        feature_vector: Optional[Any] = None,
    ) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.reset()
            return GameAction.RESET
        if getattr(latest_frame, "full_reset", False) and self.step_index == 0:
            self.reset()

        gate_res = self.step(
            latest_frame,
            causal_transition_score=causal_transition_score,
            specialist_logits=specialist_logits,
            feature_vector=feature_vector,
        )
        encoded = gate_res["encoded"]

        legal_actions: List[GameAction] = []
        for a in (latest_frame.available_actions or list(GameAction)):
            if isinstance(a, GameAction) and a is not GameAction.RESET:
                legal_actions.append(a)
            else:
                try:
                    wanted = int(a)
                    matched = next(
                        (m for m in GameAction if m.value == wanted and m is not GameAction.RESET),
                        None,
                    )
                    if matched:
                        legal_actions.append(matched)
                except (TypeError, ValueError):
                    pass
        if not legal_actions:
            legal_actions = [a for a in GameAction if a is not GameAction.RESET]

        if gate_res["ternary_state"] == 0:
            self.quarantine_pivot_offset += 1

        octant = gate_res["chair_octant"]
        chosen_idx = (
            octant + self.step_index + gate_res["key_729"] + self.quarantine_pivot_offset
        ) % len(legal_actions)
        action = legal_actions[chosen_idx]

        if action.is_complex():
            action.set_data({"x": int(encoded["target_x"]), "y": int(encoded["target_y"])})

        action.reasoning = {
            "thought": (
                f"[BAIopenclaw VLIW Gate {gate_res['gate_status']} F={gate_res['f_value']} "
                f"Ternary={gate_res['ternary_state']:+d} Causal={gate_res['causal_transition_score']:.2f} "
                f"Obs={gate_res['raw_obs_shape']}] Key={gate_res['synthesized_key']} "
                f"ChairOctant={octant} (Corner3={gate_res['corner3_hits']}) 729Key={gate_res['key_729']} "
                f"BLAKE2b={gate_res['blake2b_anchor']}"
            )[:200],
            "confidence": float(gate_res["causal_transition_score"]),
            "alternatives_considered": [
                f"PT_Outlier_Mode={gate_res['pt_outlier_mode']}",
                f"AAB_Cadence={gate_res['aab_tokens']}",
                f"Strip7Col={''.join(str(b) for b in gate_res['strip_7col'])}",
            ],
            "reasoning_tokens": 0,
        }
        return action
