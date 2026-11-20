"""E7 sensitivity sweep builder (run_pilot.build_sweep_commands).

Contract: OFAT arms by default (one factor off-base at a time + one interaction
check), full cross-product with full_grid=True, base-config arm skipped,
--set overrides present in every argv, unique experiment ids.
"""

from __future__ import annotations

from pathlib import Path

from nova_mem.config import load_yaml
from run_pilot import build_sweep_commands

REPO = Path(__file__).resolve().parents[1]


def base_cfg():
    return load_yaml(REPO / "configs" / "local.yaml")  # bm25 / session / k=5


class TestOFATDefault:
    def test_arm_set(self):
        arms = build_sweep_commands(base_cfg(), reader="primary", judge="local")
        ids = [eid for eid, _ in arms]
        # k off-base (bm25 session): k=3, k=10
        assert "pilot_sens_C_bm25_k3_session" in ids
        assert "pilot_sens_C_bm25_k10_session" in ids
        # retriever off-base (k=5 session)
        assert "pilot_sens_C_dense_k5_session" in ids
        # granularity off-base (bm25 k=5)
        assert "pilot_sens_C_bm25_k5_turn" in ids
        # one interaction check: dense x turn
        assert "pilot_sens_C_dense_k5_turn" in ids
        # base arm (bm25 k5 session) is skipped — main pilot covers it
        assert "pilot_sens_C_bm25_k5_session" not in ids
        assert len(ids) == len(set(ids)), "duplicate experiment ids"
        assert len(arms) == 5

    def test_set_flags_present(self):
        arms = dict(build_sweep_commands(base_cfg(), reader="primary", judge="local"))
        argv = arms["pilot_sens_C_bm25_k3_session"]
        assert "--set" in argv
        sets = {argv[i + 1] for i, a in enumerate(argv) if a == "--set"}
        assert sets == {
            "retrieval.system_C.retriever=bm25",
            "retrieval.system_C.top_k=3",
            "retrieval.system_C.granularity=session",
        }
        assert "--experiment-id" in argv and "--system" in argv

    def test_turn_arm_overrides_granularity(self):
        arms = dict(build_sweep_commands(base_cfg(), reader="primary", judge="local"))
        argv = arms["pilot_sens_C_bm25_k5_turn"]
        assert "retrieval.system_C.granularity=turn" in argv
        assert "retrieval.system_C.top_k=5" in argv


class TestFullGrid:
    def test_cross_product_minus_base(self):
        arms = build_sweep_commands(base_cfg(), reader="primary", judge="local", full_grid=True)
        ids = [eid for eid, _ in arms]
        # 3 k's x 2 retrievers x 2 granularities = 12 combos, minus base arm
        assert len(ids) == 11
        assert "pilot_sens_C_bm25_k5_session" not in ids
        assert "pilot_sens_C_dense_k10_turn" in ids
        assert "pilot_sens_C_bm25_k3_turn" in ids


class TestSystemD:
    def test_d_sweeps_top_k_final(self):
        arms = build_sweep_commands(base_cfg(), reader="primary", judge="local", systems=("D",))
        ids = [eid for eid, _ in arms]
        # local.yaml system_D: bm25, top_k_final=5, session -> OFAT over ks
        assert "pilot_sens_D_bm25_k3_session" in ids
        assert "pilot_sens_D_bm25_k10_session" in ids
        assert "pilot_sens_D_bm25_k5_session" not in ids  # base
        argv = dict(arms)["pilot_sens_D_bm25_k3_session"]
        assert "retrieval.system_D.top_k_final=3" in argv
        assert "retrieval.system_D.top_k=3" not in argv  # D key must not be C's

    def test_c_and_d_combined(self):
        arms = build_sweep_commands(base_cfg(), reader="primary", judge="local", systems=("C", "D"))
        ids = [eid for eid, _ in arms]
        assert len(ids) == 5 + 2  # 5 C OFAT arms + 2 D arms


class TestCustomValues:
    def test_trimmed_k_and_single_retriever(self):
        arms = build_sweep_commands(
            base_cfg(), reader="primary", judge="local", ks=(3, 10), retrievers=("bm25",)
        )
        ids = [eid for eid, _ in arms]
        # only k varies now (bm25 only, session base): k=3, k=10 + turn arm
        assert "pilot_sens_C_bm25_k3_session" in ids
        assert "pilot_sens_C_bm25_k10_session" in ids
        assert "pilot_sens_C_bm25_k5_turn" in ids
        assert not any("dense" in i for i in ids)

    def test_unknown_granularity_value_flows_through(self):
        """The builder passes values verbatim; run_experiment's apply_overrides
        + system construction validate them (fail loudly downstream)."""
        arms = build_sweep_commands(
            base_cfg(), reader="primary", judge="local", granularities=("session",)
        )
        assert all("granularity=turn" not in " ".join(a) for _, a in arms)
