"""P16 · Stage start-up planning: device groups, parallel init, rollback, admission.

init_groups(stage_devices): stages that share a device -- directly or transitively --
  form one group. Return groups as sorted stage lists, ordered by their first stage.
run_init(stage_devices, init_fn, cleanup_fn, max_workers):
  groups run in parallel (thread pool); stages inside a group run one after another in
  stage-id order. On the first failure: no NEW init_fn calls start, running ones finish,
  then cleanup_fn(stage, result) for every stage that succeeded, and raise
  InitError(failed_stage, cause). Returns {stage: init_fn(stage)} on success.
admission_check(device_total_bytes, stages, reserve_bytes_per_stage):
  for every device, sum(utilization * total + reserve) over stages placed on it must not
  exceed total; return one error string per offending device (mentioning "device <id>").
"""
from collections.abc import Callable
from typing import Any


class InitError(Exception):
    def __init__(self, stage_id: int, cause: BaseException) -> None:
        super().__init__(f"stage {stage_id} failed to initialize: {cause!r}")
        self.stage_id = stage_id


def init_groups(stage_devices: dict[int, set[int]]) -> list[list[int]]:
    raise NotImplementedError


def run_init(
    stage_devices: dict[int, set[int]],
    init_fn: Callable[[int], Any],
    cleanup_fn: Callable[[int, Any], None],
    max_workers: int = 4,
) -> dict[int, Any]:
    raise NotImplementedError


def admission_check(
    device_total_bytes: dict[int, int],
    stages: dict[int, tuple[set[int], float]],
    reserve_bytes_per_stage: int,
) -> list[str]:
    raise NotImplementedError
