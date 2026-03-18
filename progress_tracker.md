# Sidecar-Only Progress Tracker

## Goal

Implement the sidecar-only scheduling path in `/sgl-workspace/data/sglang_fresh`, using `/sgl-workspace/sglang/sidecar_only_plan.md` as the reference design, but starting from a newer upstream `sglang` that currently has no SLO-specific engine code.

## Current Baseline

- Reference repo with prior work: `/sgl-workspace/sglang`
- Reference branch with old implementation: `sidecar`
- Reference plan: `/sgl-workspace/sglang/sidecar_only_plan.md`
- Fresh target repo: `/sgl-workspace/data/sglang_fresh`
- Fresh target branch: `sidecar-only`
- Fresh target base: upstream `main` at `21c4fc6334d13cbc075504353b9abc3716cc069e`

## High-Level Conclusion

The fresh upstream tree is the right restart point because it has no legacy in-engine SLO scheduler code to unwind. The old `sidecar` branch should be used as a selective source for request metadata threading, sidecar transport, and scheduler drain logic, but not as a cherry-pick target.

The main adaptation is not conceptual. It is mechanical drift in the upstream request pipeline and scheduler internals since `1c3dbad8fee4bf1631924cc03165d60819a7a473`.

## Execution Rule

Every implementation step must end in a testable state.

- Do not stack multiple large refactors before validating behavior.
- Each phase should land with either automated coverage, a focused smoke test, or a clearly documented manual verification procedure.
- If a step cannot be tested yet, split it into a smaller preparatory change and a follow-up change that becomes testable immediately.
- Prefer preserving compile- and import-safe intermediate states over fast bulk ports.

## What Changed In New Upstream

- The fresh tree no longer contains the older SLO fields or SLO server args.
- `Scheduler`, `Req`, request tokenization, and server args have all been heavily rewritten since `1c3dbad`.
- `Scheduler` still has `get_new_batch_prefill()`, so the old plan to inject a sidecar-provided `max_prefill_tokens` into the existing budget-based path still looks valid.
- Request construction is now centralized more strongly through `tokenizer_manager.py`, so SLO metadata must be threaded through tokenizer flow, not only HTTP or gRPC ingress.
- Fresh upstream does not have `/ui_stats`, `/set_tpot`, `slo_scheduler_mode`, `prefill_schedule_mode`, or predictor config in the engine. This matches the desired sidecar-only architecture and should stay that way.

## Concrete Drift To Account For

- `python/sglang/srt/managers/scheduler.py`
  Fresh upstream has a substantially different scheduler class layout and more mixins. The sidecar hooks must be reinserted into the current init and run-loop structure, not copied at the old line offsets.

- `python/sglang/srt/managers/schedule_batch.py`
  Fresh `Req` no longer carries `target_ttft_ms`, `target_tpot_ms`, `arrival_time_ms`, `start_iteration`, `router_generation`, `router_message_id`, or `slo_violated`. Minimal sidecar metadata must be re-added in a way that fits the current request lifecycle.

- `python/sglang/srt/managers/io_struct.py`
  Fresh `GenerateReqInput` and `TokenizedGenerateReqInput` exist, but the SLO fields are gone. They need to be reintroduced here first.

- `python/sglang/srt/managers/tokenizer_manager.py`
  Fresh request-to-tokenized-request conversion now runs through `_create_tokenized_object()`. This is a required threading point for any new request metadata.

- `python/sglang/srt/entrypoints/http_server.py`
  Fresh upstream no longer applies `--slo-target-margin` or exposes SLO-only endpoints. Only the ingress margin logic should return here.

- `python/sglang/srt/entrypoints/grpc_server.py`
  gRPC request conversion must be checked and updated together with HTTP so both ingress paths set the same SLO metadata.

- `python/sglang/srt/server_args.py`
  Fresh server args are much larger and newer. Only a minimal sidecar-only arg set should be added back:
  `--slo-target-margin`, `--slo-scheduler-addr`, `--slo-scheduler-timeout-ms`.

- `python/sglang/srt/managers/schedule_policy.py`
  Keep as close to fresh upstream as possible. Do not reintroduce internal SLO chunking modes. The intended integration remains an externally supplied prefill budget.

## Scope To Keep

- Minimal request SLO metadata:
  `target_ttft_ms`, `target_tpot_ms`, `arrival_time_ms`
- Minimal request runtime state:
  `slo_violated`
- Minimal sidecar transport:
  DEALER client, timeout handling, stale-decision rejection
- Three temporal drains in the scheduler:
  `finished`, `current`, `scheduling`
- Sidecar decision application in prefill admission

## Scope To Exclude

- Internal SLO scheduling modes
- Shadow or shadow-sidecar support
- Engine-owned predictor config
- Engine-owned TPOT control
- Engine-owned `/ui_stats`
- Engine-owned `/set_tpot`
- Engine-owned iteration metrics or router metrics push
- Internal `PrefillScheduleMode` logic from the old branch

## Planned Implementation Breakdown

### Phase 0: Analysis

- [x] Read `/sgl-workspace/sglang/sidecar_only_plan.md`
- [x] Compare old `sidecar` branch against fresh upstream surfaces
- [x] Confirm fresh upstream starts without any SLO-specific engine path
- [x] Create this tracker

### Phase 1: Request Metadata Plumbing

- [x] Add `target_ttft_ms`, `target_tpot_ms`, `arrival_time_ms` to `GenerateReqInput`
- [x] Add the same fields to `TokenizedGenerateReqInput`
- [x] Thread those fields through `tokenizer_manager.py`
- [x] Re-add ingress margin handling with `--slo-target-margin`
- [ ] Update gRPC request conversion if needed
- Testability target:
  request construction and tokenization paths should preserve the new fields in unit coverage or a focused request-object smoke test before moving on.

Phase 1 status:
- Local HTTP/tokenizer-side plumbing is implemented and covered by focused unit tests.
- The remaining unchecked item is gRPC. In this tree, `python/sglang/srt/entrypoints/grpc_server.py` is only a thin wrapper around the external `smg_grpc_servicer` package.
- The installed `smg_grpc_proto` `sglang_scheduler.proto` does not currently define `target_ttft_ms`, `target_tpot_ms`, or `arrival_time_ms` fields on `GenerateRequest`.
- The installed `smg_grpc_servicer` still constructs `TokenizedGenerateReqInput` without those fields.
- As a result, full Phase 1 completion now depends on an external gRPC proto/servicer update, not additional local plumbing in this repo.

### Phase 2: Request Runtime State

- [x] Add minimal SLO fields to `Req`
- [x] Re-add `start_iteration` only if the sidecar message contract still needs it
- [x] Re-add router metadata only if the actual sidecar still depends on it
- Testability target:
  scheduler-side request objects should expose the new metadata without breaking normal request creation paths.

Phase 2 status:
- Fresh `Req` now carries the minimal runtime/request SLO state needed for the sidecar path:
  `target_ttft_ms`, `target_tpot_ms`, `arrival_time_ms`, and `slo_violated`.
- Fresh scheduler request creation and session-based request creation both preserve those fields.
- Focused unit coverage now exercises direct `Req` construction and `Session.create_req()` propagation.
- Re-checking `/sgl-workspace/data/PolyserveSidecar` showed no current need for `start_iteration` in the live sidecar contract, so it remains excluded from fresh upstream for now.
- Re-checking `/sgl-workspace/data/PolyserveSidecar` also showed router generation/message-id fields exist only as optional observability fields in sidecar messages. They are not needed for minimal Phase 2 request runtime state, so they remain excluded until Phase 4/5 proves they are required.

### Phase 3: Sidecar Transport

- [x] Add a simplified `slo_scheduler_client.py`
- [x] Keep timeout fallback to plain budget scheduling
- [x] Keep stale response rejection via `iteration_count`
- [x] Drop old fallback probe state machine unless current behavior proves it is needed
- Testability target:
  add a transport-level roundtrip or mocked client test before integrating it into the live scheduler.

Phase 3 status:
- Added `python/sglang/srt/managers/slo_scheduler_client.py` as a minimal DEALER transport client that imports sidecar serialization lazily at call time.
- Added `--slo-scheduler-addr` and `--slo-scheduler-timeout-ms` back to `server_args.py` so the transport can be configured before scheduler integration lands.
- Timeout/error behavior currently falls back by returning `None`, which preserves the intended "plain upstream budget scheduling" fallback once the scheduler starts calling the client.
- Stale responses are drained and rejected by comparing `decision.iteration_count` against the expected iteration.
- The old consecutive-failure fallback probe state machine was intentionally not ported.
- Focused unit coverage now exists for exact-match roundtrip, stale-decision rejection, timeout fallback, close behavior, and server-args parsing.

### Phase 4: Scheduler Snapshot Logic

- [ ] Add `scheduler_sidecar_mixin.py`
- [ ] Port request-to-sidecar conversion selectively from old `sidecar`
- [ ] Port the three drain points:
  `finished`, `current`, `scheduling`
- [ ] Verify each drain point against fresh overlap and non-overlap execution paths
- Testability target:
  each drain path should be exercisable independently with focused scheduler or snapshot-construction tests.

### Phase 5: Scheduler Integration

- [ ] Initialize sidecar client in fresh `Scheduler.__init__`
- [ ] Capture pre-run KV and current snapshot after launch
- [ ] Drain finished iteration after result processing
- [ ] Drain scheduling context before prefill admission
- [ ] Apply sidecar `max_prefill_tokens` inside the existing prefill path
- [ ] Keep scheduler fallback behavior as plain upstream budget scheduling
- Testability target:
  integration should land incrementally, with timeout fallback and external-budget application verified before adding more scheduler hooks.

### Phase 6: Validation

- [ ] Add at least one sidecar roundtrip test
- [ ] Add stale-decision rejection coverage
- [ ] Add timeout fallback coverage
- [ ] Validate overlap and non-overlap drain timing assumptions

## Immediate Next Steps

1. Inspect the fresh scheduler event loops around batch launch and result processing so the three drain call sites can be placed correctly.
2. Inspect fresh `schedule_policy.py` and `PrefillAdder` to find the cleanest external-budget injection point.
3. Re-read the actual sidecar message schema in `/sgl-workspace/data/PolyserveSidecar` before adding request and snapshot fields, so only truly required fields are restored.

## Risks

- The fresh scheduler now has more overlap, disaggregation, and mixin-driven code paths than the old branch, so drain timing may differ from the old assumptions.
- The old sidecar mixin assumes certain batch fields and request state are available at specific times; each assumption needs to be revalidated against fresh upstream.
- A naive port of the old branch will likely reintroduce unwanted engine-owned SLO features because the old sidecar work was built on top of a branch that still carried transition scaffolding.

## Working Rule

Prefer small, selective ports from the old `sidecar` branch into fresh upstream. If a change starts pulling internal SLO logic, router metrics, predictor config, or shadow-mode scaffolding back into the engine, stop and simplify it instead.

Also require each step to remain testable before proceeding to the next one.
