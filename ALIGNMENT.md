# ServingStudio instrumentation on upstream SGLang

`vibesim-alignment-20261108` is the maintained ServingStudio line of this fork.
It sits on official SGLang `upstream/main`
`9d547e19ff519f0d1b964b0b1c014d98dc65caf4` (2026-11-08), the commit the official
nightly image `lmsysorg/sglang:nightly-dev-cu13-20261108-9d547e19` is built from.
`ALIGNMENT_BASE` states that base, the image and its digest for the Sim's
profiler-image build; nothing derives the base from commit messages. Both
alignment forks (vLLM and SGLang) are the latest upstream commit plus its
official image plus our commits. At the next upgrade, prefer a stable release
and its image.

ServingStudio Sim pins the line at `alignment/profiler/sglang`.

Earlier lines are not moved, because recorded captures name them; every
capture's `<name>_launch.json` records the fork's `head`, `branch`, and
`status_short` under `fork_git`.

- `vibesim-alignment-20261107` (head `05ddf99b87`) sits on main `ba5906471c`
  (2026-11-07). This line is that one rebased without conflicts: the nine
  upstream commits between the two bases touch none of the files our commits
  change.
- `vibesim-alignment` (head `449cb04a53`, tagged
  `vibesim-alignment-base-449cb04a5`) sits on main `9d4be40124` (2026-08-10).

## Upstream delta since `vibesim-alignment-20261107`

`ba5906471c..9d547e19ff` is nine commits. None touches the scheduler,
`tokenizer_manager.py`, the JIT cache, DSA or MoE kernels, or the NVTX and
profiler paths, so the GLM-5.2 TP4 B200 capture that validated
`vibesim-alignment-20261107` (no iteration over 1.6 s; TTFT p90 1626 ms, TPOT
p90 28.3 ms against 1805 ms and 30.4 ms in 2026-08) was not rerun for this base.

| Commit | Change | Effect on a CUDA capture |
| --- | --- | --- |
| `c3c292b65a` | Rust server: image detail hint | none (Rust server only) |
| `391f4e6f17` | AMD cookbook doc | none |
| `9e10f79ac7` | Rust server request logging | none (Rust server; drops a Rust-only arg check) |
| `7975093bd1` | tests pin the Python frontend | none (tests) |
| `4b4dc7b01b` | Rust server cache report | none (Rust server only) |
| `ba48fc1a9e` | ROCm MI450 custom all-reduce | code under `USE_ROCM` / `torch.version.hip` only; the edited `custom_all_reduce.cuh` changes that JIT module's content hash, so a warm cache builds it once more on first use |
| `0a8ad45c12` | kv-canary ids in the graph-capture dummy forward | `None` unless the kv-canary token oracle is on |
| `6d1ce5b894` | ERNIE 5.5 imports | none (other model) |
| `9d547e19ff` | in-tree aiter tuned tables | `install_aiter_tuned_tables()` returns at once unless ROCm with aiter |

Torch 2.14.1, `sglang-kernel` 0.4.10 and FlashInfer 0.7.3 are unchanged.

## Commits

The 2026-11-07 rebase from `vibesim-alignment` kept all four commits, one for
one, and dropped none (the 2026-11-08 rebase then carried every commit
unchanged):

| `vibesim-alignment` | Here | Content |
| --- | --- | --- |
| `4fc5472fc3` | `feat(alignment): emit VibeSim alignment records` | iteration, request-timing and API-timing records; `sglang_iteration(N): <phase>` NVTX scopes |
| `29c0c7e406` | `feat(alignment): close the last three gaps` | EPLB expert-load records, token dump, NVTX forward range |
| `543f4fb993` | `fix(alignment): scope the forward range to launches` | forward range covers launches only; worker record maps device to rank |
| `449cb04a53` | `fix(serving): stabilize multi-tokenizer streaming` | four tokenizer workers by default, `return_prompt_token_ids`, token-bearing API timing, Uvicorn logger |

`402eaff452` (`report routed expert load without expert parallelism`) was never
on `vibesim-alignment`; it exists only on a local branch of a retired Sim
worktree. It is not carried. The pure-TP `expert_popularity` pass depends on it.

## Upstream fixes carried

- `96519bfe14` (`962c5b3dee` on the 2026-11-07 line) `fix(jit): resolve the
  staging dir before dropping generated units`. Upstream's content-addressed JIT cache never hits when the cache root
  is reached through a symlink (here `~/.cache` -> `/raid/kanzhu/.cache`): the
  generated `cuda.cu` is recorded under its vanishing `.staging-<uuid>` path.
  Each process then rebuilds every `sgl_kernel_jit_*` module on first use,
  serialized by the build lock, about 5 s per rank per module. Modules first
  used while serving (`sampling_argmax`, `clamp_position`) stalled GLM-5.2 TP4
  for 20 s and 18 s. Drop this commit once upstream resolves the staging path.

## Adaptations to upstream changes

- `scheduler.py`: upstream moved the scripted hook and profiler predicate into
  `_on_step_launch`. Geometry capture follows that call in `run_batch`. The
  forward range opens after the prebuilt early return and closes before the
  single `return`. `process_batch_result` keeps upstream's
  `scheduler_stage_method` decorator under the iteration-profile decorator.
  The worker record is emitted after `configure_scheduler_process`.
- `batch_result_processor.py`: request timing is also emitted at upstream's two
  new `set_completion_time` sites (the embedding discard path and the
  sampling-mask abort).
- Server arguments moved to `arg_groups/`. The `tokenizer_worker_num` default
  of 4 is in `fields/serving.py`. `handle_tokenizer_worker_compatibility` in
  `serving_hook.py` requires one worker with `api_key`, `admin_api_key`, or SSL
  refresh. It runs after `handle_return_hidden_states_mode` and is overridable.
  Server arguments resolve lazily (`ServerArgs.resolve_once()`), so the offline
  Engine sets `tokenizer_worker_num=1` before resolution
  (`_resolve_offline_server_args`). Upstream already rejects native gRPC and
  elastic EP with more than one worker, so those checks were not ported.
- `tokenizer_manager.py`: token counts come from upstream's `row_output_ids`.
  A string output batch carries output when it has token ids or text; a
  token-id batch when it has token ids; other outputs always. The first-token
  time is set only from an output that carries output, and is the scheduler's
  first-token time when upstream reports it. The API record's last-token time
  is the last batch with tokens.
- `serving_completions.py`: upstream still couples prompt token ids to
  `return_token_ids`; this line keeps `return_prompt_token_ids` separate.
- DSA indexer kernels moved to `sglang.kernels.ops.attention.dsa.indexer_q`
  and `indexer_k`. Sim runners import them from there.
- Upstream tests that start HTTP serving paths assume one tokenizer worker.
  They now pass `tokenizer_worker_num=1` (`test_crash_dump`, `test_server_info`,
  `test_load_snapshot_backends`, the gRPC and Rust-server cases in
  `test_server_args`).

## Environment

Upstream at this base pins Torch 2.14.1 (cu130), `sglang-kernel` 0.4.10 and
FlashInfer 0.7.3. Since FlashInfer 0.7, `flashinfer-jit-cache` is a
metapackage; the shared objects are in per-architecture providers such as
`flashinfer-jit-cache-sm100a`.

```bash
cd python
uv venv --python 3.12 .venv-sglang
uv pip install --python .venv-sglang/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --index-strategy unsafe-best-match -e .
uv pip install --python .venv-sglang/bin/python pytest nvtx==0.2.16
```

## Validation

Focused tests, from the repository root with `.venv-sglang`. They need no GPU,
but unset an empty `CUDA_VISIBLE_DEVICES`:

```bash
python/.venv-sglang/bin/python -m pytest -q \
  test/registered/unit/observability/test_vibesim_alignment.py \
  test/registered/unit/server_args \
  test/registered/unit/entrypoints/openai \
  test/registered/unit/managers/test_tokenizer_manager_rid_cleanup.py \
  test/registered/unit/observability/test_crash_dump.py \
  test/registered/unit/entrypoints/test_server_info.py \
  test/registered/unit/managers/test_load_snapshot_backends.py
```

Unit tests do not qualify a capture. Run a GPU capture through
`alignment-campaign` before bumping the Sim gitlink.
