#!/usr/bin/env python3
"""
Online Cycle Time Estimation Frontend

Stream records and update the online linear estimator while predicting.
Reuses the log parser from cycle_time_est.
"""

import argparse
import json
import sys
from typing import Optional

from log_parser import LogParser
from online_predictor import OnlineLinearCycleTime, PredictionInput
import numpy as np
import time
import statistics


def _parse_feature_indices(arg: Optional[str]) -> Optional[list]:
    if not arg:
        return None
    parts = [p.strip() for p in arg.split(',') if p.strip()]
    try:
        idx = [int(p) for p in parts]
        return idx if idx else None
    except ValueError:
        raise ValueError("--feature-indices must be a comma-separated list of integers, e.g. 0,1,8,16,17")


def replay_log(
    log_file: str,
    *,
    log_every: int = 1000,
    forgetting: float = 1.0,
    init_cov: float = 1e3,
    store_history: bool = False,
    max_history: int = 100000,
    max_records: Optional[int] = None,
    predictions_file: Optional[str] = None,
    epochs: int = 1,
    feature_preset: Optional[str] = None,
    feature_indices: Optional[list] = None,
):
    """
    Parse the log and stream it through the online estimator.
    Predict-first, then update, and collect metrics.
    """
    print("=" * 80)
    print("Online Cycle Time Prediction - Replay")
    print("=" * 80)
    print()

    # Parse
    print(f"Parsing log file: {log_file}")
    parser = LogParser()
    records = parser.parse_file(log_file)
    if not records:
        print("Error: No records found in log file")
        return None
    print(f"✓ Parsed {len(records)} records")
    print()

    # Estimator
    est = OnlineLinearCycleTime(
        log_every=log_every,
        forgetting=forgetting,
        init_cov=init_cov,
        store_history=store_history,
        max_history=max_history,
        feature_preset=feature_preset,
        feature_indices=feature_indices,
    )
    if feature_preset:
        print(f"Feature preset: {feature_preset}")
    if feature_indices:
        print(f"Feature indices: {feature_indices}")

    # Metrics accumulators
    n = 0
    sum_abs = 0.0
    sum_sq = 0.0
    sum_pct_abs = 0.0
    preds_dump = [] if predictions_file else None

    # Stream possibly multiple epochs over the same trace
    limit = max_records if (max_records is not None and max_records > 0) else len(records)
    limit = min(limit, len(records))
    epochs = max(1, int(epochs))

    for epoch in range(epochs):
        if epochs > 1:
            print()
            print(f"--- Epoch {epoch+1}/{epochs} ---")
        # Per-epoch metrics
        e_n = 0
        e_sum_abs = 0.0
        e_sum_sq = 0.0
        e_sum_pct_abs = 0.0
        e_abs_errs = []

        for r in records[:limit]:
            # predict-first update
            pred, abs_err = est.submit(
                batch_size_tokens=r.batch_size_tokens,
                prefill_chunk_pairs=r.prefill_chunk_pairs,
                kv_tokens_used=r.kv_tokens_used,
                iteration_time_ms=r.iteration_time_ms,
            )

            # Global accumulators
            n += 1
            sum_abs += abs_err
            diff = pred - r.iteration_time_ms
            sum_sq += diff * diff
            if r.iteration_time_ms > 0:
                sum_pct_abs += abs(diff) / r.iteration_time_ms

            # Epoch accumulators
            e_n += 1
            e_sum_abs += abs_err
            e_sum_sq += diff * diff
            if r.iteration_time_ms > 0:
                e_sum_pct_abs += abs(diff) / r.iteration_time_ms
            e_abs_errs.append(abs_err)

            if preds_dump is not None:
                preds_dump.append({
                    "epoch": epoch + 1,
                    "batch_size_tokens": r.batch_size_tokens,
                    "kv_tokens_used": r.kv_tokens_used,
                    "prefill_chunk_pairs": r.prefill_chunk_pairs,
                    "actual_time_ms": r.iteration_time_ms,
                    "predicted_time_ms": pred,
                    "error_ms": diff,
                    "abs_error_ms": abs_err,
                    "pct_error": (abs(diff) / r.iteration_time_ms * 100.0) if r.iteration_time_ms > 0 else 0.0,
                })

        # Print per-epoch summary
        if epochs > 1:
            e_mae = e_sum_abs / e_n if e_n > 0 else 0.0
            e_rmse = (e_sum_sq / e_n) ** 0.5 if e_n > 0 else 0.0
            e_mape = (e_sum_pct_abs / e_n) * 100.0 if e_n > 0 else 0.0
            p90 = float(np.percentile(e_abs_errs, 90)) if e_n > 0 else 0.0
            p99 = float(np.percentile(e_abs_errs, 99)) if e_n > 0 else 0.0
            print(f"Epoch {epoch+1}: MAE={e_mae:.2f} ms | RMSE={e_rmse:.2f} ms | MAPE={e_mape:.2f} % | P90={p90:.2f} ms | P99={p99:.2f} ms")

    # Final metrics
    if n == 0:
        print("No samples processed.")
        return None

    mae = sum_abs / n
    mse = sum_sq / n
    rmse = mse ** 0.5
    mape = (sum_pct_abs / n) * 100.0

    print()
    print("Results:")
    print(f"  Samples: {n}")
    print(f"  MAE:  {mae:.2f} ms")
    print(f"  RMSE: {rmse:.2f} ms")
    print(f"  MAPE: {mape:.2f} %")
    # Global percentiles (best effort):
    try:
        if epochs == 1:
            all_abs = np.array(e_abs_errs, dtype=np.float64)
        else:
            all_abs = np.array([item["abs_error_ms"] for item in preds_dump], dtype=np.float64) if preds_dump is not None else None
        if all_abs is not None and all_abs.size > 0:
            g_p90 = float(np.percentile(all_abs, 90))
            g_p99 = float(np.percentile(all_abs, 99))
            print(f"  P90 AbsErr: {g_p90:.2f} ms")
            print(f"  P99 AbsErr: {g_p99:.2f} ms")
    except Exception:
        pass
    print()

    if predictions_file:
        with open(predictions_file, 'w') as f:
            json.dump({
                "samples": n,
                "epochs": epochs,
                "samples_per_epoch": limit,
                "mae_ms": mae,
                "rmse_ms": rmse,
                "mape_pct": mape,
                "predictions": preds_dump,
            }, f, indent=2)
        print(f"✓ Predictions saved to {predictions_file}")

    print("=" * 80)
    print("Replay Complete!")
    print("=" * 80)
    return {
        "samples": n,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }


def interactive(
    *,
    log_every: int = 1000,
    forgetting: float = 1.0,
    init_cov: float = 1e3,
    feature_preset: Optional[str] = None,
    feature_indices: Optional[list] = None,
):
    """
    Interactive loop to manually enter samples and measured times.
    """
    est = OnlineLinearCycleTime(
        log_every=log_every,
        forgetting=forgetting,
        init_cov=init_cov,
        feature_preset=feature_preset,
        feature_indices=feature_indices,
    )
    if feature_preset:
        print(f"Feature preset: {feature_preset}")
    if feature_indices:
        print(f"Feature indices: {feature_indices}")

    print("=" * 80)
    print("Online Cycle Time - Interactive Mode")
    print("Predict first, then submit measurement; shows abs error.")
    print("Press Ctrl+C or type 'quit' to exit.")
    print("=" * 80)
    print()

    while True:
        try:
            inp = input("batch_size_tokens (or 'quit'): ").strip()
            if not inp:
                continue
            if inp.lower() == 'quit':
                break
            batch = int(inp)

            kv = int(input("kv_tokens_used: ").strip())
            pairs_str = input("prefill_chunk_pairs (JSON, e.g. [[256,256]] or blank): ").strip()
            if pairs_str:
                prefill_pairs = json.loads(pairs_str)
            else:
                prefill_pairs = []

            y = float(input("measured_iteration_time_ms: ").strip())

            pred, abs_err = est.submit(
                batch_size_tokens=batch,
                prefill_chunk_pairs=prefill_pairs,
                kv_tokens_used=kv,
                iteration_time_ms=y,
            )
            print(f"→ Predicted: {pred:.2f} ms | AbsErr: {abs_err:.2f} ms | CumMAE: {est.mean_abs_err_cum:.2f} ms (N={est.n_seen})")
            print()

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Online Cycle Time Estimation Frontend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Replay a log file with default settings
  python cycle_time_est_online/frontend.py replay worker.log

  # Replay with forgetting and dump predictions
  python cycle_time_est_online/frontend.py replay worker.log --forgetting 0.99 --predictions preds.json

  # Interactive mode
  python cycle_time_est_online/frontend.py interactive --forgetting 0.995
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Replay command
    replay_parser = subparsers.add_parser('replay', help='Stream a log through the online estimator')
    replay_parser.add_argument('log_file', type=str, help='Path to the log file')
    replay_parser.add_argument('--log-every', type=int, default=1000, help='Logging interval for avg abs error (default: 1000)')
    replay_parser.add_argument('--forgetting', type=float, default=1.0, help='RLS forgetting factor in (0,1]; 1.0 = no forgetting')
    replay_parser.add_argument('--init-cov', type=float, default=1e3, help='Initial covariance scaling for RLS (larger = faster initial learning)')
    replay_parser.add_argument('--store-history', action='store_true', help='Store bounded sample history (max 100000 by default)')
    replay_parser.add_argument('--max-history', type=int, default=100000, help='Max samples to keep if history is enabled')
    replay_parser.add_argument('--max-records', type=int, help='Limit number of records to process')
    replay_parser.add_argument('--predictions', '-p', type=str, help='Path to save predictions JSON')
    replay_parser.add_argument('--epochs', type=int, default=1, help='Number of times to replay the trace (default: 1)')
    replay_parser.add_argument('--feature-preset', type=str, choices=['all','basic','hybrid5','prod_only','key'], help='Choose a predefined feature subset')
    replay_parser.add_argument('--feature-indices', type=str, help='Comma-separated list of feature indices, e.g. 0,1,8,16,17')

    # Perf command
    perf_parser = subparsers.add_parser('perf', help='Benchmark predict and submit throughput/latency')
    perf_parser.add_argument('log_file', type=str, help='Path to the log file')
    perf_parser.add_argument('--warmup', type=int, default=1000, help='Warmup samples to submit before timing (default: 1000)')
    perf_parser.add_argument('--max-records', type=int, help='Limit number of records to time (post-warmup)')
    perf_parser.add_argument('--forgetting', type=float, default=1.0)
    perf_parser.add_argument('--init-cov', type=float, default=1e3)
    perf_parser.add_argument('--feature-preset', type=str, choices=['all','basic','hybrid5','prod_only','key'])
    perf_parser.add_argument('--feature-indices', type=str)

    # Interactive command
    inter_parser = subparsers.add_parser('interactive', help='Interactive prediction + online update')
    inter_parser.add_argument('--log-every', type=int, default=1000)
    inter_parser.add_argument('--forgetting', type=float, default=1.0)
    inter_parser.add_argument('--init-cov', type=float, default=1e3)
    inter_parser.add_argument('--feature-preset', type=str, choices=['all','basic','hybrid5','prod_only','key'])
    inter_parser.add_argument('--feature-indices', type=str)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == 'replay':
            feat_idx = _parse_feature_indices(getattr(args, 'feature_indices', None))
            replay_log(
                log_file=args.log_file,
                log_every=args.log_every,
                forgetting=args.forgetting,
                init_cov=args.init_cov,
                store_history=args.store_history,
                max_history=args.max_history,
                max_records=getattr(args, 'max_records', None),
                predictions_file=getattr(args, 'predictions', None),
                epochs=getattr(args, 'epochs', 1),
                feature_preset=getattr(args, 'feature_preset', None),
                feature_indices=feat_idx,
            )
        elif args.command == 'perf':
            # Parse
            parser_obj = LogParser()
            records = parser_obj.parse_file(args.log_file)
            if not records:
                print("No records found.")
                return 1
            # Limits
            warmup = max(0, int(getattr(args, 'warmup', 0)))
            limit = int(getattr(args, 'max_records', 0) or (len(records) - warmup))
            limit = max(0, min(limit, max(0, len(records) - warmup)))
            if warmup > len(records):
                warmup = len(records)
            # Estimator
            feat_idx = _parse_feature_indices(getattr(args, 'feature_indices', None))
            est = OnlineLinearCycleTime(
                forgetting=args.forgetting,
                init_cov=args.init_cov,
                feature_preset=getattr(args, 'feature_preset', None),
                feature_indices=feat_idx,
            )
            # Warmup submit
            for r in records[:warmup]:
                est.submit(r.batch_size_tokens, r.prefill_chunk_pairs, r.kv_tokens_used, r.iteration_time_ms)
            print(f"Warmup: {warmup} samples")
            # Measure predict-only
            t_pred = []
            for r in records[warmup:warmup+limit]:
                t0 = time.perf_counter()
                _ = est.predict(r.batch_size_tokens, r.prefill_chunk_pairs, r.kv_tokens_used)
                t1 = time.perf_counter()
                t_pred.append((t1 - t0) * 1e3)
            # Measure submit (includes internal predict)
            t_submit = []
            for r in records[warmup:warmup+limit]:
                t0 = time.perf_counter()
                _ = est.submit(r.batch_size_tokens, r.prefill_chunk_pairs, r.kv_tokens_used, r.iteration_time_ms)
                t1 = time.perf_counter()
                t_submit.append((t1 - t0) * 1e3)

            def _stats(vec):
                if not vec:
                    return {}
                arr = np.array(vec, dtype=np.float64)
                return {
                    'count': arr.size,
                    'mean_ms': float(arr.mean()),
                    'median_ms': float(np.median(arr)),
                    'p90_ms': float(np.percentile(arr, 90)),
                    'p99_ms': float(np.percentile(arr, 99)),
                    'min_ms': float(arr.min()),
                    'max_ms': float(arr.max()),
                    'ops_per_sec': float(arr.size / (arr.sum() / 1e3)) if arr.sum() > 0 else float('inf'),
                }

            sp = _stats(t_pred)
            ss = _stats(t_submit)

            print("\nPerf Results (ms):")
            if sp:
                print(f"  predict: mean={sp['mean_ms']:.3f} | p90={sp['p90_ms']:.3f} | p99={sp['p99_ms']:.3f} | median={sp['median_ms']:.3f} | min={sp['min_ms']:.3f} | max={sp['max_ms']:.3f} | ops/s={sp['ops_per_sec']:.1f}")
            if ss:
                print(f"  submit:  mean={ss['mean_ms']:.3f} | p90={ss['p90_ms']:.3f} | p99={ss['p99_ms']:.3f} | median={ss['median_ms']:.3f} | min={ss['min_ms']:.3f} | max={ss['max_ms']:.3f} | ops/s={ss['ops_per_sec']:.1f}")

            return 0
        elif args.command == 'interactive':
            feat_idx = _parse_feature_indices(getattr(args, 'feature_indices', None))
            interactive(
                log_every=args.log_every,
                forgetting=args.forgetting,
                init_cov=args.init_cov,
                feature_preset=getattr(args, 'feature_preset', None),
                feature_indices=feat_idx,
            )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
