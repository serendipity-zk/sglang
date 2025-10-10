#!/usr/bin/env python3
"""
Feature Importance Analysis

Analyzes which features are most important for cycle time prediction
and identifies features that can be removed without losing accuracy.
"""

import argparse
import json
import sys
from typing import List, Tuple, Dict
import numpy as np

from log_parser import LogParser
from predictor import CycleTimePredictor, PredictionInput


def analyze_linear_coefficients(predictor: CycleTimePredictor, feature_names: List[str]) -> Dict:
    """
    Analyze linear model coefficients to determine feature importance.

    For linear models, the absolute value of normalized coefficients
    indicates feature importance.
    """
    if predictor.model_type != 'linear':
        print("Warning: Coefficient analysis only works for linear models")
        return {}

    if not predictor.is_trained:
        raise RuntimeError("Model must be trained first")

    # Get coefficients (excluding bias term which is last)
    coeffs = predictor.model[:-1]

    # Compute importance as absolute value of normalized coefficient
    importance = np.abs(coeffs)

    # Create ranking
    ranking = []
    for i, (name, imp) in enumerate(zip(feature_names, importance)):
        ranking.append({
            'feature': name,
            'importance': float(imp),
            'coefficient': float(coeffs[i]),
            'rank': 0  # Will be filled later
        })

    # Sort by importance
    ranking.sort(key=lambda x: x['importance'], reverse=True)

    # Add ranks
    for i, item in enumerate(ranking):
        item['rank'] = i + 1

    return {
        'method': 'linear_coefficients',
        'features': ranking,
        'top_5': ranking[:5],
        'bottom_5': ranking[-5:]
    }


def analyze_feature_correlation(training_data: List[Tuple[PredictionInput, float]]) -> Dict:
    """
    Compute correlation between each feature and the target variable.
    High correlation = more predictive power.
    """
    # Extract features and targets
    X_list = []
    y_list = []

    for inp, target in training_data:
        X_list.append(inp.to_features())
        y_list.append(target)

    X = np.array(X_list)
    y = np.array(y_list)

    # Compute correlation for each feature
    feature_names = get_feature_names()
    correlations = []

    for i, name in enumerate(feature_names):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        correlations.append({
            'feature': name,
            'correlation': float(corr),
            'abs_correlation': float(abs(corr)),
            'rank': 0
        })

    # Sort by absolute correlation
    correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)

    # Add ranks
    for i, item in enumerate(correlations):
        item['rank'] = i + 1

    return {
        'method': 'correlation',
        'features': correlations,
        'top_5': correlations[:5],
        'bottom_5': correlations[-5:]
    }


def ablation_study(training_data: List[Tuple[PredictionInput, float]],
                   test_data: List[Tuple[PredictionInput, float]]) -> Dict:
    """
    Perform ablation study: remove each feature and measure performance drop.
    Larger drop = more important feature.
    """
    print("\nPerforming ablation study (leave-one-out)...")
    print("=" * 80)

    feature_names = get_feature_names()
    n_features = len(feature_names)

    # Train baseline model with all features
    print("Training baseline model with all features...")
    baseline_predictor = CycleTimePredictor(model_type='linear')
    for inp, target in training_data:
        baseline_predictor.submit(
            batch_size_tokens=inp.batch_size_tokens,
            prefill_chunk_pairs=inp.prefill_chunk_pairs,
            kv_tokens_used=inp.kv_tokens_used,
            iteration_time_ms=target
        )
    baseline_predictor.train()
    baseline_stats = baseline_predictor.evaluate(test_data)
    baseline_mae = baseline_stats['mae']

    print(f"Baseline MAE: {baseline_mae:.2f}ms")
    print()

    # Test removing each feature
    results = []

    for feature_idx in range(n_features):
        feature_name = feature_names[feature_idx]
        print(f"Testing without feature {feature_idx+1}/{n_features}: {feature_name}...", end=' ')

        # Create modified training data with feature removed
        X_train = []
        y_train = []
        for inp, target in training_data:
            features = inp.to_features()
            # Remove the feature by setting it to 0 (or mean)
            features_modified = features.copy()
            features_modified[feature_idx] = 0.0
            X_train.append(features_modified)
            y_train.append(target)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Train model with modified features
        predictor_modified = CycleTimePredictor(model_type='linear')
        # Manually set training data
        predictor_modified.training_data = []
        for features, target in zip(X_train, y_train):
            # Create a dummy input (won't be used for features)
            dummy_inp = PredictionInput(
                batch_size_tokens=0,
                prefill_chunk_pairs=[],
                kv_tokens_used=0
            )
            predictor_modified.training_data.append((dummy_inp, target))

        # Override training to use modified features
        predictor_modified.feature_mean = np.mean(X_train, axis=0)
        predictor_modified.feature_std = np.std(X_train, axis=0)
        predictor_modified.feature_std[predictor_modified.feature_std == 0] = 1.0

        X_norm = (X_train - predictor_modified.feature_mean) / predictor_modified.feature_std
        X_norm = np.hstack([X_norm, np.ones((X_norm.shape[0], 1))])

        try:
            predictor_modified.model = np.linalg.lstsq(X_norm, y_train, rcond=None)[0]
        except:
            predictor_modified.model = np.linalg.pinv(X_norm) @ y_train

        predictor_modified.is_trained = True

        # Evaluate on test set with feature removed
        test_predictions = []
        test_targets = []
        for inp, target in test_data:
            features = inp.to_features()
            features[feature_idx] = 0.0  # Remove feature
            features_norm = (features - predictor_modified.feature_mean) / predictor_modified.feature_std
            features_norm = np.append(features_norm, 1.0)
            pred = float(features_norm @ predictor_modified.model)
            pred = max(0.0, pred)
            test_predictions.append(pred)
            test_targets.append(target)

        mae_without = np.mean(np.abs(np.array(test_targets) - np.array(test_predictions)))
        mae_increase = mae_without - baseline_mae
        pct_increase = (mae_increase / baseline_mae) * 100

        print(f"MAE={mae_without:.2f}ms (Δ={mae_increase:+.2f}ms, {pct_increase:+.1f}%)")

        results.append({
            'feature': feature_name,
            'mae_without': float(mae_without),
            'mae_increase': float(mae_increase),
            'pct_increase': float(pct_increase),
            'importance_score': float(abs(mae_increase)),
            'rank': 0
        })

    # Sort by importance
    results.sort(key=lambda x: x['importance_score'], reverse=True)

    # Add ranks
    for i, item in enumerate(results):
        item['rank'] = i + 1

    return {
        'method': 'ablation',
        'baseline_mae': float(baseline_mae),
        'features': results,
        'top_5': results[:5],
        'bottom_5': results[-5:],
        'removable_features': [f for f in results if f['mae_increase'] < 0.5]  # <0.5ms impact
    }


def get_feature_names() -> List[str]:
    """Get the list of feature names in order."""
    return [
        'batch_size_tokens',
        'kv_tokens_used',
        'num_prefill_requests',
        'total_prefill_chunks',
        'total_cumulative_prefill',
        'avg_chunk_size',
        'max_chunk_size',
        'avg_cumulative_size',
        'sum_chunk_history_product',
        'max_chunk_history_product',
        'avg_chunk_history_product',
        'sum_chunk_squared',
        'sum_cumulative_squared',
        'min_chunk_size',
        'std_chunk_size',
        'kv_to_batch_ratio',
        'batch_squared',
        'kv_squared',
        'batch_times_kv',
    ]


def print_analysis_report(analysis_results: Dict):
    """Print a formatted analysis report."""
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE ANALYSIS REPORT")
    print("=" * 80)

    for method_name, results in analysis_results.items():
        print(f"\n{results['method'].upper()} ANALYSIS")
        print("-" * 80)

        if 'top_5' in results:
            print("\n🏆 TOP 5 MOST IMPORTANT FEATURES:")
            for item in results['top_5']:
                feature = item['feature']
                if 'importance' in item:
                    print(f"  {item['rank']:2d}. {feature:30s} - Importance: {item['importance']:.4f}")
                elif 'abs_correlation' in item:
                    print(f"  {item['rank']:2d}. {feature:30s} - |Correlation|: {item['abs_correlation']:.4f}")
                elif 'mae_increase' in item:
                    print(f"  {item['rank']:2d}. {feature:30s} - MAE increase: {item['mae_increase']:+.2f}ms ({item['pct_increase']:+.1f}%)")

        if 'bottom_5' in results:
            print("\n🗑️  BOTTOM 5 LEAST IMPORTANT FEATURES:")
            for item in results['bottom_5']:
                feature = item['feature']
                if 'importance' in item:
                    print(f"  {item['rank']:2d}. {feature:30s} - Importance: {item['importance']:.4f}")
                elif 'abs_correlation' in item:
                    print(f"  {item['rank']:2d}. {feature:30s} - |Correlation|: {item['abs_correlation']:.4f}")
                elif 'mae_increase' in item:
                    print(f"  {item['rank']:2d}. {feature:30s} - MAE increase: {item['mae_increase']:+.2f}ms ({item['pct_increase']:+.1f}%)")

        if 'removable_features' in results and results['removable_features']:
            print("\n✂️  FEATURES THAT CAN BE REMOVED (impact <0.5ms):")
            for item in results['removable_features']:
                print(f"  - {item['feature']:30s} (impact: {item['mae_increase']:+.2f}ms)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Feature Importance Analysis for Cycle Time Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all analyses on a log file
  python feature_analysis.py /path/to/worker.log

  # Run only correlation analysis
  python feature_analysis.py /path/to/worker.log --method correlation

  # Save results to JSON
  python feature_analysis.py /path/to/worker.log --output analysis.json
        """
    )

    parser.add_argument('log_file', type=str, help='Path to the log file')
    parser.add_argument('--method', type=str,
                       choices=['correlation', 'coefficients', 'ablation', 'all'],
                       default='all',
                       help='Analysis method to use (default: all)')
    parser.add_argument('--train-ratio', type=float, default=0.9,
                       help='Ratio for train/test split (default: 0.9)')
    parser.add_argument('--output', '-o', type=str,
                       help='Save results to JSON file')

    args = parser.parse_args()

    # Parse log file
    print(f"Parsing log file: {args.log_file}")
    log_parser = LogParser()
    records = log_parser.parse_file(args.log_file)

    if not records:
        print("Error: No records found in log file")
        return 1

    print(f"✓ Parsed {len(records)} records")

    # Split train/test
    train_records, test_records = log_parser.split_train_test(train_ratio=args.train_ratio)
    print(f"✓ Train: {len(train_records)} records, Test: {len(test_records)} records\n")

    # Prepare training data
    training_data = [
        (
            PredictionInput(
                batch_size_tokens=r.batch_size_tokens,
                prefill_chunk_pairs=r.prefill_chunk_pairs,
                kv_tokens_used=r.kv_tokens_used
            ),
            r.iteration_time_ms
        )
        for r in train_records
    ]

    test_data = [
        (
            PredictionInput(
                batch_size_tokens=r.batch_size_tokens,
                prefill_chunk_pairs=r.prefill_chunk_pairs,
                kv_tokens_used=r.kv_tokens_used
            ),
            r.iteration_time_ms
        )
        for r in test_records
    ]

    # Run analyses
    analysis_results = {}

    if args.method in ['correlation', 'all']:
        print("Running correlation analysis...")
        analysis_results['correlation'] = analyze_feature_correlation(training_data)
        print("✓ Correlation analysis complete\n")

    if args.method in ['coefficients', 'all']:
        print("Running coefficient analysis...")
        # Train linear model for coefficient analysis
        predictor = CycleTimePredictor(model_type='linear')
        for inp, target in training_data:
            predictor.submit(
                batch_size_tokens=inp.batch_size_tokens,
                prefill_chunk_pairs=inp.prefill_chunk_pairs,
                kv_tokens_used=inp.kv_tokens_used,
                iteration_time_ms=target
            )
        predictor.train()
        analysis_results['coefficients'] = analyze_linear_coefficients(predictor, get_feature_names())
        print("✓ Coefficient analysis complete\n")

    if args.method in ['ablation', 'all']:
        analysis_results['ablation'] = ablation_study(training_data, test_data)
        print("✓ Ablation study complete\n")

    # Print report
    print_analysis_report(analysis_results)

    # Save to JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        print(f"\n✓ Results saved to {args.output}")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    # Provide recommendations based on ablation study
    if 'ablation' in analysis_results:
        removable = analysis_results['ablation']['removable_features']
        if removable:
            print(f"\n✂️  You can safely remove {len(removable)} features with minimal impact:")
            for f in removable:
                print(f"  - {f['feature']}")
            print(f"\nThis would reduce from 19 to {19-len(removable)} features.")
        else:
            print("\n✅ All features contribute meaningfully to prediction accuracy.")
            print("   Consider keeping all 19 features.")

    print("\n" + "=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
