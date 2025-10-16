#!/usr/bin/env python3
"""
Cycle Time Estimation Frontend

Main entry point for parsing logs, training models, and making predictions.
"""

import argparse
import json
import sys
from typing import Optional

from log_parser import LogParser
from predictor import CycleTimePredictor, PredictionInput


def parse_and_train(log_file: str, output_model: str = None, train_ratio: float = 0.9, predictions_file: str = None, model_type: str = 'ensemble'):
    """
    Parse log file, split into train/test, and train a predictor.

    Args:
        log_file: Path to the log file
        output_model: Path to save the trained model (optional)
        train_ratio: Ratio for train/test split (default 0.9)
        predictions_file: Path to save test set predictions (optional, defaults to model_name_predictions.json)
    """
    print(f"=" * 80)
    print(f"Cycle Time Prediction - Training Pipeline")
    print(f"=" * 80)
    print()

    # Step 1: Parse the log file
    print(f"Step 1: Parsing log file: {log_file}")
    parser = LogParser()
    records = parser.parse_file(log_file)

    if not records:
        print("Error: No records found in log file")
        return None

    print(f"✓ Parsed {len(records)} records")
    print()

    # Step 2: Show statistics
    print("Step 2: Dataset Statistics")
    stats = parser.get_statistics()
    print(json.dumps(stats, indent=2))
    print()

    # Step 3: Split train/test
    print(f"Step 3: Splitting data (train={train_ratio*100:.0f}%, test={100-train_ratio*100:.0f}%)")
    train_records, test_records = parser.split_train_test(train_ratio=train_ratio)

    if not train_records:
        print("Error: No training data available")
        return None

    print(f"✓ Train: {len(train_records)} records")
    print(f"✓ Test: {len(test_records)} records")
    print()

    # Step 4: Create and train predictor
    print(f"Step 4: Training Cycle Time Predictor (model_type={model_type})")
    predictor = CycleTimePredictor(model_type=model_type)

    # Submit training data
    for record in train_records:
        predictor.submit(
            batch_size_tokens=record.batch_size_tokens,
            prefill_chunk_pairs=record.prefill_chunk_pairs,
            kv_tokens_used=record.kv_tokens_used,
            iteration_time_ms=record.iteration_time_ms
        )

    # Train the model
    train_stats = predictor.train()
    print(f"✓ Training completed")
    print(f"  - MAE: {train_stats['mae']:.2f} ms")
    print(f"  - RMSE: {train_stats['rmse']:.2f} ms")
    print(f"  - R²: {train_stats['r2_score']:.4f}")
    print()

    # Step 5: Evaluate on test set
    if test_records:
        print("Step 5: Evaluating on Test Set")
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

        test_stats = predictor.evaluate(test_data)
        print(f"✓ Test evaluation completed")
        print(f"  - MAE: {test_stats['mae']:.2f} ms")
        print(f"  - RMSE: {test_stats['rmse']:.2f} ms")
        print(f"  - R²: {test_stats['r2_score']:.4f}")
        print(f"  - MAPE: {test_stats['mape']:.2f}%")
        print()

        # Generate predictions for test set and dump to file
        if output_model or predictions_file:
            # Determine the predictions file path
            if predictions_file:
                preds_path = predictions_file
            elif output_model:
                preds_path = output_model.replace('.pkl', '_predictions.json')
            else:
                preds_path = 'predictions.json'

            print(f"Step 5b: Dumping test set predictions to {preds_path}")

            predictions_data = []
            for inp, actual in test_data:
                predicted = predictor.predict(
                    batch_size_tokens=inp.batch_size_tokens,
                    prefill_chunk_pairs=inp.prefill_chunk_pairs,
                    kv_tokens_used=inp.kv_tokens_used,
                    use_model_type=model_type
                )

                predictions_data.append({
                    "batch_size_tokens": inp.batch_size_tokens,
                    "kv_tokens_used": inp.kv_tokens_used,
                    "prefill_chunk_pairs": inp.prefill_chunk_pairs,
                    "actual_time_ms": actual,
                    "predicted_time_ms": predicted,
                    "error_ms": predicted - actual,
                    "abs_error_ms": abs(predicted - actual),
                    "pct_error": abs(predicted - actual) / actual * 100 if actual > 0 else 0.0
                })

            with open(preds_path, 'w') as f:
                json.dump({
                    "test_set_size": len(predictions_data),
                    "test_stats": test_stats,
                    "predictions": predictions_data
                }, f, indent=2)

            print(f"✓ Predictions saved to {preds_path}")
            print()

    # Step 6: Save model
    if output_model:
        print(f"Step 6: Saving model to {output_model}")
        predictor.save(output_model)
        print(f"✓ Model saved")
        print()

    print("=" * 80)
    print("Training Complete!")
    print("=" * 80)

    return predictor


def predict_interactive(model_file: str, model_type: Optional[str] = None):
    """
    Load a trained model and make interactive predictions.

    Args:
        model_file: Path to the saved model file
    """
    print(f"Loading model from {model_file}...")
    predictor = CycleTimePredictor()
    predictor.load(model_file)
    print("✓ Model loaded")
    print()

    print("=" * 80)
    print("Interactive Prediction Mode")
    print("=" * 80)
    print()
    print("Enter prediction inputs (or 'quit' to exit):")
    print()

    while True:
        try:
            # Get batch size
            batch_size = input("batch_size_tokens: ").strip()
            if batch_size.lower() == 'quit':
                break
            batch_size = int(batch_size)

            # Get KV tokens
            kv_tokens = int(input("kv_tokens_used: ").strip())

            # Get prefill chunk pairs
            pairs_str = input("prefill_chunk_pairs (JSON format, e.g., [[256,256]]): ").strip()
            if pairs_str:
                prefill_pairs = json.loads(pairs_str)
            else:
                prefill_pairs = []

            # Make prediction
            pred_time = predictor.predict(
                batch_size_tokens=batch_size,
                prefill_chunk_pairs=prefill_pairs,
                kv_tokens_used=kv_tokens,
                use_model_type=model_type
            )

            print(f"➜ Predicted iteration time: {pred_time:.2f} ms")
            print()

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Cycle Time Estimation Frontend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model from log file
  python frontend.py train worker.log --output model.pkl

  # Train with custom train/test split and save predictions
  python frontend.py train worker.log --output model.pkl --train-ratio 0.8 --predictions preds.json

  # Load model and make predictions interactively
  python frontend.py predict model.pkl

  # Parse log and show statistics only
  python frontend.py parse worker.log
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train a cycle time predictor')
    train_parser.add_argument('log_file', type=str, help='Path to the log file')
    train_parser.add_argument('--output', '-o', type=str,
                             help='Path to save the trained model')
    train_parser.add_argument('--train-ratio', type=float, default=0.9,
                             help='Ratio for train/test split (default: 0.9)')
    train_parser.add_argument('--predictions', '-p', type=str,
                             help='Path to save test set predictions (default: <model>_predictions.json)')
    train_parser.add_argument('--model-type', type=str, default='linear', choices=['linear','polynomial','ensemble','hybrid'],
                             help='Model type to train (default: ensemble)')

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions using trained model')
    predict_parser.add_argument('model_file', type=str, help='Path to the trained model file')
    predict_parser.add_argument('--model-type', type=str, choices=['linear','polynomial','ensemble','hybrid'],
                               help='Override model type used for prediction')

    # Parse command
    parse_parser = subparsers.add_parser('parse', help='Parse log file and show statistics')
    parse_parser.add_argument('log_file', type=str, help='Path to the log file')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == 'train':
            parse_and_train(
                log_file=args.log_file,
                output_model=args.output,
                train_ratio=args.train_ratio,
                predictions_file=args.predictions,
                model_type=args.model_type
            )

        elif args.command == 'predict':
            predict_interactive(model_file=args.model_file, model_type=getattr(args, 'model_type', None))

        elif args.command == 'parse':
            parser_obj = LogParser()
            records = parser_obj.parse_file(args.log_file)

            print("\nStatistics:")
            stats = parser_obj.get_statistics()
            print(json.dumps(stats, indent=2))

            train, test = parser_obj.split_train_test()
            print(f"\nDefault 90/10 split:")
            print(f"  Train: {len(train)} records")
            print(f"  Test: {len(test)} records")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
