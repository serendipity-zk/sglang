import knn_workload_predictor
import time
import random

def test_performance():
    # 1. Initialize
    print("Initializing predictor...")
    # Matches your Rust config: K=10, History=1000
    predictor = knn_workload_predictor.KNNPredictor(k_neighbors=10, max_history=1000)

    # 2. Load Grid (Optional)
    # predictor.load_grid("grid3d.json")

    # 3. Fill History (Simulate Training)
    print("Filling history with 1000 items...")
    batches = [random.randint(1, 128) for _ in range(1000)]
    prefills = ["[]"] * 1000
    kvs = [random.randint(0, 8000) for _ in range(1000)]
    times = [random.uniform(5.0, 100.0) for _ in range(1000)]
    
    # Use the fast batch update
    predictor.update_batch(batches, prefills, kvs, times)

    # 4. Benchmarking Prediction
    num_queries = 100_000
    print(f"Generating {num_queries} queries...")
    q_batches = [random.randint(1, 128) for _ in range(num_queries)]
    q_prefills = ["[]"] * num_queries
    q_kvs = [random.randint(0, 8000) for _ in range(num_queries)]

    print("Running predict_batch (Rust + Rayon)...")
    start = time.time()
    
    # This calls the Rust parallel implementation
    results = predictor.predict_batch(q_batches, q_prefills, q_kvs)
    
    end = time.time()
    duration = end - start
    rps = num_queries / duration

    print(f"Processed {num_queries} queries in {duration:.4f}s")
    print(f"Throughput: {rps:.2f} req/s")
    
    # Print a few results
    print("Sample results:", results[:5])

if __name__ == "__main__":
    test_performance()