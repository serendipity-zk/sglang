pip install maturin
maturin build --release
python3 -m pip install --user target/wheels/knn_workload_predictor-0.1.0-cp312-cp312-manylinux_2_34_x86_64.whl