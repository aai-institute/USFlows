#!/bin/bash
# Start TensorBoard in the background
tensorboard --logdir=ray_results --bind_all &
mkdir ray_results
mkdir reports
poetry run python scripts/run_experiment.py --config experiments/mnist/mnist.yaml --report_dir reports --storage_path ray_results
