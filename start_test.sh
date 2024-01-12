#!/bin/bash
# Start TensorBoard in the background
poetry run tensorboard --logdir=ray_results --bind_all &
mkdir ray_results
mkdir reports
poetry run python scripts/run-experiment.py --config experiments/mnist/mnist.yaml --report_dir reports --storage_path ./ray_results
