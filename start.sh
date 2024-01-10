#!/bin/bash
# Start TensorBoard in the background
tensorboard --logdir=/path/to/logs --bind_all &

# Run your main application
poetry run python scripts/run-experiment.py --config experiments/mnist/mnist.yaml
