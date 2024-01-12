FROM ubuntu:latest

RUN apt clean

# Update aptitude with new repo
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git python3 python3-pip nvidia-cuda-toolkit \
    nvidia-cuda-toolkit-gcc \
    gnupg2 \
    curl \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Add the NVIDIA package repositories
# Add the GPG key for the NVIDIA repository
RUN curl -fsSL https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub | apt-key add - && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/ /" > /etc/apt/sources.list.d/nvidia-ml.list

# Install libcudnn8
RUN apt-get update && apt-get install -y --no-install-recommends \
        libcudnn8 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && pip install poetry

RUN mkdir /root/.ssh/ && \
    ssh-keyscan github.com >> /root/.ssh/known_hosts && \
    git clone https://github.com/aai-institute/VeriFlow.git

WORKDIR /VeriFlow
# TODO delete before merging!
RUN git checkout dockerize-experiments
RUN poetry install

# Expose port for TensorBoard
EXPOSE 6006

# TODO: change to start.sh
RUN chmod a+rwx start_test.sh
CMD ["./start_test.sh"]
#CMD ["poetry", "run", "python", "scripts/run-eperiment.py", "--config", "experiments/mnist/mnist.yaml"]
