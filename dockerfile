FROM ubuntu:latest

# Update aptitude with new repo
RUN apt-get update

# Install software 
RUN apt-get install -y git python3 python3-pip

RUN pip install --upgrade pip && pip install poetry

RUN mkdir /root/.ssh/
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

RUN git clone https://github.com/aai-institute/VeriFlow.git

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
