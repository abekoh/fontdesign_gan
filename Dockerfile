FROM tensorflow/tensorflow:1.3.0-gpu-py3
RUN pip install keras tqdm imageio hyperopt
