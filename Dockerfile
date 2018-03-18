FROM nvidia/cuda:8.0-cudnn6-runtime
RUN apt-get update
RUN apt-get -y install python3-dev python3-pip python3-tk curl git cmake build-essential
RUN pip3 install -U pip
RUN pip3 install tensorflow-gpu==1.4.1 tqdm imageio h5py matplotlib scikit-learn scipy opencv-python
RUN git clone https://github.com/DmitryUlyanov/Multicore-TSNE.git
RUN pip3 install Multicore-TSNE/
RUN ln -s /usr/bin/python3 /usr/bin/python
