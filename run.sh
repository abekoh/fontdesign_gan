nvidia-docker run -it --rm -p 6006:6006 --volume `pwd`:/workdir -w /workdir/ --name fdgan abekoh/fdgan $*
