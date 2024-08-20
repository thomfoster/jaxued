FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

# random useful stuff
ENV SHELL=/bin/bash
ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# install python3.11 and pip
RUN chmod 1777 /tmp
RUN apt-get update --fix-missing
RUN apt upgrade -y
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update
RUN apt install python3.11 python3.11-distutils -y
RUN apt-get install curl -y
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 
# can also install python-dev for C++ headers etc if you want
# RUN apt install python3.11-dev -y --fix-missing

RUN apt-get install git -y
RUN git config --global user.email "fosterthom16@gmail.com"
RUN git config --global user.name "Thom Foster"

# install tmux for running scripts in background
RUN apt install tmux -y

# (Optional) install sqlite for DB stuff
# RUN apt install sqlite3 -y

# If you want jax memory profiling
add-apt-repository ppa:longsleep/golang-backports
apt update
apt install golang-go
apt install graphviz
go install github.com/google/pprof@latest

# (Optional) install ipython (i know its not minimal but its so much nicer for interactive mode)
RUN python3.11 -m pip install ipython
RUN alias ipython='python3.11 -m IPython'

# Make /shared directory that we'll mount shared data into
RUN mkdir /shared

# so that changes made to mounted file when inside container are correct ownership
RUN useradd -d /project -u 3549 --create-home thomf

# install our package as editable repo
ADD ./jaxued /project/jaxued
WORKDIR /project/jaxued
RUN python3.11 -m pip install --default-timeout=500 -e .
WORKDIR /project

USER thomf
CMD ["/bin/bash"]