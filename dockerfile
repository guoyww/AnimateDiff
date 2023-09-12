FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04

# RUN sed -i s/archive.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list
# RUN sed -i s/security.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list
# RUN sed -i s/ports.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list
RUN apt update && apt-get -y install git wget \
    python3.10 python3.10-venv python3-pip \
    build-essential libgl-dev libglib2.0-0 wget git git-lfs
RUN ln -s /usr/bin/python3.10 /usr/bin/python
RUN useradd -ms /bin/bash usera
WORKDIR /app

# RUN pip config set global.index-url https://mirrors.cloud.tencent.com/pypi/simple
# RUN pip config set global.trusted-host mirrors.cloud.tencent.com
# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH
ADD condarc ~/.condarc
ADD environment.yaml .
RUN conda env create -f environment.yaml
# Initialize conda in bash config fiiles:
SHELL ["conda", "run", "-n", "animatediff", "/bin/bash", "-c"]
RUN echo "source activate animatediff" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH
RUN conda run -n animatediff pip install xformers --no-cache-dir
RUN conda install -n animatediff -y imageio[pyav] imageio[ffmpeg]