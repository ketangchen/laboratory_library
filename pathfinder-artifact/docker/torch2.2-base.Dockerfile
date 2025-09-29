FROM starlabunist/pathfinder:base


####################
# Install Anaconda #
####################

RUN apt-get update && \
    apt-get install -y \
    libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 \
    libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 \
    libxtst6

ENV CONDA_HOME=$HOME/anaconda3
ENV CONDA_BIN=$CONDA_HOME/bin
ENV CONDA=$CONDA_HOME/bin/conda
ENV PATH=$CONDA_BIN:$PATH
ENV CONDA_ENV=base
ENV CONDA_RUN="conda run -n ${CONDA_ENV}"

RUN cd $HOME && \
    curl -O https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh && \
    bash Anaconda3-2024.02-1-Linux-x86_64.sh -b -p $CONDA_HOME && \
    rm $HOME/Anaconda3-2024.02-1-Linux-x86_64.sh

RUN $CONDA install -y python=3.9
RUN $CONDA init && \
    conda install -y -c conda-forge libstdcxx-ng=12 && \
    conda install -y mkl mkl-include


########################
# Clone PyTorch Source #
########################

ENV PYTORCH=$HOME/pytorch
RUN git clone https://github.com/pytorch/pytorch $PYTORCH && \
    cd $PYTORCH && git checkout 115da02 && \
    git submodule sync && \
    git submodule update --init --recursive

COPY ../docker/torch2.2-requirements.txt.patch $HOME/torch2.2-requirements.txt.patch
RUN patch $PYTORCH/requirements.txt $HOME/torch2.2-requirements.txt.patch && \
    rm $HOME/torch2.2-requirements.txt.patch

RUN cd $PYTORCH && \
    $CONDA_RUN pip install -r requirements.txt
