FROM starlabunist/pathfinder:base


#######################
# Install Python 3.11 #
#######################

RUN apt update && \
    apt install -y software-properties-common && \
    apt install -y python3.11 && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.11 /usr/bin/python3


#################
# Install Bazel #
#################

RUN cd $HOME && \
    wget https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64 && \
    mkdir $HOME/bin && \
    mv $HOME/bazelisk-linux-amd64 $HOME/bin/bazelisk-linux-amd64 && \
    chmod +x $HOME/bin/bazelisk-linux-amd64 && \
    ln -s $HOME/bin/bazelisk-linux-amd64 $HOME/bin/bazel
ENV PATH=$HOME/bin:$PATH


####################
# Clone TensorFlow #
####################

ENV TENSORFLOW=$HOME/tensorflow
RUN git clone https://github.com/tensorflow/tensorflow.git $TENSORFLOW && \
    cd $TENSORFLOW && git checkout 02d96c3

COPY ../docker/tf2.16-WORKSPACE.patch $HOME/tf2.16-WORKSPACE.patch
RUN patch $TENSORFLOW/WORKSPACE $HOME/tf2.16-WORKSPACE.patch && \
    rm $HOME/tf2.16-WORKSPACE.patch

RUN python3 -m pip install -U pip numpy wheel packaging requests opt_einsum patchelf && \
    python3 -m pip install -U keras_preprocessing --no-deps
