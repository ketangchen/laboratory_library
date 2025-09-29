FROM starlabunist/pathfinder:tf2.16-base


#########################
# Generate Fuzz Drivers #
#########################

RUN $PDG/build/bin/pdg --dll tf --dll_version 2.16 --output $TENSORFLOW/tensorflow/core/kernels/pathfinder && \
    mv $TENSORFLOW/tensorflow/core/kernels/pathfinder/driver $TENSORFLOW/tensorflow/core/kernels/pathfinder/default

RUN $PDG/build/bin/pdg --dll tf --dll_version 2.16 --wo_staged --output $HOME/pathfinder-wo_staged && \
    mv $HOME/pathfinder-wo_staged/driver $TENSORFLOW/tensorflow/core/kernels/pathfinder/wo_staged && \
    rm -rf $HOME/pathfinder-wo_staged


###################################
# Build TensorFlow & Fuzz Drivers #
###################################

ENV PYTHON_BIN_PATH=/usr/bin/python3
ENV USE_DEFAULT_PYTHON_LIB_PATH=1
ENV TF_NEED_ROCM=0
ENV TF_NEED_CUDA=0
ENV TF_NEED_CLANG=0
ENV CC_OPT_FLAGS=-Wno-sign-compare
ENV TF_SET_ANDROID_WORKSPACE=0
RUN cd $TENSORFLOW && \
    ./configure && \
    bazel build \
    --per_file_copt=+tensorflow.*,+external.*,-tensorflow/core/kernels/pathfinder.*@--coverage,-fprofile-abs-path \
    --linkopt=--coverage                                                                                          \
    //tensorflow/core/kernels/pathfinder/default/...                                                              \
    //tensorflow/core/kernels/pathfinder/wo_staged/...
