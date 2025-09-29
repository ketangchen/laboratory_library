FROM starlabunist/pathfinder:torch1.11-base


####################
# Install PyTorch #
####################

RUN cd $PYTORCH && \
    CC=gcc CXX=g++ CMAKE_PREFIX_PATH=$CONDA_HOME CMAKE_C_COMPILER=gcc CMAKE_CXX_COMPILER=g++ BUILD_TEST=0 USE_MKLDNN=0 USE_OPENMP=0 USE_GLOO=0 USE_BREAKPAD=0 USE_FBGEMM=0 USE_CUDA=0 USE_NCCL=0 USE_CPP_CODE_COVERAGE=1 $CONDA_RUN python3 setup.py install


###################################
# Generate & Compile Fuzz Drivers #
###################################

RUN $PDG/build/bin/pdg --dll torch --dll_version 1.11 --output $HOME/pathfinder-torch-default && \
    cd $HOME/pathfinder-torch-default && mkdir build && cd build && \
    CMAKE_PREFIX_PATH=$PYTORCH/torch/share/cmake CC=clang CXX=clang++ cmake -GNinja -DGEN_DRIVER=1 -DCMAKE_BUILD_TYPE=Debug .. && \
    ninja
