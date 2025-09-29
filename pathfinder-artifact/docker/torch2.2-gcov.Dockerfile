FROM starlabunist/pathfinder:torch2.2-base


####################
# Install PyTorch #
####################

RUN cd $PYTORCH && \
    CC=gcc CXX=g++ CMAKE_PREFIX_PATH=$CONDA_HOME CMAKE_C_COMPILER=gcc CMAKE_CXX_COMPILER=g++ BUILD_TEST=0 USE_MKLDNN=0 USE_OPENMP=0 USE_CUDA=0 USE_NCCL=0 USE_CPP_CODE_COVERAGE=1 $CONDA_RUN python3 setup.py install


###################################
# Generate & Compile Fuzz Drivers #
###################################

RUN $PDG/build/bin/pdg --dll torch --dll_version 2.2 --output $HOME/pathfinder-torch-default && \
    cd $HOME/pathfinder-torch-default && mkdir build && cd build && \
    CMAKE_PREFIX_PATH=$PYTORCH/torch/share/cmake CC=clang CXX=clang++ cmake -GNinja -DGEN_DRIVER=1 -DCMAKE_BUILD_TYPE=Debug .. && \
    ninja

RUN $PDG/build/bin/pdg --dll torch --dll_version 2.2 --wo_staged --output $HOME/pathfinder-torch-wo_staged && \
    cd $HOME/pathfinder-torch-wo_staged && mkdir build && cd build && \
    CMAKE_PREFIX_PATH=$PYTORCH/torch/share/cmake CC=clang CXX=clang++ cmake -GNinja -DGEN_DRIVER=1 -DCMAKE_BUILD_TYPE=Debug .. && \
    ninja
