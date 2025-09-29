FROM starlabunist/pathfinder:torch2.2-base


####################
# Install PyTorch #
####################

ENV LIBASAN_RT=$LLVM_HOME/lib/clang/16/lib/x86_64-unknown-linux-gnu/libclang_rt.asan.so
ENV LD_PRELOAD=$LLVM_HOME/lib/clang/16/lib/x86_64-unknown-linux-gnu/libclang_rt.asan.so

ENV LDSHARED="clang --shared"
ENV LDFLAGS="-L/root/anaconda3/lib:-stdlib=libstdc++ -fsanitize=address -shared-libasan"
ENV CFLAGS="-fsanitize=address -fno-sanitize-recover=all -shared-libasan -pthread"
ENV CXX_FLAGS="-pthread"
ENV UBSAN_FLAGS="-fno-sanitize-recover=all"

ENV BUILD_CAFFE2_OPS=0
ENV USE_DISTRIBUTED=0

ENV ASAN_OPTIONS=detect_leaks=0:symbolize=1:strict_init_order=true
ENV ASAN_SYMBOLIZER_PATH=$LLVM_HOME/bin/llvm-symbolizer

RUN cd $PYTORCH && \
    CC=clang CXX=clang++ CMAKE_PREFIX_PATH=$CONDA_HOME CMAKE_C_COMPILER=clang CMAKE_CXX_COMPILER=clang++ DEBUG=1 BUILD_TEST=0 USE_MKLDNN=0 USE_OPENMP=0 USE_CUDA=0 USE_NCCL=0 CXXFLAGS=-fsanitize-coverage=edge,no-prune,trace-pc-guard $CONDA_RUN python3 setup.py install && \
    rm -rf $PYTORCH/build


###################################
# Generate & Compile Fuzz Drivers #
###################################

RUN $PDG/build/bin/pdg --dll torch --dll_version 2.2 --output $HOME/pathfinder-torch-default && \
    cd $HOME/pathfinder-torch-default && mkdir build && cd build && \
    CMAKE_PREFIX_PATH=$PYTORCH/torch/share/cmake CC=clang CXX=clang++ CXXFLAGS=-fsanitize-coverage=edge,no-prune,trace-pc-guard cmake -GNinja -DGEN_DRIVER=1 -DCMAKE_BUILD_TYPE=Debug .. && \
    ninja
