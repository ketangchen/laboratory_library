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
ENV TF_NEED_CLANG=1
ENV CLANG_COMPILER_PATH=$LLVM_HOME/bin/clang
ENV CC_OPT_FLAGS=-Wno-sign-compare
ENV TF_SET_ANDROID_WORKSPACE=0

ENV CC=$LLVM_HOME/bin/clang
ENV CXX=$LLVM_HOME/bin/clang++
ENV LLVM_ROOT=$LLVM_HOME
ENV LIBASAN_RT=$LLVM_HOME/lib/clang/16/lib/x86_64-unknown-linux-gnu/libclang_rt.asan.so
ENV LD_PRELOAD=$LLVM_HOME/lib/clang/16/lib/x86_64-unknown-linux-gnu/libclang_rt.asan.so
ENV ASAN_SYMBOLIZER_PATH=$LLVM_HOME/bin/llvm-symbolizer
ENV ASAN_OPTIONS=detect_leaks=0:symbolize=1

RUN cd $TENSORFLOW && \
    ./configure && \
    bazel build --config=dbg --copt=-O0 \
    --define=with_xla_support=false     \
    --per_file_copt=+tensorflow.*,+external.*@-fsanitize-coverage=edge           \
    --per_file_copt=+tensorflow.*,+external.*@-fsanitize-coverage=trace-pc-guard \
    --per_file_copt=+tensorflow.*,+external.*@-fsanitize-coverage=no-prune       \
    --per_file_copt=+tensorflow.*,+external.*@-fsanitize=address                 \
    --per_file_copt=+tensorflow.*,+external.*@-fno-omit-frame-pointer            \
    --per_file_copt=+tensorflow.*,+external.*@-mllvm                             \
    --per_file_copt=+tensorflow.*,+external.*@-asan-globals=0                    \
    --per_file_copt=+tensorflow.*,+external.*@-fno-sanitize-recover=all          \
    --linkopt=-fsanitize=address                                                 \
    --linkopt=-shared-libasan                                                    \
    //tensorflow/core/kernels/pathfinder/default/...                             \
    //tensorflow/core/kernels/pathfinder/wo_staged/...
