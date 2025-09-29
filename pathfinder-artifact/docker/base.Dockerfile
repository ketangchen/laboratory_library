FROM ubuntu:jammy

ENV HOME=/root


################
# Install LCOV #
################

RUN apt-get update && \
    apt-get install -y wget make cpanminus gcc && \
    cd $HOME && \
    wget https://github.com/linux-test-project/lcov/releases/download/v2.1/lcov-2.1.tar.gz && \
    tar -xvzf lcov-2.1.tar.gz && \
    rm lcov-2.1.tar.gz && \
    cd lcov-2.1 && make install

RUN perl -MCPAN -e 'install(Capture::Tiny)' && \
    perl -MCPAN -e 'install(DateTime)'      && \
    perl -MCPAN -e 'install(Devel::Cover)'  && \
    perl -MCPAN -e 'install(JSON::XS)'      && \
    perl -MCPAN -e 'install(Memory::Process)'


################
# Install Duet #
################

RUN apt-get update && \
    apt-get install -y \
    git curl bubblewrap patch make unzip bzip2 opam libgmp-dev python2.7

RUN opam init --bare --disable-sandboxing -y && \
    git clone https://github.com/wslee/duet.git $HOME/duet && \
    cd $HOME/duet && git checkout 3f0eced && \
    bash build && eval $(opam env) && make

ENV LD_LIBRARY_PATH=$HOME/.opam/4.08.0/lib/z3:${LD_LIBRARY_PATH}
ENV DUET_BIN_PATH=$HOME/duet/main.native


######################
# Install Basic Deps #
######################

RUN apt-get update && \
    apt-get install -y \
    vim xz-utils g++ libstdc++-12-dev pip cmake ninja-build


#################
# Install Clang #
#################

RUN cd $HOME && \
    wget https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.4/clang+llvm-16.0.4-x86_64-linux-gnu-ubuntu-22.04.tar.xz && \
    tar -xf clang+llvm-16.0.4-x86_64-linux-gnu-ubuntu-22.04.tar.xz && \
    rm clang+llvm-16.0.4-x86_64-linux-gnu-ubuntu-22.04.tar.xz
ENV LLVM_HOME=$HOME/clang+llvm-16.0.4-x86_64-linux-gnu-ubuntu-22.04
ENV PATH=$LLVM_HOME/bin:$PATH


##############
# Install Z3 #
##############

RUN git clone https://github.com/Z3Prover/z3.git $HOME/z3 && \
    cd $HOME/z3 && mkdir build && cd build && \
    CC=clang CXX=clang++ cmake -GNinja -DCMAKE_BUILD_TYPE=Release .. && \
    ninja && \
    cmake --install .
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib


######################
# Install PathFinder #
######################

ENV PATHFINDER=$HOME/pathfinder
COPY ../pathfinder $PATHFINDER
RUN cd $PATHFINDER && mkdir build && cd build && \
    CC=clang CXX=clang++ cmake -GNinja -DDUET_BIN_PATH=${DUET_BIN_PATH} -DCMAKE_BUILD_TYPE=Release .. && \
    ninja && \
    cmake --install .


############################
# Install Driver Generator #
############################

ENV PDG=$HOME/pathfinder-driver-generator
COPY ../pathfinder-driver-generator $PDG
RUN cd $PDG && mkdir build && cd build && \
    CC=clang CXX=clang++ cmake -GNinja -DCMAKE_BUILD_TYPE=Debug .. && \
    ninja


#########################################
# Copy Scripts for Coverage Measurement #
#########################################

COPY ../scripts/coverage.py $HOME/coverage.py
COPY ../scripts/pathfinder_coverage.py $HOME/pathfinder_coverage.py
