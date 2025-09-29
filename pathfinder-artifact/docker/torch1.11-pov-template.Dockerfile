FROM starlabunist/pathfinder:torch1.11-pov-base


###########################
# Generate & Compile PoVs #
###########################

COPY ../_buggy_input $HOME/buggy_input
RUN $PDG/build/bin/pdg --dll torch --dll_version 1.11 --mode pov --buggy_input_dir $HOME/buggy_input --output $HOME/pathfinder-torch && \
    cd $HOME/pathfinder-torch && mkdir build && cd build && \
    CMAKE_PREFIX_PATH=$PYTORCH/torch/share/cmake CC=clang-13 CXX=clang++-13 cmake -GNinja -DGEN_POV=1 -DCMAKE_BUILD_TYPE=Debug .. && \
    ninja
