# PathFinder-artifact

This artifact, accompanying the ICSE 2025 paper titled *Lightweight Concolic Testing via Path-Condition Synthesis for Deep Learning Libraries*, is a script designed to automate the setup and execution of experiments.

For PathFinder and PathFinder-Driver-Generator, the core components of this work, please refer to the following links.
- [PathFinder](https://github.com/starlab-unist/pathfinder)
- [PathFinder Driver Generator](https://github.com/starlab-unist/pathfinder-driver-generator)

## 1. Setup

Since all experimental environments are prepared in Docker containers, you only need Docker and the minimal Python dependencies to run the script. This script has been tested on an environment equipped with an Intel Xeon Platinum 8468 CPU, Ubuntu 22.04, and Docker 26.1.0.
- Docker
- Python3

You can either pull pre-built images from Docker Hub or build them from Dockerfiles. In either case, you may want to pull/build selectively, as each image requires a substantial amount of storage.

### 1.1. (Recommended) Pull Pre-built Docker Images
- PyTorch 2.2

    ```bash
    # For Fuzzing & Coverage Mesearuing (RQ1 and RQ2, ~46.2GB)
    docker pull starlabunist/pathfinder:torch2.2-fuzz
    docker pull starlabunist/pathfinder:torch2.2-gcov

    # For Fuzzing with ASAN & Generating PoVs (RQ3, ~36.7GB)
    docker pull starlabunist/pathfinder:torch2.2-asan
    docker pull starlabunist/pathfinder:torch2.2-pov-base
    ```

- PyTorch 1.11 (For comparison with IvySyn)

    ```bash
    # For Fuzzing & Coverage Mesearuing (RQ1 and RQ2, ~30.6GB)
    docker pull starlabunist/pathfinder:torch1.11-fuzz
    docker pull starlabunist/pathfinder:torch1.11-gcov

    # For Fuzzing with ASAN & Generating PoVs (RQ3, ~30.0GB)
    docker pull starlabunist/pathfinder:torch1.11-asan
    docker pull starlabunist/pathfinder:torch1.11-pov-base
    ```

- TensorFlow 2.16

    ```bash
    # For Fuzzing & Coverage Mesearuing (RQ1 and RQ2, ~44.0GB)
    docker pull starlabunist/pathfinder:tf2.16-fuzz
    docker pull starlabunist/pathfinder:tf2.16-gcov

    # For Fuzzing with ASAN & Generating PoVs (RQ3, ~28.4GB)
    docker pull starlabunist/pathfinder:tf2.16-asan
    ```

### 1.2. (Alternative) Build Docker Images from Dockerfiles

- (necessary) PathFinder Base Image

    ```bash
    # Pull PathFinder and PDG
    git submodule sync
    git submodule update --init --recursive

    docker build -f $PWD/docker/base.Dockerfile -t starlabunist/pathfinder:base .
    ```

- PyTorch 2.2

    ```bash
    # PyTorch 2.2 Base Image
    docker build -f $PWD/docker/torch2.2-base.Dockerfile -t starlabunist/pathfinder:torch2.2-base .

    # For Fuzzing & Coverage Mesearuing (RQ1 and RQ2, ~46.2GB)
    docker build -f $PWD/docker/torch2.2-fuzz.Dockerfile -t starlabunist/pathfinder:torch2.2-fuzz .
    docker build -f $PWD/docker/torch2.2-gcov.Dockerfile -t starlabunist/pathfinder:torch2.2-gcov .

    # For Fuzzing with ASAN & Generating PoVs (RQ3, ~36.7GB)
    docker build -f $PWD/docker/torch2.2-asan.Dockerfile -t starlabunist/pathfinder:torch2.2-asan .
    docker build -f $PWD/docker/torch2.2-pov-base.Dockerfile -t starlabunist/pathfinder:torch2.2-pov-base .
    ```

- PyTorch 1.11 (For comparison with IvySyn)

    ```bash
    # PyTorch 1.11 Base Image
    docker build -f $PWD/docker/torch1.11-base.Dockerfile -t starlabunist/pathfinder:torch1.11-base .

    # For Fuzzing & Coverage Mesearuing (RQ1 and RQ2, ~30.6GB)
    docker build -f $PWD/docker/torch1.11-fuzz.Dockerfile -t starlabunist/pathfinder:torch1.11-fuzz .
    docker build -f $PWD/docker/torch1.11-gcov.Dockerfile -t starlabunist/pathfinder:torch1.11-gcov .

    # For Fuzzing with ASAN & Generating PoVs (RQ3, ~30.0GB)
    docker build -f $PWD/docker/torch1.11-asan.Dockerfile -t starlabunist/pathfinder:torch1.11-asan .
    docker build -f $PWD/docker/torch1.11-pov-base.Dockerfile -t starlabunist/pathfinder:torch1.11-pov-base .
    ```

- TensorFlow 2.16

    ```bash
    # TensorFlow 2.16 Base Image
    docker build -f $PWD/docker/tf2.16-base.Dockerfile -t starlabunist/pathfinder:tf2.16-base .

    # For Fuzzing & Coverage Mesearuing (RQ1 and RQ2, ~44.0GB)
    docker build -f $PWD/docker/tf2.16-fuzz.Dockerfile -t starlabunist/pathfinder:tf2.16-fuzz .
    docker build -f $PWD/docker/tf2.16-gcov.Dockerfile -t starlabunist/pathfinder:tf2.16-gcov .

    # For Fuzzing with ASAN & Generating PoVs (RQ3, ~28.4GB)
    docker build -f $PWD/docker/tf2.16-asan.Dockerfile -t starlabunist/pathfinder:tf2.16-asan .
    ```

## 2. Basic Usage
- Common Flags
    - `--dll`: Target DL library. Should be one of {`torch`, `tf`}.
    - `--version`: Version of target DLL. Default value is `2.2` for `torch` and `2.16` for `tf`.
    - `--mode`: Running mode. Should be one of {`fuzz`, `gcov`, `asan`}.
    - `--apis`: One or more APIs to fuzz. Should be from `api_list/*.txt`. If not set, run all possible APIs.
    - `--time_budget`: Time budget for each API (in seconds).
    - `--cpu_capacity`: Number of available cores. Fuzzing of a single API operates on a single core. If not specified, it utilizes half of system cores.

### 2.1. Fuzzing & Coverage Measuring (RQ1 and RQ2)
- Fuzzing

    ```bash
    python3 -u run.py --dll torch --mode fuzz \
        --apis torch.nn.functional.conv1d     \
        --time_budget 300 --cpu_capacity 1
    ```
    - Results are stored in `_fuzz_result`.

- Coverage Measuring

    ```bash
    python3 -u run.py --dll torch --mode gcov \
        --apis torch.nn.functional.conv1d     \
        --time_budget 300 --cpu_capacity 1
    ```

    - A summary similar to the following is printed, and the results are stored in `_gcov_result`.

        ```bash
        GCOV Summary:
        +------------+-----------------+
        | time (sec) | branch coverage |
        +------------+-----------------+
        |        300 |            7170 |
        +------------+-----------------+
        ```

    - Related Flags
        - `--gen_html`: If set, generates a visualized coverage report.

### 2.2. Fuzzing with ASAN & Generating PoVs (RQ3)
- Fuzzing with ASAN

    ```bash
    python3 -u run.py --dll torch --mode fuzz --asan \
        --apis torch.Tensor.topk                     \
        --time_budget 300 --cpu_capacity 1
    ```

    - An instruction for running generated PoVs similar to the following is printed, and the results are stored in `_asan_result`.

        ```bash
        Generated docker image `torch2.2-pov-300sec-0`.
        In the docker image, generated PoV source codes are in `/root/pathfinder-torch/pov`.

        How to execute PoVs:
            # Run docker container.
            docker run -it --rm torch2.2-pov-300sec-0 bash
            # In the docker container, execute each PoV binaries.
            /root/pathfinder-torch/build/bin/<POV_BIN>
        ```
        
    - Related Flags
        - `--asan`: If set, run with AddressSanitizer.

### 2.3. Counting Valid Inputs (Discussion)
- Example command
    ```bash
    python3 scripts/count_input.py --dll torch --result_path ./_fuzz_result/torch2.2-fuzz-default-300sec/
    ```
- Flags
    - `--dll`: Target DL library. Should be one of {`torch`, `tf`}.
    - `--result_path`: Path of fuzz result directory (under `./_fuzz_result`)

## 3. Reproducing RQs

> **Note**
>
> This script is designed to minimize redundant fuzzing. APIs shared across experiments can reuse existing fuzzing results for coverage measurement. If an API already has results from a previous run, the script will output `Note: Skip <CONTAINER_NAME>, as fuzz result exists in <RESULT_DIRECTORY>` and skip fuzzing for that API. 

Common Flags:
- `--vs`: For RQ1. A baseline tool for comparison. The target APIs are selected from those common to both PathFinder and the baseline. Should be one of {`freefuzz`, `deeprel`, `titanfuzz`, `acetest`, `ivysyn`}.
- `--ablation`: For RQ2. Should be one of {`default`, `wo_nbp`, `wo_staged`}. Default is `default`.

### 3.1. RQ1-Branch Coverage Analysis

- PyTorch

    ```bash
    # vs FreeFuzz
    python3 -u run.py --dll torch --mode fuzz --vs freefuzz > rq1-torch-vs_freefuzz-fuzz.log 2>&1
    python3 -u run.py --dll torch --mode gcov --vs freefuzz > rq1-torch-vs_freefuzz-gcov.log 2>&1

    # vs DeepREL
    python3 -u run.py --dll torch --mode fuzz --vs deeprel > rq1-torch-vs_deeprel-fuzz.log 2>&1
    python3 -u run.py --dll torch --mode gcov --vs deeprel > rq1-torch-vs_deeprel-gcov.log 2>&1

    # vs TitanFuzz
    python3 -u run.py --dll torch --mode fuzz --vs titanfuzz > rq1-torch-vs_titanfuzz-fuzz.log 2>&1
    python3 -u run.py --dll torch --mode gcov --vs titanfuzz > rq1-torch-vs_titanfuzz-gcov.log 2>&1

    # vs ACETest
    python3 -u run.py --dll torch --mode fuzz --vs acetest > rq1-torch-vs_acetest-fuzz.log 2>&1
    python3 -u run.py --dll torch --mode gcov --vs acetest > rq1-torch-vs_acetest-gcov.log 2>&1

    # vs IvySyn
    python3 -u run.py --dll torch --mode fuzz --vs ivysyn > rq1-torch-vs_ivysyn-fuzz.log 2>&1
    python3 -u run.py --dll torch --mode gcov --vs ivysyn > rq1-torch-vs_ivysyn-gcov.log 2>&1
    ```

- TensorFlow

    ```bash
    # vs FreeFuzz
    python3 -u run.py --dll tf --mode fuzz --vs freefuzz > rq1-tf-vs_freefuzz-fuzz.log 2>&1
    python3 -u run.py --dll tf --mode gcov --vs freefuzz > rq1-tf-vs_freefuzz-gcov.log 2>&1

    # vs DeepREL
    python3 -u run.py --dll tf --mode fuzz --vs deeprel > rq1-tf-vs_deeprel-fuzz.log 2>&1
    python3 -u run.py --dll tf --mode gcov --vs deeprel > rq1-tf-vs_deeprel-gcov.log 2>&1

    # vs TitanFuzz
    python3 -u run.py --dll tf --mode fuzz --vs titanfuzz > rq1-tf-vs_titanfuzz-fuzz.log 2>&1
    python3 -u run.py --dll tf --mode gcov --vs titanfuzz > rq1-tf-vs_titanfuzz-gcov.log 2>&1

    # vs ACETest
    python3 -u run.py --dll tf --mode fuzz --vs acetest > rq1-tf-vs_acetest-fuzz.log 2>&1
    python3 -u run.py --dll tf --mode gcov --vs acetest > rq1-tf-vs_acetest-gcov.log 2>&1
    ```

### 3.2. RQ2-Ablation Study

- PyTorch

    ```bash
    # Default
    python3 -u run.py --dll torch --mode fuzz --ablation default > rq2-torch-default-fuzz.log 2>&1
    python3 -u run.py --dll torch --mode gcov --ablation default > rq2-torch-default-gcov.log 2>&1

    # w/o NBP
    python3 -u run.py --dll torch --mode fuzz --ablation wo_nbp > rq2-torch-wo_nbp-fuzz.log 2>&1
    python3 -u run.py --dll torch --mode gcov --ablation wo_nbp > rq2-torch-wo_nbp-gcov.log 2>&1

    # w/o Staged
    python3 -u run.py --dll torch --mode fuzz --ablation wo_staged > rq2-torch-wo_staged-fuzz.log 2>&1
    python3 -u run.py --dll torch --mode gcov --ablation wo_staged > rq2-torch-wo_staged-gcov.log 2>&1
    ```

- TensorFlow

    ```bash
    # Default
    python3 -u run.py --dll tf --mode fuzz --ablation default > rq2-tf-default-fuzz.log 2>&1
    python3 -u run.py --dll tf --mode gcov --ablation default > rq2-tf-default-gcov.log 2>&1

    # w/o NBP
    python3 -u run.py --dll tf --mode fuzz --ablation wo_nbp > rq2-tf-wo_nbp-fuzz.log 2>&1
    python3 -u run.py --dll tf --mode gcov --ablation wo_nbp > rq2-tf-wo_nbp-gcov.log 2>&1

    # w/o Staged
    python3 -u run.py --dll tf --mode fuzz --ablation wo_staged > rq2-tf-wo_staged-fuzz.log 2>&1
    python3 -u run.py --dll tf --mode gcov --ablation wo_staged > rq2-tf-wo_staged-gcov.log 2>&1
    ```

### 3.3. RQ3-Bug Detection Analysis

- PyTorch

    ```bash
    # ALL APIs
    python3 -u run.py --dll torch --mode fuzz --asan > rq3-torch-all.log 2>&1

    # vs FreeFuzz
    python3 -u run.py --dll torch --mode fuzz --asan --vs freefuzz > rq3-torch-vs_freefuzz.log 2>&1

    # vs DeepREL
    python3 -u run.py --dll torch --mode fuzz --asan --vs deeprel > rq3-torch-vs_deeprel.log 2>&1

    # vs TitanFuzz
    python3 -u run.py --dll torch --mode fuzz --asan --vs titanfuzz > rq3-torch-vs_titanfuzz.log 2>&1

    # vs ACETest
    python3 -u run.py --dll torch --mode fuzz --asan --vs acetest > rq3-torch-vs_acetest.log 2>&1

    # vs IvySyn
    python3 -u run.py --dll torch --mode fuzz --asan --vs ivysyn > rq3-torch-vs_ivysyn.log 2>&1
    ```

- TensorFlow

    ```bash
    # ALL APIs
    python3 -u run.py --dll tf --mode fuzz --asan > rq3-tf-all.log 2>&1

    # vs FreeFuzz
    python3 -u run.py --dll tf --mode fuzz --asan --vs freefuzz > rq3-tf-vs_freefuzz.log 2>&1

    # vs DeepREL
    python3 -u run.py --dll tf --mode fuzz --asan --vs deeprel > rq3-tf-vs_deeprel.log 2>&1

    # vs TitanFuzz
    python3 -u run.py --dll tf --mode fuzz --asan --vs titanfuzz > rq3-tf-vs_titanfuzz.log 2>&1

    # vs ACETest
    python3 -u run.py --dll tf --mode fuzz --asan --vs acetest > rq3-tf-vs_acetest.log 2>&1
    ```

## 4. How to Extend to other DL Libraries

### 4.1. Prerequisite

If you want to extend this script, which facilitates PathFinder fuzzing and coverage measurement, to another DL library, you should first do the following:

- [Import PathFinder into your DL library](https://github.com/starlab-unist/pathfinder/blob/main/README.md#3-how-to-import-pathfinder).
- [Extend PDG for your DL library](https://github.com/starlab-unist/pathfinder-driver-generator/blob/main/README.md#3-how-to-extend-to-other-dl-libraries).

### 4.2. Requirements

You need to provide the following three types of components:

- List of target APIs
    - Examples are available at [api_list](./api_list/) directory.

- Dockerfile(s)
    - Examples are available at [docker](./docker/) directory.
    - You may use [base.Dockerfile](./docker/base.Dockerfile) as your base image.

- Extending `DLLInfo`
    - Inherit and override [`DLLInfo`](./dll_info.py) for your DL library.
    - You may refer to `TorchInfo` and `TFInfo`, which correspond to ones for PyTorch and TensorFlow, respectively.
