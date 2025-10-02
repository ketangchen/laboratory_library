# laboratory_library
laboratory_library

论文题目：《Towards More Complete Constraints for Deep Learning Library Testing via Complementary Set Guided Refinement》
论文总结：
1. 核心问题
现有深度学习（DL）库测试中，约束引导的测试方法多关注约束的“可靠性”（生成的测试用例均有效），却忽视“完整性”（未覆盖所有有效输入空间），导致测试不全面。
2. 问题重要性
DL库是AI系统核心组件，约束不完整会遗漏大量有效测试场景，可能导致库中隐藏的漏洞（如自动驾驶系统决策错误）未被发现，引发安全风险与经济损失。
3. 方法与贡献
方法：提出“补集引导的约束优化”方法，设计工具DeepConstr。先通过错误信息（绑定约束目标了）将复杂约束拆分为独立子约束，再生成子约束对应测试集的补集，若补集中存在有效测试用例则说明约束不完整，最后用遗传算法迭代优化约束（探边找精准范围，结合LLM生成原始约束、约束合成与适配度评估）。
有效性：通过补集精准定位约束遗漏区域，拆分约束避免优化干扰，遗传算法平衡可靠性与完整性，相比人工或现有自动方法，能覆盖更全面的有效输入空间。
4. 实验结果
在PyTorch和TensorFlow上发现84个未知漏洞，72个被确认、51个已修复；对NeuRI支持的算子，PyTorch和TensorFlow上分别有58.27%、62.1%的分支覆盖率提升，对NNSmith支持的算子则分别提升49.15%、38.1%。
5. 局限与优化
局限：无法处理错误描述与根源无关的错误信息，不支持张量元素级约束（如“元素数量需小于n”）。
优化：结合静态分析修正错误信息映射，引入轻量化元素级约束建模（如抽样验证元素条件），平衡精度与资源消耗。

基于对代码库的分析，这是一个名为 **DeepConstr** 的深度学习库测试工具。为该论文的实现，总结如何运行这个代码：

## 代码运行指南

### 1. 环境准备

**使用Docker（推荐）:**
```bash
# 安装Docker
docker --version

# 拉取预配置的Docker镜像
docker pull gwihwan/artifact-issta24:latest
```

**或者本地安装:**
```bash
# 安装依赖
pip install -r requirements.txt

# 设置环境变量（如果需要使用OpenAI API）
# 创建 .env 文件
echo "OPENAI_API_KEY1='sk-********'" > .env
```

### 2. 主要运行方式

#### A. 约束提取（Constraint Extraction）

**PyTorch约束提取:**
```bash
cd DeepConstr
PYTHONPATH=$(pwd):$(pwd)/deepconstr:$(pwd)/nnsmith \
python deepconstr/train/run.py \
train.record_path=repro/records/torch \
backend.type=torchcomp \
model.type=torch \
train.parallel=1 \
train.num_eval=500 \
train.pass_rate=95 \
train.retrain=false \
train.target='["torch.add","torch.abs"]'
```

**TensorFlow约束提取:**
```bash
PYTHONPATH=$(pwd):$(pwd)/deepconstr:$(pwd)/nnsmith \
python deepconstr/train/run.py \
train.record_path=repro/records/tf \
backend.type=xla \
model.type=tensorflow \
train.parallel=1 \
train.num_eval=300 \
train.pass_rate=95 \
train.retrain=false \
train.target='["tf.add","tf.abs"]'
```

#### B. 模糊测试（Fuzzing）

**PyTorch模糊测试:**
```bash
PYTHONPATH=$(pwd):$(pwd)/deepconstr:$(pwd)/nnsmith \
python nnsmith/cli/fuzz.py \
fuzz.time=15m \
mgen.record_path=$(pwd)/data/records/torch \
fuzz.root=$(pwd)/outputs/torch-deepconstr-n5 \
fuzz.save_test=$(pwd)/outputs/torch-deepconstr-n5.models \
model.type=torch \
backend.type=torchcomp \
filter.type=[nan,dup,inf] \
debug.viz=true \
hydra.verbose=['fuzz'] \
fuzz.resume=true \
mgen.method=deepconstr \
mgen.max_nodes=5 \
mgen.test_pool=[torch.abs,torch.add] \
mgen.pass_rate=10
```

**TensorFlow模糊测试:**
```bash
PYTHONPATH=$(pwd):$(pwd)/deepconstr:$(pwd)/nnsmith \
python nnsmith/cli/fuzz.py \
fuzz.time=4h \
mgen.record_path=$(pwd)/data/records/tf \
fuzz.root=$(pwd)/outputs/tensorflow-deepconstr-n5 \
fuzz.save_test=$(pwd)/outputs/tensorflow-deepconstr-n5.models \
model.type=tensorflow \
backend.type=xla \
filter.type=[nan,dup,inf] \
debug.viz=true \
hydra.verbose=['fuzz'] \
fuzz.resume=true \
mgen.method=deepconstr \
mgen.max_nodes=5 \
mgen.pass_rate=10
```

#### C. 使用便捷脚本

**使用fuzz.sh脚本:**
```bash
# 参数: NSIZE METHOD MODEL BACKEND TIME TESTPOOL
./fuzz.sh 5 deepconstr torch torchcomp 15m torch.abs,torch.add
```

### 3. 实验复现

**覆盖率比较实验:**
```bash
# PyTorch
PYTHONPATH=$(pwd):$(pwd)/nnsmith:$(pwd)/deepconstr \
python experiments/evaluate_apis.py \
exp.save_dir=exp/torch \
mgen.record_path=$(pwd)/data/records/torch/ \
mgen.pass_rate=0.05 \
model.type=torch \
backend.type=torchjit \
fuzz.time=15m \
exp.parallel=16 \
mgen.noise=0.8 \
exp.targets=$(pwd)/data/torch_dc_neuri.json \
exp.baselines="['deepconstr', 'neuri', 'symbolic-cinit', 'deepconstr_2']"
```

### 4. 关键参数说明

- **model.type**: 选择深度学习框架 (`torch`, `tensorflow`)
- **backend.type**: 选择后端 (`torchcomp`, `torchjit`, `xla`)
- **mgen.method**: 约束生成方法 (`deepconstr`, `neuri`, `symbolic-cinit`)
- **fuzz.time**: 测试时间 (`4h`, `15m`, `30s`)
- **mgen.max_nodes**: 每个生成图中的操作符数量
- **mgen.test_pool**: 指定要测试的API列表

### 5. 输出结果

- **约束文件**: 保存在 `train.record_path` 指定的目录
- **测试用例**: 保存在 `fuzz.save_test` 指定的目录  
- **发现的bug**: 保存在 `fuzz.root` 指定的目录

这个工具主要用于深度学习库的约束提取和模糊测试，可以帮助发现深度学习框架中的bug。


# Towards More Complete Constraints for Deep Learning Library Testing via Complementary Set Guided Refinement

<p align="center">
    <!-- <a href="https://arxiv.org/abs/2302.02261"><img src="https://img.shields.io/badge/arXiv-2302.02261-b31b1b.svg?style=for-the-badge"> -->
    <a href="https://doi.org/10.5281/zenodo.12669927"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.8319975-8A2BE2?style=for-the-badge">
    <a href="https://github.com/THU-WingTecher/DeepConstr/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge"></a>
    <a href="https://hub.docker.com/repository/docker/gwihwan/artifact-issta24/tags"><img src="https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white"></a>
</p>

Welcome to the artifact repository of the DeepConstr paper which is accepted by ISSTA 2024.

### Source Code Structure 
```
|-- build        # Directory for compiling PyTorch and TensorFlow
|-- data         # Data directory, contains records of constraints and intersected operator names
|-- deepconstr   # Main implementation of DeepConstr
|   |-- error.py     # Error handling module for DeepConstr
|   |-- gen          # Implementation for test case generation from SMT-expression
|   |-- grammar      # Implementation for SMT-expression grammar to convert natural language into SMT-expression
|   |-- train        # Implementation for constraint extraction and refinement
|   |-- logger.py    # Logging module for DeepConstr
|   `-- utils.py     # Utility functions for DeepConstr
|-- docs         # Documentation for the project
|-- experiments  # Scripts for conducting experiments
|-- nnsmith      # Main implementation of NNSmith
|-- requirements.txt # List of Python dependencies
|-- outputs      # Directory for output files generated by the project
|-- tests        # Test scripts for verifying the functionality of the project
|-- collect_cov.sh
|-- collect_env.py
|-- fuzz.sh
|-- LICENSE
|-- README.md
`-- requirements.txt
```

### Bug Finding Evidence (RQ3)

You can find the bug finding evidence [here](docs/bug_list.md).

### Get Ready

Before you start, please make sure you have [Docker](https://docs.docker.com/engine/install/) installed.
To check the installation:
```bash
docker --version # Test docker availability
```
Get Docker image from Docker Hub
```bash
docker pull gwihwan/artifact-issta24:latest
``` 
Navigate to the DeepConstr project directory.
```bash
cd ../DeepConstr
```
### Start fuzzing

You can start fuzzing with the `fuzz.py` script.

> [!NOTE]
>
> **Command usage of**: `python nnsmith/cli/fuzz.py`
>
> **Arguments**:
> - `mgen.max_nodes`: the number of operators in each generated graph.
> - `mgen.method`: approach of generated constraints, choose from `["deepconstr", "neuri", "symbolic-cinit"]`.
> - `model.type`: generated model type, choose from `["tensorflow", "torch"]`.
> - `backend.type`: generated backend type, choose from `["xla", "torchjit"]`.
> - `fuzz.time`: fuzzing time in formats such as `4h`, `1m`, `30s`.
> - `mgen.record_path`: the directory that constraints are saved, such as `$(pwd)/data/records/torch`.
> - `fuzz.save_test`: the directory that generated test cases are saved, such as `$(pwd)/bugs/${model.type}-${mgen.method}-n${mgen.max_nodes}`.
> - `fuzz.root`: the directory that buggy test cases are saved, such as `$(pwd)/bugs/${model.type}-${mgen.method}-n${mgen.max_nodes}-buggy`.
> - `mgen.test_pool`(Optional): specific API for fuzzing. If not specified, fuzzing will be conducted across all prepared APIs.
>
> **Outputs**:
> The buggy test cases will be saved in the directory specified by `fuzz.root`, while every generated test case will be saved in the directory specified by `fuzz.save_test`.

#### Quick start for PyTorch

First, activate the conda environment created for this project.
```bash 
conda activate std
``` 

For PyTorch, you can specify the APIs to be tested by setting the `mgen.test_pool` argument, such as `[torch.abs,torch.add]`. For example, following code will fuzz `torch.abs` and `torch.add` for 15 minutes.
```bash
PYTHONPATH=$(pwd):$(pwd)/deepconstr:$(pwd)/nnsmith python nnsmith/cli/fuzz.py fuzz.time=15m \
mgen.record_path=/DeepConstr/data/records/torch fuzz.root=/DeepConstr/outputs/torch-deepconstr-n5-torch.abs-torch.add \
fuzz.save_test=/DeepConstr/outputs/torch-deepconstr-n5-torch.abs-torch.add.models \
model.type=torch backend.type=torchcomp filter.type=[nan,dup,inf] \
debug.viz=true hydra.verbose=['fuzz'] fuzz.resume=true \
mgen.method=deepconstr mgen.max_nodes=5 mgen.test_pool=[torch.abs,torch.add] mgen.pass_rate=10
```
If the `mgen.test_pool` is not specified, the program will fuzz all APIs that deepconstr supports. Following code will fuzz all APIs that deepconstr support for 4 hours.
```bash
PYTHONPATH=$(pwd):$(pwd)/deepconstr:$(pwd)/nnsmith python nnsmith/cli/fuzz.py fuzz.time=4h \
mgen.record_path=/DeepConstr/data/records/torch \
fuzz.root=/DeepConstr/outputs/torch-deepconstr-n5-torch.abs-torch.add \
fuzz.save_test=/DeepConstr/outputs/torch-deepconstr-n5-torch.abs-torch.add.models \
model.type=torch backend.type=torchcomp filter.type=[nan,dup,inf] debug.viz=true hydra.verbose=['fuzz'] fuzz.resume=true mgen.method=deepconstr mgen.max_nodes=5 mgen.pass_rate=10
```

#### Quick start for TensorFlow
First, activate the conda environment created for this project.
```bash
conda activate std
```
Then, execute the following commands to start fuzzing. Following code will fuzz all APIs that deepconstr supports for 4 hours.
```bash 
PYTHONPATH=$(pwd):$(pwd)/deepconstr:$(pwd)/nnsmith python nnsmith/cli/fuzz.py fuzz.time=4h \
mgen.record_path=/DeepConstr/data/records/tf \
fuzz.root=/DeepConstr/outputs/tensorflow-deepconstr-n5- fuzz.save_test=/DeepConstr/outputs/tensorflow-deepconstr-n5-.models \
model.type=tensorflow backend.type=xla filter.type=[nan,dup,inf] \
debug.viz=true hydra.verbose=['fuzz'] \
fuzz.resume=true mgen.method=deepconstr mgen.max_nodes=5 mgen.pass_rate=10
```

#### Generate python code

The test case of deepconstr is saved as the format of `gir.pkl`. To convert the `git.pkl` into python code, you can utilize below code. You can specify the code with the option of compiler. For now, we support "torchcomp" compiler with pytorch. You can use following code to convert the `gir.pkl` which is saved at `code_saved_dir` into python code. If you followed above quick start, you can use below code to convert the code.

```bash
PYTHONPATH=$(pwd):$(pwd)/deepconstr:$(pwd)/nnsmith python nnsmith/materialize/torch/program.py /DeepConstr/outputs/torch-deepconstr-n5-torch.abs-torch.add torchcomp
```

# Extract Constraints

### Setup Instructions

1. (optional) If you are not using docker, install required packages:
```bash 
pip install -r requirements.txt
```
2. Generate a `.env` file in your workspace directory `$(pwd)/.env` and populate it with your specific values:
- OpenAI API Key:
```OPENAI_API_KEY1 ='sk-********'```
- Proxy Setting (Optional):
```MYPROXY ='166.***.***.***:****'```

3. Testing Your Configuration: After setting your environment variables, you can verify your configuration by running:
```bash
PYTHONPATH=$(pwd):$(pwd)/deepconstr:$(pwd)/nnsmith python tests/proxy.py
# INFO    llm    - Output(Ptk12-OtkPtk9) : 
# Hello! How can I assist you today? 
# Time cost : 1.366152286529541 seconds 
```
If configured correctly, you will receive a response from the OpenAI API, such as: "Hello! How can I assist you today?"

### Start Extraction
You can extract constraints by running `deepconstr/train/run.py` script.

> [!NOTE]
>
> **Command usage of**: `python deepconstr/train/run.py`
>
> **Important Arguments**:
> - `tran.target`: Specifies the API name or path to extract. This can be a single API name (e.g., `"torch.add"`), a list containing multiple API names (e.g., `["torch.add", "torch.abs"]`), or a JSON file path containing the list.
> - `train.retrain`: A boolean value that determines whether to reconduct constraint extraction. If set to false, the tool will only collect APIs that haven't been extracted. If set to true, the tool collects all APIs except those where the pass rate exceeds the preset target pass rate (`train.pass_rate`).
> - `train.pass_rate`: The target pass rate to filter out APIs that have a pass rate higher than this target.
> - `train.parallel`: The number of parallel processes used to validate the constraints. We do not recommend to set this argument to 1.
> - `train.record_path`: The path where the extracted constraints are saved. This directlry should be the same as the `mgen.record_path` in the fuzzing step.
> - `hydra.verbose`: Set the logging level of Hydra for specific modules ("smt", "train", "convert", "constr", "llm", etc). If you want to see all the log messages, you can set it to `True`.
> - `train.num_eval`: The number of evaluations performed to validate the constraints (default: 500).
> - `model.type`: Choose from `["tensorflow", "torch"]`.
> - `backend.type`: Choose from `["xla", "torchjit"]`.
>
> **Other Arguments**:
> For additional details, refer to the values under train at `/DeepConstr/nnsmith/config/main.yaml`.
>
> **Outputs**:
> - `$(pwd)/${train.record_path}/torch` if `model.type` is `torch`
> - `$(pwd)/${train.record_path}/tf` if `model.type` is `tensorflow`


#### Quick Start :

Please set your `train.record_path` to the desired location that you want to store. For instance, `$(pwd)/repro/records/torch`

##### for PyTorch 
Below command will extract constraints from `"torch.add","torch.abs"`. The extracted constraints are stored to `$(pwd)/repro/records/torch`. We recommand to set `train.parallel` to larger than 1. The approximate cost will be $0.40 for extracting constraints from 2 operators.

```bash
PYTHONPATH=/DeepConstr/:/DeepConstr/nnsmith/:/DeepConstr/deepconstr/:$PYTHONPATH \
python deepconstr/train/run.py train.record_path=repro/records/torch backend.type=torchcomp \
model.type=torch hydra.verbose=train train.parallel=1 train.num_eval=500 \
train.pass_rate=95 hydra.verbose=['train'] \
train.retrain=false train.target='["torch.add","torch.abs"]'
```
By specifying the path to a JSON file, you can target a specific set of APIs for processing. This JSON file should contain a list of API names.

```bash
PYTHONPATH=/DeepConstr/:/DeepConstr/nnsmith/:/DeepConstr/deepconstr/:$PYTHONPATH \
python deepconstr/train/run.py train.record_path=repro/records/torch backend.type=torchcomp \
model.type=torch hydra.verbose=train train.parallel=1 train.num_eval=500 \
train.pass_rate=95 hydra.verbose=['train'] \
train.retrain=false train.target='/your/json/path'
```

##### for TensorFlow 

Below command will extract constraints from `"tf.add", "tf.abs"`. The extracted constraints are stored to `$(pwd)/repro/records/tf`. The approximate cost will be less than $0.01 since these operators does not contain complicate constraints.

```bash
PYTHONPATH=/DeepConstr/:/DeepConstr/nnsmith/:/DeepConstr/deepconstr/:$PYTHONPATH \
python deepconstr/train/run.py train.record_path=repro/records/tf backend.type=xla \
model.type=tensorflow hydra.verbose=train train.parallel=1 train.num_eval=300 train.pass_rate=95 hydra.verbose=['train'] \
train.retrain=false train.target='["tf.add","tf.abs"]'
```

##### For NumPy

Below command will extract constraints from `"numpy.add"` and extracted constraints are stored to `$(pwd)/repro/records/numpy`. The approximate cost will be less than $0.01 for running below commands.

```bash
PYTHONPATH=/DeepConstr/:/DeepConstr/nnsmith/:/DeepConstr/deepconstr/:$PYTHONPATH \
python deepconstr/train/run.py train.record_path=repro/records/numpy backend.type=numpy \
model.type=numpy hydra.verbose=train train.parallel=1 train.num_eval=300 train.pass_rate=95 hydra.verbose=['train'] \
train.retrain=false train.target='["numpy.add"]'
```
# Reproduce Experiments

### Comparative Experiment (RQ1) 

#### Check trained operators( table 1)

You can inspect the number of trained APIs by executing the following commands:

```bash 
python experiments/apis_overview.py /DeepConstr/data/records
# Number of trained tf apis:  258
# Number of trained torch apis:  843
```

#### Coverage Comparison Experiment
> [!NOTE]
> To reproduce this experiment, please pull our Docker image.

We have four baselines for conducting experiments. Additionally, approximately 700 operators (programs) require testing for PyTorch and 150 operators for TensorFlow. Given that each operator needs to be tested for 15 minutes, completing the experiment will be time-intensive. To expedite the process, we recommend using the `exp.parallel` argument to enable multiple threads during the experiment(We set this to 16 when running the experiment). The experiment results will be saved in the folder specified by `exp.save_dir`.

##### for PyTorch 

First, change the environment to the conda environment created for this project. We strongly recommend to set `exp.parallel` larger than 1.

```bash
conda activate cov
```

```bash
PYTHONPATH=/DeepConstr/:/DeepConstr/nnsmith/:/DeepConstr/deepconstr/:$PYTHONPATH \
python experiments/evaluate_apis.py \
exp.save_dir=exp/torch mgen.record_path=$(pwd)/data/records/torch/ mgen.pass_rate=0.05 model.type=torch backend.type=torchjit fuzz.time=15m exp.parallel=16 mgen.noise=0.8 exp.targets=/DeepConstr/data/torch_dc_neuri.json exp.baselines="['deepconstr', 'neuri', 'symbolic-cinit', 'deepconstr_2']"
```

<!-- for testing acetest
```bash
PYTHONPATH=/DeepConstr/:/DeepConstr/nnsmith/:/DeepConstr/deepconstr/:$PYTHONPATH python experiments/evaluate_apis.py exp.save_dir= mgen.max_nodes=1 mgen.pass_rate=0.05 model.type=torch backend.type=torchjit fuzz.time=5m exp.parallel=1 mgen.noise=0.8 exp.targets=/DeepConstr/data/tf_dc_acetest.json exp.baselines=['acetest']
``` -->
##### for TensorFlow 

<!-- First, change the environment to the conda environment created for this project.
```bash
conda activate cov
``` -->

```bash
PYTHONPATH=/DeepConstr/:/DeepConstr/nnsmith/:/DeepConstr/deepconstr/:$PYTHONPATH \
python experiments/evaluate_apis.py \
exp.save_dir=exp/tf mgen.record_path=$(pwd)/data/records/tf/ mgen.pass_rate=0.05 model.type=tensorflow backend.type=xla fuzz.time=15m exp.parallel=16 mgen.noise=0.8 exp.targets=/DeepConstr/data/tf_dc_neuri.json exp.baselines="['deepconstr', 'neuri', 'symbolic-cinit', 'deepconstr_2']"
```

<!-- for testing acetest
```bash
PYTHONPATH=/DeepConstr/:/DeepConstr/nnsmith/:/DeepConstr/deepconstr/:$PYTHONPATH python experiments/evaluate_apis.py exp.save_dir=exp/aceteonstr_1/tf mgen.max_nodes=1 mgen.records/tf mgen.max_nodes=1 mgen.record=onstr_1/tf mgen.max_nodes=1 mgen.record_path=$(pwd)/data/records/tf/ mgen.pass_rate=0.05 model.type=tensorflow backend.type=xla fuzz.time=5m exp.parallel=64 mgen.noise=0.8 exp.targets=/DeepConstr/data/tf_dc_acetest.json exp.baselines=['acetest']
``` -->


##### Summarize the results

Specify the folder name that you used in a previous experiment. Use the -o option to name the output file. The final experiment results will be saved in the path that is specified through -o.

For example, to specify a folder named pt_gen and save the results to pt_gen.csv, use the following command:
```bash
python experiments/summarize_merged_cov.py -f exp/torch -o torch_exp -p deepconstr -k torch
# Result will be saved at /DeepConstr/results/torch_exp.csv
python experiments/summarize_merged_cov.py -f exp/tf -o tf_exp -p deepconstr -k tf
# Result will be saved at /DeepConstr/results/tf_exp.csv
```

##### When encounters with unnormal values 

Occasionally, you may encounter abnormal coverage values, such as 0. In such cases, please refer to the list of abnormal values saved at `$(pwd)/results/unnormal_val*`. To address these issues, re-run the experiment with the following adjustments to your arguments: `mode=fix exp.targets=$(pwd)/results/unnormal_val*`.

```bash
PYTHONPATH=/DeepConstr/:/DeepConstr/nnsmith/:/DeepConstr/deepconstr/:$PYTHONPATH \
python experiments/evaluate_apis.py \
exp.save_dir=pt_gen mgen.record_path=$(pwd)/data/records/torch/ mgen.pass_rate=0 model.type=torch backend.type=torchjit fuzz.time=15m exp.parallel=1 mgen.noise=0.8 exp.targets=/DeepConstr/results/unnormal_val_deepconstr_torch.json
```

### Constraint Assessment (RQ2) 

You can review the overall scores of constraints by executing the following script:
You can look into the overall scores of constraints by running below scripts.
```bash
python experiments/eval_constr.py
#######  torch  ####### 
#DeepConstr
## Num of Sub Constraints :  7072 from 929 number of operators
## Mean :  7.61248654467169  Median  6
#DeepConstr^s
## Num of Sub Constraints :  7540 from 855 number of operators
## Mean :  8.818713450292398  Median  7
#...
```
This script will automatically gather the constraints from the  default locations(`/DeepConstr/data/records/`). The resulting plots will be saved at`/DeepConstr/results/5_dist_tf.png` for TensorFlow and `/DeepConstr/results/5_dist_torch.png` for PyTorch.



### 优化实践：LP‑CCR 开关与伪代码（对比可复现）

> 目标：在不大改现有代码的前提下，加入 LP‑CCR（PAC‑Bayes 先验引导的补集约束精炼）相关开关与打分逻辑，便于与原 DeepConstr 做 A/B 对比与消融。

#### 1. 统一开关与参数（示例命名，建议通过 Hydra 配置）

- `lpccr.enable`：是否启用 LP‑CCR（默认 false）。
- `lpccr.lambda`：KL 惩罚系数 λ（默认 0.3～1.0 之间根据时间预算调节）。
- `lpccr.prior`：先验来源（`off` | `template-llm`）。
- `lpccr.active`：是否启用主动补集采样（`true|false`）。
- `lpccr.acq`：主动采样策略（`disagreement|entropy|random`）。
- `lpccr.noise_fix`：是否启用“静态传播+LLM 反事实”降噪对齐（`true|false`）。
- `lpccr.prop_test.m`：元素级性质测试的抽样大小 m（如 64）。
- `lpccr.prop_test.gamma`：性质测试的置信间隔宽度 γ（如 0.05）。
- `lpccr.prop_test.tau`：可接受的违例阈值 τ（常取 0 或极小正数）。

以上开关可置于现有 Hydra 配置中，例如：`hydra.verbose=['fuzz','train'] lpccr.enable=true lpccr.lambda=0.5 ...`。

#### 2. 评分函数（替换/扩展子约束的适配度打分）

```python
def compute_constraint_score(
    candidate_h,
    validation_samples,
    prior_score: float,  # 由先验（LLM 模板一致度等）映射到 [0,1]
    lambda_: float,
    lpccr_enabled: bool,
):
    # 经验无效率 R_hat：候选约束对验证样本判为 invalid 的比例
    invalid_cnt = 0
    for x in validation_samples:
        if not candidate_h.validate(x):
            invalid_cnt += 1
    R_hat = invalid_cnt / max(1, len(validation_samples))

    if not lpccr_enabled:
        return R_hat  # 兼容原逻辑：仅用经验无效率

    # KL 代理项：以先验一致度作为负惩罚，越一致惩罚越小
    # 将 prior_score∈[0,1] 转成 KL_proxy≥0，可采用：KL_proxy = 1 - prior_score
    KL_proxy = max(0.0, 1.0 - float(prior_score))

    # 目标：R_hat + lambda * KL_proxy
    return R_hat + lambda_ * KL_proxy
```

最小改动：在 `deepconstr/train/*` 子约束搜索/遗传迭代中的“适配度/排序”处调用 `compute_constraint_score`，当 `lpccr.enable=false` 时与原结果保持一致。

先验分数 `prior_score` 的一个简单实现：

```python
def llm_prior_score(api_doc: str, err_msg: str, sig: str, template: str) -> float:
    """返回与模板的一致度分数 [0,1]，可用 LLM/规则打分、描述长度惩罚或关键词匹配实现。
    示例：更简单、包含广播/形状/类型关键谓词的模板得分更高。"""
    # 伪代码占位：工程中可缓存、降采样调用 LLM
    return heuristic_score(api_doc, err_msg, sig, template)
```

#### 3. 主动补集采样（集成于 fuzz 循环）

```python
def select_next_candidates(candidates, committee_of_h, acq: str):
    # candidates：补集生成的输入候选；committee_of_h：多个候选约束形成的“委员会”
    def disagreement(x):
        votes = [h.validate(x) for h in committee_of_h]
        p = sum(votes) / max(1, len(votes))
        return p * (1 - p)  # 方差作为不确定性

    def entropy(x):
        import math
        p = sum(h.validate(x) for h in committee_of_h) / max(1, len(committee_of_h))
        if p in (0, 1):
            return 0.0
        return -p * math.log(p + 1e-12) - (1 - p) * math.log(1 - p + 1e-12)

    score_fn = disagreement if acq == 'disagreement' else (entropy if acq == 'entropy' else (lambda _: 0.0))
    return sorted(candidates, key=score_fn, reverse=True)
```

将上述函数用于 `nnsmith/cli/fuzz.py` 的“候选样本队列”排序；当 `lpccr.active=false` 时跳过该排序，保持原有顺序。

#### 4. 降噪对齐（错误信息 → 根因）

```python
def align_error_label(sample, raw_error_msg):
    # 先行静态传播（形状/类型）定位高疑似根因字段
    cause = static_infer_shape_type(sample)
    # LLM 生成反事实最小修复（不改 dtype 时能否通过？不改 shape 时能否通过？）
    cf = llm_counterfactual_fix(sample, raw_error_msg, cause)
    # 回填到“子约束标签”（例如将问题从 dtype 对齐到 broadcast 形状）
    return corrected_label_from(cf)
```

将其作为 `deepconstr/error.py` 的前置步骤，`lpccr.noise_fix=false` 时跳过。

#### 5. 元素级性质测试（轻量概率保证）

```python
def property_test_elements(tensor, prop, m: int, gamma: float, tau: float) -> bool:
    # 随机采样 m 个元素，估计违例率 p_hat
    elems = random_sample_elements(tensor, m)
    violations = sum(1 for v in elems if not prop(v))
    p_hat = violations / max(1, m)
    # 判定：若 p_hat > tau - gamma 则拒绝
    return p_hat <= max(0.0, tau - gamma)
```

在 fuzz 后验校验处调用；当 `lpccr.prop_test.m=0` 或不开启相关断言时直接跳过。

#### 6. 最小改动的函数签名（便于落地对接）

- 训练/精炼（`deepconstr/train/...`）
  - `compute_constraint_score(candidate_h, validation_samples, prior_score, lambda_, lpccr_enabled) -> float`
  - `llm_prior_score(api_doc, err_msg, sig, template) -> float`

- 采样/评估（`nnsmith/cli/fuzz.py` 或 `mgen.*`）
  - `select_next_candidates(candidates, committee_of_h, acq: str) -> List[x]`
  - `property_test_elements(tensor, prop, m, gamma, tau) -> bool`
  - `align_error_label(sample, raw_error_msg) -> CorrectedLabel`

#### 7. 调用示例（对比与消融）

以下命令仅演示新增开关，其他参数与 README 示例保持一致。

```bash
# Baseline（原 DeepConstr）
PYTHONPATH=$(pwd):$(pwd)/deepconstr:$(pwd)/nnsmith \
python nnsmith/cli/fuzz.py \
  fuzz.time=15m \
  mgen.record_path=$(pwd)/data/records/torch \
  fuzz.root=$(pwd)/outputs/torch-deepconstr-n5 \
  fuzz.save_test=$(pwd)/outputs/torch-deepconstr-n5.models \
  model.type=torch backend.type=torchcomp \
  filter.type=[nan,dup,inf] hydra.verbose=['fuzz'] fuzz.resume=true \
  mgen.method=deepconstr mgen.max_nodes=5 mgen.pass_rate=10

# LP‑CCR（开启所有子模块）
PYTHONPATH=$(pwd):$(pwd)/deepconstr:$(pwd)/nnsmith \
python nnsmith/cli/fuzz.py \
  fuzz.time=15m \
  mgen.record_path=$(pwd)/data/records/torch \
  fuzz.root=$(pwd)/outputs/torch-lpccr-n5 \
  fuzz.save_test=$(pwd)/outputs/torch-lpccr-n5.models \
  model.type=torch backend.type=torchcomp \
  filter.type=[nan,dup,inf] hydra.verbose=['fuzz'] fuzz.resume=true \
  mgen.method=deepconstr mgen.max_nodes=5 mgen.pass_rate=10 \
  lpccr.enable=true lpccr.lambda=0.5 lpccr.prior=template-llm \
  lpccr.active=true lpccr.acq=disagreement \
  lpccr.noise_fix=true \
  lpccr.prop_test.m=64 lpccr.prop_test.gamma=0.05 lpccr.prop_test.tau=0.0

# 消融示例：去掉先验
PYTHONPATH=$(pwd):$(pwd)/deepconstr:$(pwd)/nnsmith \
python nnsmith/cli/fuzz.py ... lpccr.enable=true lpccr.prior=off lpccr.active=true

# 消融示例：去掉主动采样
PYTHONPATH=$(pwd):$(pwd)/deepconstr:$(pwd)/nnsmith \
python nnsmith/cli/fuzz.py ... lpccr.enable=true lpccr.prior=template-llm lpccr.active=false
```

建议记录随机种子并使用 `experiments/summarize_merged_cov.py` 汇总覆盖与缺陷数据，以确保统计意义和可复现性。
