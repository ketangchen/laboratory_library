# laboratory_library
laboratory_library


基于对代码库的分析，这是一个名为 **DeepConstr** 的深度学习库测试工具。总结如何运行这个代码：

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