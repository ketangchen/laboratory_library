# Lattice-MAB 使用指南和对比测试

## 快速开始

### 1. 编译 AFL++

```bash
cd /Users/ketangchen/Documents/codeTry/AFLplusplus-stable
make
```

编译完成后，`afl-fuzz` 将自动包含 Lattice-MAB 功能。

### 2. 基本使用

Lattice-MAB 默认启用，无需额外配置：

```bash
# 正常使用，Lattice-MAB 自动启用
afl-fuzz -i input_dir -o output_dir -- ./target @@
```

### 3. 禁用 Lattice-MAB（用于对比）

如果需要使用原始策略进行对比，可以通过环境变量禁用：

```bash
# 禁用 Lattice-MAB，使用原始随机选择策略
AFL_DISABLE_LATTICE_MAB=1 afl-fuzz -i input_dir -o output_dir -- ./target @@
```

## 查看统计信息

在 fuzzer 运行过程中，UI 界面会显示 Lattice-MAB 的统计信息：

```
lattice-mab  : MAB:12345 Lattice:3456 Hybrid:8901 avg_reward:12.34 best:5
```

其中：
- `MAB`: MAB 算法选择的次数
- `Lattice`: 格理论选择的次数
- `Hybrid`: 混合模式选择的次数
- `avg_reward`: 平均奖励值
- `best`: 最佳变异类型ID

## 对比测试

### 使用对比测试脚本

我们提供了一个自动化对比测试脚本：

```bash
./utils/compare_lattice_mab.sh -t ./target -i ./seeds -T 600 -n 3
```

参数说明：
- `-t, --target`: 目标二进制文件路径（必需）
- `-i, --input`: 输入种子目录（必需）
- `-o, --output`: 输出基础目录（默认: ./comparison_results）
- `-T, --time`: 每个测试的运行时间（秒，默认: 300）
- `-n, --runs`: 每个配置的运行次数（默认: 3）

### 手动对比测试

#### 步骤 1: 准备测试环境

```bash
# 创建测试目录
mkdir -p comparison_test
cd comparison_test

# 准备目标程序和种子
# 假设目标程序是 test_target，种子在 seeds/ 目录
```

#### 步骤 2: 运行启用 Lattice-MAB 的测试

```bash
# 测试 1: 启用 Lattice-MAB（默认）
afl-fuzz -i seeds -o output_with_lm -- ./test_target @@

# 运行一段时间后（例如 10 分钟），按 Ctrl+C 停止
```

#### 步骤 3: 运行禁用 Lattice-MAB 的测试

```bash
# 测试 2: 禁用 Lattice-MAB
AFL_DISABLE_LATTICE_MAB=1 afl-fuzz -i seeds -o output_without_lm -- ./test_target @@

# 运行相同时间后停止
```

#### 步骤 4: 对比结果

查看 `fuzzer_stats` 文件中的关键指标：

```bash
# 启用 Lattice-MAB 的统计
cat output_with_lm/default/fuzzer_stats | grep -E "execs_done|paths_total|unique_crashes|execs_per_sec"

# 禁用 Lattice-MAB 的统计
cat output_without_lm/default/fuzzer_stats | grep -E "execs_done|paths_total|unique_crashes|execs_per_sec"
```

## 关键指标对比

### 1. 执行次数 (execs_done)

**含义**: 总执行次数，反映 fuzzer 的整体工作量

**对比方法**:
```bash
execs_with=$(grep "^execs_done" output_with_lm/default/fuzzer_stats | awk '{print $3}')
execs_without=$(grep "^execs_done" output_without_lm/default/fuzzer_stats | awk '{print $3}')
improvement=$(echo "scale=2; ($execs_with - $execs_without) * 100 / $execs_without" | bc)
echo "执行次数提升: ${improvement}%"
```

### 2. 发现路径数 (paths_total)

**含义**: 发现的唯一路径数量，反映覆盖率提升

**对比方法**:
```bash
paths_with=$(grep "^paths_total" output_with_lm/default/fuzzer_stats | awk '{print $3}')
paths_without=$(grep "^paths_total" output_without_lm/default/fuzzer_stats | awk '{print $3}')
improvement=$(echo "scale=2; ($paths_with - $paths_without) * 100 / $paths_without" | bc)
echo "路径发现提升: ${improvement}%"
```

### 3. 崩溃数 (unique_crashes)

**含义**: 发现的唯一崩溃数量，反映漏洞发现能力

**对比方法**:
```bash
crashes_with=$(grep "^unique_crashes" output_with_lm/default/fuzzer_stats | awk '{print $3}')
crashes_without=$(grep "^unique_crashes" output_without_lm/default/fuzzer_stats | awk '{print $3}')
improvement=$(echo "scale=2; ($crashes_with - $crashes_without) * 100 / (${crashes_without:-1})" | bc)
echo "崩溃发现提升: ${improvement}%"
```

### 4. 执行速度 (execs_per_sec)

**含义**: 每秒执行次数，反映 fuzzer 的效率

**对比方法**:
```bash
speed_with=$(grep "^execs_per_sec" output_with_lm/default/fuzzer_stats | awk '{print $3}')
speed_without=$(grep "^execs_per_sec" output_without_lm/default/fuzzer_stats | awk '{print $3}')
improvement=$(echo "scale=2; ($speed_with - $speed_without) * 100 / $speed_without" | bc)
echo "执行速度提升: ${improvement}%"
```

### 5. 覆盖率 (plot_data)

**含义**: 代码覆盖率，可以从 plot_data 文件中提取

**对比方法**:
```bash
# 提取最后一条记录的覆盖率
coverage_with=$(tail -1 output_with_lm/default/plot_data | awk '{print $4}')
coverage_without=$(tail -1 output_without_lm/default/plot_data | awk '{print $4}')
improvement=$(echo "scale=2; ($coverage_with - $coverage_without) * 100 / $coverage_without" | bc)
echo "覆盖率提升: ${improvement}%"
```

## 完整对比脚本示例

```bash
#!/bin/bash

TARGET="./test_target"
SEEDS="./seeds"
TEST_TIME=600  # 10分钟

echo "=== 对比测试开始 ==="

# 测试 1: 启用 Lattice-MAB
echo "测试 1: 启用 Lattice-MAB..."
timeout $TEST_TIME afl-fuzz -i $SEEDS -o output_with_lm -- $TARGET @@ || true

# 测试 2: 禁用 Lattice-MAB
echo "测试 2: 禁用 Lattice-MAB..."
AFL_DISABLE_LATTICE_MAB=1 timeout $TEST_TIME afl-fuzz -i $SEEDS -o output_without_lm -- $TARGET @@ || true

# 提取并对比指标
echo ""
echo "=== 对比结果 ==="

extract_stat() {
    local file=$1
    local key=$2
    grep "^$key" "$file" | awk '{print $3}'
}

with_stats="output_with_lm/default/fuzzer_stats"
without_stats="output_without_lm/default/fuzzer_stats"

execs_with=$(extract_stat "$with_stats" "execs_done")
execs_without=$(extract_stat "$without_stats" "execs_done")
paths_with=$(extract_stat "$with_stats" "paths_total")
paths_without=$(extract_stat "$without_stats" "paths_total")
crashes_with=$(extract_stat "$with_stats" "unique_crashes")
crashes_without=$(extract_stat "$without_stats" "unique_crashes")
speed_with=$(extract_stat "$with_stats" "execs_per_sec")
speed_without=$(extract_stat "$without_stats" "execs_per_sec")

echo "执行次数: $execs_with (启用) vs $execs_without (禁用)"
echo "发现路径: $paths_with (启用) vs $paths_without (禁用)"
echo "发现崩溃: $crashes_with (启用) vs $crashes_without (禁用)"
echo "执行速度: $speed_with (启用) vs $speed_without (禁用)"
```

## 分析结果

### 预期改进

Lattice-MAB 优化预期在以下方面带来改进：

1. **路径发现率**: 通过智能选择变异策略，可能提高 5-15%
2. **崩溃发现**: 通过优化策略选择，可能提高 10-20%
3. **执行效率**: 通过减少无效变异，可能提高 2-5%

### 注意事项

1. **测试时间**: 建议至少运行 10-30 分钟以获得稳定结果
2. **多次运行**: 建议每个配置运行 3-5 次取平均值
3. **目标程序**: 不同目标程序可能表现不同
4. **随机性**: 模糊测试具有随机性，结果可能有波动

## 故障排除

### 问题: 看不到 Lattice-MAB 统计信息

**解决**: 检查是否被禁用：
```bash
# 确保环境变量未设置
unset AFL_DISABLE_LATTICE_MAB
```

### 问题: 编译错误

**解决**: 确保所有依赖已安装：
```bash
# 检查数学库链接
make clean
make
```

### 问题: 性能下降

**解决**: Lattice-MAB 在初期可能需要学习时间，建议：
- 运行更长时间（>30分钟）
- 检查是否有足够的种子文件
- 查看日志确认系统正常工作

## 进一步分析

### 查看详细日志

```bash
# 查看 fuzzer 日志
cat output_dir/default/fuzzer.log

# 查看 plot 数据（可用于绘制图表）
cat output_dir/default/plot_data
```

### 绘制对比图表

可以使用 `afl-plot` 工具生成可视化图表：

```bash
afl-plot output_with_lm/default plot_with_lm
afl-plot output_without_lm/default plot_without_lm
```

然后在浏览器中打开生成的 HTML 文件进行对比。

