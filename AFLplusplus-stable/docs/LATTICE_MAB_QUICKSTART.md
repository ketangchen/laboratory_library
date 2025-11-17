# Lattice-MAB 快速开始指南

## 一、编译和运行

### 1. 编译 AFL++

```bash
cd /Users/ketangchen/Documents/codeTry/AFLplusplus-stable
make clean
make
```

### 2. 基本运行（Lattice-MAB 默认启用）

```bash
# 编译目标程序（使用 afl-cc）
CC=afl-cc CXX=afl-c++ ./configure
make

# 运行 fuzzer（Lattice-MAB 自动启用）
afl-fuzz -i seeds/ -o output/ -- ./target @@
```

### 3. 禁用 Lattice-MAB（用于对比）

```bash
# 使用环境变量禁用
AFL_DISABLE_LATTICE_MAB=1 afl-fuzz -i seeds/ -o output_original/ -- ./target @@
```

## 二、查看效果

### 在 UI 中查看统计信息

运行 fuzzer 时，在 UI 界面中会显示 Lattice-MAB 统计信息：

```
lattice-mab  : MAB:12345 Lattice:3456 Hybrid:8901 avg_reward:12.34 best:5
```

### 查看 fuzzer_stats 文件

```bash
# 查看关键指标
cat output/default/fuzzer_stats | grep -E "execs_done|paths_total|unique_crashes|execs_per_sec"
```

## 三、自动化对比测试

### 使用对比脚本（推荐）

```bash
# 运行自动化对比测试
./utils/compare_lattice_mab.sh -t ./target -i ./seeds -T 600 -n 3

# 参数说明：
# -t: 目标二进制文件
# -i: 输入种子目录
# -T: 每个测试运行时间（秒），默认300秒
# -n: 每个配置运行次数，默认3次
```

脚本会自动：
1. 运行启用 Lattice-MAB 的测试（N次）
2. 运行禁用 Lattice-MAB 的测试（N次）
3. 计算平均值和提升百分比
4. 生成对比报告

### 手动对比测试

#### 步骤 1: 准备测试环境

```bash
mkdir -p comparison_test
cd comparison_test

# 准备目标程序和种子
# 假设目标程序是 test_target，种子在 seeds/ 目录
```

#### 步骤 2: 运行两个测试

**终端 1 - 启用 Lattice-MAB:**
```bash
afl-fuzz -i seeds -o output_with_lm -- ./test_target @@
```

**终端 2 - 禁用 Lattice-MAB:**
```bash
AFL_DISABLE_LATTICE_MAB=1 afl-fuzz -i seeds -o output_without_lm -- ./test_target @@
```

运行相同时间后（建议至少10分钟），按 Ctrl+C 停止两个测试。

#### 步骤 3: 提取和对比指标

```bash
# 创建对比脚本
cat > compare.sh << 'EOF'
#!/bin/bash

echo "=== 对比结果 ==="
echo ""

# 提取指标函数
extract() {
    grep "^$1" "$2" | awk '{print $3}'
}

with="output_with_lm/default/fuzzer_stats"
without="output_without_lm/default/fuzzer_stats"

execs_with=$(extract "execs_done" "$with")
execs_without=$(extract "execs_done" "$without")
paths_with=$(extract "paths_total" "$with")
paths_without=$(extract "paths_total" "$without")
crashes_with=$(extract "unique_crashes" "$with")
crashes_without=$(extract "unique_crashes" "$without")
speed_with=$(extract "execs_per_sec" "$with")
speed_without=$(extract "execs_per_sec" "$without")

echo "指标对比:"
echo "  执行次数: $execs_with (启用) vs $execs_without (禁用)"
echo "  发现路径: $paths_with (启用) vs $paths_without (禁用)"
echo "  发现崩溃: $crashes_with (启用) vs $crashes_without (禁用)"
echo "  执行速度: $speed_with (启用) vs $speed_without (禁用)"
echo ""

# 计算提升百分比
exec_imp=$(echo "scale=2; ($execs_with - $execs_without) * 100 / $execs_without" | bc)
paths_imp=$(echo "scale=2; ($paths_with - $paths_without) * 100 / $paths_without" | bc)
crashes_imp=$(echo "scale=2; ($crashes_with - $crashes_without) * 100 / (${crashes_without:-1})" | bc)
speed_imp=$(echo "scale=2; ($speed_with - $speed_without) * 100 / $speed_without" | bc)

echo "提升百分比:"
echo "  执行次数: ${exec_imp}%"
echo "  路径发现: ${paths_imp}%"
echo "  崩溃发现: ${crashes_imp}%"
echo "  执行速度: ${speed_imp}%"
EOF

chmod +x compare.sh
./compare.sh
```

## 四、关键指标说明

### 1. 执行次数 (execs_done)
- **含义**: 总执行次数
- **重要性**: 反映 fuzzer 的整体工作量
- **预期**: Lattice-MAB 可能提升 2-5%

### 2. 发现路径数 (paths_total)
- **含义**: 发现的唯一路径数量
- **重要性**: 反映代码覆盖率
- **预期**: Lattice-MAB 可能提升 5-15%

### 3. 崩溃数 (unique_crashes)
- **含义**: 发现的唯一崩溃数量
- **重要性**: 反映漏洞发现能力
- **预期**: Lattice-MAB 可能提升 10-20%

### 4. 执行速度 (execs_per_sec)
- **含义**: 每秒执行次数
- **重要性**: 反映 fuzzer 效率
- **预期**: Lattice-MAB 可能提升 2-5%（通过减少无效变异）

## 五、示例测试

### 使用 test-instr.c 进行快速测试

```bash
# 1. 编译测试程序
afl-cc -o test-instr test-instr.c

# 2. 准备种子
mkdir -p seeds
echo "hello" > seeds/seed1.txt

# 3. 运行对比测试（快速测试，5分钟）
./utils/compare_lattice_mab.sh -t ./test-instr -i ./seeds -T 300 -n 2

# 4. 查看结果
cat comparison_results/comparison_*/summary.txt
```

## 六、结果分析

### 查看详细报告

对比脚本会在 `comparison_results/comparison_<timestamp>/` 目录下生成：

- `summary.txt`: 文本格式的对比报告
- `results.json`: JSON 格式的详细数据
- `with_lattice_mab_run*/`: 启用 Lattice-MAB 的每次运行结果
- `without_lattice_mab_run*/`: 禁用 Lattice-MAB 的每次运行结果

### 解读结果

**正提升（+）**: 表示 Lattice-MAB 在该指标上表现更好
**负提升（-）**: 表示原始策略在该指标上表现更好

**注意**: 
- 模糊测试具有随机性，建议多次运行取平均值
- 不同目标程序可能表现不同
- 初期可能需要学习时间，建议运行至少 10-30 分钟

## 七、常见问题

### Q: 看不到 Lattice-MAB 统计信息？
A: 检查是否被禁用：`unset AFL_DISABLE_LATTICE_MAB`

### Q: 编译错误？
A: 确保安装了数学库：`sudo apt-get install libc6-dev` (Ubuntu/Debian)

### Q: 性能没有提升？
A: 
- 确保运行时间足够长（>30分钟）
- 检查种子文件质量
- 不同目标程序表现可能不同

### Q: 如何调整 Lattice-MAB 参数？
A: 目前参数在 `include/afl-lattice-mab.h` 中定义，可以修改：
- `MAB_EXPLORATION_C`: 探索常数（默认 2.0）
- `MAB_ALPHA`: 学习率（默认 0.1）
- 混合比例在 `src/afl-fuzz-lattice-mab.c` 中（默认 70% MAB, 30% Lattice）

## 八、下一步

- 查看详细文档: `docs/LATTICE_MAB.md`
- 查看使用指南: `docs/LATTICE_MAB_USAGE.md`
- 查看源代码: `src/afl-fuzz-lattice-mab.c`

