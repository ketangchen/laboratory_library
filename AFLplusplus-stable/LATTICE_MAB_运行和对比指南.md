# Lattice-MAB 运行和对比测试完整指南

本指南详细介绍如何运行 Lattice-MAB 并与 AFL++ 原始变异选择策略进行对比测试。

## 目录
1. [环境准备](#环境准备)
2. [测试程序准备](#测试程序准备)
3. [运行 Lattice-MAB](#运行-lattice-mab)
4. [对比测试方法](#对比测试方法)
5. [结果分析](#结果分析)
6. [完整示例](#完整示例)

---

## 环境准备

### 1. 确认工作目录

```bash
# AFL++ 项目根目录
WORKSPACE="/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable"
cd "$WORKSPACE"
```

### 2. 编译 AFL++

```bash
cd "/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable"
make clean
make
```

编译完成后，`afl-fuzz` 将自动包含 Lattice-MAB 功能。

### 3. 验证编译结果

```bash
# 检查 afl-fuzz 是否存在
ls -lh "./afl-fuzz"

# 检查 afl-cc 是否存在
ls -lh "./afl-cc"
```

---

## 测试程序准备

### 方法 1: 使用内置测试程序（推荐用于快速测试）

**测试程序源代码路径：**
```
/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable/test-instr.c
```

**编译命令：**
```bash
cd "/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable"
./afl-cc -o test-instr test-instr.c
```

**准备种子文件：**
```bash
mkdir -p seeds
echo "0" > seeds/seed1.txt
echo "1" > seeds/seed2.txt
echo "hello" > seeds/seed3.txt
```

**文件路径总结：**
- 测试程序: `/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable/test-instr`
- 种子目录: `/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable/seeds`

### 方法 2: 创建带崩溃的测试程序

**创建测试程序：**
```bash
cd "/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable"

cat > test_crash.c << 'EOF'
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    char buf[100];
    
    if (argc < 2) {
        if (fread(buf, 1, 100, stdin) < 1) {
            return 1;
        }
    } else {
        FILE *f = fopen(argv[1], "r");
        if (!f) return 1;
        fread(buf, 1, 100, f);
        fclose(f);
    }
    
    // 触发崩溃的条件
    if (strncmp(buf, "CRASH", 5) == 0) {
        *(volatile int*)0 = 0;  // 段错误
    }
    
    // 其他路径
    if (strncmp(buf, "TEST", 4) == 0) {
        printf("Found test path\n");
    }
    
    return 0;
}
EOF

# 编译
./afl-cc -o test_crash test_crash.c

# 准备种子
mkdir -p seeds
echo "hello" > seeds/seed1.txt
echo "world" > seeds/seed2.txt
```

**文件路径：**
- 测试程序: `/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable/test_crash`

---

## 运行 Lattice-MAB

### 基本运行（Lattice-MAB 默认启用）

**命令：**
```bash
cd "/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable"
./afl-fuzz -i seeds -o output_with_lm -- ./test-instr @@
```

**完整路径命令：**
```bash
"/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable/afl-fuzz" \
    -i "/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable/seeds" \
    -o "/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable/output_with_lm" \
    -- "/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable/test-instr" @@
```

### 查看 Lattice-MAB 统计信息

在 fuzzer 运行过程中，UI 界面会显示 Lattice-MAB 统计信息：

```
lattice-mab  : MAB:12345 Lattice:3456 Hybrid:8901 avg_reward:12.34 best:5
```

其中：
- `MAB`: MAB 算法选择的次数
- `Lattice`: 格理论选择的次数
- `Hybrid`: 混合模式选择的次数
- `avg_reward`: 平均奖励值
- `best`: 最佳变异类型ID

---

## 对比测试方法

### 方法 1: 自动化对比测试（推荐）

**使用对比脚本：**

```bash
cd "/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable"

# 确保脚本有执行权限
chmod +x utils/compare_lattice_mab.sh

# 运行自动化对比测试
./utils/compare_lattice_mab.sh -t ./test-instr -i ./seeds -T 600 -n 3
```

**完整路径命令：**
```bash
"/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable/utils/compare_lattice_mab.sh" \
    -t "/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable/test-instr" \
    -i "/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable/seeds" \
    -T 600 \
    -n 3 \
    -o "./comparison_results"
```

**参数说明：**
- `-t, --target`: 目标二进制文件路径（必需）
- `-i, --input`: 输入种子目录（必需）
- `-T, --time`: 每个测试的运行时间（秒），默认 300 秒（5分钟）
- `-n, --runs`: 每个配置的运行次数，默认 3 次
- `-o, --output`: 输出基础目录，默认 `./comparison_results`

**脚本功能：**
1. 自动运行启用 Lattice-MAB 的测试 N 次
2. 自动运行禁用 Lattice-MAB 的测试 N 次
3. 计算平均值和提升百分比
4. 生成对比报告（文本和 JSON 格式）

**查看结果：**
```bash
# 查看最新对比结果目录
ls -lt comparison_results/ | head -2

# 查看摘要报告
cat comparison_results/comparison_*/summary.txt

# 查看 JSON 格式的详细数据
cat comparison_results/comparison_*/results.json | python3 -m json.tool
```

### 方法 2: 手动对比测试

#### 步骤 1: 创建测试目录

```bash
cd "/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable"
mkdir -p comparison_test
cd comparison_test

# 复制测试程序
cp ../test-instr ./

# 准备种子目录
mkdir -p seeds
echo "0" > seeds/seed1.txt
echo "1" > seeds/seed2.txt
echo "hello" > seeds/seed3.txt
```

#### 步骤 2: 运行启用 Lattice-MAB 的测试

**终端 1 - 启用 Lattice-MAB（默认）：**

```bash
cd "/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable/comparison_test"

# 确保未设置禁用环境变量
unset AFL_DISABLE_LATTICE_MAB

# 运行 fuzzer（Lattice-MAB 自动启用）
../afl-fuzz -i seeds -o output_with_lm -- ./test-instr @@
```

**完整路径命令：**
```bash
cd "/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable/comparison_test"
"/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable/afl-fuzz" \
    -i "/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable/comparison_test/seeds" \
    -o "/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable/comparison_test/output_with_lm" \
    -- "/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable/comparison_test/test-instr" @@
```

#### 步骤 3: 运行禁用 Lattice-MAB 的测试

**终端 2 - 禁用 Lattice-MAB（原始策略）：**

```bash
cd "/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable/comparison_test"

# 设置环境变量禁用 Lattice-MAB
export AFL_DISABLE_LATTICE_MAB=1

# 运行 fuzzer（使用原始策略）
../afl-fuzz -i seeds -o output_without_lm -- ./test-instr @@
```

**完整路径命令：**
```bash
cd "/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable/comparison_test"
AFL_DISABLE_LATTICE_MAB=1 "/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable/afl-fuzz" \
    -i "/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable/comparison_test/seeds" \
    -o "/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable/comparison_test/output_without_lm" \
    -- "/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable/comparison_test/test-instr" @@
```

#### 步骤 4: 运行相同时间后停止

建议运行至少 **10-30 分钟** 以获得有意义的结果。在两个终端中按 `Ctrl+C` 停止测试。

#### 步骤 5: 提取和对比指标

**创建对比脚本：**

```bash
cd "/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable/comparison_test"

cat > compare_results.sh << 'EOF'
#!/bin/bash

echo "=========================================="
echo "Lattice-MAB 对比测试结果"
echo "=========================================="
echo ""

# 提取指标函数
extract() {
    local file=$1
    local key=$2
    if [ -f "$file" ]; then
        grep "^$key" "$file" | awk '{print $3}' || echo "0"
    else
        echo "0"
    fi
}

with_stats="output_with_lm/default/fuzzer_stats"
without_stats="output_without_lm/default/fuzzer_stats"

if [ ! -f "$with_stats" ] || [ ! -f "$without_stats" ]; then
    echo "错误: 找不到 fuzzer_stats 文件"
    echo "请确保两个测试都已运行完成"
    exit 1
fi

# 提取关键指标
execs_with=$(extract "$with_stats" "execs_done")
execs_without=$(extract "$without_stats" "execs_done")
paths_with=$(extract "$with_stats" "paths_total")
paths_without=$(extract "$without_stats" "paths_total")
crashes_with=$(extract "$with_stats" "unique_crashes")
crashes_without=$(extract "$without_stats" "unique_crashes")
hangs_with=$(extract "$with_stats" "unique_hangs")
hangs_without=$(extract "$without_stats" "unique_hangs")
speed_with=$(extract "$with_stats" "execs_per_sec")
speed_without=$(extract "$without_stats" "execs_per_sec")
cycles_with=$(extract "$with_stats" "cycles_done")
cycles_without=$(extract "$without_stats" "cycles_done")

echo "指标对比:"
echo "----------------------------------------"
printf "%-25s %15s %15s\n" "指标" "启用Lattice-MAB" "禁用Lattice-MAB"
echo "----------------------------------------"
printf "%-25s %15s %15s\n" "总执行次数" "$execs_with" "$execs_without"
printf "%-25s %15s %15s\n" "发现路径数" "$paths_with" "$paths_without"
printf "%-25s %15s %15s\n" "发现崩溃数" "$crashes_with" "$crashes_without"
printf "%-25s %15s %15s\n" "发现挂起数" "$hangs_with" "$hangs_without"
printf "%-25s %15.2f %15.2f\n" "执行速度(execs/s)" "$speed_with" "$speed_without"
printf "%-25s %15s %15s\n" "完成周期数" "$cycles_with" "$cycles_without"
echo ""

# 计算提升百分比
echo "提升百分比:"
echo "----------------------------------------"

if [ "$execs_without" != "0" ]; then
    exec_imp=$(echo "scale=2; ($execs_with - $execs_without) * 100 / $execs_without" | bc)
    printf "%-25s %15.2f%%\n" "执行次数" "$exec_imp"
fi

if [ "$paths_without" != "0" ]; then
    paths_imp=$(echo "scale=2; ($paths_with - $paths_without) * 100 / $paths_without" | bc)
    printf "%-25s %15.2f%%\n" "路径发现" "$paths_imp"
fi

if [ "$crashes_without" != "0" ] || [ "$crashes_with" != "0" ]; then
    crashes_imp=$(echo "scale=2; ($crashes_with - $crashes_without) * 100 / (${crashes_without:-1})" | bc)
    printf "%-25s %15.2f%%\n" "崩溃发现" "$crashes_imp"
fi

if [ "$speed_without" != "0" ]; then
    speed_imp=$(echo "scale=2; ($speed_with - $speed_without) * 100 / $speed_without" | bc)
    printf "%-25s %15.2f%%\n" "执行速度" "$speed_imp"
fi

echo ""
echo "详细统计文件位置:"
echo "  启用 Lattice-MAB: $(pwd)/$with_stats"
echo "  禁用 Lattice-MAB: $(pwd)/$without_stats"
EOF

chmod +x compare_results.sh
./compare_results.sh
```

---

## 结果分析

### 关键指标说明

#### 1. 执行次数 (execs_done)
- **含义**: 总执行次数，反映 fuzzer 的整体工作量
- **预期提升**: Lattice-MAB 可能提升 2-5%
- **查看方法**: 
  ```bash
  grep "execs_done" output_with_lm/default/fuzzer_stats
  ```

#### 2. 发现路径数 (paths_total)
- **含义**: 发现的唯一路径数量，反映代码覆盖率
- **预期提升**: Lattice-MAB 可能提升 5-15%
- **查看方法**:
  ```bash
  grep "paths_total" output_with_lm/default/fuzzer_stats
  ```

#### 3. 崩溃数 (unique_crashes)
- **含义**: 发现的唯一崩溃数量，反映漏洞发现能力
- **预期提升**: Lattice-MAB 可能提升 10-20%
- **查看方法**:
  ```bash
  grep "unique_crashes" output_with_lm/default/fuzzer_stats
  ls output_with_lm/default/crashes/ | wc -l
  ```

#### 4. 执行速度 (execs_per_sec)
- **含义**: 每秒执行次数，反映 fuzzer 效率
- **预期提升**: Lattice-MAB 可能提升 2-5%（通过减少无效变异）
- **查看方法**:
  ```bash
  grep "execs_per_sec" output_with_lm/default/fuzzer_stats
  ```

#### 5. 完成周期数 (cycles_done)
- **含义**: 完成的队列周期数
- **查看方法**:
  ```bash
  grep "cycles_done" output_with_lm/default/fuzzer_stats
  ```

### 查看详细统计文件

```bash
# 查看完整的 fuzzer_stats
cat output_with_lm/default/fuzzer_stats

# 查看关键指标
cat output_with_lm/default/fuzzer_stats | grep -E "execs_done|paths_total|unique_crashes|execs_per_sec|cycles_done"
```

### 统计文件路径

**启用 Lattice-MAB 的统计文件：**
```
/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable/comparison_test/output_with_lm/default/fuzzer_stats
```

**禁用 Lattice-MAB 的统计文件：**
```
/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable/comparison_test/output_without_lm/default/fuzzer_stats
```

---

## 完整示例

### 快速测试（5分钟）

```bash
# 1. 进入工作目录
cd "/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable"

# 2. 编译测试程序
./afl-cc -o test-instr test-instr.c

# 3. 准备种子
mkdir -p seeds
echo "0" > seeds/seed1.txt
echo "1" > seeds/seed2.txt

# 4. 运行自动化对比（5分钟，每个配置2次）
./utils/compare_lattice_mab.sh -t ./test-instr -i ./seeds -T 300 -n 2

# 5. 查看结果
cat comparison_results/comparison_*/summary.txt
```

### 完整测试（30分钟）

```bash
# 1. 准备测试环境
cd "/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable"
mkdir -p comparison_test
cd comparison_test
cp ../test-instr ./
mkdir -p seeds
echo "0" > seeds/seed1.txt
echo "1" > seeds/seed2.txt
echo "hello" > seeds/seed3.txt

# 2. 终端1：启用 Lattice-MAB（运行30分钟）
../afl-fuzz -i seeds -o output_with_lm -- ./test-instr @@

# 3. 终端2：禁用 Lattice-MAB（运行30分钟）
AFL_DISABLE_LATTICE_MAB=1 ../afl-fuzz -i seeds -o output_without_lm -- ./test-instr @@

# 4. 30分钟后，在两个终端按 Ctrl+C 停止

# 5. 对比结果
./compare_results.sh
```

---

## 路径总结

### 关键文件路径

| 文件/目录 | 路径 |
|----------|------|
| AFL++ 根目录 | `/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable` |
| afl-fuzz 可执行文件 | `/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable/afl-fuzz` |
| afl-cc 编译器 | `/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable/afl-cc` |
| 测试程序源码 | `/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable/test-instr.c` |
| 对比测试脚本 | `/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable/utils/compare_lattice_mab.sh` |

### 输出目录路径

| 输出类型 | 路径 |
|---------|------|
| 启用 Lattice-MAB 输出 | `./output_with_lm/default/` |
| 禁用 Lattice-MAB 输出 | `./output_without_lm/default/` |
| 自动化对比结果 | `./comparison_results/comparison_<timestamp>/` |
| 统计文件 | `./output_with_lm/default/fuzzer_stats` |

---

## 注意事项

1. **运行时间**: 建议至少运行 10-30 分钟以获得有意义的结果
2. **多次运行**: 由于模糊测试的随机性，建议多次运行取平均值
3. **系统资源**: 确保有足够的 CPU 和内存资源
4. **种子质量**: 使用高质量的种子文件可以提高测试效果
5. **环境一致性**: 确保两个测试在相同的系统环境下运行

---

## 故障排除

### 问题 1: 找不到 afl-fuzz

```bash
# 确保在 AFL++ 目录中
cd "/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-stable"

# 使用完整路径
./afl-fuzz -i seeds -o output -- ./target @@
```

### 问题 2: Lattice-MAB 统计信息不显示

```bash
# 检查是否被禁用
echo $AFL_DISABLE_LATTICE_MAB

# 确保未设置禁用
unset AFL_DISABLE_LATTICE_MAB
```

### 问题 3: 测试程序崩溃

```bash
# 检查测试程序是否正常
./test-instr < seeds/seed1.txt

# 检查是否使用 afl-cc 编译
file test-instr | grep -i "not stripped"
```

---

## 参考文档

- 快速开始: `docs/LATTICE_MAB_QUICKSTART.md`
- 详细文档: `docs/LATTICE_MAB.md`
- 使用指南: `docs/LATTICE_MAB_USAGE.md`
- 对比指南: `LATTICE_MAB_COMPARISON_GUIDE.md`
- 源代码: `src/afl-fuzz-lattice-mab.c`

