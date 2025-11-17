# AFL++ 仓库介绍与变异操作详解

## 📚 仓库概述

**AFL++ (American Fuzzy Lop plus plus)** 是一个强大的模糊测试（Fuzzing）框架，是 Google 的 AFL 的增强版本。它通过自动生成和变异测试用例来发现软件中的漏洞和bug。

### 核心特性

- ✅ **更快的执行速度** - 优化的插桩和运行机制
- ✅ **37+ 种变异操作** - 丰富的变异策略
- ✅ **多种插桩模式** - LLVM、GCC、QEMU、Frida、Unicorn等
- ✅ **自定义变异器** - 支持 C/C++、Python、Rust
- ✅ **覆盖率引导** - 基于代码覆盖率的智能变异
- ✅ **多种目标支持** - 二进制文件、网络服务、GUI程序等

### 主要组件

| 工具 | 功能 |
|------|------|
| `afl-fuzz` | 主要的模糊测试工具 |
| `afl-cc` | 编译器包装器，用于插桩 |
| `afl-showmap` | 显示程序的覆盖率映射 |
| `afl-cmin` | 最小化测试用例集 |
| `afl-plot` | 生成测试统计图表 |

## 🔬 变异操作详解

AFL++ 使用多种变异操作来生成新的测试用例。这些操作分为几个阶段：

### 阶段1: 确定性变异 (Deterministic)

对每个输入执行所有可能的单步变异，确保覆盖所有基础变异：

#### 1. 位级操作

**MUT_FLIPBIT** - 翻转单个位
```c
// 代码实现 (include/afl-mutations.h:1871-1879)
case MUT_FLIPBIT: {
    u8  bit = rand_below(afl, 8);      // 随机选择位 (0-7)
    u32 off = rand_below(afl, len);    // 随机选择字节位置
    buf[off] ^= 1 << bit;              // 翻转指定位
    break;
}
```

**示例效果：**
- 原始: `Hello` (48 65 6c 6c 6f)
- 变异: `Hdllo` (48 64 6c 6c 6f) - 第2字节第0位翻转

**MUT_FLIP8** - 翻转整个字节
```c
case MUT_FLIP8: {
    buf[rand_below(afl, len)] ^= 0xff;  // 所有位取反
    break;
}
```

#### 2. 算术操作

**MUT_ARITH8** - 字节加法/减法
```c
case MUT_ARITH8: {
    item = 1 + rand_below(afl, ARITH_MAX);  // 随机增量 (1-35)
    buf[rand_below(afl, len)] += item;      // 加法
    break;
}
case MUT_ARITH8_: {
    item = 1 + rand_below(afl, ARITH_MAX);
    buf[rand_below(afl, len)] -= item;      // 减法
    break;
}
```

**示例效果：**
- 原始: `Hello` (48 65 6c 6c 6f)
- 变异: `Hgllo` (48 67 6c 6c 6f) - 第2字节 +2

**MUT_ARITH16/32** - 字/双字算术运算（支持大小端）

**MUT_BYTEADD/BYTESUB** - 字节加1/减1
```c
case MUT_BYTEADD: {
    buf[rand_below(afl, len)]++;  // 加1
    break;
}
case MUT_BYTESUB: {
    buf[rand_below(afl, len)]--;  // 减1
    break;
}
```

#### 3. 特殊值替换

**MUT_INTERESTING8/16/32** - 替换为"有趣"的值

这些值包括边界值，如：
- `0x00`, `0x01`, `0x7F`, `0x80`, `0xFF`
- `0x0000`, `0x0001`, `0x7FFF`, `0x8000`, `0xFFFF`
- `0x00000000`, `0x00000001`, `0x7FFFFFFF`, `0x80000000`, `0xFFFFFFFF`

```c
case MUT_INTERESTING8: {
    item = rand_below(afl, sizeof(interesting_8));
    buf[rand_below(afl, len)] = interesting_8[item];
    break;
}
```

**示例效果：**
- 原始: `Hello` (48 65 6c 6c 6f)
- 变异: `\x00ello` (00 65 6c 6c 6f) - 替换为0x00

### 阶段2: Havoc 阶段 (随机变异)

随机选择多种变异操作，可以执行1-16步变异：

#### 4. 数据块操作

**MUT_CLONE_COPY** - 克隆数据块
```c
case MUT_CLONE_COPY: {
    u32 clone_len = choose_block_len(afl, len);
    u32 clone_from = rand_below(afl, len - clone_len + 1);
    u32 clone_to = rand_below(afl, len);
    // 将 clone_from 开始的 clone_len 字节复制到 clone_to
    memcpy(tmp_buf + clone_to, buf + clone_from, clone_len);
    break;
}
```

**示例效果：**
- 原始: `Hello` (48 65 6c 6c 6f)
- 变异: `HeHeo` (48 65 48 65 6f) - 克隆前2字节到位置2

**MUT_OVERWRITE_COPY** - 覆盖数据块
```c
case MUT_OVERWRITE_COPY: {
    u32 copy_len = choose_block_len(afl, len - 1);
    u32 copy_from = rand_below(afl, len - copy_len + 1);
    u32 copy_to = rand_below(afl, len - copy_len + 1);
    memmove(buf + copy_to, buf + copy_from, copy_len);
    break;
}
```

**MUT_DEL** - 删除数据块
```c
case MUT_DEL: {
    u32 del_len = choose_block_len(afl, len - 1);
    u32 del_from = rand_below(afl, len - del_len + 1);
    memmove(buf + del_from, buf + del_from + del_len, 
            len - del_from - del_len);
    len -= del_len;
    break;
}
```

**示例效果：**
- 原始: `Hello` (48 65 6c 6c 6f)
- 变异: `Hlo` (48 6c 6f) - 删除位置1-2的2字节

**MUT_SWITCH** - 交换数据块位置
```c
case MUT_SWITCH: {
    u32 switch_from = rand_below(afl, len);
    u32 switch_to = rand_below(afl, len);
    // 交换两个位置的数据块
    memcpy(tmp_buf, buf + switch_from, switch_len);
    memcpy(buf + switch_from, buf + switch_to, switch_len);
    memcpy(buf + switch_to, tmp_buf, switch_len);
    break;
}
```

**MUT_SHUFFLE** - 打乱数据块内的字节顺序

#### 5. 插入操作

**MUT_INSERTONE** - 插入单个字节
```c
case MUT_INSERTONE: {
    u32 clone_to = rand_below(afl, len);
    u32 strat = rand_below(afl, 2);
    item = strat ? rand_below(afl, 256) : buf[clone_to - 1];
    // 在 clone_to 位置插入 item
    break;
}
```

**示例效果：**
- 原始: `Hello` (48 65 6c 6c 6f)
- 变异: `HeXllo` (48 65 58 6c 6c 6f) - 在位置2插入'X'

**MUT_EXTRA_INSERT** - 从字典插入数据
```c
case MUT_EXTRA_INSERT: {
    u32 use_extra = rand_below(afl, afl->extras_cnt);
    u32 extra_len = afl->extras[use_extra].len;
    u32 insert_at = rand_below(afl, len + 1);
    // 在 insert_at 位置插入字典中的条目
    break;
}
```

#### 6. 文本特定操作

**MUT_ASCIINUM** - 修改 ASCII 数字
- 查找输入中的数字字符串
- 对数字进行变异（+1, -1, *2, /2, 随机值等）

**MUT_INSERTASCIINUM** - 插入 ASCII 数字

### 阶段3: 拼接阶段 (Splicing)

**MUT_SPLICE_OVERWRITE** - 从另一个输入覆盖数据
```c
case MUT_SPLICE_OVERWRITE: {
    copy_len = choose_block_len(afl, splice_len - 1);
    copy_from = rand_below(afl, splice_len - copy_len + 1);
    copy_to = rand_below(afl, len - copy_len + 1);
    memmove(buf + copy_to, splice_buf + copy_from, copy_len);
    break;
}
```

**MUT_SPLICE_INSERT** - 从另一个输入插入数据
```c
case MUT_SPLICE_INSERT: {
    clone_len = choose_block_len(afl, splice_len);
    clone_from = rand_below(afl, splice_len - clone_len + 1);
    clone_to = rand_below(afl, len + 1);
    // 在 clone_to 位置插入 splice_buf 中从 clone_from 开始的 clone_len 字节
    break;
}
```

**示例效果：**
- 输入1: `Hello` (48 65 6c 6c 6f)
- 输入2: `World` (57 6f 72 6c 64)
- 结果: `HeWorllo` (48 65 57 6f 72 6c 6c 6f) - 在位置2插入'World'的前3字节

## 🎯 变异策略选择

AFL++ 根据以下因素选择变异策略：

1. **输入类型**
   - 文本输入：使用 `mutation_strategy_exploration_text` 或 `mutation_strategy_exploitation_text`
   - 二进制输入：使用 `mutation_strategy_exploration_binary` 或 `mutation_strategy_exploitation_binary`

2. **探索模式 vs 利用模式**
   - 探索模式：更激进的变异，寻找新的代码路径
   - 利用模式：更保守的变异，深入探索已知路径

3. **覆盖率反馈**
   - 根据覆盖率信息调整变异策略
   - 优先变异能触发新覆盖的输入

## 🚀 快速开始示例

### 1. 编译目标程序

```bash
# 使用 AFL++ 编译器（推荐，会进行插桩）
./afl-cc -o demo_target demo_target.c

# 或使用普通编译器（仅用于测试）
gcc -o demo_target demo_target.c
```

### 2. 准备种子文件

```bash
mkdir seeds
echo "Hello" > seeds/seed1.txt
```

### 3. 运行模糊测试

```bash
./afl-fuzz -i seeds -o output -- ./demo_target @@
```

### 4. 查看结果

- 测试用例: `output/default/queue/`
- 崩溃: `output/default/crashes/`
- 超时: `output/default/hangs/`

## 📊 演示脚本

本仓库包含以下演示文件：

1. **`demo_target.c`** - 演示目标程序
   - 检查特定关键字（PASSWORD, SECRET, CRASH等）
   - 可用于测试变异操作效果

2. **`show_mutations.py`** - 变异操作可视化
   - 展示各种变异操作的实际效果
   - 运行: `python3 show_mutations.py`

3. **`demo_mutation.py`** - 完整演示脚本
   - 展示如何运行 AFL++ 模糊测试
   - 运行: `python3 demo_mutation.py --run`

## 📖 相关文档

- [官方文档](docs/README.md)
- [安装指南](docs/INSTALL.md)
- [模糊测试深度指南](docs/fuzzing_in_depth.md)
- [自定义变异器](docs/custom_mutators.md)
- [最佳实践](docs/best_practices.md)

## 🔍 代码位置

变异操作的核心实现：
- 头文件: `include/afl-mutations.h`
- 实现: `include/afl-mutations.h` (内联函数)
- 使用: `src/afl-fuzz-one.c`, `custom_mutators/aflpp/aflpp.c`

## 📝 总结

AFL++ 通过多种变异操作和覆盖率引导，能够高效地发现软件漏洞。理解这些变异操作有助于：

1. **优化种子文件** - 提供更好的初始输入
2. **编写自定义变异器** - 针对特定格式的输入
3. **调试模糊测试** - 理解为什么某些输入被生成
4. **提高测试效率** - 选择合适的变异策略

通过不断变异输入并观察程序行为，AFL++ 能够发现传统测试方法难以找到的边界条件和bug。

