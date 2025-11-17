# AFL++ 仓库介绍与变异操作演示

## 仓库介绍

**AFL++ (American Fuzzy Lop plus plus)** 是一个先进的模糊测试（Fuzzing）工具，是 Google 的 AFL 的增强版本。

### 主要特点

1. **更快的速度** - 优化的执行引擎
2. **更多更好的变异策略** - 支持 37+ 种变异操作
3. **更好的插桩** - 支持多种插桩模式（LLVM、GCC、QEMU、Frida等）
4. **自定义模块支持** - 支持 C/C++、Python、Rust 自定义变异器
5. **多种运行模式** - 支持二进制文件、网络服务、GUI程序等

### 核心组件

- `afl-fuzz` - 主要的模糊测试工具
- `afl-cc` - 编译器包装器，用于插桩目标程序
- `afl-showmap` - 显示程序的覆盖率映射
- `afl-cmin` - 最小化测试用例集

## 变异操作类型

AFL++ 支持多种变异操作，主要包括：

### 1. 位级操作
- **MUT_FLIPBIT** - 翻转单个位
- **MUT_FLIP8** - 翻转整个字节

### 2. 算术操作
- **MUT_ARITH8/16/32** - 对字节/字/双字进行加减运算
- **MUT_BYTEADD/BYTESUB** - 字节加1或减1

### 3. 特殊值替换
- **MUT_INTERESTING8/16/32** - 替换为"有趣"的值（如 0, -1, MAX_INT 等）

### 4. 数据块操作
- **MUT_CLONE_COPY** - 克隆数据块
- **MUT_OVERWRITE_COPY** - 覆盖数据块
- **MUT_DEL** - 删除数据块
- **MUT_SWITCH** - 交换数据块位置
- **MUT_SHUFFLE** - 打乱数据块

### 5. 插入操作
- **MUT_INSERTONE** - 插入单个字节
- **MUT_EXTRA_INSERT** - 从字典插入数据

### 6. 文本特定操作
- **MUT_ASCIINUM** - 修改 ASCII 数字
- **MUT_INSERTASCIINUM** - 插入 ASCII 数字

### 7. 拼接操作
- **MUT_SPLICE_OVERWRITE** - 从另一个输入覆盖数据
- **MUT_SPLICE_INSERT** - 从另一个输入插入数据

## 变异操作演示

下面我们将创建一个简单的演示程序来展示变异操作的效果。

