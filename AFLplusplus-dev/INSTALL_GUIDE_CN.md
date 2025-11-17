# AFL++ macOS 安装指南

## 当前状态

检测到您使用的是 **macOS (Darwin)** 系统，AFL++ 尚未编译。

## 快速安装步骤

### 1. 安装依赖

首先安装 Homebrew（如果还没有安装）：
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

然后安装必要的依赖：
```bash
brew install wget git make cmake llvm gdb coreutils
```

### 2. 配置环境变量

根据您的 Homebrew 安装位置设置环境变量：

**对于 Apple Silicon (M1/M2/M3) Mac:**
```bash
export HOMEBREW_BASE="/opt/homebrew/opt"
```

**对于 Intel Mac:**
```bash
export HOMEBREW_BASE="/usr/local/opt"
```

设置 PATH 和编译器：
```bash
export PATH="$HOMEBREW_BASE/coreutils/libexec/gnubin:/usr/local/bin:$HOMEBREW_BASE/llvm/bin:$PATH"
export CC=clang
export CXX=clang++
```

### 3. 配置系统设置

macOS 需要增加共享内存限制：
```bash
sudo ./afl-system-config
```

这会增加系统共享内存限制，这是 AFL++ 运行所必需的。

### 4. 编译 AFL++

在 AFL++ 源码目录中执行：

```bash
# 进入 AFL++ 目录
cd "/Users/ketangchen/Library/Mobile Documents/com~apple~CloudDocs/Documents/000_20250825dev/laboratory_library/AFLplusplus-dev"

# 更新子模块
git submodule update --init

# 编译（仅编译基础功能，适合快速开始）
make source-only

# 或者编译完整版本（包括所有模式，需要更长时间）
# make distrib
```

### 5. 安装（可选）

将编译好的工具安装到系统路径：
```bash
sudo make install
```

或者直接使用当前目录中的二进制文件：
```bash
# 使用 ./afl-fuzz 而不是 afl-fuzz
./afl-fuzz -i seeds -o output -- ./demo_target @@
```

### 6. 验证安装

检查是否编译成功：
```bash
./afl-fuzz -h
./afl-cc -h
```

## 常见问题

### Q: 编译时出现错误
A: 确保：
- 已安装所有依赖：`brew install wget git make cmake llvm gdb coreutils`
- 已设置正确的环境变量（见步骤2）
- 使用的是 Homebrew 的 clang，不是 Xcode 的 clang

### Q: 运行时出现共享内存错误
A: 运行 `sudo ./afl-system-config` 来增加共享内存限制

### Q: fork() 相关错误
A: macOS 的 fork() 语义与 Linux 不同，如果遇到问题，设置：
```bash
export AFL_NO_FORKSRV=1
```

### Q: 性能较慢
A: macOS 上的模糊测试性能通常比 Linux 慢。考虑：
- 在 Linux VM 中运行
- 使用 FRIDA 模式（`-O`）进行二进制模糊测试

## 快速测试

编译完成后，可以测试演示程序：

```bash
# 1. 编译演示目标程序
gcc -o demo_target demo_target.c

# 2. 创建种子目录
mkdir -p seeds
echo "Hello" > seeds/seed1.txt

# 3. 运行模糊测试（按 Ctrl+C 停止）
./afl-fuzz -i seeds -o output -- ./demo_target @@
```

## 仅查看变异操作演示（无需编译 AFL++）

如果您只是想查看变异操作的效果，可以运行：

```bash
# 查看变异操作可视化
python3 show_mutations.py

# 查看演示说明
python3 demo_mutation.py
```

这些脚本不需要 AFL++ 已编译，可以直接运行。

## 下一步

安装完成后，您可以：
1. 查看 `README_CN.md` 了解变异操作详情
2. 运行 `python3 show_mutations.py` 查看变异效果
3. 使用 `demo_target.c` 进行实际模糊测试

