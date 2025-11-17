# Python 库加载问题解决方案

## 问题描述

运行 `afl-fuzz` 时出现错误：
```
./afl-fuzz: error while loading shared libraries: libpython3.13.so.1.0: cannot open shared object file: No such file or directory
```

这是因为编译时链接了 Python 库，但运行时系统找不到该库。

## 快速解决方案

### 方案 1: 使用包装脚本（推荐）

使用提供的 `run_afl_fuzz.sh` 脚本，它会自动检测并设置 Python 库路径：

```bash
# 使用包装脚本运行
./run_afl_fuzz.sh -i seeds -o output -- ./test-instr @@

# 对比测试时也使用包装脚本
./run_afl_fuzz.sh -i seeds -o output_with_lm -- ./test-instr @@
AFL_DISABLE_LATTICE_MAB=1 ./run_afl_fuzz.sh -i seeds -o output_without_lm -- ./test-instr @@
```

### 方案 2: 手动设置环境变量

#### Linux 系统

```bash
# 查找 Python 库路径
python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR') or sysconfig.get_path('stdlib').replace('/lib/python' + sysconfig.get_python_version(), '').replace('/lib', '') + '/lib')"

# 设置 LD_LIBRARY_PATH（临时）
export LD_LIBRARY_PATH="/home/apulis-dev/miniconda3/lib:$LD_LIBRARY_PATH"

# 然后运行 afl-fuzz
./afl-fuzz -i seeds -o output -- ./test-instr @@
```

#### macOS 系统

```bash
# 设置 DYLD_LIBRARY_PATH
export DYLD_LIBRARY_PATH="$(python3-config --prefix)/lib:$DYLD_LIBRARY_PATH"

# 然后运行 afl-fuzz
./afl-fuzz -i seeds -o output -- ./test-instr @@
```

### 方案 3: 在命令中直接设置

```bash
# Linux
LD_LIBRARY_PATH="/home/apulis-dev/miniconda3/lib:$LD_LIBRARY_PATH" ./afl-fuzz -i seeds -o output -- ./test-instr @@

# macOS
DYLD_LIBRARY_PATH="$(python3-config --prefix)/lib:$DYLD_LIBRARY_PATH" ./afl-fuzz -i seeds -o output -- ./test-instr @@
```

### 方案 4: 永久设置（在 ~/.bashrc 或 ~/.zshrc 中）

```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
export LD_LIBRARY_PATH="/home/apulis-dev/miniconda3/lib:$LD_LIBRARY_PATH"

# 然后重新加载
source ~/.bashrc  # 或 source ~/.zshrc
```

### 方案 5: 禁用 Python 支持重新编译（如果不需要 Python 功能）

如果不需要 Python 自定义变异器功能，可以禁用 Python 支持：

```bash
make clean
make NO_PYTHON=1
```

## 检测 Python 库路径

运行诊断脚本：

```bash
./fix_python_lib.sh
```

这会显示：
1. 检测到的 Python 库路径
2. 多种解决方案
3. 自动创建包装脚本（如果可能）

## 常见 Python 库路径

### Conda/Miniconda
- Linux: `~/miniconda3/lib` 或 `~/anaconda3/lib`
- macOS: `~/miniconda3/lib` 或 `~/anaconda3/lib`

### 系统 Python
- Linux: `/usr/lib` 或 `/usr/local/lib`
- macOS: `/usr/local/lib` 或 `/opt/homebrew/lib`

### 虚拟环境
- `$VIRTUAL_ENV/lib`
- `$CONDA_PREFIX/lib`

## 验证修复

运行以下命令验证库路径设置正确：

```bash
# Linux
ldd ./afl-fuzz | grep python

# macOS
otool -L ./afl-fuzz | grep python
```

如果显示库路径，说明设置成功。

## 更新对比脚本

`quick_compare.sh` 和 `utils/compare_lattice_mab.sh` 已更新，会自动处理 Python 库路径问题。

## 注意事项

1. **临时设置**: 使用 `export` 设置的变量只在当前终端会话有效
2. **永久设置**: 添加到 `~/.bashrc` 或 `~/.zshrc` 后需要重新加载
3. **Docker/容器**: 在容器中可能需要挂载 Python 库目录
4. **多 Python 版本**: 确保使用编译时使用的 Python 版本

