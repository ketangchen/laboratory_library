#!/bin/bash
#
# 修复 AFL++ Python 库加载问题
#

echo "正在检测 Python 库路径..."

# 方法1: 使用 python3-config
PYTHON_LIBDIR=$(python3-config --prefix 2>/dev/null)/lib
if [ ! -d "$PYTHON_LIBDIR" ]; then
    # 方法2: 使用 sysconfig
    PYTHON_LIBDIR=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR') or sysconfig.get_path('stdlib').replace('/lib/python' + sysconfig.get_python_version(), '').replace('/lib', '') + '/lib')" 2>/dev/null)
fi

if [ ! -d "$PYTHON_LIBDIR" ]; then
    # 方法3: 从环境变量中查找
    if [ -n "$CONDA_PREFIX" ]; then
        PYTHON_LIBDIR="$CONDA_PREFIX/lib"
    elif [ -n "$VIRTUAL_ENV" ]; then
        PYTHON_LIBDIR="$VIRTUAL_ENV/lib"
    fi
fi

if [ -d "$PYTHON_LIBDIR" ] && [ -f "$PYTHON_LIBDIR/libpython"*.so* ] 2>/dev/null || [ -f "$PYTHON_LIBDIR/libpython"*.dylib* ] 2>/dev/null; then
    echo "找到 Python 库目录: $PYTHON_LIBDIR"
    echo ""
    echo "解决方案 1: 设置 LD_LIBRARY_PATH（临时）"
    echo "----------------------------------------"
    echo "export LD_LIBRARY_PATH=\"$PYTHON_LIBDIR:\$LD_LIBRARY_PATH\""
    echo ""
    echo "解决方案 2: 在运行命令时设置（推荐）"
    echo "----------------------------------------"
    echo "LD_LIBRARY_PATH=\"$PYTHON_LIBDIR:\$LD_LIBRARY_PATH\" ./afl-fuzz -i seeds -o output -- ./test-instr @@"
    echo ""
    echo "解决方案 3: 创建包装脚本"
    echo "----------------------------------------"
    cat > run_afl_fuzz.sh << EOF
#!/bin/bash
export LD_LIBRARY_PATH="$PYTHON_LIBDIR:\${LD_LIBRARY_PATH}"
exec ./afl-fuzz "\$@"
EOF
    chmod +x run_afl_fuzz.sh
    echo "已创建 run_afl_fuzz.sh，使用: ./run_afl_fuzz.sh -i seeds -o output -- ./test-instr @@"
    echo ""
    echo "解决方案 4: 禁用 Python 支持重新编译（如果不需要 Python 功能）"
    echo "----------------------------------------"
    echo "make clean"
    echo "make NO_PYTHON=1"
else
    echo "警告: 无法自动检测 Python 库路径"
    echo ""
    echo "请手动查找 Python 库:"
    echo "  find /usr -name 'libpython*.so*' 2>/dev/null"
    echo "  find \$HOME -name 'libpython*.so*' 2>/dev/null"
    echo ""
    echo "然后设置:"
    echo "  export LD_LIBRARY_PATH=/path/to/python/lib:\$LD_LIBRARY_PATH"
fi

