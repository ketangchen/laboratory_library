#!/bin/bash
#
# AFL++ 运行包装脚本
# 自动设置 Python 库路径
#

# 检测 Python 库路径
detect_python_lib() {
    # 方法1: 使用 python3-config
    local libdir=$(python3-config --prefix 2>/dev/null)/lib
    if [ -d "$libdir" ] && ([ -f "$libdir/libpython"*.so* ] 2>/dev/null || [ -f "$libdir/libpython"*.dylib* ] 2>/dev/null); then
        echo "$libdir"
        return
    fi
    
    # 方法2: 使用 sysconfig
    libdir=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR') or sysconfig.get_path('stdlib').replace('/lib/python' + sysconfig.get_python_version(), '').replace('/lib', '') + '/lib')" 2>/dev/null)
    if [ -d "$libdir" ] && ([ -f "$libdir/libpython"*.so* ] 2>/dev/null || [ -f "$libdir/libpython"*.dylib* ] 2>/dev/null); then
        echo "$libdir"
        return
    fi
    
    # 方法3: 从环境变量
    if [ -n "$CONDA_PREFIX" ] && [ -d "$CONDA_PREFIX/lib" ]; then
        echo "$CONDA_PREFIX/lib"
        return
    fi
    
    if [ -n "$VIRTUAL_ENV" ] && [ -d "$VIRTUAL_ENV/lib" ]; then
        echo "$VIRTUAL_ENV/lib"
        return
    fi
    
    # 方法4: 常见路径
    for path in "/usr/lib" "/usr/local/lib" "$HOME/miniconda3/lib" "$HOME/anaconda3/lib"; do
        if [ -d "$path" ] && ([ -f "$path/libpython"*.so* ] 2>/dev/null || [ -f "$path/libpython"*.dylib* ] 2>/dev/null); then
            echo "$path"
            return
        fi
    done
    
    echo ""
}

# 获取 Python 库目录
PYTHON_LIBDIR=$(detect_python_lib)

# 设置库路径
if [ -n "$PYTHON_LIBDIR" ]; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        export DYLD_LIBRARY_PATH="$PYTHON_LIBDIR:${DYLD_LIBRARY_PATH}"
    else
        # Linux
        export LD_LIBRARY_PATH="$PYTHON_LIBDIR:${LD_LIBRARY_PATH}"
    fi
fi

# 运行 afl-fuzz
exec ./afl-fuzz "$@"

