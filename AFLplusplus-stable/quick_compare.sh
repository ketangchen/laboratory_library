#!/bin/bash
#
# Lattice-MAB 快速对比脚本
# 用于快速对比启用和禁用 Lattice-MAB 的效果
#

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 默认参数
TARGET=""
INPUT_DIR=""
TEST_TIME=600  # 默认10分钟
AFL_FUZZ="./afl-fuzz"

# 使用说明
usage() {
    echo "用法: $0 -t <target> -i <input_dir> [选项]"
    echo ""
    echo "选项:"
    echo "  -t, --target     目标二进制文件路径（必需）"
    echo "  -i, --input      输入种子目录（必需）"
    echo "  -T, --time       测试时间（秒，默认: 600）"
    echo "  -a, --afl-fuzz   afl-fuzz 路径（默认: ./afl-fuzz）"
    echo "  -h, --help       显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 -t ./test-instr -i ./seeds -T 300"
    exit 1
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--target)
            TARGET="$2"
            shift 2
            ;;
        -i|--input)
            INPUT_DIR="$2"
            shift 2
            ;;
        -T|--time)
            TEST_TIME="$2"
            shift 2
            ;;
        -a|--afl-fuzz)
            AFL_FUZZ="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "未知选项: $1"
            usage
            ;;
    esac
done

# 检查必需参数
if [[ -z "$TARGET" || -z "$INPUT_DIR" ]]; then
    echo -e "${RED}错误: 必须指定目标程序和输入目录${NC}"
    usage
fi

if [[ ! -f "$TARGET" ]]; then
    echo -e "${RED}错误: 目标文件不存在: $TARGET${NC}"
    exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo -e "${RED}错误: 输入目录不存在: $INPUT_DIR${NC}"
    exit 1
fi

if [[ ! -f "$AFL_FUZZ" ]]; then
    echo -e "${RED}错误: afl-fuzz 不存在: $AFL_FUZZ${NC}"
    exit 1
fi

# 创建输出目录
OUTPUT_DIR="./quick_comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Lattice-MAB 快速对比测试${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "目标程序: $TARGET"
echo "输入目录: $INPUT_DIR"
echo "测试时间: ${TEST_TIME}秒 ($(($TEST_TIME / 60))分钟)"
echo "输出目录: $OUTPUT_DIR"
echo ""

# 提取指标函数
extract_metric() {
    local file=$1
    local key=$2
    if [[ -f "$file" ]]; then
        grep "^$key" "$file" | awk '{print $3}' || echo "0"
    else
        echo "0"
    fi
}

# 检测并设置 Python 库路径
setup_python_lib() {
    # 方法1: 使用 python3-config
    local libdir=$(python3-config --prefix 2>/dev/null)/lib
    if [[ -d "$libdir" ]] && ([[ -f "$libdir/libpython"*.so* ]] 2>/dev/null || [[ -f "$libdir/libpython"*.dylib* ]] 2>/dev/null); then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            export DYLD_LIBRARY_PATH="$libdir:${DYLD_LIBRARY_PATH}"
        else
            export LD_LIBRARY_PATH="$libdir:${LD_LIBRARY_PATH}"
        fi
        return
    fi
    
    # 方法2: 使用 sysconfig
    libdir=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR') or sysconfig.get_path('stdlib').replace('/lib/python' + sysconfig.get_python_version(), '').replace('/lib', '') + '/lib')" 2>/dev/null)
    if [[ -d "$libdir" ]] && ([[ -f "$libdir/libpython"*.so* ]] 2>/dev/null || [[ -f "$libdir/libpython"*.dylib* ]] 2>/dev/null); then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            export DYLD_LIBRARY_PATH="$libdir:${DYLD_LIBRARY_PATH}"
        else
            export LD_LIBRARY_PATH="$libdir:${LD_LIBRARY_PATH}"
        fi
        return
    fi
    
    # 方法3: 从环境变量
    if [[ -n "$CONDA_PREFIX" ]] && [[ -d "$CONDA_PREFIX/lib" ]]; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            export DYLD_LIBRARY_PATH="$CONDA_PREFIX/lib:${DYLD_LIBRARY_PATH}"
        else
            export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"
        fi
    fi
}

# 运行测试函数
run_fuzz() {
    local output=$1
    local disable_lm=$2
    local desc=$3
    
    echo -e "${YELLOW}运行测试: $desc${NC}"
    
    # 设置 Python 库路径
    setup_python_lib
    
    if [[ "$disable_lm" == "1" ]]; then
        export AFL_DISABLE_LATTICE_MAB=1
    else
        unset AFL_DISABLE_LATTICE_MAB
    fi
    
    timeout "$TEST_TIME" "$AFL_FUZZ" -i "$INPUT_DIR" -o "$output" -- "$TARGET" @@ > "$output/fuzzer.log" 2>&1 || true
    
    # 等待 fuzzer 完全停止
    sleep 2
}

# 测试 1: 启用 Lattice-MAB
echo -e "${GREEN}[1/2] 测试启用 Lattice-MAB...${NC}"
OUTPUT_WITH="$OUTPUT_DIR/with_lattice_mab"
run_fuzz "$OUTPUT_WITH" "0" "启用 Lattice-MAB"

# 测试 2: 禁用 Lattice-MAB
echo -e "${GREEN}[2/2] 测试禁用 Lattice-MAB...${NC}"
OUTPUT_WITHOUT="$OUTPUT_DIR/without_lattice_mab"
run_fuzz "$OUTPUT_WITHOUT" "1" "禁用 Lattice-MAB"

# 提取指标
echo ""
echo -e "${BLUE}提取统计信息...${NC}"

WITH_STATS="$OUTPUT_WITH/default/fuzzer_stats"
WITHOUT_STATS="$OUTPUT_WITHOUT/default/fuzzer_stats"

if [[ ! -f "$WITH_STATS" ]] || [[ ! -f "$WITHOUT_STATS" ]]; then
    echo -e "${RED}错误: 无法找到统计文件${NC}"
    exit 1
fi

execs_with=$(extract_metric "$WITH_STATS" "execs_done")
execs_without=$(extract_metric "$WITHOUT_STATS" "execs_done")
paths_with=$(extract_metric "$WITH_STATS" "paths_total")
paths_without=$(extract_metric "$WITHOUT_STATS" "paths_total")
crashes_with=$(extract_metric "$WITH_STATS" "unique_crashes")
crashes_without=$(extract_metric "$WITHOUT_STATS" "unique_crashes")
hangs_with=$(extract_metric "$WITH_STATS" "unique_hangs")
hangs_without=$(extract_metric "$WITHOUT_STATS" "unique_hangs")
speed_with=$(extract_metric "$WITH_STATS" "execs_per_sec")
speed_without=$(extract_metric "$WITHOUT_STATS" "execs_per_sec")
cycles_with=$(extract_metric "$WITH_STATS" "cycles_done")
cycles_without=$(extract_metric "$WITHOUT_STATS" "cycles_done")

# 生成报告
REPORT_FILE="$OUTPUT_DIR/comparison_report.txt"

{
    echo "=========================================="
    echo "Lattice-MAB 对比测试报告"
    echo "=========================================="
    echo ""
    echo "测试时间: $(date)"
    echo "目标程序: $TARGET"
    echo "输入目录: $INPUT_DIR"
    echo "测试时长: ${TEST_TIME}秒 ($(($TEST_TIME / 60))分钟)"
    echo ""
    echo "----------------------------------------"
    echo "指标对比"
    echo "----------------------------------------"
    echo ""
    printf "%-30s %20s %20s %15s\n" "指标" "启用Lattice-MAB" "禁用Lattice-MAB" "提升/下降"
    echo "----------------------------------------"
    
    # 执行次数
    if [[ "$execs_without" != "0" ]]; then
        exec_imp=$(echo "scale=2; ($execs_with - $execs_without) * 100 / $execs_without" | bc)
        printf "%-30s %20s %20s %15.2f%%\n" "总执行次数" "$execs_with" "$execs_without" "$exec_imp"
    else
        printf "%-30s %20s %20s %15s\n" "总执行次数" "$execs_with" "$execs_without" "N/A"
    fi
    
    # 发现路径数
    if [[ "$paths_without" != "0" ]]; then
        paths_imp=$(echo "scale=2; ($paths_with - $paths_without) * 100 / $paths_without" | bc)
        printf "%-30s %20s %20s %15.2f%%\n" "发现路径数" "$paths_with" "$paths_without" "$paths_imp"
    else
        printf "%-30s %20s %20s %15s\n" "发现路径数" "$paths_with" "$paths_without" "N/A"
    fi
    
    # 崩溃数
    crashes_imp=$(echo "scale=2; ($crashes_with - $crashes_without) * 100 / (${crashes_without:-1})" | bc)
    printf "%-30s %20s %20s %15.2f%%\n" "发现崩溃数" "$crashes_with" "$crashes_without" "$crashes_imp"
    
    # 挂起数
    hangs_imp=$(echo "scale=2; ($hangs_with - $hangs_without) * 100 / (${hangs_without:-1})" | bc)
    printf "%-30s %20s %20s %15.2f%%\n" "发现挂起数" "$hangs_with" "$hangs_without" "$hangs_imp"
    
    # 执行速度
    if [[ "$speed_without" != "0" ]]; then
        speed_imp=$(echo "scale=2; ($speed_with - $speed_without) * 100 / $speed_without" | bc)
        printf "%-30s %20.2f %20.2f %15.2f%%\n" "执行速度(execs/s)" "$speed_with" "$speed_without" "$speed_imp"
    else
        printf "%-30s %20.2f %20.2f %15s\n" "执行速度(execs/s)" "$speed_with" "$speed_without" "N/A"
    fi
    
    # 完成周期数
    if [[ "$cycles_without" != "0" ]]; then
        cycles_imp=$(echo "scale=2; ($cycles_with - $cycles_without) * 100 / $cycles_without" | bc)
        printf "%-30s %20s %20s %15.2f%%\n" "完成周期数" "$cycles_with" "$cycles_without" "$cycles_imp"
    else
        printf "%-30s %20s %20s %15s\n" "完成周期数" "$cycles_with" "$cycles_without" "N/A"
    fi
    
    echo ""
    echo "----------------------------------------"
    echo "文件位置"
    echo "----------------------------------------"
    echo "启用 Lattice-MAB 统计: $WITH_STATS"
    echo "禁用 Lattice-MAB 统计: $WITHOUT_STATS"
    echo "完整输出目录: $OUTPUT_DIR"
    echo ""
    
} | tee "$REPORT_FILE"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}测试完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "报告已保存到: ${BLUE}$REPORT_FILE${NC}"
echo -e "查看报告: ${YELLOW}cat $REPORT_FILE${NC}"
echo ""

