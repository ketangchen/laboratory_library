#!/bin/bash
#
# AFL++ Lattice-MAB 对比测试脚本
# 用于对比启用和禁用 Lattice-MAB 的性能差异
#

set -e

# macOS 兼容性：实现 timeout 功能
timeout_cmd() {
    local duration=$1
    shift
    
    if command -v timeout &> /dev/null; then
        # Linux 系统有 timeout 命令
        timeout "$duration" "$@"
    elif command -v gtimeout &> /dev/null; then
        # macOS 安装了 coreutils
        gtimeout "$duration" "$@"
    else
        # macOS 原生实现：使用后台进程和 kill
        # 注意：这里不使用 set -e，因为我们需要处理超时
        "$@" &
        local pid=$!
        (
            sleep "$duration"
            kill -TERM "$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
        ) &
        local killer=$!
        # 等待进程结束或超时
        if wait "$pid" 2>/dev/null; then
            local exit_code=$?
        else
            local exit_code=124  # timeout 退出码
        fi
        kill "$killer" 2>/dev/null || true
        return $exit_code
    fi
}

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认参数
TARGET_BINARY=""
INPUT_DIR=""
OUTPUT_BASE_DIR="./comparison_results"
TEST_TIME=300  # 默认测试5分钟
NUM_RUNS=3     # 每个配置运行3次

# 解析命令行参数
usage() {
    echo "用法: $0 -t <target_binary> -i <input_dir> [选项]"
    echo ""
    echo "选项:"
    echo "  -t, --target     目标二进制文件路径（必需）"
    echo "  -i, --input      输入种子目录（必需）"
    echo "  -o, --output     输出基础目录（默认: $OUTPUT_BASE_DIR）"
    echo "  -T, --time       测试时间（秒，默认: $TEST_TIME）"
    echo "  -n, --runs       每个配置的运行次数（默认: $NUM_RUNS）"
    echo "  -h, --help       显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 -t ./test_target -i ./seeds -T 600 -n 5"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--target)
            TARGET_BINARY="$2"
            shift 2
            ;;
        -i|--input)
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_BASE_DIR="$2"
            shift 2
            ;;
        -T|--time)
            TEST_TIME="$2"
            shift 2
            ;;
        -n|--runs)
            NUM_RUNS="$2"
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
if [[ -z "$TARGET_BINARY" || -z "$INPUT_DIR" ]]; then
    echo -e "${RED}错误: 必须指定目标二进制文件和输入目录${NC}"
    usage
fi

if [[ ! -f "$TARGET_BINARY" ]]; then
    echo -e "${RED}错误: 目标二进制文件不存在: $TARGET_BINARY${NC}"
    exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo -e "${RED}错误: 输入目录不存在: $INPUT_DIR${NC}"
    exit 1
fi

# 查找 afl-fuzz
AFL_FUZZ=""
if command -v afl-fuzz &> /dev/null; then
    # 如果在 PATH 中找到
    AFL_FUZZ="afl-fuzz"
elif [[ -f "./afl-fuzz" ]]; then
    # 如果在当前目录找到
    AFL_FUZZ="./afl-fuzz"
elif [[ -f "../afl-fuzz" ]]; then
    # 如果在上一级目录找到（从 utils/ 目录运行）
    AFL_FUZZ="../afl-fuzz"
else
    # 尝试从脚本位置查找
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    AFL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    if [[ -f "$AFL_ROOT/afl-fuzz" ]]; then
        AFL_FUZZ="$AFL_ROOT/afl-fuzz"
    else
        echo -e "${RED}错误: 找不到 afl-fuzz 命令，请先编译 AFL++${NC}"
        echo "请确保在 AFL++ 根目录运行此脚本，或确保 afl-fuzz 在 PATH 中"
        exit 1
    fi
fi

echo -e "${GREEN}使用 afl-fuzz: $AFL_FUZZ${NC}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}AFL++ Lattice-MAB 对比测试${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "目标程序: $TARGET_BINARY"
echo "输入目录: $INPUT_DIR"
echo "测试时间: ${TEST_TIME}秒"
echo "运行次数: $NUM_RUNS"
echo "输出目录: $OUTPUT_BASE_DIR"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_BASE_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$OUTPUT_BASE_DIR/comparison_$TIMESTAMP"
mkdir -p "$RESULTS_DIR"

# 创建结果文件
SUMMARY_FILE="$RESULTS_DIR/summary.txt"
JSON_FILE="$RESULTS_DIR/results.json"

# 初始化 JSON 文件
echo "{" > "$JSON_FILE"
echo "  \"target\": \"$TARGET_BINARY\"," >> "$JSON_FILE"
echo "  \"input_dir\": \"$INPUT_DIR\"," >> "$JSON_FILE"
echo "  \"test_time\": $TEST_TIME," >> "$JSON_FILE"
echo "  \"num_runs\": $NUM_RUNS," >> "$JSON_FILE"
echo "  \"timestamp\": \"$TIMESTAMP\"," >> "$JSON_FILE"
echo "  \"results\": {" >> "$JSON_FILE"
echo "    \"with_lattice_mab\": []," >> "$JSON_FILE"
echo "    \"without_lattice_mab\": []" >> "$JSON_FILE"
echo "  }" >> "$JSON_FILE"
echo "}" >> "$JSON_FILE"

# 提取指标的函数
extract_metrics() {
    local output_dir=$1
    local run_num=$2
    
    # 从 fuzzer_stats 文件读取指标
    local stats_file="$output_dir/fuzzer_stats"
    if [[ ! -f "$stats_file" ]]; then
        echo "{}"
        return
    fi
    
    # 提取关键指标
    local execs=$(grep "^execs_done" "$stats_file" | awk '{print $3}' || echo "0")
    local paths=$(grep "^paths_total" "$stats_file" | awk '{print $3}' || echo "0")
    local crashes=$(grep "^unique_crashes" "$stats_file" | awk '{print $3}' || echo "0")
    local hangs=$(grep "^unique_hangs" "$stats_file" | awk '{print $3}' || echo "0")
    local exec_speed=$(grep "^execs_per_sec" "$stats_file" | awk '{print $3}' || echo "0")
    local cycles=$(grep "^cycles_done" "$stats_file" | awk '{print $3}' || echo "0")
    
    # 计算覆盖率（如果可用）
    local coverage=0
    if [[ -f "$output_dir/plot_data" ]]; then
        coverage=$(tail -1 "$output_dir/plot_data" | awk '{print $4}' || echo "0")
    fi
    
    echo "{\"execs\": $execs, \"paths\": $paths, \"crashes\": $crashes, \"hangs\": $hangs, \"exec_speed\": $exec_speed, \"cycles\": $cycles, \"coverage\": $coverage}"
}

# 运行测试的函数
run_test() {
    local config_name=$1
    local disable_lm=$2
    local run_num=$3
    
    local output_dir="$RESULTS_DIR/${config_name}_run${run_num}"
    mkdir -p "$output_dir"
    
    echo -e "${YELLOW}运行 $config_name (第 $run_num 次)...${NC}"
    echo "  输出目录: $output_dir"
    echo "  运行时间: ${TEST_TIME}秒"
    
    # 设置环境变量
    if [[ "$disable_lm" == "1" ]]; then
        export AFL_DISABLE_LATTICE_MAB=1
        echo "  配置: 禁用 Lattice-MAB"
    else
        unset AFL_DISABLE_LATTICE_MAB
        echo "  配置: 启用 Lattice-MAB"
    fi
    
    # 运行 afl-fuzz（使用兼容的 timeout 函数）
    echo "  开始运行 afl-fuzz..."
    # 使用 -n 选项以支持非插桩二进制文件（用于对比测试）
    timeout_cmd "$TEST_TIME" "$AFL_FUZZ" -n -i "$INPUT_DIR" -o "$output_dir" -- "$TARGET_BINARY" @@ 2>&1 | tee "$output_dir/fuzzer.log" || true
    echo -e "${GREEN}  完成${NC}"
    
    # 提取指标
    local metrics=$(extract_metrics "$output_dir" "$run_num")
    echo "$metrics"
}

# 计算平均值
calculate_average() {
    local values=("$@")
    local sum=0
    local count=${#values[@]}
    
    for val in "${values[@]}"; do
        sum=$(echo "$sum + $val" | bc)
    done
    
    if [[ $count -gt 0 ]]; then
        echo "scale=2; $sum / $count" | bc
    else
        echo "0"
    fi
}

# 运行对比测试
echo -e "${GREEN}开始对比测试...${NC}"
echo ""

# 测试启用 Lattice-MAB
echo -e "${BLUE}=== 测试 1: 启用 Lattice-MAB ===${NC}"
with_lm_execs=()
with_lm_paths=()
with_lm_crashes=()
with_lm_speed=()

for i in $(seq 1 $NUM_RUNS); do
    metrics=$(run_test "with_lattice_mab" "0" "$i")
    execs=$(echo "$metrics" | grep -o '"execs": [0-9]*' | grep -o '[0-9]*')
    paths=$(echo "$metrics" | grep -o '"paths": [0-9]*' | grep -o '[0-9]*')
    crashes=$(echo "$metrics" | grep -o '"crashes": [0-9]*' | grep -o '[0-9]*')
    speed=$(echo "$metrics" | grep -o '"exec_speed": [0-9.]*' | grep -o '[0-9.]*')
    
    with_lm_execs+=("${execs:-0}")
    with_lm_paths+=("${paths:-0}")
    with_lm_crashes+=("${crashes:-0}")
    with_lm_speed+=("${speed:-0}")
    
    sleep 2
done

# 测试禁用 Lattice-MAB
echo ""
echo -e "${BLUE}=== 测试 2: 禁用 Lattice-MAB（原始策略）===${NC}"
without_lm_execs=()
without_lm_paths=()
without_lm_crashes=()
without_lm_speed=()

for i in $(seq 1 $NUM_RUNS); do
    metrics=$(run_test "without_lattice_mab" "1" "$i")
    execs=$(echo "$metrics" | grep -o '"execs": [0-9]*' | grep -o '[0-9]*')
    paths=$(echo "$metrics" | grep -o '"paths": [0-9]*' | grep -o '[0-9]*')
    crashes=$(echo "$metrics" | grep -o '"crashes": [0-9]*' | grep -o '[0-9]*')
    speed=$(echo "$metrics" | grep -o '"exec_speed": [0-9.]*' | grep -o '[0-9.]*')
    
    without_lm_execs+=("${execs:-0}")
    without_lm_paths+=("${paths:-0}")
    without_lm_crashes+=("${crashes:-0}")
    without_lm_speed+=("${speed:-0}")
    
    sleep 2
done

# 计算平均值
avg_with_execs=$(calculate_average "${with_lm_execs[@]}")
avg_with_paths=$(calculate_average "${with_lm_paths[@]}")
avg_with_crashes=$(calculate_average "${with_lm_crashes[@]}")
avg_with_speed=$(calculate_average "${with_lm_speed[@]}")

avg_without_execs=$(calculate_average "${without_lm_execs[@]}")
avg_without_paths=$(calculate_average "${without_lm_paths[@]}")
avg_without_crashes=$(calculate_average "${without_lm_crashes[@]}")
avg_without_speed=$(calculate_average "${without_lm_speed[@]}")

# 生成对比报告
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}对比测试结果${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

{
    echo "AFL++ Lattice-MAB 对比测试报告"
    echo "================================"
    echo ""
    echo "测试时间: $(date)"
    echo "目标程序: $TARGET_BINARY"
    echo "输入目录: $INPUT_DIR"
    echo "测试时长: ${TEST_TIME}秒"
    echo "运行次数: $NUM_RUNS"
    echo ""
    echo "----------------------------------------"
    echo "指标对比（平均值）"
    echo "----------------------------------------"
    echo ""
    printf "%-30s %15s %15s %15s\n" "指标" "启用Lattice-MAB" "禁用Lattice-MAB" "提升/下降"
    echo "----------------------------------------"
    
    # 执行次数
    exec_improvement=$(echo "scale=2; ($avg_with_execs - $avg_without_execs) * 100 / $avg_without_execs" | bc)
    printf "%-30s %15.0f %15.0f %15.2f%%\n" "总执行次数" "$avg_with_execs" "$avg_without_execs" "$exec_improvement"
    
    # 发现路径数
    paths_improvement=$(echo "scale=2; ($avg_with_paths - $avg_without_paths) * 100 / $avg_without_paths" | bc)
    printf "%-30s %15.0f %15.0f %15.2f%%\n" "发现路径数" "$avg_with_paths" "$avg_without_paths" "$paths_improvement"
    
    # 崩溃数
    crashes_improvement=$(echo "scale=2; ($avg_with_crashes - $avg_without_crashes) * 100 / (${avg_without_crashes:-1})" | bc)
    printf "%-30s %15.0f %15.0f %15.2f%%\n" "发现崩溃数" "$avg_with_crashes" "$avg_without_crashes" "$crashes_improvement"
    
    # 执行速度
    speed_improvement=$(echo "scale=2; ($avg_with_speed - $avg_without_speed) * 100 / $avg_without_speed" | bc)
    printf "%-30s %15.2f %15.2f %15.2f%%\n" "执行速度(execs/s)" "$avg_with_speed" "$avg_without_speed" "$speed_improvement"
    
    echo ""
    echo "----------------------------------------"
    echo "详细数据"
    echo "----------------------------------------"
    echo ""
    echo "启用 Lattice-MAB:"
    echo "  执行次数: ${with_lm_execs[*]}"
    echo "  路径数: ${with_lm_paths[*]}"
    echo "  崩溃数: ${with_lm_crashes[*]}"
    echo "  速度: ${with_lm_speed[*]}"
    echo ""
    echo "禁用 Lattice-MAB:"
    echo "  执行次数: ${without_lm_execs[*]}"
    echo "  路径数: ${without_lm_paths[*]}"
    echo "  崩溃数: ${without_lm_crashes[*]}"
    echo "  速度: ${without_lm_speed[*]}"
    
} | tee "$SUMMARY_FILE"

echo ""
echo -e "${GREEN}测试完成！结果已保存到: $RESULTS_DIR${NC}"
echo -e "${BLUE}查看详细报告: cat $SUMMARY_FILE${NC}"

