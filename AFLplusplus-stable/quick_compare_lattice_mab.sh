#!/bin/bash
#
# Lattice-MAB 快速对比测试脚本
# 自动准备测试环境并运行对比测试
#

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Lattice-MAB 快速对比测试${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 默认参数
TEST_TIME=600   # 默认10分钟
NUM_RUNS=3      # 默认运行3次
TARGET_BINARY=""
INPUT_DIR=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -T|--time)
            TEST_TIME="$2"
            shift 2
            ;;
        -n|--runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        -t|--target)
            TARGET_BINARY="$2"
            shift 2
            ;;
        -i|--input)
            INPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  -T, --time       测试时间（秒，默认: 600）"
            echo "  -n, --runs       每个配置的运行次数（默认: 3）"
            echo "  -t, --target     目标二进制文件路径（可选，默认使用 test-instr）"
            echo "  -i, --input      输入种子目录（可选，默认使用 ./seeds）"
            echo "  -h, --help       显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0 -T 300 -n 2"
            echo "  $0 -t ./my_target -i ./my_seeds -T 600 -n 3"
            exit 0
            ;;
        *)
            echo -e "${RED}未知选项: $1${NC}"
            exit 1
            ;;
    esac
done

# 检查 afl-cc 和 afl-fuzz
if [ ! -f "./afl-cc" ]; then
    echo -e "${RED}错误: 找不到 afl-cc，请先编译 AFL++${NC}"
    echo "运行: make clean && make"
    exit 1
fi

if [ ! -f "./afl-fuzz" ]; then
    echo -e "${RED}错误: 找不到 afl-fuzz，请先编译 AFL++${NC}"
    echo "运行: make clean && make"
    exit 1
fi

# 准备测试程序
if [ -z "$TARGET_BINARY" ]; then
    echo -e "${YELLOW}准备测试程序...${NC}"
    
    if [ ! -f "./test-instr" ] || [ "./test-instr.c" -nt "./test-instr" ]; then
        echo "编译 test-instr..."
        ./afl-cc -o test-instr test-instr.c
        if [ $? -ne 0 ]; then
            echo -e "${RED}错误: 编译 test-instr 失败${NC}"
            exit 1
        fi
        echo -e "${GREEN}✓ 测试程序编译成功${NC}"
    else
        echo -e "${GREEN}✓ 测试程序已存在${NC}"
    fi
    
    TARGET_BINARY="./test-instr"
else
    if [ ! -f "$TARGET_BINARY" ]; then
        echo -e "${RED}错误: 目标二进制文件不存在: $TARGET_BINARY${NC}"
        exit 1
    fi
fi

# 准备种子目录
if [ -z "$INPUT_DIR" ]; then
    echo -e "${YELLOW}准备种子文件...${NC}"
    
    if [ ! -d "./seeds" ]; then
        mkdir -p seeds
        echo "0" > seeds/seed1.txt
        echo "1" > seeds/seed2.txt
        echo "hello" > seeds/seed3.txt
        echo -e "${GREEN}✓ 种子目录已创建${NC}"
    else
        # 检查种子目录是否为空
        if [ -z "$(ls -A ./seeds)" ]; then
            echo "0" > seeds/seed1.txt
            echo "1" > seeds/seed2.txt
            echo "hello" > seeds/seed3.txt
            echo -e "${GREEN}✓ 种子文件已添加${NC}"
        else
            echo -e "${GREEN}✓ 种子目录已存在${NC}"
        fi
    fi
    
    INPUT_DIR="./seeds"
else
    if [ ! -d "$INPUT_DIR" ]; then
        echo -e "${RED}错误: 输入目录不存在: $INPUT_DIR${NC}"
        exit 1
    fi
fi

# 检查对比脚本
if [ ! -f "./utils/compare_lattice_mab.sh" ]; then
    echo -e "${RED}错误: 找不到对比脚本: ./utils/compare_lattice_mab.sh${NC}"
    exit 1
fi

# 确保对比脚本有执行权限
chmod +x ./utils/compare_lattice_mab.sh

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}开始对比测试${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "目标程序: $TARGET_BINARY"
echo "输入目录: $INPUT_DIR"
echo "测试时间: ${TEST_TIME}秒"
echo "运行次数: $NUM_RUNS"
echo ""

# 运行对比测试
./utils/compare_lattice_mab.sh \
    -t "$TARGET_BINARY" \
    -i "$INPUT_DIR" \
    -T "$TEST_TIME" \
    -n "$NUM_RUNS"

echo ""
echo -e "${GREEN}测试完成！${NC}"
echo ""
echo "查看结果:"
echo "  cat comparison_results/comparison_*/summary.txt"

