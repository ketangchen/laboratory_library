#!/bin/bash
# AFL++ 模糊测试快速演示脚本

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║        AFL++ 模糊测试完全入门演示                          ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# 1. 编译目标程序
echo "【步骤 1】编译目标程序..."
if gcc -o demo_target demo_target.c 2>/dev/null; then
    echo "✓ 编译成功: demo_target"
else
    echo "✗ 编译失败，请检查 demo_target.c 是否存在"
    exit 1
fi
echo ""

# 2. 创建种子目录
echo "【步骤 2】准备种子文件（初始输入）..."
mkdir -p seeds
echo "Hello" > seeds/seed1.txt
echo "test" > seeds/seed2.txt
echo "12345" > seeds/seed3.txt
echo "✓ 创建了 3 个种子文件在 seeds/ 目录"
echo "  种子文件内容："
cat seeds/seed1.txt | sed 's/^/    - /'
cat seeds/seed2.txt | sed 's/^/    - /'
cat seeds/seed3.txt | sed 's/^/    - /'
echo ""

# 3. 手动测试程序
echo "【步骤 3】手动测试程序（了解程序行为）..."
echo ""
echo "测试 1: 普通输入"
echo "  输入: Hello"
echo -n "  输出: "
echo "Hello" | ./demo_target 2>&1 | head -1
echo ""

echo "测试 2: 包含关键字 'PASSWORD'"
echo "  输入: PASSWORD"
echo -n "  输出: "
echo "PASSWORD" | ./demo_target 2>&1 | head -1
echo ""

echo "测试 3: 包含关键字 'CRASH'（会崩溃）"
echo "  输入: CRASH"
echo -n "  输出: "
echo "CRASH" | ./demo_target 2>&1 | head -1 || echo "  (程序崩溃 - 这是预期的！)"
echo ""

# 4. 解释模糊测试过程
echo "【步骤 4】模糊测试会做什么？"
echo ""
echo "AFL++ 会："
echo "  1. 读取种子文件: seeds/seed1.txt, seed2.txt, seed3.txt"
echo "  2. 对种子进行变异（37+ 种变异操作）:"
echo "     - 位翻转: 'Hello' → 'Hdllo'"
echo "     - 插入字符: 'Hello' → 'HeXllo'"
echo "     - 删除字符: 'Hello' → 'Hlo'"
echo "     - 拼接: 'Hello' + 'test' → 'Hetestlo'"
echo "     - ... 等等"
echo "  3. 用变异后的输入运行程序"
echo "  4. 观察程序是否："
echo "     - 崩溃了？ → 保存到 output/default/crashes/"
echo "     - 触发了新代码路径？ → 保存到 output/default/queue/"
echo "     - 超时了？ → 保存到 output/default/hangs/"
echo "  5. 对'有趣'的输入进行更多变异"
echo ""

# 5. 检查 AFL++ 是否可用
echo "【步骤 5】检查 AFL++ 是否可用..."
if [ -f "./afl-fuzz" ]; then
    echo "✓ 找到 AFL++ (./afl-fuzz)"
    AFL_AVAILABLE=1
elif command -v afl-fuzz >/dev/null 2>&1; then
    echo "✓ 找到 AFL++ (系统 PATH 中)"
    AFL_AVAILABLE=1
else
    echo "⚠ AFL++ 未找到"
    echo "  您可以："
    echo "    1. 查看安装指南: cat INSTALL_GUIDE_CN.md"
    echo "    2. 或先查看变异操作演示: python3 show_mutations.py"
    AFL_AVAILABLE=0
fi
echo ""

# 6. 运行命令提示
if [ $AFL_AVAILABLE -eq 1 ]; then
    echo "【步骤 6】运行模糊测试"
    echo ""
    echo "执行以下命令开始模糊测试："
    echo ""
    echo "  ./afl-fuzz -i seeds -o output -- ./demo_target"
    echo ""
    echo "参数说明："
    echo "  -i seeds      : 输入种子目录"
    echo "  -o output     : 输出目录（存放结果）"
    echo "  --            : 分隔符"
    echo "  ./demo_target : 要测试的程序"
    echo ""
    echo "运行后会显示实时界面，按 Ctrl+C 停止"
    echo ""
    echo "查看结果："
    echo "  ls output/default/crashes/  # 查看崩溃"
    echo "  ls output/default/queue/    # 查看测试用例"
else
    echo "【步骤 6】查看变异操作演示（无需 AFL++）"
    echo ""
    echo "运行以下命令查看变异操作效果："
    echo ""
    echo "  python3 show_mutations.py"
    echo ""
    echo "这会展示各种变异操作如何修改输入数据"
fi

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  更多信息请查看: FUZZING_TUTORIAL_CN.md                  ║"
echo "╚═══════════════════════════════════════════════════════════╝"

