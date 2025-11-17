#!/usr/bin/env python3
"""
AFL++ 变异操作演示脚本
展示如何使用 AFL++ 的变异功能对输入进行变异
"""

import os
import sys
import subprocess
import tempfile
import shutil

def print_section(title):
    """打印章节标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")

def create_seed_file(content, filename):
    """创建种子文件"""
    with open(filename, 'wb') as f:
        f.write(content.encode('utf-8') if isinstance(content, str) else content)
    print(f"✓ 创建种子文件: {filename}")
    print(f"  内容: {content[:50]}... (长度: {len(content)} 字节)")

def show_mutation_examples():
    """展示变异操作示例"""
    print_section("变异操作类型说明")
    
    mutations = [
        ("MUT_FLIPBIT", "翻转单个位", "Hello → Hfllo (第2个字节的第1位翻转)"),
        ("MUT_ARITH8", "字节算术运算", "ABC → ADC (B+2=D)"),
        ("MUT_INTERESTING8", "替换为特殊值", "A → \\x00 或 \\xff"),
        ("MUT_CLONE_COPY", "克隆数据块", "Hello → HeHellollo"),
        ("MUT_DEL", "删除数据块", "Hello → Hlo"),
        ("MUT_INSERTONE", "插入字节", "Hello → Hxello"),
        ("MUT_SWITCH", "交换数据块", "Hello → leHlo"),
        ("MUT_SPLICE_INSERT", "拼接插入", "Hello + World → HeWorllold"),
    ]
    
    for mut_type, desc, example in mutations:
        print(f"• {mut_type:20} - {desc}")
        print(f"  示例: {example}\n")

def demonstrate_mutations():
    """演示变异操作"""
    print_section("变异操作演示")
    
    # 创建临时目录
    demo_dir = tempfile.mkdtemp(prefix="aflpp_demo_")
    seeds_dir = os.path.join(demo_dir, "seeds")
    output_dir = os.path.join(demo_dir, "output")
    os.makedirs(seeds_dir)
    
    print(f"工作目录: {demo_dir}\n")
    
    # 创建种子文件
    seed_content = "Hello World"
    seed_file = os.path.join(seeds_dir, "seed1.txt")
    create_seed_file(seed_content, seed_file)
    
    # 检查是否有编译好的目标程序
    target_binary = "./demo_target"
    if not os.path.exists(target_binary):
        print(f"\n⚠ 警告: 未找到编译好的目标程序 {target_binary}")
        print("请先运行以下命令编译:")
        print(f"  ./afl-cc -o {target_binary} demo_target.c")
        print("\n或者使用普通编译器:")
        print(f"  gcc -o {target_binary} demo_target.c")
        return
    
    print(f"\n✓ 找到目标程序: {target_binary}")
    
    # 检查是否有 afl-fuzz
    afl_fuzz = "./afl-fuzz"
    if not os.path.exists(afl_fuzz):
        afl_fuzz = "afl-fuzz"
        try:
            subprocess.run([afl_fuzz, "-h"], 
                         capture_output=True, 
                         timeout=2, 
                         check=True)
        except:
            print(f"\n⚠ 警告: 未找到 afl-fuzz 工具")
            print("请确保 AFL++ 已正确安装并在 PATH 中")
            print("\n💡 提示:")
            print("  1. 如果 AFL++ 已编译但未安装，使用: ./afl-fuzz")
            print("  2. 查看安装指南: cat INSTALL_GUIDE_CN.md")
            print("  3. 或查看官方文档: cat docs/INSTALL.md")
            print("\n📝 您仍然可以查看变异操作演示（无需编译）:")
            print("  python3 show_mutations.py")
            return
    
    print(f"✓ 找到 AFL++ 工具: {afl_fuzz}")
    
    # 显示运行命令
    print_section("运行 AFL++ 模糊测试")
    cmd = [
        afl_fuzz,
        "-i", seeds_dir,
        "-o", output_dir,
        "--",
        target_binary, "@@"
    ]
    
    print("命令:")
    print("  " + " ".join(cmd))
    print("\n说明:")
    print("  -i: 输入种子目录")
    print("  -o: 输出目录（包含发现的测试用例、崩溃等）")
    print("  @@: 占位符，AFL++ 会用生成的测试文件替换")
    print("\n按 Ctrl+C 停止模糊测试")
    print("\n" + "-" * 60)
    
    try:
        # 运行 afl-fuzz（只运行几秒钟作为演示）
        print("\n开始运行 AFL++ (演示模式，5秒后自动停止)...\n")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # 读取输出
        import time
        start_time = time.time()
        while time.time() - start_time < 5:
            if process.poll() is not None:
                break
            line = process.stdout.readline()
            if line:
                print(line.rstrip())
            time.sleep(0.1)
        
        process.terminate()
        process.wait(timeout=2)
        
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
    finally:
        # 显示结果
        print_section("演示结果")
        if os.path.exists(output_dir):
            queue_dir = os.path.join(output_dir, "default", "queue")
            crashes_dir = os.path.join(output_dir, "default", "crashes")
            
            if os.path.exists(queue_dir):
                queue_files = [f for f in os.listdir(queue_dir) 
                             if f.startswith("id:")]
                print(f"✓ 生成的测试用例: {len(queue_files)} 个")
                if queue_files:
                    print(f"  示例: {queue_files[0]}")
            
            if os.path.exists(crashes_dir):
                crash_files = [f for f in os.listdir(crashes_dir) 
                             if f.startswith("id:")]
                if crash_files:
                    print(f"✓ 发现的崩溃: {len(crash_files)} 个")
                    print(f"  示例: {crash_files[0]}")
        
        print(f"\n完整输出目录: {output_dir}")
        print(f"清理命令: rm -rf {demo_dir}")

def show_mutation_details():
    """显示变异操作详细说明"""
    print_section("变异操作详细说明")
    
    print("""
AFL++ 的变异操作分为几个阶段：

1. **确定性阶段 (Deterministic)**
   - 对每个输入执行所有可能的确定性变异
   - 包括位翻转、算术运算、特殊值替换等
   - 确保覆盖所有可能的单步变异

2. **Havoc 阶段 (随机变异)**
   - 随机选择多种变异操作组合
   - 可以执行多次变异操作
   - 变异步数: 1-16 步（可配置）

3. **拼接阶段 (Splicing)**
   - 将两个不同的输入拼接
   - 生成新的测试用例
   - 有助于发现复杂的bug

变异策略会根据以下因素调整：
- 输入类型（文本 vs 二进制）
- 探索模式 vs 利用模式
- 覆盖率反馈
    """)

if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════╗
║          AFL++ 变异操作演示                                ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    show_mutation_examples()
    show_mutation_details()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        demonstrate_mutations()
    else:
        print_section("运行完整演示")
        print("要运行完整的模糊测试演示，请执行:")
        print("  python3 demo_mutation.py --run")
        print("\n或者直接运行 AFL++:")
        print("  ./afl-fuzz -i seeds_dir -o output_dir -- ./demo_target @@")

