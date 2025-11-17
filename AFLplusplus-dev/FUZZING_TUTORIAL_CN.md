# 模糊测试（Fuzzing）完全入门教程

## 🤔 什么是模糊测试？

**模糊测试（Fuzzing）** 就像是一个"自动测试员"，它会：
1. 自动生成大量随机的、异常的输入数据
2. 把这些数据喂给程序
3. 观察程序是否会崩溃、出错或产生异常行为
4. 如果发现问题，就记录下来

**简单比喻**：就像让一个机器人不停地用各种奇怪的方式敲键盘，看看程序会不会被"搞崩溃"。

## 🎯 为什么要用模糊测试？

传统测试需要人工编写测试用例，但：
- ❌ 人工测试覆盖有限
- ❌ 难以想到所有边界情况
- ❌ 耗时耗力

模糊测试可以：
- ✅ 自动生成成千上万的测试用例
- ✅ 发现人工难以想到的bug
- ✅ 24小时不间断运行
- ✅ 发现安全漏洞

## 📚 完整示例：从零开始

让我们用一个简单的例子，一步步演示模糊测试的完整流程。

### 步骤1：准备一个待测试的程序

假设我们有一个程序，它会读取用户输入并检查密码：

```c
// vulnerable_program.c
#include <stdio.h>
#include <string.h>

int main() {
    char password[10];
    char input[100];
    
    strcpy(password, "secret123");
    
    printf("请输入密码: ");
    fgets(input, 100, stdin);
    
    // 这里有个bug：没有检查输入长度！
    if (strcmp(input, password) == 0) {
        printf("密码正确！\n");
        return 0;
    } else {
        printf("密码错误！\n");
        return 1;
    }
}
```

这个程序有个潜在问题：如果输入超过10个字符，可能会溢出。

### 步骤2：编译程序（使用 AFL++ 插桩）

```bash
# 使用 AFL++ 的编译器编译（会插入代码来追踪程序执行）
./afl-cc -o vulnerable_program vulnerable_program.c

# 如果没有 AFL++，先用普通编译器（仅用于理解）
gcc -o vulnerable_program vulnerable_program.c
```

### 步骤3：准备种子文件

种子文件是模糊测试的"起点"，就像给机器人一个初始的"敲键盘"模板：

```bash
# 创建输入目录
mkdir seeds

# 创建一些初始输入（种子）
echo "test" > seeds/seed1.txt
echo "password" > seeds/seed2.txt
echo "12345" > seeds/seed3.txt
```

### 步骤4：运行模糊测试

```bash
./afl-fuzz -i seeds -o output -- ./vulnerable_program
```

**参数解释：**
- `-i seeds`：输入种子目录
- `-o output`：输出目录（存放发现的测试用例、崩溃等）
- `-- ./vulnerable_program`：要测试的程序

### 步骤5：观察结果

AFL++ 会显示一个实时界面，显示：
- 执行了多少次测试
- 发现了多少独特的路径（代码覆盖率）
- 发现了多少崩溃
- 当前执行速度

### 步骤6：查看发现的bug

```bash
# 查看发现的崩溃
ls output/default/crashes/

# 查看导致崩溃的输入
cat output/default/crashes/id:000000,sig:11,src:000000,op:flip1,pos:0
```

## 🎬 实际演示：用我们的 demo_target 程序

让我们用仓库中的 `demo_target.c` 来做一个完整的演示：

### 1. 查看目标程序

```bash
cat demo_target.c
```

这个程序会：
- 读取输入
- 检查是否包含 "PASSWORD"、"SECRET"、"CRASH" 等关键字
- 如果输入是 "CRASH"，程序会崩溃（故意设计的）

### 2. 编译程序

```bash
# 如果有 AFL++ 编译器
./afl-cc -o demo_target demo_target.c

# 或者用普通编译器（仅用于演示）
gcc -o demo_target demo_target.c
```

### 3. 手动测试一下

```bash
# 测试普通输入
echo "Hello" | ./demo_target
# 输出: 普通输入，长度: 5

# 测试关键字
echo "PASSWORD" | ./demo_target
# 输出: 发现密码关键字！

# 测试崩溃
echo "CRASH" | ./demo_target
# 输出: 发现崩溃关键字！然后程序崩溃
```

### 4. 准备种子文件

```bash
mkdir -p seeds
echo "Hello" > seeds/seed1.txt
echo "test" > seeds/seed2.txt
```

### 5. 运行模糊测试

```bash
./afl-fuzz -i seeds -o output -- ./demo_target
```

**AFL++ 会做什么？**

1. **读取种子文件**：从 `seeds/` 目录读取初始输入
2. **变异输入**：对种子进行各种变异操作
   - 翻转位：`Hello` → `Hdllo`
   - 插入字符：`Hello` → `HeXllo`
   - 删除字符：`Hello` → `Hlo`
   - 拼接：`Hello` + `test` → `Hetestlo`
   - ... 等等37+种变异操作
3. **执行程序**：用变异后的输入运行程序
4. **观察行为**：
   - 程序是否崩溃？
   - 是否触发了新的代码路径？
   - 是否发现了新的bug？
5. **保存有趣的输入**：
   - 导致崩溃的输入 → `output/default/crashes/`
   - 触发新路径的输入 → `output/default/queue/`
   - 导致超时的输入 → `output/default/hangs/`

### 6. 查看结果

运行一段时间后（可以按 Ctrl+C 停止），查看结果：

```bash
# 查看生成的测试用例
ls output/default/queue/
# 会看到很多文件，如：id:000000,orig:seed1.txt

# 查看是否发现崩溃
ls output/default/crashes/
# 如果发现了包含 "CRASH" 的输入，这里会有文件

# 查看一个测试用例的内容
cat output/default/queue/id:000000,orig:seed1.txt
```

## 🔍 理解 AFL++ 的工作原理

### 覆盖率引导（Coverage-Guided）

AFL++ 的核心是**覆盖率引导**：

1. **插桩（Instrumentation）**：
   - 编译时在程序中插入代码
   - 这些代码会记录"哪些代码被执行了"

2. **追踪执行路径**：
   - 每次运行程序，记录执行的代码路径
   - 用哈希值表示不同的路径

3. **优先变异**：
   - 如果某个输入触发了新的代码路径，说明它"有趣"
   - 优先对这个输入进行更多变异
   - 这样更容易发现bug

### 变异策略

AFL++ 使用多种变异操作：

```
原始输入: "Hello"
         ↓
    [变异操作]
         ↓
┌────────┴────────┐
│                 │
位翻转           插入字符
"Hello" → "Hdllo"  "Hello" → "HeXllo"
│                 │
删除字符         拼接
"Hello" → "Hlo"   "Hello" + "World" → "HeWorllo"
```

## 📊 实际运行示例

让我们创建一个可以立即运行的完整示例：

### 快速开始脚本

```bash
#!/bin/bash
# quick_fuzz_demo.sh

echo "=== AFL++ 模糊测试快速演示 ==="

# 1. 编译目标程序
echo "1. 编译目标程序..."
gcc -o demo_target demo_target.c
echo "✓ 编译完成"

# 2. 创建种子目录
echo "2. 准备种子文件..."
mkdir -p seeds
echo "Hello" > seeds/seed1.txt
echo "test" > seeds/seed2.txt
echo "✓ 种子文件准备完成"

# 3. 手动测试
echo "3. 手动测试程序..."
echo "输入 'Hello':"
echo "Hello" | ./demo_target
echo ""
echo "输入 'PASSWORD':"
echo "PASSWORD" | ./demo_target
echo ""

# 4. 说明如何运行模糊测试
echo "4. 运行模糊测试（需要 AFL++ 已编译）:"
echo "   ./afl-fuzz -i seeds -o output -- ./demo_target"
echo ""
echo "或者查看变异操作演示（无需 AFL++）:"
echo "   python3 show_mutations.py"
```

## 🎓 关键概念总结

| 概念 | 解释 | 类比 |
|------|------|------|
| **种子（Seed）** | 初始输入文件 | 给机器人的"初始模板" |
| **变异（Mutation）** | 修改输入数据 | 机器人"改变敲键盘的方式" |
| **覆盖率（Coverage）** | 执行了哪些代码 | 机器人"探索了哪些房间" |
| **崩溃（Crash）** | 程序异常退出 | 程序"被搞崩溃了" |
| **队列（Queue）** | 有趣的测试用例 | 机器人"发现的有趣输入" |

## 🚀 下一步

1. **运行演示脚本**：
   ```bash
   python3 show_mutations.py  # 查看变异操作效果
   ```

2. **编译 AFL++**（如果需要）：
   ```bash
   cat INSTALL_GUIDE_CN.md  # 查看安装指南
   ```

3. **运行实际模糊测试**：
   ```bash
   ./afl-fuzz -i seeds -o output -- ./demo_target
   ```

4. **分析结果**：
   - 查看 `output/default/crashes/` 中的崩溃
   - 用调试器分析崩溃原因
   - 修复bug

## 💡 常见问题

**Q: 模糊测试要运行多久？**
A: 通常运行几小时到几天。可以随时按 Ctrl+C 停止，已发现的结果会保存。

**Q: 如何知道模糊测试是否有效？**
A: 观察：
- 代码覆盖率是否在增长
- 是否发现了崩溃
- 执行速度是否正常

**Q: 模糊测试一定能找到bug吗？**
A: 不一定，但它是发现bug的有效方法。配合其他测试方法使用效果更好。

**Q: 需要多少种子文件？**
A: 通常1-10个就够了。AFL++ 会从这些种子开始变异。

## 📖 更多资源

- `README_CN.md` - 详细的变异操作说明
- `show_mutations.py` - 可视化变异操作
- `INSTALL_GUIDE_CN.md` - 安装指南

---

**记住**：模糊测试的核心就是"让程序处理各种奇怪输入，看它会不会出错"！

