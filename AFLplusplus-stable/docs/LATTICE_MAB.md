# Lattice-based Mutation Strategy with Multi-Armed Bandit

## 概述

本模块实现了基于格理论（Lattice Theory）的变异策略优化系统，结合多臂老虎机（Multi-Armed Bandit, MAB）算法来智能选择变异操作。

## 核心概念

### 1. 变异向量（Mutation Vector）

将每个变异操作形式化为一个8维向量：
- `mut_type`: 变异类型（MUT_FLIPBIT, MUT_ARITH8等）
- `position`: 位置（归一化到0-100）
- `magnitude`: 变异幅度（0-100）
- `dimension`: 影响的维度（bit/byte/word/dword）
- `direction`: 方向（add/sub/flip/overwrite）
- `context`: 上下文（text/binary/ascii）
- `frequency`: 使用频率
- `effectiveness`: 历史有效性得分

### 2. 格（Lattice）结构

策略空间被建模为高维离散格，每个格点代表一个变异向量。格结构支持：
- **最近邻搜索**：找到与当前向量最相似的变异策略
- **正交性计算**：衡量两个变异向量的相似性
- **密度分析**：分析策略空间的分布

### 3. 多臂老虎机（MAB）

使用UCB（Upper Confidence Bound）算法进行策略选择：
- **探索-利用平衡**：在探索新策略和利用已知有效策略之间平衡
- **自适应学习**：根据历史奖励动态调整策略选择概率
- **奖励机制**：基于发现新路径、崩溃和执行时间计算奖励

## 使用方法

系统在初始化时自动启用，无需额外配置。默认使用混合模式（70% MAB + 30% Lattice）。

### 配置选项

可以通过修改 `lattice_mab_state_t` 结构中的标志来控制行为：

```c
lm->use_lattice = 1;    // 启用格理论选择
lm->use_mab = 1;        // 启用MAB选择
lm->use_hybrid = 1;     // 启用混合模式
lm->adaptive_mode = 1;  // 自适应模式
```

## 算法流程

1. **初始化阶段**：
   - 构建初始格结构（从现有变异数组）
   - 初始化MAB的每个臂（对应每种变异类型）

2. **选择阶段**（在havoc_stage）：
   - 根据配置选择使用MAB、Lattice或混合模式
   - MAB模式：使用UCB算法选择最优臂
   - Lattice模式：找到最近邻的变异向量
   - 混合模式：70%概率使用MAB，30%概率使用Lattice

3. **执行阶段**：
   - 执行选定的变异操作
   - 运行目标程序
   - 收集执行结果（新路径、崩溃、执行时间）

4. **更新阶段**：
   - 计算奖励（基于执行结果）
   - 更新MAB臂的统计信息
   - 更新格中向量的有效性得分

## 奖励函数

奖励计算考虑以下因素：

```c
double reward = 0.0;

if (found_crash) {
    reward += 100.0;  // 发现崩溃：高奖励
}

if (found_new_path) {
    reward += 10.0;   // 发现新路径：中等奖励
}

// 执行时间惩罚
if (exec_time > base_time) {
    reward -= penalty;  // 慢速执行：小惩罚
}
```

## 统计信息

系统跟踪以下统计信息：
- `lattice_selections`: 格理论选择的次数
- `mab_selections`: MAB选择的次数
- `hybrid_selections`: 混合选择的次数
- `avg_reward`: 平均奖励
- `best_reward`: 最佳奖励
- `best_arm_id`: 最佳臂ID

## 性能考虑

- 格结构最大支持1024个向量点
- MAB最多支持256个臂（对应MUT_MAX个变异类型）
- 选择算法的时间复杂度：O(n)，其中n是变异数组大小
- 更新操作的时间复杂度：O(1)

## 未来改进方向

1. **动态格构建**：根据实际使用情况动态调整格结构
2. **更复杂的奖励函数**：考虑更多因素（如路径深度、覆盖率等）
3. **并行MAB**：为不同的输入类型维护独立的MAB实例
4. **格优化**：使用更高级的格算法（如LLL算法）优化向量分布

## 参考文献

- Upper Confidence Bound (UCB) Algorithm
- Lattice Theory in Optimization
- Multi-Armed Bandit Problems

