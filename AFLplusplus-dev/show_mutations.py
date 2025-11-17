#!/usr/bin/env python3
"""
展示 AFL++ 变异操作的实际效果
模拟各种变异操作对输入数据的影响
"""

import random
import struct

def flip_bit(data, pos, bit):
    """MUT_FLIPBIT: 翻转单个位"""
    byte_pos = pos // 8
    bit_pos = pos % 8
    if byte_pos < len(data):
        data[byte_pos] ^= (1 << bit_pos)
    return data

def arith8_add(data, pos, value):
    """MUT_ARITH8: 字节加法"""
    if pos < len(data):
        data[pos] = (data[pos] + value) & 0xFF
    return data

def interesting8_replace(data, pos, value):
    """MUT_INTERESTING8: 替换为特殊值"""
    if pos < len(data):
        data[pos] = value
    return data

def clone_copy(data, from_pos, to_pos, length):
    """MUT_CLONE_COPY: 克隆数据块"""
    if from_pos + length <= len(data) and to_pos + length <= len(data):
        chunk = data[from_pos:from_pos+length]
        data[to_pos:to_pos+length] = chunk
    return data

def delete_block(data, pos, length):
    """MUT_DEL: 删除数据块"""
    if pos + length <= len(data):
        data = data[:pos] + data[pos+length:]
    return data

def insert_one(data, pos, value):
    """MUT_INSERTONE: 插入单个字节"""
    data.insert(pos, value)
    return data

def switch_blocks(data, pos1, pos2, length):
    """MUT_SWITCH: 交换数据块"""
    if pos1 + length <= len(data) and pos2 + length <= len(data):
        block1 = data[pos1:pos1+length]
        block2 = data[pos2:pos2+length]
        data[pos1:pos1+length] = block2
        data[pos2:pos2+length] = block1
    return data

def splice_insert(data1, data2, pos, length):
    """MUT_SPLICE_INSERT: 从另一个输入插入数据"""
    if pos <= len(data1) and length <= len(data2):
        chunk = data2[:length]
        data1 = data1[:pos] + chunk + data1[pos:]
    return data1

def show_mutation(original, mutated, operation_name):
    """显示变异前后的对比"""
    orig_str = ''.join(chr(b) if 32 <= b < 127 else f'\\x{b:02x}' for b in original)
    mut_str = ''.join(chr(b) if 32 <= b < 127 else f'\\x{b:02x}' for b in mutated)
    
    print(f"\n操作: {operation_name}")
    print(f"原始: {orig_str}")
    print(f"变异: {mut_str}")
    print(f"原始(hex): {' '.join(f'{b:02x}' for b in original)}")
    print(f"变异(hex): {' '.join(f'{b:02x}' for b in mutated)}")

def demonstrate_mutations():
    """演示各种变异操作"""
    print("=" * 70)
    print("AFL++ 变异操作效果演示")
    print("=" * 70)
    
    # 原始输入
    original = bytearray(b"Hello")
    print(f"\n原始输入: {original.decode()}")
    print(f"原始(hex): {' '.join(f'{b:02x}' for b in original)}")
    
    # 1. 位翻转
    data = bytearray(original)
    flip_bit(data, 8, 0)  # 翻转第2个字节的第0位
    show_mutation(original, data, "MUT_FLIPBIT (翻转第2字节第0位)")
    
    # 2. 字节算术运算
    data = bytearray(original)
    arith8_add(data, 1, 2)  # 第2个字节加2
    show_mutation(original, data, "MUT_ARITH8 (第2字节+2)")
    
    # 3. 特殊值替换
    data = bytearray(original)
    interesting8_replace(data, 0, 0x00)  # 替换为0
    show_mutation(original, data, "MUT_INTERESTING8 (替换为0x00)")
    
    data = bytearray(original)
    interesting8_replace(data, 0, 0xFF)  # 替换为0xFF
    show_mutation(original, data, "MUT_INTERESTING8 (替换为0xFF)")
    
    # 4. 克隆数据块
    data = bytearray(original)
    clone_copy(data, 0, 2, 2)  # 从位置0克隆2字节到位置2
    show_mutation(original, data, "MUT_CLONE_COPY (克隆前2字节到位置2)")
    
    # 5. 删除数据块
    data = bytearray(original)
    data = delete_block(data, 1, 2)  # 从位置1删除2字节
    show_mutation(original, data, "MUT_DEL (删除位置1-2的2字节)")
    
    # 6. 插入字节
    data = bytearray(original)
    insert_one(data, 2, ord('X'))  # 在位置2插入'X'
    show_mutation(original, data, "MUT_INSERTONE (在位置2插入'X')")
    
    # 7. 交换数据块
    data = bytearray(b"HelloWorld")
    switch_blocks(data, 0, 5, 3)  # 交换前3字节和后3字节
    show_mutation(bytearray(b"HelloWorld"), data, "MUT_SWITCH (交换前3字节和后3字节)")
    
    # 8. 拼接插入
    data1 = bytearray(b"Hello")
    data2 = bytearray(b"World")
    result = splice_insert(data1, data2, 2, 3)  # 在位置2插入data2的前3字节
    show_mutation(data1, result, "MUT_SPLICE_INSERT (在位置2插入'World'的前3字节)")
    
    # 9. 组合变异 - 模拟 Havoc 阶段
    print("\n" + "=" * 70)
    print("组合变异示例 (模拟 Havoc 阶段)")
    print("=" * 70)
    
    data = bytearray(original)
    print(f"\n原始: {data.decode()}")
    
    # 执行多次随机变异
    mutations = [
        ("位翻转", lambda d: flip_bit(d, random.randint(0, len(d)*8-1), random.randint(0, 7))),
        ("字节加", lambda d: arith8_add(d, random.randint(0, len(d)-1), random.randint(1, 10))),
        ("插入字节", lambda d: insert_one(d, random.randint(0, len(d)), random.randint(32, 126))),
    ]
    
    for i in range(3):
        mut_name, mut_func = random.choice(mutations)
        data = mut_func(data)
        print(f"步骤{i+1} ({mut_name}): {data.decode() if all(32 <= b < 127 for b in data) else '包含非打印字符'}")

if __name__ == "__main__":
    demonstrate_mutations()
    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)

