/*
 * 演示目标程序 - 用于展示 AFL++ 的变异操作
 * 这个程序会读取输入并检查特定的字符串模式
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv) {
    char buffer[1024];
    size_t len;
    
    // 从标准输入读取数据
    if (argc > 1) {
        // 从文件读取
        FILE *f = fopen(argv[1], "rb");
        if (!f) {
            fprintf(stderr, "无法打开文件: %s\n", argv[1]);
            return 1;
        }
        len = fread(buffer, 1, sizeof(buffer) - 1, f);
        fclose(f);
    } else {
        // 从标准输入读取
        len = read(0, buffer, sizeof(buffer) - 1);
    }
    
    if (len <= 0) {
        printf("输入为空\n");
        return 1;
    }
    
    buffer[len] = '\0';
    
    // 检查各种模式
    if (strstr(buffer, "PASSWORD")) {
        printf("发现密码关键字！\n");
    } else if (strstr(buffer, "SECRET")) {
        printf("发现秘密关键字！\n");
    } else if (strstr(buffer, "CRASH")) {
        printf("发现崩溃关键字！\n");
        // 故意触发崩溃
        abort();
    } else if (strstr(buffer, "BUG")) {
        printf("发现BUG关键字！\n");
    } else if (memcmp(buffer, "MAGIC", 5) == 0) {
        printf("发现魔法数字！\n");
    } else {
        printf("普通输入，长度: %zu\n", len);
    }
    
    return 0;
}

