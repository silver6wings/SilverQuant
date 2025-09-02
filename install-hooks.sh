#!/bin/bash
# 自动安装Git钩子：将scripts/git-hooks下的脚本链接到.git/hooks

HOOKS_DIR=".git/hooks"
SOURCE_HOOKS_DIR="scripts/git-hooks"

# 确保源钩子目录存在
if [ ! -d "$SOURCE_HOOKS_DIR" ]; then
    echo "❌ 钩子源目录 $SOURCE_HOOKS_DIR 不存在"
    exit 1
fi

# 遍历源钩子目录，创建符号链接
for hook in "$SOURCE_HOOKS_DIR"/*; do
    hook_name=$(basename "$hook")
    target="$HOOKS_DIR/$hook_name"

    # 如果目标不存在或不是符号链接，创建链接
    if [ ! -L "$target" ]; then
        ln -s ../../"$hook" "$target"
        chmod +x "$target"  # 确保钩子可执行
        echo "🔗 已安装钩子: $hook_name"
    else
        echo "ℹ️ 钩子已存在: $hook_name"
    fi
done

echo "✅ 钩子安装完成"
