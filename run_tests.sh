#!/bin/bash
# 运行所有单元测试

echo "运行 bert4torch 单元测试..."
echo "================================"
echo ""

PYTHONPATH=bert4torch:$PYTHONPATH python -m unittest discover tests -v

echo ""
echo "================================"
echo "测试完成！"
