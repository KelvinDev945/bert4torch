#!/bin/bash
# 运行所有示例

cd "$(dirname "$0")"

echo "======================================"
echo "运行所有示例"
echo "======================================"

echo -e "\n[1/6] 基础功能测试..."
python basic_test.py 2>&1 | tail -20

echo -e "\n[2/6] 文本分类示例..."
python task_sentiment_classification.py 2>&1 | tail -15

echo -e "\n[3/6] 序列标注示例..."
python task_ner_crf.py 2>&1 | tail -15

echo -e "\n[4/6] GPT 文本生成示例..."
python task_text_generation_gpt.py 2>&1 | tail -15

echo -e "\n[5/6] T5 Seq2Seq 示例..."
python task_seq2seq_t5.py 2>&1 | tail -15

echo -e "\n[6/6] 关系抽取示例..."
python task_relation_extraction.py 2>&1 | tail -15

echo -e "\n======================================"
echo "所有示例运行完成！"
echo "======================================"
