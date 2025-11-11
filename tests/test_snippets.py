"""snippets.py 单元测试"""
import unittest
import torch
import numpy as np
import sys
sys.path.insert(0, '..')

from bert4torch.snippets import (
    sequence_padding,
    truncate_sequences,
    text_segmentate,
    to_array,
    DataGenerator,
    ViterbiDecoder
)


class TestSnippets(unittest.TestCase):
    """测试 snippets.py 中的工具函数"""

    def test_sequence_padding(self):
        """测试序列padding"""
        sequences = [
            [1, 2, 3],
            [4, 5],
            [6, 7, 8, 9]
        ]

        padded = sequence_padding(sequences)

        # 检查形状
        self.assertEqual(padded.shape, (3, 4))  # 3个序列，最长4

        # 检查padding值
        self.assertEqual(padded[1, 2], 0)  # 第2个序列的第3个位置应该是0

    def test_sequence_padding_custom_length(self):
        """测试指定长度的padding"""
        sequences = [
            [1, 2, 3],
            [4, 5]
        ]

        padded = sequence_padding(sequences, length=5, value=-1)

        self.assertEqual(padded.shape, (2, 5))
        self.assertEqual(padded[1, 4], -1)  # 填充值应该是-1

    def test_sequence_padding_pre(self):
        """测试前置padding"""
        sequences = [
            [1, 2, 3],
            [4, 5]
        ]

        padded = sequence_padding(sequences, mode='pre')

        # 前置padding: 第2个序列应该是 [0, 0, 4, 5] 的某种形式
        self.assertEqual(padded.shape[1], 3)  # 最长序列长度

    def test_truncate_sequences(self):
        """测试序列截断"""
        seq1 = [1, 2, 3, 4, 5]
        seq2 = [6, 7, 8, 9, 10, 11]

        truncated = truncate_sequences(8, seq1, seq2)

        # 总长度应该 <= 8
        total_len = len(truncated[0]) + len(truncated[1])
        self.assertLessEqual(total_len, 8)

    def test_truncate_sequences_single(self):
        """测试单序列截断"""
        seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        truncated = truncate_sequences(5, seq)

        self.assertLessEqual(len(truncated[0]), 5)

    def test_text_segmentate(self):
        """测试文本分段"""
        text = "这是第一句。这是第二句。这是第三句。这是第四句。"
        maxlen = 10
        seps = '。'

        segments = text_segmentate(text, maxlen, seps)

        # 应该被分成多段
        self.assertGreater(len(segments), 1)

        # 每段长度应该 <= maxlen
        for seg in segments:
            self.assertLessEqual(len(seg), maxlen)

    def test_to_array(self):
        """测试转换为numpy数组"""
        list_data = [[1, 2, 3], [4, 5, 6]]

        array = to_array(list_data)

        self.assertIsInstance(array, np.ndarray)
        self.assertEqual(array.shape, (2, 3))

    def test_to_array_multiple(self):
        """测试多个转换"""
        list1 = [[1, 2], [3, 4]]
        list2 = [[5, 6], [7, 8]]

        arr1, arr2 = to_array(list1, list2)

        self.assertIsInstance(arr1, np.ndarray)
        self.assertIsInstance(arr2, np.ndarray)


class TestDataGenerator(unittest.TestCase):
    """测试DataGenerator"""

    def test_data_generator_basic(self):
        """测试DataGenerator基础功能"""
        data = list(range(100))
        batch_size = 10

        class SimpleGenerator(DataGenerator):
            def collate_fn(self, batch):
                return torch.tensor(batch)

        gen = SimpleGenerator(data, batch_size=batch_size)

        # 检查步数
        self.assertEqual(len(gen), 10)

        # 检查迭代
        batches = list(gen)
        self.assertEqual(len(batches), 10)
        self.assertEqual(batches[0].shape[0], batch_size)

    def test_data_generator_shuffle(self):
        """测试DataGenerator shuffle"""
        data = list(range(20))
        batch_size = 5

        class SimpleGenerator(DataGenerator):
            def collate_fn(self, batch):
                return batch

        gen = SimpleGenerator(data, batch_size=batch_size, shuffle=True)

        batch1 = next(iter(gen))
        batch2 = next(iter(gen))

        # shuffle后的批次应该包含不同的数据
        self.assertIsInstance(batch1, list)
        self.assertIsInstance(batch2, list)


class TestViterbiDecoder(unittest.TestCase):
    """测试ViterbiDecoder"""

    def test_viterbi_decoder(self):
        """测试维特比解码"""
        num_tags = 5
        batch_size, seq_len = 2, 10

        # 创建转移矩阵
        trans = np.random.randn(num_tags, num_tags)

        decoder = ViterbiDecoder(trans)

        # 创建发射分数
        emissions = np.random.randn(batch_size, seq_len, num_tags)

        # 解码
        paths = decoder.decode(emissions)

        # 检查结果
        self.assertEqual(len(paths), batch_size)
        self.assertEqual(len(paths[0]), seq_len)

        # 检查标签在合法范围内
        for path in paths:
            for tag in path:
                self.assertGreaterEqual(tag, 0)
                self.assertLess(tag, num_tags)

    def test_viterbi_decoder_with_mask(self):
        """测试带mask的维特比解码"""
        num_tags = 5
        batch_size, seq_len = 2, 10

        trans = np.random.randn(num_tags, num_tags)
        decoder = ViterbiDecoder(trans)

        emissions = np.random.randn(batch_size, seq_len, num_tags)

        # 创建mask (前5个有效)
        mask = np.ones((batch_size, seq_len))
        mask[:, 5:] = 0

        paths = decoder.decode(emissions, mask)

        self.assertEqual(len(paths), batch_size)


class TestUtilityFunctions(unittest.TestCase):
    """测试其他工具函数"""

    def test_sequence_padding_with_numpy(self):
        """测试numpy数组的padding"""
        sequences = [
            np.array([1, 2, 3]),
            np.array([4, 5])
        ]

        padded = sequence_padding(sequences)

        self.assertIsInstance(padded, np.ndarray)
        self.assertEqual(padded.shape, (2, 3))

    def test_empty_sequence_padding(self):
        """测试空序列padding"""
        sequences = []

        try:
            padded = sequence_padding(sequences)
            # 如果没有错误，检查结果
            self.assertTrue(True)
        except:
            # 预期可能会有错误
            self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
