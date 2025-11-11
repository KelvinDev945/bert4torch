"""tokenizers.py 单元测试"""
import unittest
import tempfile
import os
import sys
sys.path.insert(0, '..')

from bert4torch.tokenizers import (
    load_vocab,
    save_vocab,
    TokenizerBase,
    Tokenizer
)


class TestVocabIO(unittest.TestCase):
    """测试词表加载和保存"""

    def setUp(self):
        """创建临时词表文件"""
        self.temp_dir = tempfile.mkdtemp()
        self.vocab_path = os.path.join(self.temp_dir, 'vocab.txt')

        # 创建简单的词表
        self.vocab = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[CLS]': 2,
            '[SEP]': 3,
            '[MASK]': 4,
            'hello': 5,
            'world': 6,
            '##ing': 7,
            '中': 8,
            '文': 9,
        }

        # 保存词表
        with open(self.vocab_path, 'w', encoding='utf-8') as f:
            for token, _ in sorted(self.vocab.items(), key=lambda x: x[1]):
                f.write(token + '\n')

    def tearDown(self):
        """清理临时文件"""
        if os.path.exists(self.vocab_path):
            os.remove(self.vocab_path)
        os.rmdir(self.temp_dir)

    def test_load_vocab(self):
        """测试加载词表"""
        token_to_id = load_vocab(self.vocab_path)

        self.assertEqual(len(token_to_id), len(self.vocab))
        self.assertEqual(token_to_id['[CLS]'], 2)
        self.assertEqual(token_to_id['hello'], 5)
        self.assertEqual(token_to_id['中'], 8)

    def test_save_vocab(self):
        """测试保存词表"""
        new_vocab_path = os.path.join(self.temp_dir, 'vocab_new.txt')
        save_vocab(self.vocab, new_vocab_path)

        # 重新加载并验证
        loaded_vocab = load_vocab(new_vocab_path)
        self.assertEqual(loaded_vocab, self.vocab)

        # 清理
        os.remove(new_vocab_path)


class TestTokenizerBase(unittest.TestCase):
    """测试TokenizerBase基类"""

    def setUp(self):
        """创建测试词表"""
        self.vocab = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[CLS]': 2,
            '[SEP]': 3,
            'hello': 4,
            'world': 5,
        }
        self.tokenizer = TokenizerBase(self.vocab)

    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.tokenizer.vocab_size, 6)
        self.assertEqual(self.tokenizer.id_to_token[2], '[CLS]')
        self.assertEqual(self.tokenizer.token_to_id['hello'], 4)

    def test_tokens_to_ids(self):
        """测试tokens转ids"""
        tokens = ['[CLS]', 'hello', 'world', '[SEP]']
        ids = self.tokenizer.tokens_to_ids(tokens)

        self.assertEqual(ids, [2, 4, 5, 3])

    def test_tokens_to_ids_with_unk(self):
        """测试未知token转换"""
        tokens = ['[CLS]', 'unknown', '[SEP]']
        ids = self.tokenizer.tokens_to_ids(tokens)

        self.assertEqual(ids, [2, 1, 3])  # unknown -> [UNK]

    def test_ids_to_tokens(self):
        """测试ids转tokens"""
        ids = [2, 4, 5, 3]
        tokens = self.tokenizer.ids_to_tokens(ids)

        self.assertEqual(tokens, ['[CLS]', 'hello', 'world', '[SEP]'])

    def test_ids_to_tokens_with_invalid_id(self):
        """测试无效id转换"""
        ids = [2, 999, 3]
        tokens = self.tokenizer.ids_to_tokens(ids)

        self.assertEqual(tokens, ['[CLS]', '[UNK]', '[SEP]'])

    def test_decode(self):
        """测试解码"""
        ids = [4, 5]  # hello world
        text = self.tokenizer.decode(ids)

        self.assertEqual(text, 'helloworld')


class TestTokenizer(unittest.TestCase):
    """测试Tokenizer分词器"""

    def setUp(self):
        """创建测试词表和分词器"""
        self.temp_dir = tempfile.mkdtemp()
        self.vocab_path = os.path.join(self.temp_dir, 'vocab.txt')

        # 创建完整的词表（不包含'testing'，只有'test'和'##ing'）
        self.vocab = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[CLS]': 2,
            '[SEP]': 3,
            '[MASK]': 4,
            'hello': 5,
            'world': 6,
            'test': 7,
            '##ing': 8,
            '中': 9,
            '文': 10,
            '分': 11,
            '词': 12,
        }

        # 保存词表
        with open(self.vocab_path, 'w', encoding='utf-8') as f:
            for token, _ in sorted(self.vocab.items(), key=lambda x: x[1]):
                f.write(token + '\n')

        self.tokenizer = Tokenizer(self.vocab_path, do_lower_case=True)

    def tearDown(self):
        """清理临时文件"""
        if os.path.exists(self.vocab_path):
            os.remove(self.vocab_path)
        os.rmdir(self.temp_dir)

    def test_tokenize_english(self):
        """测试英文分词"""
        text = "hello world"
        tokens = self.tokenizer.tokenize(text)

        self.assertIn('hello', tokens)
        self.assertIn('world', tokens)

    def test_tokenize_english_lowercase(self):
        """测试英文大小写转换"""
        text = "Hello World"
        tokens = self.tokenizer.tokenize(text)

        # 应该被转为小写
        self.assertIn('hello', tokens)
        self.assertIn('world', tokens)

    def test_tokenize_chinese(self):
        """测试中文分词"""
        text = "中文"
        tokens = self.tokenizer.tokenize(text)

        self.assertIn('中', tokens)
        self.assertIn('文', tokens)

    def test_tokenize_mixed(self):
        """测试中英混合分词"""
        text = "hello中文"
        tokens = self.tokenizer.tokenize(text)

        self.assertIn('hello', tokens)
        self.assertIn('中', tokens)
        self.assertIn('文', tokens)

    def test_wordpiece_tokenize(self):
        """测试WordPiece分词"""
        # testing 应该被分为 test + ##ing
        tokens = self.tokenizer._wordpiece_tokenize('testing')

        self.assertEqual(tokens, ['test', '##ing'])

    def test_wordpiece_tokenize_whole_word(self):
        """测试整词WordPiece"""
        # hello 在词表中，不应该被拆分
        tokens = self.tokenizer._wordpiece_tokenize('hello')

        self.assertEqual(tokens, ['hello'])

    def test_wordpiece_tokenize_unknown(self):
        """测试未知词WordPiece"""
        # xyz 不在词表中
        tokens = self.tokenizer._wordpiece_tokenize('xyz')

        # 应该包含[UNK]
        self.assertTrue(any('[UNK]' in t for t in tokens))

    def test_is_chinese_char(self):
        """测试中文字符判断"""
        # 中文字符
        self.assertTrue(self.tokenizer._is_chinese_char(ord('中')))
        self.assertTrue(self.tokenizer._is_chinese_char(ord('文')))

        # 英文字符
        self.assertFalse(self.tokenizer._is_chinese_char(ord('a')))
        self.assertFalse(self.tokenizer._is_chinese_char(ord('A')))

    def test_tokenize_chinese_chars(self):
        """测试中文字符空格处理"""
        text = "hello中文"
        result = self.tokenizer._tokenize_chinese_chars(text)

        # 中文字符周围应该有空格
        self.assertIn(' 中 ', result)
        self.assertIn(' 文', result)

    def test_encode_single_text(self):
        """测试单文本编码"""
        text = "hello world"
        token_ids, segment_ids = self.tokenizer.encode(text, max_length=10)

        # 应该包含[CLS]和[SEP]
        self.assertEqual(token_ids[0], self.vocab['[CLS]'])
        self.assertEqual(token_ids[-1], self.vocab['[SEP]'])

        # segment_ids应该全为0
        self.assertTrue(all(s == 0 for s in segment_ids))

        # 长度应该匹配
        self.assertEqual(len(token_ids), len(segment_ids))

    def test_encode_two_texts(self):
        """测试双文本编码"""
        text1 = "hello"
        text2 = "world"
        token_ids, segment_ids = self.tokenizer.encode(text1, text2, max_length=20)

        # 应该包含[CLS]和两个[SEP]
        self.assertEqual(token_ids[0], self.vocab['[CLS]'])
        self.assertEqual(token_ids.count(self.vocab['[SEP]']), 2)

        # segment_ids前半部分为0，后半部分为1
        first_sep_idx = token_ids.index(self.vocab['[SEP]'])
        self.assertTrue(all(s == 0 for s in segment_ids[:first_sep_idx + 1]))
        self.assertTrue(all(s == 1 for s in segment_ids[first_sep_idx + 1:]))

    def test_encode_truncation(self):
        """测试截断"""
        # 创建很长的文本
        long_text = " ".join(['hello'] * 100)
        token_ids, segment_ids = self.tokenizer.encode(long_text, max_length=10)

        # 长度应该不超过max_length
        self.assertLessEqual(len(token_ids), 10)
        self.assertLessEqual(len(segment_ids), 10)

    def test_encode_two_texts_truncation(self):
        """测试双文本截断"""
        text1 = " ".join(['hello'] * 50)
        text2 = " ".join(['world'] * 50)
        token_ids, segment_ids = self.tokenizer.encode(text1, text2, max_length=20)

        # 总长度应该不超过max_length
        self.assertLessEqual(len(token_ids), 20)
        self.assertEqual(len(token_ids), len(segment_ids))

    def test_rematch(self):
        """测试token映射"""
        text = "hello world"
        tokens = ['[CLS]', 'hello', 'world', '[SEP]']

        mapping = self.tokenizer.rematch(text, tokens)

        # [CLS]和[SEP]应该映射为空
        self.assertEqual(mapping[0], [])
        self.assertEqual(mapping[-1], [])

        # hello应该映射到0-5
        self.assertEqual(mapping[1][0], 0)
        self.assertEqual(mapping[1][1], 5)

        # world应该映射到6-11
        self.assertEqual(mapping[2][0], 6)
        self.assertEqual(mapping[2][1], 11)

    def test_rematch_with_wordpiece(self):
        """测试带WordPiece的token映射"""
        text = "testing"
        tokens = ['[CLS]', 'test', '##ing', '[SEP]']

        mapping = self.tokenizer.rematch(text, tokens)

        # [CLS]映射为空
        self.assertEqual(mapping[0], [])

        # test映射到0-4
        self.assertEqual(mapping[1][0], 0)
        self.assertEqual(mapping[1][1], 4)

        # ##ing映射到4-7
        self.assertEqual(mapping[2][0], 4)
        self.assertEqual(mapping[2][1], 7)


class TestTokenizerEdgeCases(unittest.TestCase):
    """测试边界情况"""

    def setUp(self):
        """创建测试词表和分词器"""
        self.temp_dir = tempfile.mkdtemp()
        self.vocab_path = os.path.join(self.temp_dir, 'vocab.txt')

        self.vocab = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[CLS]': 2,
            '[SEP]': 3,
            'a': 4,
        }

        with open(self.vocab_path, 'w', encoding='utf-8') as f:
            for token, _ in sorted(self.vocab.items(), key=lambda x: x[1]):
                f.write(token + '\n')

        self.tokenizer = Tokenizer(self.vocab_path, do_lower_case=True)

    def tearDown(self):
        """清理临时文件"""
        if os.path.exists(self.vocab_path):
            os.remove(self.vocab_path)
        os.rmdir(self.temp_dir)

    def test_empty_text(self):
        """测试空文本"""
        text = ""
        tokens = self.tokenizer.tokenize(text)

        # 空文本应该产生空token列表
        self.assertEqual(len(tokens), 0)

    def test_whitespace_only(self):
        """测试只有空格的文本"""
        text = "   "
        tokens = self.tokenizer.tokenize(text)

        # 只有空格应该产生空token列表
        self.assertEqual(len(tokens), 0)

    def test_special_characters(self):
        """测试特殊字符"""
        text = "!@#$%"
        tokens = self.tokenizer.tokenize(text)

        # 应该包含一些token（可能是[UNK]）
        self.assertIsInstance(tokens, list)

    def test_encode_empty_text(self):
        """测试编码空文本"""
        text = ""
        token_ids, segment_ids = self.tokenizer.encode(text, max_length=10)

        # 应该只有[CLS]和[SEP]
        self.assertEqual(len(token_ids), 2)
        self.assertEqual(token_ids[0], self.vocab['[CLS]'])
        self.assertEqual(token_ids[1], self.vocab['[SEP]'])


if __name__ == '__main__':
    unittest.main()
