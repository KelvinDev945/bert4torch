import re
import unicodedata


def load_vocab(vocab_path):
    """加载词表"""
    token_to_id = {}
    with open(vocab_path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            token = line.strip()
            token_to_id[token] = i
    return token_to_id


def save_vocab(token_to_id, vocab_path):
    """保存词表"""
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for token, _ in sorted(token_to_id.items(), key=lambda x: x[1]):
            f.write(token + '\n')


class TokenizerBase:
    """分词器基类"""
    def __init__(self, token_to_id, do_lower_case=False):
        self.token_to_id = token_to_id
        self.id_to_token = {v: k for k, v in token_to_id.items()}
        self.do_lower_case = do_lower_case
        self.vocab_size = len(token_to_id)

    def tokenize(self, text):
        raise NotImplementedError

    def encode(self, text, max_length=None):
        tokens = self.tokenize(text)
        if max_length:
            tokens = tokens[:max_length]
        token_ids = self.tokens_to_ids(tokens)
        return token_ids

    def decode(self, token_ids):
        tokens = [self.id_to_token.get(i, '[UNK]') for i in token_ids]
        text = ''.join(tokens).replace('##', '')
        return text

    def tokens_to_ids(self, tokens):
        return [self.token_to_id.get(t, self.token_to_id.get('[UNK]', 0)) for t in tokens]

    def ids_to_tokens(self, ids):
        return [self.id_to_token.get(i, '[UNK]') for i in ids]


class Tokenizer(TokenizerBase):
    """BERT分词器"""
    def __init__(self, vocab_path, do_lower_case=True):
        token_to_id = load_vocab(vocab_path)
        super().__init__(token_to_id, do_lower_case)

    def tokenize(self, text):
        """分词"""
        if self.do_lower_case:
            text = text.lower()
            text = unicodedata.normalize('NFD', text)
            text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])

        tokens = []
        for word in self._tokenize_chinese_chars(text).split():
            tokens.extend(self._wordpiece_tokenize(word))

        return tokens

    def encode(self, first_text, second_text=None, max_length=None):
        """编码为token_ids和segment_ids"""
        first_tokens = self.tokenize(first_text)
        second_tokens = self.tokenize(second_text) if second_text else None

        if second_tokens:
            # truncate
            while len(first_tokens) + len(second_tokens) > max_length - 3:
                if len(first_tokens) > len(second_tokens):
                    first_tokens = first_tokens[:-1]
                else:
                    second_tokens = second_tokens[:-1]

            tokens = ['[CLS]'] + first_tokens + ['[SEP]'] + second_tokens + ['[SEP]']
            segment_ids = [0] * (len(first_tokens) + 2) + [1] * (len(second_tokens) + 1)
        else:
            if max_length:
                first_tokens = first_tokens[:max_length - 2]
            tokens = ['[CLS]'] + first_tokens + ['[SEP]']
            segment_ids = [0] * len(tokens)

        token_ids = self.tokens_to_ids(tokens)
        return token_ids, segment_ids

    def _tokenize_chinese_chars(self, text):
        """在中文字符周围添加空格"""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(' ')
                output.append(char)
                output.append(' ')
            else:
                output.append(char)
        return ''.join(output)

    def _is_chinese_char(self, cp):
        """判断是否为中文字符"""
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or
            (cp >= 0x3400 and cp <= 0x4DBF) or
            (cp >= 0x20000 and cp <= 0x2A6DF) or
            (cp >= 0x2A700 and cp <= 0x2B73F) or
            (cp >= 0x2B740 and cp <= 0x2B81F) or
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or
            (cp >= 0x2F800 and cp <= 0x2FA1F)):
            return True
        return False

    def _wordpiece_tokenize(self, word):
        """WordPiece分词"""
        if word in self.token_to_id:
            return [word]

        tokens = []
        start = 0
        while start < len(word):
            end = len(word)
            cur_substr = None
            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = '##' + substr
                if substr in self.token_to_id:
                    cur_substr = substr
                    break
                end -= 1

            if cur_substr is None:
                tokens.append('[UNK]')
                start += 1
            else:
                tokens.append(cur_substr)
                start = end

        return tokens

    def rematch(self, text, tokens):
        """将tokens映射回原文本位置"""
        if self.do_lower_case:
            text = text.lower()

        normalized_text = ''
        char_mapping = []
        for i, ch in enumerate(text):
            if self.do_lower_case:
                ch = unicodedata.normalize('NFD', ch)
                ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

        text = normalized_text

        token_mapping = []
        offset = 0
        for token in tokens:
            if token in ['[CLS]', '[SEP]']:
                token_mapping.append([])
            else:
                token = token.replace('##', '')
                start = text[offset:].index(token) + offset
                end = start + len(token)
                token_mapping.append([char_mapping[start], char_mapping[end - 1] + 1])
                offset = end

        return token_mapping
