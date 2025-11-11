import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


def sequence_padding(inputs, length=None, value=0, mode='post'):
    """序列padding"""
    if length is None:
        length = max([len(x) for x in inputs])

    outputs = []
    for x in inputs:
        x = list(x)
        if len(x) < length:
            if mode == 'post':
                x = x + [value] * (length - len(x))
            else:
                x = [value] * (length - len(x)) + x
        else:
            x = x[:length]
        outputs.append(x)

    return np.array(outputs)


def truncate_sequences(maxlen, *sequences):
    """截断序列"""
    sequences = [s for s in sequences if s]
    if not isinstance(sequences[0], (list, tuple)):
        sequences = [sequences]

    while True:
        lengths = [len(s) for s in sequences]
        if sum(lengths) <= maxlen:
            return sequences
        idx = lengths.index(max(lengths))
        sequences[idx] = sequences[idx][:-1]


def text_segmentate(text, maxlen, seps='\n', strips=None):
    """文本分段"""
    text = text.strip().strip(strips)
    if seps and len(text) > maxlen:
        pieces = text.split(seps[0])
        text, texts = '', []
        for i, p in enumerate(pieces):
            if text and p and len(text) + len(p) > maxlen - 1:
                texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
                text = ''
            if i + 1 == len(pieces):
                text = text + p
            else:
                text = text + p + seps[0]
        if text:
            texts.extend(text_segmentate(text, maxlen, seps[1:], strips))
        return texts
    else:
        return [text]


def to_array(*args):
    """转换为numpy数组"""
    results = []
    for x in args:
        if isinstance(x, (list, tuple)):
            x = np.array(x)
        results.append(x)
    return results if len(results) > 1 else results[0]


class DataGenerator:
    """数据生成器基类"""
    def __init__(self, data, batch_size=32, shuffle=False):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.steps = len(data) // batch_size
        if len(data) % batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        idxs = list(range(len(self.data)))
        if self.shuffle:
            np.random.shuffle(idxs)

        batch_data = []
        for idx in idxs:
            batch_data.append(self.data[idx])
            if len(batch_data) == self.batch_size:
                yield self.collate_fn(batch_data)
                batch_data = []

        if batch_data:
            yield self.collate_fn(batch_data)

    def collate_fn(self, batch):
        """需要子类实现"""
        raise NotImplementedError


class AutoRegressiveDecoder:
    """自回归解码器基类"""
    def __init__(self, start_id, end_id, maxlen, minlen=1, device='cuda'):
        self.start_id = start_id
        self.end_id = end_id
        self.maxlen = maxlen
        self.minlen = minlen
        self.device = device

    @torch.no_grad()
    def predict(self, inputs, output_ids):
        """预测下一个token，需要子类实现"""
        raise NotImplementedError

    @torch.no_grad()
    def beam_search(self, inputs, topk=1, topp=None):
        """束搜索"""
        output_ids = torch.tensor([[self.start_id]], device=self.device)
        output_scores = torch.zeros(1, device=self.device)

        for step in range(self.maxlen):
            scores = self.predict(inputs, output_ids)  # [batch*beam, vocab]
            scores = torch.log_softmax(scores, dim=-1)

            if step < self.minlen:
                scores[:, self.end_id] = -float('inf')

            # add previous scores
            scores = output_scores.unsqueeze(1) + scores  # [batch*beam, vocab]

            if step == 0:
                scores = scores[0]
            else:
                scores = scores.view(-1)

            # top-k
            indices = scores.argsort(descending=True)[:topk]
            beam_indices = indices // scores.shape[-1]
            token_indices = indices % scores.shape[-1]

            # update
            output_ids = torch.cat([
                output_ids[beam_indices],
                token_indices.unsqueeze(1)
            ], dim=1)
            output_scores = scores[indices]

            # check if all beams end
            ends = (token_indices == self.end_id).all()
            if ends:
                break

        # select best
        best_idx = output_scores.argmax()
        return output_ids[best_idx].cpu().numpy()

    @torch.no_grad()
    def random_sample(self, inputs, topk=None, topp=None, temperature=1.0):
        """随机采样"""
        output_ids = [[self.start_id]]

        for step in range(self.maxlen):
            scores = self.predict(inputs, torch.tensor(output_ids, device=self.device))
            scores = scores[0, -1] / temperature

            if step < self.minlen:
                scores[self.end_id] = -float('inf')

            # top-k
            if topk:
                indices = scores.argsort(descending=True)[:topk]
                scores = torch.full_like(scores, -float('inf'))
                scores[indices] = scores[indices]

            # top-p
            if topp:
                sorted_scores, sorted_indices = scores.sort(descending=True)
                cumsum_probs = torch.cumsum(torch.softmax(sorted_scores, dim=-1), dim=-1)
                mask = cumsum_probs > topp
                mask[0] = False
                sorted_scores[mask] = -float('inf')
                scores = torch.full_like(scores, -float('inf'))
                scores[sorted_indices] = sorted_scores

            # sample
            probs = torch.softmax(scores, dim=-1)
            token_id = torch.multinomial(probs, 1).item()
            output_ids[0].append(token_id)

            if token_id == self.end_id:
                break

        return output_ids[0]


class ViterbiDecoder:
    """维特比解码器"""
    def __init__(self, trans, starts=None, ends=None):
        self.trans = trans
        self.starts = starts
        self.ends = ends
        self.num_tags = len(trans)

    def decode(self, emissions, mask=None):
        """维特比解码"""
        if isinstance(emissions, torch.Tensor):
            emissions = emissions.cpu().numpy()

        batch_size, seq_len, num_tags = emissions.shape
        paths = []

        for i in range(batch_size):
            nodes = emissions[i]
            if mask is not None:
                nodes = nodes[:int(mask[i].sum())]

            path = self._viterbi(nodes)
            paths.append(path)

        return paths

    def _viterbi(self, nodes):
        """单条序列的维特比解码"""
        seq_len = len(nodes)
        paths = np.zeros((seq_len, self.num_tags), dtype=int)
        scores = nodes[0].copy()

        if self.starts is not None:
            scores += self.starts

        # forward
        for t in range(1, seq_len):
            scores_with_trans = scores.reshape(-1, 1) + self.trans
            paths[t] = scores_with_trans.argmax(axis=0)
            scores = scores_with_trans.max(axis=0) + nodes[t]

        # add ends
        if self.ends is not None:
            scores += self.ends

        # backtrack
        best_path = [scores.argmax()]
        for t in range(seq_len - 1, 0, -1):
            best_path.insert(0, paths[t, best_path[0]])

        return best_path


def parallel_apply(func, items, workers=8):
    """并行处理"""
    from multiprocessing import Pool

    if workers == 0 or workers == 1:
        return [func(item) for item in items]

    with Pool(workers) as pool:
        return pool.map(func, items)


def insert_arguments(**arguments):
    """插入参数装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            kwargs.update(arguments)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def delete_arguments(*arguments):
    """删除参数装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for arg in arguments:
                kwargs.pop(arg, None)
            return func(*args, **kwargs)
        return wrapper
    return decorator
