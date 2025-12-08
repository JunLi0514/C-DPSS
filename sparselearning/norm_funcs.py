import numpy as np

import numpy as np

class ScoreNormalizer:
    def __init__(self, method='minmax'):
        assert method in ['minmax', 'zscore', 'rank'], "method must be 'minmax', 'zscore', or 'rank'"
        self.method = method
        self._stats = {}  # 用于保存每次 blend 的归一化参数

    def normalize(self, x, key=None, save=False):
        x = np.array(x)
        if self.method == 'minmax':
            min_val = np.min(x)
            max_val = np.max(x)
            normed = np.zeros_like(x) if max_val == min_val else (x - min_val) / (max_val - min_val)
            if save and key:
                self._stats[key] = {'min': min_val, 'max': max_val}
            return normed

        elif self.method == 'zscore':
            mean = np.mean(x)
            std = np.std(x)
            normed = np.zeros_like(x) if std == 0 else (x - mean) / std
            if save and key:
                self._stats[key] = {'mean': mean, 'std': std}
            return normed

        elif self.method == 'rank':
            ranks = x.argsort().argsort()
            normed = ranks / (len(x) - 1)
            if save and key:
                self._stats[key] = {'rank_map': dict(zip(x, normed))}
            return normed

    def blend(self, score_a, score_b, alpha=0.5):
        """
        Normalize and blend two score arrays.
        Saves normalization parameters for use in blend2().
        """
        norm_a = self.normalize(score_a, key='a', save=True)
        norm_b = self.normalize(score_b, key='b', save=True)
        return alpha * norm_a + (1 - alpha) * norm_b

    def blend2(self, val_a, val_b, alpha=0.5):
        """
        Restore a single blended value from original (unnormalized) val_a and val_b
        using stored normalization stats.
        """
        if self.method == 'minmax':
            stat_a = self._stats.get('a', {})
            stat_b = self._stats.get('b', {})
            norm_a = 0 if stat_a['max'] == stat_a['min'] else (val_a - stat_a['min']) / (stat_a['max'] - stat_a['min'])
            norm_b = 0 if stat_b['max'] == stat_b['min'] else (val_b - stat_b['min']) / (stat_b['max'] - stat_b['min'])
        elif self.method == 'zscore':
            stat_a = self._stats.get('a', {})
            stat_b = self._stats.get('b', {})
            norm_a = 0 if stat_a['std'] == 0 else (val_a - stat_a['mean']) / stat_a['std']
            norm_b = 0 if stat_b['std'] == 0 else (val_b - stat_b['mean']) / stat_b['std']
        elif self.method == 'rank':
            # rank-based恢复精度有限，只做近似
            stat_a = self._stats.get('a', {}).get('rank_map', {})
            stat_b = self._stats.get('b', {}).get('rank_map', {})
            norm_a = stat_a.get(val_a, 0)
            norm_b = stat_b.get(val_b, 0)
        else:
            raise ValueError("Unsupported method")

        return alpha * norm_a + (1 - alpha) * norm_b
