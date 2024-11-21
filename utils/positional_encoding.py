import torch

class Embedder:
    """
    实现位置编码（Positional Encoding）
    """
    def __init__(self, input_dims, num_freqs, include_input=True, log_sampling=True):
        self.include_input = include_input
        self.input_dims = input_dims
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling
        self._create_embedding_fn()

    def _create_embedding_fn(self):
        self.embed_fns = []
        out_dim = 0

        if self.include_input:
            self.embed_fns.append(lambda x: x)
            out_dim += self.input_dims

        if self.log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, self.num_freqs - 1, self.num_freqs)
        else:
            freq_bands = torch.linspace(1.0, 2.0 ** (self.num_freqs - 1), self.num_freqs)

        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))
            out_dim += 2 * self.input_dims

        self.out_dim = out_dim

    def embed(self, x):
        """
        返回位置编码后的结果
        """
        return torch.cat([fn(x) for fn in self.embed_fns], dim=-1)


def get_embedder(multires, input_dims, include_input=True):
    """
    创建位置编码器
    """
    embedder = Embedder(
        input_dims=input_dims,
        num_freqs=multires,
        include_input=include_input,
    )
    return embedder.embed, embedder.out_dim
