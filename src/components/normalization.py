import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """Applies layer normalization to the input tensor.

    Arguments
    ---------
    input_size : int
        The expected size of the dimension to be normalized.
    input_shape : tuple
        The expected shape of the input.
    eps : float
        This value is added to std deviation estimation to improve the numerical
        stability.
    elementwise_affine : bool
        If True, this module has learnable per-element affine parameters
        initialized to ones (for weights) and zeros (for biases).

    Example
    -------
    >>> input = torch.randn(100, 101, 128)
    >>> norm = LayerNorm(input_shape=input.shape)
    >>> output = norm(input)
    >>> output.shape
    torch.Size([100, 101, 128])
    """

    def __init__(
        self,
        input_size=None,
        input_shape=None,
        eps=1e-05,
        elementwise_affine=True,
    ):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if input_shape is not None:
            input_size = input_shape[2:]

        self.norm = torch.nn.LayerNorm(
            input_size,
            eps=self.eps,
            elementwise_affine=self.elementwise_affine,
        )

    def forward(self, x):
        """Returns the normalized input tensor.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channels)
            input to normalize. 3d or 4d tensors are expected.

        Returns
        -------
        The normalized outputs.
        """
        return self.norm(x)


class LayerNormCN(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """  # noqa: E501

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x