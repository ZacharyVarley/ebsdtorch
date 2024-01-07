""" 
The conventional brute force code is taken from KeOps benchmarks found at:
https://www.kernel-operations.io/keops/_auto_benchmarks/benchmark_KNN.html
Feydy, Jean, Alexis Glaunès, Benjamin Charlier, and Michael Bronstein. "Fast
geometric learning with symbolic matrices." Advances in Neural Information
Processing Systems 33 (2020): 14448-14462.

CPU quantization (doing arithmetic in int8) is also supported, notes here: 

PyTorch provides a very nice interface with FBGEMM for quantized matrix
multiplication. This is used to accelerate the distance matrix calculation
tremendously. FBGEMM is developed by Facebook, and both x86 and ARM CPUs are
supported. PyTorch plans to support GPU quantization in the future on NVIDIA
GPUs, via "Tensor Cores" and Apple silicon via the "Neural Engine". This will
provide a 10x speedup over the current GPU implementation which indexes around
10,000 patterns per second at 1000 PCA components, and a high symmetry Laue
group. The quantized CPU implementation is currently around 1000 patterns per
second under the same conditions. This means for the CPU it takes longer to
project the dictionary than it does to index most small EBSD datasets.

I have explored many approximation methods such as HNSW, IVF-Flat, etc (on the
GPU too).

They are not worth running on the raw EBSD images because the time invested in
building the index is large. For cubic materials, a dictionary around 100,000
images is needed, and most methods will take 10s of seconds if the patterns are
larger than 60x60 = 3600 dimensions. PCA requires <10 seconds investment, even
using Online covariance matrix estimation approaches that stream pattern
projection.

 """

import torch
from torch.nn import Linear, Module
from torch import Tensor
from torch.quantization import quantize_dynamic
from ebsdtorch.ebsd_dictionary_indexing.utils_progress_bar import progressbar


class LinearLayer(Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(LinearLayer, self).__init__()
        self.fc = Linear(in_dim, out_dim, bias=False)

    def forward(self, inp: Tensor) -> Tensor:
        return self.fc(inp)


def quant_model(t: Tensor) -> Module:
    layer = LinearLayer(t.shape[1], t.shape[0])
    layer.fc.weight.data = t
    layer.eval()
    return quantize_dynamic(
        model=layer, qconfig_spec={Linear}, dtype=torch.qint8, inplace=False
    )


def topk_quant(q_batch: Tensor, q_model: Module, k: int) -> Tensor:
    res = torch.topk(q_model(q_batch), k, dim=1, largest=True, sorted=True)
    return res.indices


def calc_norm(t: Tensor) -> Tensor:
    return (t**2).sum(-1)


@torch.jit.script
def knn_batch(
    data: Tensor, data_norm: Tensor, query: Tensor, topk: int, metric: str
) -> Tensor:
    lgst = True if metric == "angular" else False
    if metric == "euclidean":
        """
        This could be done in FP16 precision but the norms must be precomputed
        in FP32. It may seem odd to transpose the dictionary, but this is not
        computed and a "pretransposed" version of GEMM will be called by PyTorch
        after graph generation / optimization.
        """
        q_norm = calc_norm(query.float())
        dist = (
            q_norm.view(-1, 1)
            + data_norm.view(1, -1)
            - (2.0 * query @ data.t()).float()
        )
    elif metric == "manhattan":
        dist = (query[:, None, :] - data[None, :, :]).abs().sum(dim=2)
    elif metric == "angular":
        dist = query @ data.t()
    else:
        raise NotImplementedError(f"'{metric}' not supported.")
    return dist.topk(topk, dim=1, largest=lgst, sorted=True).indices


def knn(
    data: Tensor,
    query: Tensor,
    topk: int,
    match_device: torch.device,
    available_ram_GB: float,
    distance_metric: str = "euclidean",
    match_dtype: torch.dtype = torch.float16,
) -> Tensor:
    # in device
    device_in = data.device

    # find the number of queries that can be processed in a batch
    n_bytes = torch.tensor([], dtype=match_dtype).element_size()

    n_data, n_query = data.shape[0], query.shape[0]

    # we can precompute the norms of the data in FP32
    # only used if the metric is euclidean
    data_norm = calc_norm(data.float())

    # cast to matching dtype
    data, query = data.to(match_dtype), query.to(match_dtype)

    # send to matching device
    data, query = data.to(match_device), query.to(match_device)

    # estimate the largest reasonable batch size
    batch_size = int(available_ram_GB * 1e9 / (n_data * n_bytes))

    # make sure there is at least one pattern per batch and that the batch size is
    # not larger than the number of patterns
    batch_size = max(1, batch_size)
    batch_size = min(batch_size, n_query)

    knn_idxs = []

    pb = progressbar(torch.split(query, batch_size), prefix="{:<20}".format("Indexing"))

    for b in pb:
        knn_idxs.append(knn_batch(data, data_norm, b, topk, distance_metric))

    # concatenate the list of tensors
    knn_indxs = torch.cat(knn_idxs, dim=0).to(device_in)

    return knn_indxs


def knn_quantized(
    data: Tensor, query: Tensor, topk: int, available_ram_GB: float
) -> Tensor:
    # get quantized model (will dynamically rescale data entries and cast to int8)
    # the output will then be transformed back to the input datatype
    data_read_for_gemm = quant_model(data)

    # find the number of queries that can be processed in a batch
    n_data, dim, n_query = data.shape[0], data.shape[1], query.shape[0]
    batch_size = int(available_ram_GB * 1e9 / (n_data * dim))
    batch_size = max(1, batch_size)
    batch_size = min(batch_size, n_query)

    # make sure there is at least one pattern per batch and that the batch size is
    # not larger than the number of patterns
    batch_size = max(1, batch_size)
    batch_size = min(batch_size, n_query)

    knn_idxs = []

    pb = progressbar(torch.split(query, batch_size), "Indexing")

    for b in pb:
        knn_idxs.append(topk_quant(b, data_read_for_gemm, topk))
    return torch.cat(knn_idxs, dim=0)
