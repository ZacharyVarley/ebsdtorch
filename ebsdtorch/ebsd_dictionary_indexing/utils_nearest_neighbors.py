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

from typing import Tuple
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
    return res.values, res.indices


def calc_norm(t: Tensor) -> Tensor:
    return (t**2).sum(-1)


@torch.jit.script
def knn_batch(
    data: Tensor, data_norm: Tensor, query: Tensor, topk: int, metric: str
) -> Tuple[Tensor, Tensor]:
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
    topk_distances, topk_indices = dist.topk(topk, dim=1, largest=lgst, sorted=False)
    return topk_distances, topk_indices


@torch.jit.script
def knn_batches(
    data: Tensor,
    data_norm: Tensor,
    query: Tensor,
    data_chunk_idxs: Tensor,
    data_chunks_n: int,
    data_chunk_size: int,
    query_chunk_idxs: Tensor,
    query_chunk_size: int,
    topk: int,
    metric: str,
    lrgst: bool,
    match_device: torch.device,
    match_dtype: torch.dtype,
):
    n_query = query.shape[0]

    knn_indices_unreduced = torch.empty(
        (n_query, topk * data_chunks_n),
        device=match_device,
        dtype=torch.int64,
    )
    knn_distances_unreduced = torch.empty(
        (n_query, topk * data_chunks_n),
        device=match_device,
        dtype=match_dtype,
    )

    for query_start in query_chunk_idxs:
        # get the query chunk
        query_end = query_start + query_chunk_size
        query_chunk = query[query_start:query_end]

        for data_start in data_chunk_idxs:
            # get the data chunk
            data_end = data_start + data_chunk_size
            data_chunk = data[data_start:data_end]

            # get the indices of the nearest neighbors
            knn_distances_chunk, knn_indices_chunk = knn_batch(
                data_chunk,
                data_norm[data_start:data_end],
                query_chunk,
                topk,
                metric,
            )

            # store the indices
            knn_indices_unreduced[
                query_start:query_end, data_start:data_end
            ] = knn_indices_chunk

            # store the distances
            knn_distances_unreduced[
                query_start:query_end, data_start:data_end
            ] = knn_distances_chunk

    # reduce the results
    knn_indices_into_unreduced = torch.topk(
        knn_distances_unreduced, topk, dim=1, largest=lrgst, sorted=True
    ).indices

    # get the indices of the nearest neighbors using the topk indices into the unreduced indices
    knn_indices = knn_indices_unreduced.gather(1, knn_indices_into_unreduced)
    return knn_indices


def knn(
    data: Tensor,
    data_chunk_size: int,
    query: Tensor,
    query_chunk_size: int,
    topk: int,
    match_device: torch.device,
    distance_metric: str = "angular",
    match_dtype: torch.dtype = torch.float16,
) -> Tensor:
    """
    This function is a wrapper for the brute force KNN algorithm. It splits the
    data and query into chunks and calls the knn_batch function on each chunk.
    The results are concatenated, reduced, and returned.

    Args:
        data (Tensor): The dictionary of patterns to be indexed.
        data_chunk_size (int): The number of dictioanry patterns to process per batch.
        query (Tensor): The patterns to be indexed.
        query_chunk_size (int): The number of experimental patterns to process per batch.
        topk (int): The number of nearest neighbors to return.
        match_device (torch.device): The device to use for indexing.
        available_ram_GB (float): The amount of RAM available for indexing.
        distance_metric (str, optional): The distance metric to use. Defaults to "euclidean".
        match_dtype (torch.dtype, optional): The datatype to use for indexing. Defaults to torch.float16.

    Returns:
        Tensor: The indices of the nearest neighbors.

    """
    # in device
    device_in = data.device

    # largest if angular
    lrgst = True if distance_metric == "angular" else False

    # shapes
    n_data = data.shape[0]
    n_query = query.shape[0]

    # if the query chunk size exceeds the number of queries, set it to the number of queries
    query_chunk_size = min(query_chunk_size, n_query)

    # if the data chunk size exceeds the number of data, set it to the number of data
    data_chunk_size = min(data_chunk_size, n_data)

    # we can precompute the norms of the data in FP32
    # only used if the metric is euclidean
    data_norm = calc_norm(data.float())

    # cast to matching dtype and device
    data, query = data.to(match_dtype), query.to(match_dtype)
    data, query = data.to(match_device), query.to(match_device)

    # find the number of chunks
    n_data_chunks = int(n_data / data_chunk_size + 1)

    # get the indices of the nearest neighbors using the topk indices into the unreduced indices
    knn_indices = knn_batches(
        data,
        data_norm,
        query,
        torch.arange(0, n_data, data_chunk_size, device=match_device),
        n_data_chunks,
        data_chunk_size,
        torch.arange(0, n_query, query_chunk_size, device=match_device),
        query_chunk_size,
        topk,
        distance_metric,
        lrgst,
        match_device,
        match_dtype,
    ).to(device_in)

    return knn_indices


def knn_quantized_batches(
    data: Tensor,
    query: Tensor,
    data_chunk_idxs: Tensor,
    data_chunks_n: int,
    data_chunk_size: int,
    query_chunk_idxs: Tensor,
    query_chunk_size: int,
    topk: int,
    lrgst: bool,
    match_device: torch.device,
    match_dtype: torch.dtype,
) -> Tensor:
    n_query = query.shape[0]

    print(f"Initializing output tensor of shape {n_query, topk * data_chunks_n}")

    knn_indices_unreduced = torch.empty(
        (n_query, topk * data_chunks_n),
        device=match_device,
        dtype=torch.int64,
    )
    knn_distances_unreduced = torch.empty(
        (n_query, topk * data_chunks_n),
        device=match_device,
        dtype=match_dtype,
    )

    pb = progressbar(
        query_chunk_idxs,
        "Indexing",
    )

    for query_start in pb:
        # get the query chunk
        query_end = query_start + query_chunk_size
        query_chunk = query[query_start:query_end]

        for data_start in data_chunk_idxs:
            # get the data chunk
            data_end = data_start + data_chunk_size
            data_chunk = data[data_start:data_end]

            data_quant_model = quant_model(data_chunk)

            # get the indices of the nearest neighbors
            knn_distances_chunk, knn_indices_chunk = topk_quant(
                query_chunk, data_quant_model, topk
            )

            # store the indices
            knn_indices_unreduced[
                query_start:query_end, data_start:data_end
            ] = knn_indices_chunk

            # store the distances
            knn_distances_unreduced[
                query_start:query_end, data_start:data_end
            ] = knn_distances_chunk

    # reduce the results
    knn_indices_into_unreduced = torch.topk(
        knn_distances_unreduced, topk, dim=1, largest=lrgst, sorted=True
    ).indices

    # get the indices of the nearest neighbors using the topk indices into the unreduced indices
    knn_indices = knn_indices_unreduced.gather(1, knn_indices_into_unreduced)
    return knn_indices


def knn_quantized(
    data: Tensor, data_chunk_size: int, query: Tensor, query_chunk_size: int, topk: int
) -> Tensor:
    """
    This function is a wrapper for the brute force KNN algorithm. It splits the
    data and query into chunks and calls the knn_batch function on each chunk.
    The results are concatenated, reduced, and returned.

    Args:
        data (Tensor): The dictionary of patterns to be indexed.
        data_chunk_size (int): The number of dictioanry patterns to process per batch.
        query (Tensor): The patterns to be indexed.
        query_chunk_size (int): The number of experimental patterns to process per batch.
        topk (int): The number of nearest neighbors to return.

    Returns:
        Tensor: The indices of the nearest neighbors.

    """

    # in device
    device_in = data.device

    # shapes
    n_data = data.shape[0]
    n_query = query.shape[0]

    # if the query chunk size exceeds the number of queries, set it to the number of queries
    query_chunk_size = min(query_chunk_size, n_query)

    # if the data chunk size exceeds the number of data, set it to the number of data
    data_chunk_size = min(data_chunk_size, n_data)

    # cast to matching dtype and device
    data, query = data.to(torch.float32), query.to(torch.float32)
    data, query = data.to(device_in), query.to(device_in)

    # find the number of chunks
    n_data_chunks = int(n_data / data_chunk_size + 1)

    # get the indices of the nearest neighbors using the topk indices into the unreduced indices
    knn_indices = knn_quantized_batches(
        data,
        query,
        torch.arange(0, n_data, data_chunk_size, device=device_in),
        n_data_chunks,
        data_chunk_size,
        torch.arange(0, n_query, query_chunk_size, device=device_in),
        query_chunk_size,
        topk,
        True,
        device_in,
        torch.float32,
    ).to(device_in)

    return knn_indices


# def knn_quantized(
#     data: Tensor, query: Tensor, topk: int, available_ram_GB: float
# ) -> Tensor:
#     # get quantized model (will dynamically rescale data entries and cast to int8)
#     # the output will then be transformed back to the input datatype
#     data_read_for_gemm = quant_model(data)

#     # find the number of queries that can be processed in a batch
#     n_data, dim, n_query = data.shape[0], data.shape[1], query.shape[0]
#     batch_size = int(available_ram_GB * 1e9 / (n_data * dim))
#     batch_size = max(1, batch_size)
#     batch_size = min(batch_size, n_query)

#     # make sure there is at least one pattern per batch and that the batch size is
#     # not larger than the number of patterns
#     batch_size = max(1, batch_size)
#     batch_size = min(batch_size, n_query)

#     knn_idxs = []

#     pb = progressbar(torch.split(query, batch_size), "Indexing")

#     for b in pb:
#         knn_idxs.append(topk_quant(b, data_read_for_gemm, topk))
#     return torch.cat(knn_idxs, dim=0)
