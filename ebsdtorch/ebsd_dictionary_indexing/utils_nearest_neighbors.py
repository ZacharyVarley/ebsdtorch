""" 
The conventional brute force code is taken from KeOps benchmarks found at:
https://www.kernel-operations.io/keops/_auto_benchmarks/benchmark_KNN.html
Feydy, Jean, Alexis Glaunès, Benjamin Charlier, and Michael Bronstein. "Fast
geometric learning with symbolic matrices." Advances in Neural Information
Processing Systems 33 (2020): 14448-14462.

CPU quantization (doing arithmetic in int8) is also supported, notes here: 

PyTorch provides a very nice interface with FBGEMM for quantized matrix
multiplication. This is used to accelerate the distance matrix calculation
tremendously. PyTorch plans to support GPU quantization in the future on NVIDIA
GPUs. This will provide a 10x speedup over the current GPU implementation. Apple
silicon *CPUs* are not yet supported by the ARM64 alternative of FBGEMM called
QNNPACK (I am not absolutely certain.), but a Metal Performance Shaders (MPS)
PyTorch backend provies Apple Silicon *GPU* support today.

I have explored many KNN search approximation methods such as HNSW, IVF-Flat,
etc (on the GPU too).

They are not worth running on the raw EBSD images because the time invested in
building the index is large. For cubic materials, a dictionary around 100,000
images is needed, and graph building will take many minutes if the patterns are
larger than 60x60 = 3600 dimensions. PCA is fast and leverages the compactness
of the dictionary in image space.

 """

from typing import Tuple
import torch
from torch.nn import Linear, Module
from torch import Tensor
from torch.quantization import quantize_dynamic
from ebsdtorch.ebsd_dictionary_indexing.utils_progress_bar import progressbar


class LinearLayer(Module):
    """
    This is a wrapper around torch.nn.Linear defining a no bias linear layer
    model. The 8-bit quantized arithmetic libraries take neural networks as
    inputs. This allows us to quantize a simple matrix multiplication.

    """

    def __init__(self, in_dim: int, out_dim: int):
        super(LinearLayer, self).__init__()
        self.fc = Linear(in_dim, out_dim, bias=False)

    def forward(self, inp: Tensor) -> Tensor:
        return self.fc(inp)


def quant_model_data(data: Tensor) -> Module:
    layer = LinearLayer(data.shape[1], data.shape[0])
    layer.fc.weight.data = data
    layer.eval()
    return quantize_dynamic(
        model=layer, qconfig_spec={Linear}, dtype=torch.qint8, inplace=True
    )


def quant_model_query(query: Tensor) -> Module:
    layer = LinearLayer(query.shape[1], query.shape[0])
    layer.fc.weight.data = query
    layer.eval()
    return quantize_dynamic(
        model=layer, qconfig_spec={Linear}, dtype=torch.qint8, inplace=True
    )


def topk_quantized_data(query_batch: Tensor, quantized_data: Module, k: int) -> Tensor:
    res = torch.topk(quantized_data(query_batch), k, dim=1, largest=True, sorted=False)
    return res.values, res.indices


def topk_quantized_query(data_batch: Tensor, quantized_query: Module, k: int) -> Tensor:
    res = torch.topk(quantized_query(data_batch), k, dim=0, largest=True, sorted=False)
    return res.values.t(), res.indices.t()


@torch.jit.script
def knn_batch(
    data: Tensor, query: Tensor, topk: int, metric: str
) -> Tuple[Tensor, Tensor]:
    lgst = True if metric == "angular" else False
    if metric == "euclidean":
        # norms must be precomputed in FP32.
        data_norm = (data.float() ** 2).sum(dim=-1)
        q_norm = (query.float() ** 2).sum(dim=-1)
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
    topk_distances, topk_indices = torch.topk(
        dist, topk, dim=1, largest=lgst, sorted=False
    )
    return topk_distances, topk_indices


def knn(
    data: Tensor,
    query: Tensor,
    data_chunk_size: int,
    query_chunk_size: int,
    topk: int,
    match_device: torch.device,
    distance_metric: str = "angular",
    match_dtype: torch.dtype = torch.float16,
    quantized: bool = True,
) -> Tensor:
    """
    This function is a wrapper for the brute force KNN algorithm. It splits the
    data and query into chunks and calls the knn_batch function on each chunk.
    The results are concatenated, reduced, and returned.

    Args:
        data (Tensor): The dictionary of patterns.
        query (Tensor): The patterns to be indexed.
        data_chunk_size (int): The number of dictioanry patterns to process per batch.
        query_chunk_size (int): The number of experimental patterns to process per batch.
        topk (int): The number of nearest neighbors to return.
        match_device (torch.device): The device to cast the data and query to.
        distance_metric (str): The distance metric to use. Options are "angular", "euclidean", and "manhattan".
        match_dtype (torch.dtype): The dtype to cast the data and query to.
        quantized (bool): Whether to use quantized arithmetic.

    Returns:
        Tensor: The indices of the nearest neighbors.

    """
    # in device
    device_in = data.device

    # largest if angular
    larger_better = distance_metric == "angular"

    # if quantized, raise an error and say only CPU is supported
    if quantized and match_device != torch.device("cpu"):
        raise NotImplementedError(
            "Quantized arithmetic is only supported on x86 CPUs at this time."
        )

    # shapes
    n_query = query.shape[0]
    n_data = data.shape[0]

    # correct chunk sizes if they exceed the number of patterns
    query_chunk_size = min(query_chunk_size, n_query)
    data_chunk_size = min(data_chunk_size, n_data)

    # cast to matching dtype and device
    data, query = data.to(match_dtype), query.to(match_dtype)
    data, query = data.to(match_device), query.to(match_device)

    # find the number of chunks
    n_data = data.shape[0]
    n_data_chunks = int(n_data / data_chunk_size) + int(n_data % data_chunk_size > 0)

    # get the indices of the nearest neighbors using the topk indices into the unreduced indices
    # shapes
    n_data = data.shape[0]
    n_query = query.shape[0]

    knn_i_global = torch.empty(
        (n_query, topk * n_data_chunks),
        device=match_device,
        dtype=torch.int64,
    )
    knn_d_global = torch.empty(
        (n_query, topk * n_data_chunks),
        device=match_device,
        dtype=match_dtype,
    )

    data_batch_num = 0

    for data_start in progressbar(range(0, n_data, data_chunk_size), "Indexing"):
        # get the data chunk
        data_chunk = data[data_start : data_start + data_chunk_size]
        if quantized and match_device == torch.device("cpu"):
            data_quantized = quant_model_data(data_chunk)

        for query_start in range(0, n_query, query_chunk_size):
            # get the query chunk
            query_chunk = query[query_start : query_start + query_chunk_size]
            # get the indices of the nearest neighbors
            if quantized and match_device == torch.device("cpu"):
                knn_distances_chunk, knn_indices_chunk = topk_quantized_data(
                    query_chunk,
                    data_quantized,
                    topk,
                )
            else:
                knn_distances_chunk, knn_indices_chunk = knn_batch(
                    data_chunk,
                    query_chunk,
                    topk,
                    distance_metric,
                )

            # Offset the indices to be relative to the entire dataset
            knn_indices_chunk += data_start

            # store the indices
            knn_i_global[
                query_start : query_start + query_chunk_size,
                data_batch_num * topk : (data_batch_num + 1) * topk,
            ] = knn_indices_chunk

            # store the distances
            knn_d_global[
                query_start : query_start + query_chunk_size,
                data_batch_num * topk : (data_batch_num + 1) * topk,
            ] = knn_distances_chunk

        # update batch number
        data_batch_num += 1

    # find the best indices of the k of the concatenated distances topk from each dictionary batch
    _, knn_i_into_global = torch.topk(knn_d_global, topk, dim=1, largest=larger_better)

    # get the indices into the actual dictionary using the topk indices with the best distances in the global array
    knn_indices_into_dict = torch.gather(knn_i_global, 1, knn_i_into_global)

    return knn_indices_into_dict.to(device_in)


class KNNIndexer:
    def __init__(
        self,
        data_size: int,
        query_size: int,
        topk: int,
        match_device: torch.device,
        distance_metric: str = "angular",
        match_dtype: torch.dtype = torch.float16,
        quantized: bool = True,
    ):
        """
        Initialize the KNNIndexer.

        Args:
            data_size (int): The total number of data entries.
            query_size (int): The total number of query entries.
            topk (int): The number of nearest neighbors to return.
            match_device (torch.device): The device to use for matching.
            distance_metric (str): The distance metric to use ('angular', 'euclidean', or 'manhattan').
            match_dtype (torch.dtype): The data type to use for matching.
            quantized (bool): Whether to use quantized arithmetic.
        """
        self.data_size = data_size
        self.query_size = query_size
        self.topk = topk
        self.match_device = match_device
        self.distance_metric = distance_metric
        self.match_dtype = match_dtype
        self.quantized = quantized

        self.larger_better = self.distance_metric == "angular"
        self.data_chunk = None
        self.data_start = 0
        self.data_quantized = None
        self.knn_indices = torch.empty(
            (self.query_size, self.topk), device=self.match_device, dtype=torch.int64
        )
        self.knn_distances = torch.full(
            (self.query_size, self.topk),
            -torch.inf if self.larger_better else torch.inf,
            device=self.match_device,
            dtype=self.match_dtype,
        )

    def set_data_chunk(self, data_chunk: Tensor, data_start: int):
        """
        Set the current data chunk.

        Args:
            data_chunk (Tensor): The data chunk to set.
            data_start (int): The starting index of the data chunk in the full data tensor.
        """
        data_chunk = data_chunk.to(self.match_dtype).to(self.match_device)
        self.data_start = data_start
        if self.quantized and self.match_device.type == "cpu":
            self.data_quantized = quant_model_data(data_chunk)
            self.data_chunk = None
        else:
            self.data_quantized = None
            self.data_chunk = data_chunk

    def query(self, query_chunk: Tensor, query_start: int):
        """
        Perform k-nearest neighbors search on a query chunk.

        Args:
            query_chunk (Tensor): The query chunk to search.
            query_start (int): The starting index of the query chunk in the full query tensor.

        """
        query_end = query_start + query_chunk.shape[0]
        query_chunk = query_chunk.to(self.match_dtype).to(self.match_device)

        if self.data_quantized is not None:
            knn_distances_chunk, knn_indices_chunk = topk_quantized_data(
                query_chunk, self.data_quantized, self.topk
            )
        else:
            knn_distances_chunk, knn_indices_chunk = knn_batch(
                self.data_chunk, query_chunk, self.topk, self.distance_metric
            )

        knn_indices_chunk += self.data_start

        # Merge the old and new top-k indices and distances
        old_knn_indices = self.knn_indices[query_start:query_end]
        old_knn_distances = self.knn_distances[query_start:query_end]
        merged_knn_distances = torch.cat(
            (old_knn_distances, knn_distances_chunk), dim=1
        )
        merged_knn_indices = torch.cat((old_knn_indices, knn_indices_chunk), dim=1)
        topk_indices = torch.topk(
            merged_knn_distances, self.topk, dim=1, largest=self.larger_better
        )[1]
        self.knn_indices[query_start:query_end] = torch.gather(
            merged_knn_indices, 1, topk_indices
        )
        self.knn_distances[query_start:query_end] = torch.gather(
            merged_knn_distances, 1, topk_indices
        )

    def retrieve_topk(
        self,
    ) -> Tensor:
        """
        Retrieve the top-k nearest neighbors.

        Args:
            device (torch.device): The device to return the results on.

        Returns:
            Tensor: The indices of the nearest neighbors.
        """
        return self.knn_indices, self.knn_distances


# test the KNNIndexer against cdist
if __name__ == "__main__":
    from torch import cdist
    from time import time

    # set the seed
    torch.manual_seed(0)

    # set the sizes
    data_size = 1000
    query_size = 5
    data_dim = 50
    query_dim = 50
    topk = 3

    # set the data and query
    data = torch.randn(data_size, data_dim)
    query = torch.randn(query_size, query_dim)

    # set the indexer
    indexer = KNNIndexer(
        data_size=data_size,
        query_size=query_size,
        topk=topk,
        match_device=torch.device("cpu"),
        distance_metric="angular",
        match_dtype=torch.float32,
        quantized=True,
    )

    # set the chunk sizes
    data_chunk_size = 10
    query_chunk_size = 5

    # set the start time
    start = time()

    # index the patterns
    for data_start in range(0, data_size, data_chunk_size):
        data_chunk = data[data_start : data_start + data_chunk_size]
        indexer.set_data_chunk(data_chunk, data_start)
        for query_start in range(0, query_size, query_chunk_size):
            query_chunk = query[query_start : query_start + query_chunk_size]
            indexer.query(query_chunk, query_start)

    # get the indices
    indices, distances = indexer.retrieve_topk()

    # get the end time
    end = time()

    # print the time
    print(f"KNNIndexer time: {end - start}")

    # set the start time
    start = time()

    # get the indices using cdist
    # dist = cdist(query, data)

    dist = query @ data.t()

    indices_cdist = torch.topk(dist, topk, dim=1, largest=True)[1]

    # get the end time
    end = time()

    # print the time
    print(f"cdist time: {end - start}")

    print(indices)
    print(indices_cdist)

    # check the indices
    assert torch.allclose(indices, indices_cdist)
    print("Success!")
