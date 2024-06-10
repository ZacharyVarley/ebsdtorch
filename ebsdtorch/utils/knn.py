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
from torch.ao.quantization import quantize_dynamic


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


class ChunkedKNN:
    def __init__(
        self,
        data_size: int,
        query_size: int,
        topk: int,
        match_device: torch.device,
        distance_metric: str = "angular",
        match_dtype: torch.dtype = torch.float16,
        quantized_via_ao: bool = True,
    ):
        """
        Initialize the ChunkedKNN indexer to do batched k-nearest neighbors search.

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

        if (
            quantized_via_ao
            and match_dtype != torch.float32
            and match_device.type == "cpu"
        ):
            print(
                "CPU Quantization requires float32 data type. Forcing float32 match_dtype."
            )
            match_dtype = torch.float32

        self.match_dtype = match_dtype
        self.quantized = quantized_via_ao

        self.big_better = self.distance_metric == "angular"
        ind_dtype = torch.int64 if self.data_size < 2**31 else torch.int32
        self.knn_indices = torch.empty(
            (self.query_size, self.topk), device=self.match_device, dtype=ind_dtype
        )
        self.knn_distances = torch.full(
            (self.query_size, self.topk),
            -torch.inf if self.big_better else torch.inf,
            device=self.match_device,
            dtype=self.match_dtype,
        )

        self.prepared_data = None
        self.data_start = 0

    def set_data_chunk(self, data_chunk: Tensor):
        """
        Set the current data chunk. Shape (N, D).

        Args:
            data_chunk (Tensor): The data chunk to set.
        """
        data_chunk = data_chunk.to(self.match_dtype).to(self.match_device)

        if self.prepared_data is not None:
            # Update data start index by size of the previous chunk
            # that is about to be discarded
            self.data_start += data_chunk.shape[0]

        # Quantize the data if needed
        if self.quantized and self.match_device.type == "cpu":
            self.prepared_data = quant_model_data(data_chunk)
        else:
            self.prepared_data = data_chunk

    def query_all(self, query: Tensor):
        """
        Perform k-nearest neighbors search on all queries.

        Args:
            query (Tensor): The queries to search.
        """
        # throw an error if the data chunk is not set
        if self.prepared_data is None:
            raise ValueError("Data chunk is not set.")

        # send query to match device and dtype
        query = query.to(self.match_dtype).to(self.match_device)

        if self.quantized and self.match_device.type == "cpu":
            knn_dists_chunk, knn_inds_chunk = topk_quantized_data(
                query, self.prepared_data, self.topk
            )
        else:
            knn_dists_chunk, knn_inds_chunk = knn_batch(
                self.prepared_data, query, self.topk, self.distance_metric
            )

        # chunk indices -> global indices
        knn_inds_chunk += self.data_start

        # Merge the old and new top-k indices and distances
        merged_knn_dists = torch.cat((self.knn_distances, knn_dists_chunk), dim=1)
        merged_knn_inds = torch.cat((self.knn_indices, knn_inds_chunk), dim=1)

        # get the overall topk
        topk_indices = torch.topk(
            merged_knn_dists, self.topk, dim=1, largest=self.big_better, sorted=False
        )[1]
        self.knn_indices = torch.gather(merged_knn_inds, 1, topk_indices)
        self.knn_distances = torch.gather(merged_knn_dists, 1, topk_indices)

    def query_chunk(self, query_chunk: Tensor, query_start: int):
        """
        Perform k-nearest neighbors search on a contiguous query chunk.

        Args:
            query_chunk (Tensor): The query chunk to search.
            query_start (int): The start index of the query chunk.

        """

        # throw an error if the data chunk is not set
        if self.prepared_data is None:
            raise ValueError("Data chunk is not set.")

        query_end = query_start + query_chunk.shape[0]
        query_chunk = query_chunk.to(self.match_dtype).to(self.match_device)

        if self.quantized and self.match_device.type == "cpu":
            knn_distances_chunk, knn_indices_chunk = topk_quantized_data(
                query_chunk, self.prepared_data, self.topk
            )
        else:
            knn_distances_chunk, knn_indices_chunk = knn_batch(
                self.prepared_data, query_chunk, self.topk, self.distance_metric
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
            merged_knn_distances,
            self.topk,
            dim=1,
            largest=self.big_better,
            sorted=False,
        )[1]
        self.knn_indices[query_start:query_end] = torch.gather(
            merged_knn_indices, 1, topk_indices
        )
        self.knn_distances[query_start:query_end] = torch.gather(
            merged_knn_distances, 1, topk_indices
        )

    def retrieve_topk(
        self,
    ) -> Tuple[Tensor, Tensor]:
        """
        Retrieve the top-k nearest neighbors indices and distances.

        Args:
            device (torch.device): The device to return the results on.

        Returns:
            Tensor: The indices of the nearest neighbors.
        """
        # sort the topk indices and distances
        topk_indices = torch.topk(
            self.knn_distances, self.topk, dim=1, largest=self.big_better, sorted=True
        )[1]
        knn_indices = torch.gather(self.knn_indices, 1, topk_indices)
        knn_distances = torch.gather(self.knn_distances, 1, topk_indices)

        return knn_indices, knn_distances
