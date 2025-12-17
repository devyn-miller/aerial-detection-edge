"""
Benchmarking Utilities

Functions for measuring inference latency, memory usage, and throughput.
These help you build the accuracy vs. speed tradeoff analysis.
"""

import time
import gc
from contextlib import contextmanager
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class BenchmarkResult:
    """Container for benchmark metrics."""
    mean_latency_ms: float
    std_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_fps: float
    peak_memory_mb: float
    model_size_mb: float
    
    def __str__(self):
        return (
            f"Latency: {self.mean_latency_ms:.2f} ± {self.std_latency_ms:.2f} ms\n"
            f"P50/P95/P99: {self.p50_latency_ms:.2f} / {self.p95_latency_ms:.2f} / {self.p99_latency_ms:.2f} ms\n"
            f"Throughput: {self.throughput_fps:.1f} FPS\n"
            f"Peak Memory: {self.peak_memory_mb:.1f} MB\n"
            f"Model Size: {self.model_size_mb:.1f} MB"
        )
    
    def to_dict(self):
        return {
            'mean_latency_ms': self.mean_latency_ms,
            'std_latency_ms': self.std_latency_ms,
            'p50_latency_ms': self.p50_latency_ms,
            'p95_latency_ms': self.p95_latency_ms,
            'p99_latency_ms': self.p99_latency_ms,
            'throughput_fps': self.throughput_fps,
            'peak_memory_mb': self.peak_memory_mb,
            'model_size_mb': self.model_size_mb
        }


@contextmanager
def constrained_cpu(num_threads: int = 2):
    """
    Context manager to simulate edge CPU constraints.
    
    Usage:
        with constrained_cpu(num_threads=2):
            # Your inference code here
    """
    original_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    try:
        yield
    finally:
        torch.set_num_threads(original_threads)


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Get model size in MB by summing parameter sizes."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def get_model_file_size_mb(path: str) -> float:
    """Get saved model file size in MB."""
    import os
    return os.path.getsize(path) / (1024 ** 2)


def measure_inference_latency(
    model,
    input_tensor: torch.Tensor,
    n_warmup: int = 10,
    n_iterations: int = 100,
    device: str = 'cpu'
) -> list[float]:
    """
    Measure inference latency over multiple iterations.
    
    Returns list of latencies in milliseconds.
    Always do warmup runs - first inference is always slow due to lazy initialization.
    """
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    model.eval()
    
    # Warmup (don't measure these)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(input_tensor)
    
    # Synchronize if using CUDA
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Actual measurements
    latencies = []
    with torch.no_grad():
        for _ in range(n_iterations):
            start = time.perf_counter()
            _ = model(input_tensor)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
    
    return latencies


def measure_peak_memory_cpu(
    model,
    input_tensor: torch.Tensor,
    n_iterations: int = 10
) -> float:
    """
    Estimate peak memory usage on CPU.
    
    Note: This is approximate. For accurate memory profiling, use tracemalloc
    or memory_profiler in a separate script.
    """
    import tracemalloc
    
    model = model.to('cpu')
    input_tensor = input_tensor.to('cpu')
    model.eval()
    
    gc.collect()
    tracemalloc.start()
    
    with torch.no_grad():
        for _ in range(n_iterations):
            _ = model(input_tensor)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return peak / (1024 ** 2)  # Convert to MB


def measure_peak_memory_cuda(
    model,
    input_tensor: torch.Tensor,
    n_iterations: int = 10
) -> float:
    """Measure peak GPU memory usage."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    
    model = model.to('cuda')
    input_tensor = input_tensor.to('cuda')
    model.eval()
    
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        for _ in range(n_iterations):
            _ = model(input_tensor)
            torch.cuda.synchronize()
    
    peak_memory = torch.cuda.max_memory_allocated()
    return peak_memory / (1024 ** 2)  # Convert to MB


def benchmark_model(
    model,
    input_tensor: torch.Tensor,
    device: str = 'cpu',
    n_warmup: int = 10,
    n_iterations: int = 100,
    model_path: str = None
) -> BenchmarkResult:
    """
    Full benchmark: latency, memory, throughput.
    
    Args:
        model: PyTorch model to benchmark
        input_tensor: Example input (batch_size, channels, height, width)
        device: 'cpu' or 'cuda'
        n_warmup: Number of warmup iterations (not measured)
        n_iterations: Number of measured iterations
        model_path: Path to saved model file (for file size)
    
    Returns:
        BenchmarkResult with all metrics
    """
    # Measure latency
    latencies = measure_inference_latency(
        model, input_tensor, n_warmup, n_iterations, device
    )
    
    # Compute latency statistics
    latencies = np.array(latencies)
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    throughput = 1000 / mean_latency  # FPS
    
    # Measure memory
    if device == 'cuda':
        peak_memory = measure_peak_memory_cuda(model, input_tensor)
    else:
        peak_memory = measure_peak_memory_cpu(model, input_tensor)
    
    # Model size
    if model_path:
        model_size = get_model_file_size_mb(model_path)
    else:
        model_size = get_model_size_mb(model)
    
    return BenchmarkResult(
        mean_latency_ms=mean_latency,
        std_latency_ms=std_latency,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        p99_latency_ms=p99,
        throughput_fps=throughput,
        peak_memory_mb=peak_memory,
        model_size_mb=model_size
    )


def compare_models(
    models: dict[str, torch.nn.Module],
    input_tensor: torch.Tensor,
    device: str = 'cpu',
    **kwargs
) -> dict[str, BenchmarkResult]:
    """
    Benchmark multiple models and return comparison.
    
    Args:
        models: Dict of {name: model}
        input_tensor: Example input
        device: 'cpu' or 'cuda'
    
    Returns:
        Dict of {name: BenchmarkResult}
    """
    results = {}
    
    for name, model in models.items():
        print(f"Benchmarking {name}...")
        results[name] = benchmark_model(model, input_tensor, device, **kwargs)
        print(f"  {results[name].mean_latency_ms:.2f} ms, {results[name].throughput_fps:.1f} FPS")
    
    return results


def results_to_dataframe(results: dict[str, BenchmarkResult]):
    """Convert benchmark results to pandas DataFrame for easy comparison."""
    import pandas as pd
    
    data = {name: result.to_dict() for name, result in results.items()}
    df = pd.DataFrame(data).T
    df.index.name = 'model'
    return df


def plot_pareto_frontier(
    results: dict[str, BenchmarkResult],
    accuracy_metric: dict[str, float],
    accuracy_label: str = 'mAP@0.5',
    figsize: tuple = (10, 6)
):
    """
    Plot accuracy vs latency with Pareto frontier.
    
    Args:
        results: Dict of {name: BenchmarkResult}
        accuracy_metric: Dict of {name: accuracy_value}
        accuracy_label: Label for y-axis
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    names = list(results.keys())
    latencies = [results[n].mean_latency_ms for n in names]
    accuracies = [accuracy_metric[n] for n in names]
    
    # Plot points
    ax.scatter(latencies, accuracies, s=100, zorder=5)
    
    # Label points
    for name, lat, acc in zip(names, latencies, accuracies):
        ax.annotate(
            name, (lat, acc),
            textcoords="offset points", xytext=(5, 5),
            fontsize=9
        )
    
    # Find and plot Pareto frontier
    # A point is on the frontier if no other point is better in both dimensions
    pareto_points = []
    for i, (lat, acc) in enumerate(zip(latencies, accuracies)):
        dominated = False
        for j, (lat2, acc2) in enumerate(zip(latencies, accuracies)):
            if i != j and lat2 <= lat and acc2 >= acc and (lat2 < lat or acc2 > acc):
                dominated = True
                break
        if not dominated:
            pareto_points.append((lat, acc, names[i]))
    
    if pareto_points:
        pareto_points.sort(key=lambda x: x[0])  # Sort by latency
        pareto_lats = [p[0] for p in pareto_points]
        pareto_accs = [p[1] for p in pareto_points]
        ax.plot(pareto_lats, pareto_accs, 'r--', alpha=0.7, label='Pareto frontier')
    
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel(accuracy_label)
    ax.set_title('Accuracy vs Latency Tradeoff')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# Example usage in notebook:
"""
from src.evaluate.benchmark import benchmark_model, compare_models, constrained_cpu

# Single model benchmark
input_tensor = torch.randn(1, 3, 640, 640)

with constrained_cpu(num_threads=2):
    result = benchmark_model(model, input_tensor, device='cpu')
    print(result)

# Compare multiple models
models = {
    'FP32': model_fp32,
    'FP16': model_fp16,
    'INT8': model_int8
}

results = compare_models(models, input_tensor, device='cpu')

# Plot with accuracy
accuracy = {'FP32': 0.45, 'FP16': 0.44, 'INT8': 0.41}
plot_pareto_frontier(results, accuracy)
"""
