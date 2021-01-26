from transformers import PyTorchBenchmark, PyTorchBenchmarkArguments


args = PyTorchBenchmarkArguments(models=["examples/text-classification/saved_models/debug/"],
                                 batch_sizes=[8],  verbose=True,
                                 training=True, sequence_lengths=[128])


benchmark = PyTorchBenchmark(args)

print(benchmark.run())