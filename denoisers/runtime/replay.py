import torch


class Replay:
    def __init__(self, fn):
        self.fn = fn
        self.graphs = {}

    def __call__(self, *args) -> None:
        key = tuple(None if arg is None else (arg.shape, arg.dtype) for arg in args)
        entry = self.graphs.get(key)
        if entry is None:
            inputs = [None if arg is None else arg.clone() for arg in args]
            self.fn(*inputs)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                self.fn(*inputs)
            self.graphs[key] = (graph, inputs)
            return
        graph, inputs = entry
        for static, arg in zip(inputs, args):
            if arg is not None:
                static.copy_(arg)
        graph.replay()
