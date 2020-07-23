import tensorflow as tf
import Proto/Linux/example_pb2 as example
import matplotlib.pyplot as plt
import numpy as np

data = example.Data()
f = open("Comparison/heuristics_nodes_size_50_leftest_node.comparison", "rb")
data.ParseFromString(f.read())
f.close()

comp = data.graphs

for i in range(100):
    print(comp[i].learned_leftest_node)