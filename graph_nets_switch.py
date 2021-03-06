from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from graph_nets import utils_np
from graph_nets import utils_tf
from graph_nets import modules
from graph_nets.demos_tf2 import models
import matplotlib.pyplot as plt
import numpy as np

import graph_pb2
import example_pb2
import time
import random

import matplotlib.pyplot as plt
import networkx as nx
import sonnet as snt
import tensorflow as tf

import os

SEED = 1
np.random.seed(SEED)
tf.random.set_seed(SEED)

data = graph_pb2.Data()
f = open("training.switch", "rb")
data.ParseFromString(f.read())
f.close()
all_graphs = list(data.graph)
print(len(all_graphs))
all_graphs = [graph for graph in all_graphs if len([node for node in graph.nodes if len(node.connectedto)>0])>0]

f = open("test.switch", "rb")
data.ParseFromString(f.read())
f.close()
test_graphs = list(data.graph)
test_graphs = [graph for graph in test_graphs if len([node for node in graph.nodes if len(node.connectedto)>0])>0]

class Initializer(snt.initializers.Initializer):
  def __init__(self, variables):
    self._variables = variables
    self._index = 0
  
  def __call__(self, shape, dtype):
    variables = self._variables[self._index]
    self._index = self._index + 1
    return variables

def get_variable(size):
  return tf.Variable(np.random.rand(size).astype('float32'))

def make_mlp_model(output_sizes):
  return snt.Sequential([
      snt.nets.MLP(output_sizes, activate_final=True),
      snt.LayerNorm(axis=-1, create_offset=True, create_scale=True)
  ])

def save_mlp_model(path, module):
  variables = [var.numpy() for var in module.variables]
  biases = [variables[i] for i in range(len(variables) - 2) if i % 2 == 0]
  weights = [variables[i] for i in range(len(variables) - 2) if i % 2 == 1]
  offset = [variables[len(variables) - 2]]
  scale = [variables[len(variables) - 1]]
  np.save(path + "_biases", biases)
  np.save(path + "_weights", weights)
  np.save(path + "_offset", offset)
  np.save(path + "_scale", scale)

def load_mlp_model(path):
  biases = np.load(path + "_biases.npy", allow_pickle=True)
  output_sizes = [bias.shape[0] for bias in biases]
  weights = np.load(path + "_weights.npy", allow_pickle=True)
  offset = np.load(path + "_offset.npy", allow_pickle=True)
  scale = np.load(path + "_scale.npy", allow_pickle=True)
  return snt.Sequential([
      snt.nets.MLP(output_sizes, activate_final=True,w_init=Initializer(weights), b_init=Initializer(biases)),
      snt.LayerNorm(axis=-1, create_offset=True, create_scale=True, scale_init=Initializer(scale), offset_init=Initializer(offset))])


def make_linear_model(output_size):
  return snt.Linear(output_size)

def save_linear_model(path, linear):
  variables = [var.read_value().numpy() for var in linear.variables]
  np.save(path, variables)

def load_linear_model(path):
  variables = np.load(path + ".npy", allow_pickle=True)
  bias = [variables[0]]
  weight = [variables[1]]
  return snt.Linear(1, w_init=Initializer(weight), b_init=Initializer(bias))

def make_path(dir, name):
    return dir + "/" + name + ".npy"
	
class GraphNN():
  #typ kann sein "GraphNetwork" oder "GraphIndependent" oder "GraphOutput"
  def __init__(self, output, name, typ):
    self._name = name
    self._typ = typ

    if (self._typ == "GraphNetwork" or self._typ == "GraphIndependent"):
      self._edge_mlp = make_mlp_model(output)
      self._node_mlp = make_mlp_model(output)
      self._global_mlp = make_mlp_model(output)
    elif (self._typ == "GraphOutput"):
      self._edge_mlp = make_linear_model(output)
      self._node_mlp = make_linear_model(output)
      self._global_mlp = make_linear_model(output)
    else:
      print("unkown type: " + self._typ)

    self.create_network()
  
  def get_variables(self):
    return self._network.trainable_variables

  def create_network(self):
    if (self._typ == "GraphNetwork"):
      self._network = modules.GraphNetwork(lambda: self._edge_mlp, lambda: self._node_mlp, lambda: self._global_mlp, name=self._name)
    elif (self._typ == "GraphIndependent" or "GraphOutput"):
      self._network = modules.GraphIndependent(lambda: self._edge_mlp, lambda: self._node_mlp, lambda: self._global_mlp, name=self._name)
    else:
      print("unkown type: " + self._typ)

  def __call__(self, input):
    return self._network(input)

  #htis is the directory to save all the mlps
  def save(self, dir):
    path = dir + "/" + self._name
    try:
      os.makedirs(path)
    except FileExistsError:
      print(path, "already exists")

    if (self._typ == "GraphNetwork" or self._typ == "GraphIndependent"):
      save_mlp_model(path + "/edge_mlp", self._edge_mlp)
      save_mlp_model(path + "/node_mlp", self._node_mlp)
      save_mlp_model(path + "/global_mlp", self._global_mlp)
    elif (self._typ == "GraphOutput"):
      save_linear_model(path + "/edge_linear", self._edge_mlp)
      save_linear_model(path + "/node_linear", self._node_mlp)
      save_linear_model(path + "/global_linear", self._global_mlp)
    else:
      print("unkown type: " + self._typ)
  
  def load(self, dir):
    path = dir + "/" + self._name

    if (self._typ == "GraphNetwork" or self._typ == "GraphIndependent"):
      self._edge_mlp = load_mlp_model(path + "/edge_mlp")
      self._node_mlp = load_mlp_model(path + "/node_mlp")
      self._global_mlp = load_mlp_model(path + "/global_mlp")
    elif (self._typ == "GraphOutput"):
      self._edge_mlp = load_linear_model(path + "/edge_linear")
      self._node_mlp = load_linear_model(path + "/node_linear")
      self._global_mlp = load_linear_model(path + "/global_linear")
    else:
      print("unkown type: " + self._typ)

    self.create_network()
	
class n_layer_Process_Encode_Decode():
  def __init__(self, num_layers, latent_size, num_processing_steps):
    self._num_processing_steps = num_processing_steps

    self._nodes_left_init = get_variable(latent_size - 3)
    self._nodes_right_init = get_variable(latent_size - 3)

    self._global_init = get_variable(latent_size - 2)

    self._edges_left_to_right_init = get_variable(latent_size - 3)
    self._edges_right_to_left_init = get_variable(latent_size - 3)

    self._edges_left_left_to_right_init = get_variable(latent_size - 3)
    self._edges_left_right_to_left_init = get_variable(latent_size - 3)

    self._edges_right_right_to_left_init = get_variable(latent_size - 3)
    self._edges_right_left_to_right_init = get_variable(latent_size - 3)

    output_sizes = [latent_size] * num_layers

    self._encoder = GraphNN(output_sizes, "encoder", "GraphIndependent")
    self._cores = [GraphNN(output_sizes, "core" + str(i), "GraphNetwork") for i in range(num_processing_steps)]
    self._decoder = GraphNN(output_sizes, "decoder", "GraphIndependent")
    self._output_transform = GraphNN(1, "output_transform", "GraphOutput")

    self._optimizer = tf.keras.optimizers.Adam(0.0001)
  def get_variables(self):
    variables = [self._nodes_left_init, self._nodes_right_init, self._global_init, self._edges_left_to_right_init, self._edges_right_to_left_init, self._edges_left_left_to_right_init, self._edges_left_right_to_left_init, self._edges_right_right_to_left_init, self._edges_right_left_to_right_init]
    for var in self._encoder.get_variables():
      variables.append(var)
    for net in self._cores:
      for var in net.get_variables():
        variables.append(var)
    for var in self._decoder.get_variables():
      variables.append(var)
    for var in self._output_transform.get_variables():
      variables.append(var)
    return variables

  def save(self, path):
    try:
      os.makedirs(path + "/embeddings")
    except FileExistsError:
      print(path, "already exists")

    np.save(path + "/embeddings/nodes_left_init", self._nodes_left_init.numpy())
    np.save(path + "/embeddings/nodes_right_init", self._nodes_right_init.numpy())

    np.save(path + "/embeddings/global_init", self._global_init.numpy())

    np.save(path + "/embeddings/edges_left_to_right_init", self._edges_left_to_right_init.numpy())
    np.save(path + "/embeddings/edges_right_to_left_init", self._edges_right_to_left_init.numpy())

    np.save(path + "/embeddings/edges_right_right_to_left_init", self._edges_right_right_to_left_init.numpy())
    np.save(path + "/embeddings/edges_right_left_to_right_init", self._edges_right_left_to_right_init.numpy())

    np.save(path + "/embeddings/edges_left_right_to_left_init", self._edges_left_right_to_left_init.numpy())
    np.save(path + "/embeddings/edges_left_left_to_right_init", self._edges_left_left_to_right_init.numpy())



    self._encoder.save(path)
    for core in self._cores:
      core.save(path)
    self._decoder.save(path)
    self._output_transform.save(path)

  def load(self, path):  
    self._nodes_left_init = tf.Variable(np.load(path + "/embeddings/nodes_left_init.npy", allow_pickle=True))
    self._nodes_right_init = tf.Variable(np.load(path + "/embeddings/nodes_right_init.npy", allow_pickle=True))

    self._global_init = tf.Variable(np.load(path + "/embeddings/global_init.npy", allow_pickle=True))

    self._edges_left_to_right_init = tf.Variable(np.load(path + "/embeddings/edges_left_to_right_init.npy", allow_pickle=True))
    self._edges_right_to_left_init = tf.Variable(np.load(path + "/embeddings/edges_right_to_left_init.npy", allow_pickle=True))

    self._edges_right_right_to_left_init = tf.Variable(np.load(path + "/embeddings/edges_right_right_to_left_init.npy", allow_pickle=True))
    self._edges_right_left_to_right_init = tf.Variable(np.load(path + "/embeddings/edges_right_left_to_right_init.npy", allow_pickle=True))

    self._edges_left_right_to_left_init = tf.Variable(np.load(path + "/embeddings/edges_left_right_to_left_init.npy", allow_pickle=True))
    self._edges_left_left_to_right_init = tf.Variable(np.load(path + "/embeddings/edges_left_left_to_right_init.npy", allow_pickle=True))

    self._encoder.load(path)
    for core in self._cores:
      core.load(path)
    self._decoder.load(path)
    self._output_transform.load(path)


  def create_graphs_tuple(self, graphs):
    data_dicts = [self.data_dict_from_graph(graph) for graph in graphs]
    graphs_tuple = utils_tf.data_dicts_to_graphs_tuple(data_dicts)
    return graphs_tuple

  def data_dict_from_graph(self, graph):
    num_nodes_left = len(graph)
    num_nodes_right = np.amax([np.amax(nodes) for nodes in graph if len(nodes) > 0]) + 1

    nodes_left = [tf.concat([self._nodes_left_init,[i],[num_nodes_left,num_nodes_right]],0) for i in range(num_nodes_left)]
    nodes_right = [tf.concat([self._nodes_right_init,[i],[num_nodes_left,num_nodes_right]],0) for i in range(num_nodes_right)]
  
    nodes = nodes_left + nodes_right

    global_features = tf.concat([self._global_init,[num_nodes_left,num_nodes_right]],0)

      #kanten hinzufügen
    edges = []
    senders = []
    receivers = []

    for i in range(num_nodes_left - 1):
      senders.append(i)
      receivers.append(i + 1)
      edges.append(tf.concat([self._edges_left_left_to_right_init,[num_nodes_left,num_nodes_right]],0))
    
    for i in range(num_nodes_left - 1):
      senders.append(i + 1)
      receivers.append(i)
      edges.append(tf.concat([self._edges_left_right_to_left_init,[num_nodes_left,num_nodes_right]],0))

    for i in range(num_nodes_right - 1):
      senders.append(num_nodes_left + i)
      receivers.append(num_nodes_left + i + 1)
      edges.append(tf.concat([self._edges_right_left_to_right_init,[num_nodes_left,num_nodes_right]],0))

      senders.append(num_nodes_left + i + 1)
      receivers.append(num_nodes_left + i)
      edges.append(tf.concat([self._edges_right_right_to_left_init,[num_nodes_left,num_nodes_right]],0))
    
      #restlichen kanten hinzufügen
    for i in range(num_nodes_left):
      for j in graph[i]:
        edges.append(tf.concat([self._edges_left_to_right_init,[num_nodes_left,num_nodes_right]],0))
        senders.append(i)
        receivers.append(num_nodes_left + j)

        edges.append(tf.concat([self._edges_right_to_left_init,[num_nodes_left,num_nodes_right]],0))
        senders.append(num_nodes_left + j)
        receivers.append(i)

    data_dict = {"globals": global_features, "nodes": nodes, "edges": edges, "senders": senders, "receivers": receivers}
    return data_dict

  def __call__(self, graphs):
    graphs_tuple = self.create_graphs_tuple(graphs)
    latent = self._encoder(graphs_tuple)
    latent0 = latent
    for i in range(self._num_processing_steps):
      core_input = utils_tf.concat([latent0, latent], axis=1)
      latent = self._cores[i](core_input)
    decode = self._decoder(latent)
    return self._output_transform(decode)

  #muss array von graphen sein wie states = [[[0],[1,2]]]
  def predict(self, states):
    output = self(states)

    return [self._predict(states[i], utils_tf.get_graph(output, i).edges) for i in range(len(states))]

  def _predict(self, state, edges):
    num_nodes_left = len(state)
    num_nodes_right = np.amax([np.amax(nodes) for nodes in state if len(nodes) > 0]) + 1
    num_edges = 2 * (num_nodes_left - 1) + 2 * (num_nodes_right - 1) + 2 * sum([len(node) for node in state])

    matrix = []

    for i in range(num_nodes_left - 1):
      row = np.zeros(num_edges).astype('float32')
      row[i] = 1
      matrix.append(row)

    return tf.reshape(tf.matmul(matrix, edges), [-1])
  
  def loss_fn(self,predictions, targets):
    return tf.math.reduce_mean([tf.keras.losses.mean_squared_error(targets[num],predictions[num]) for num in range(len(targets))])

  def train(self, states, targets):

    with tf.GradientTape() as tape:
      predictions = self.predict(states)
      loss = self.loss_fn(predictions, targets)
      print(loss.numpy())

    variables = self.get_variables()
    gradients = tape.gradient(loss, variables)
    self._optimizer.apply_gradients(zip(gradients, variables))
	
def switch_score(graphs):
	return [graph.score[:-1] for graph in graphs]

def accuracy(model, acc_graphs):
  states = [[node.connectedto for node in graph.nodes] for graph in acc_graphs]
  predictions = model.predict(states)
  correct = 0

  for i in range(len(states)):
    prediction = [1 if pred > 0.5 else 0 for pred in predictions[i]]
    right = True
    for index in range(len(prediction)):
      if prediction[index] != acc_graphs[i].score[index]:
        right = False
    if right == True:
      correct = correct + 1
  return correct / len(states)

def train_model(batchsize_nodes, model, training_graphs, testing_graphs, target_fn, path, acc_every = 10, num_graphs_to_test = 1000):
  temp_accuracy = 0
  start_time = time.time()

  iteration =0
  while time.time() - start_time < 10.9*3600:
    random.shuffle(training_graphs)
    i = 0
    graphs = []
    num_nodes = 0
    while True:
      graphs.append(training_graphs[i])
      num_nodes = num_nodes + len(training_graphs[i].score) + training_graphs[i].right
      i = i + 1
      if num_nodes > batchsize_nodes:
        break
    
    states = [[node.connectedto for node in graph.nodes]  for graph in graphs]
    targets = target_fn(graphs)
    model.train(states, targets)

    if iteration % acc_every == 0:
      this_accuracy = accuracy(model, testing_graphs[:num_graphs_to_test])
      print(this_accuracy)
      if this_accuracy > temp_accuracy:
        print("new highest accuracy")
        temp_accuracy = this_accuracy
        model.save(path)
    iteration = iteration + 1

  model.load(path)
    
  testing_graphs_partitions = [testing_graphs[i*1000:(i+1)*1000]  for i in range(10)]
  
  final_accuracy = [accuracy(model,partition) for partition in testing_graphs_partitions]

  final_loss = [model.loss_fn(model.predict([[node.connectedto for node in graph.nodes] for graph in partition]), target_fn(partition)).numpy() for partition in testing_graphs_partitions]


  text_file = open(path + "/results.txt", "w")
  text_file.write("accuracy: " + str(np.mean(final_accuracy)) + ", loss: " + str(np.mean(final_loss)) + ", loss: " + str(np.mean(final_loss)), + ", batchsize_nodes: " + str(batchsize_nodes))
  text_file.close()
  
a = n_layer_Process_Encode_Decode(2,80,5)
train_model(300*12.5,a,all_graphs,test_graphs,switch_score, "models/n_layer_Process_Encode_Decode/switch(2,80,5)", 15,500)