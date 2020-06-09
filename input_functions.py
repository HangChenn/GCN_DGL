#in this file, all input related functions are added

import networkx as nx
import random

# this functions takes two arguments
# graph: this is a networkx graph
# file_name: this is the output file
# this functions write the graph in the file in txt format
# later we can read this file using the functions defined later
# note that, the positions are selected in a random way
def write_networx_graph(graph, file_name):
 file = open(file_name,"w")
 file.write(str(graph.number_of_nodes())+"\n");
 for j in range(graph.number_of_nodes()):
  file.write(str(random.randint(1,300))+" "+str(random.randint(1,300))+"\n")
 edges = graph.edges()
 for e in edges:
  file.write(str(e[0])+" "+str(e[1])+"\n")
 file.close()

def is_comment(x):
 if x[0]=='#':
  return True
 return False

def take_input(input_file):
 file = open(input_file,"r")
 #print("File name: "+input_file)
 while True:
  l = file.readline()
  #print(l)
  if not is_comment(l):
   break
 nodes_weight = {}
 if l == "Nodes\n":
  l = file.readline()
  l = file.readline().rstrip('\n').split()
  while len(l) == 2:
   nodes_weight[int(l[0])] = {'node_weight' :int(l[1]) }
   l = file.readline().rstrip('\n').split()

 if bool(nodes_weight):
  l = file.readline()
 m = int(l)
 edge_list = list()
 for i in range(m):
    while True:
     l = file.readline()
     if len(l) == 0:
      break
     if not is_comment(l):
      break
    t_arr1 = []
    t_arr2 = l.split()
    if(len(t_arr2)<3):break
    #t_arr1.append(int(t_arr2[0])-1)
    #t_arr1.append(int(t_arr2[1])-1)
    t_arr1.append(int(t_arr2[0]))
    t_arr1.append(int(t_arr2[1]))
    #t_arr1.append(int(t_arr2[2]))
    t_arr1.append(float(t_arr2[2]))
    edge_list.append(t_arr1)

 levels = int(file.readline())
 tree_ver=[]
 #tree_ver = [(int(x)-1) for x in raw_input().split()]
 for l in range(levels):
  #print "Steiner tree vertices of level "+str(l+1)+":"
  #tree_ver.append([(int(x)-1) for x in file.readline().split()])
  tree_ver.append([(int(x)) for x in file.readline().split()])

 file.close()
 return edge_list, tree_ver, nodes_weight

def take_non_uniform_input(input_file):
 file = open(input_file,"r")
 #print("File name: "+input_file)
 while True:
  l = file.readline()
  #print(l)
  if not is_comment(l):
   break
 m = int(l)
 levels = int(file.readline())
 edge_list = list()
 for i in range(m):
    while True:
     l = file.readline()
     if len(l) == 0:
      break
     if not is_comment(l):
      break
    t_arr1 = []
    t_arr2 = l.split()
    if(len(t_arr2)<3):break
    #t_arr1.append(int(t_arr2[0])-1)
    #t_arr1.append(int(t_arr2[1])-1)
    t_arr1.append(int(t_arr2[0]))
    t_arr1.append(int(t_arr2[1]))
    #t_arr1.append(int(t_arr2[2]))
    for j in range(levels):
     t_arr1.append(float(t_arr2[2+j]))
    edge_list.append(t_arr1)

 #levels = int(file.readline())
 tree_ver=[]
 #tree_ver = [(int(x)-1) for x in raw_input().split()]
 for l in range(levels):
  #print "Steiner tree vertices of level "+str(l+1)+":"
  #tree_ver.append([(int(x)-1) for x in file.readline().split()])
  tree_ver.append([(int(x)) for x in file.readline().split()])

 file.close()
 return edge_list, tree_ver


import json

def take_input_from_json(input_file):
 file = open(input_file,"r")
 #print("File name: "+input_file)
 graph = json.loads(file.read())
 #print(graph)

 n = len(graph['nodes'])
 coord_list = list()
 edge_list = list()
 matrix = [[0] * n for i in range(n)]

 for v in graph['nodes']:
  arr = []
  arr.append(float(v['x']))
  arr.append(float(v['y']))
  coord_list.append(arr)

 for e in graph['edges']:
  matrix[e['target']][e['source']] = matrix[e['source']][e['target']] = 1
  t_arr1 = []
  t_arr1.append(e['source'])
  t_arr1.append(e['target'])
  edge_list.append(t_arr1)

 file.close()
 return n, coord_list, edge_list

def txt_to_json(input_file, output_file):
 n, coord_list, edge_list = take_input(input_file)
 graph = {}
 nodes = []
 max_x = 0
 max_y = 0
 for i in range(len(coord_list)):
  node = {}
  node['id'] = i
  node['x'] = coord_list[i][0]
  if max_x<coord_list[i][0]:max_x=coord_list[i][0]
  node['y'] = coord_list[i][1]
  if max_y<coord_list[i][1]:max_y=coord_list[i][1]
  nodes.append(node)
 graph['nodes'] = nodes
 edges = []
 for i in range(len(edge_list)):
  edge = {}
  edge['source']=edge_list[i][0]
  edge['target']=edge_list[i][1]
  edges.append(edge)
 graph['edges'] = edges
 graph['xdimension'] = max_x
 graph['ydimension'] = max_y
 with open(output_file, 'w') as outfile:
  json.dump(graph, outfile)


# this function directly builds a networkx graph from a txt file
def build_networkx_graph(filename):
 edge_list, tree_vers, nodes_weight  = take_input(filename)
 G=nx.Graph()
 for e in edge_list:
  G.add_weighted_edges_from([(e[0], e[1], e[2])])
 # node_weight is empty, them don't set node attribute
 if bool(nodes_weight):
  nx.set_node_attributes(G, nodes_weight)

 return G, tree_vers

# this function directly builds a networkx graph from a txt file
def build_networkx_non_uniform_graph(filename):
 edge_list, tree_vers = take_non_uniform_input(filename)
 Gs = []
 for i in range(len(tree_vers)):
  Gs.append(nx.Graph())
 for e in edge_list:
  for l in range(len(tree_vers)):
   Gs[l].add_weighted_edges_from([(e[0], e[1], e[l+2])])
 return Gs, tree_vers

def parse_dot_file(file_name):
 file = open(file_name, 'r')
 arr = file.read().split(';')
 nodes = []
 edges = []
 node_coords = []
 edge_list = []
 for i in range(len(arr)):
 #for i in range(10):
  arr[i] = arr[i].strip()
  elmnt = arr[i].split()
  if len(elmnt)>2:
   if elmnt[1][0]=='[':
    #print(elmnt[0])
    nodes.append(elmnt)
   else:
    #print(elmnt[0]+elmnt[1]+elmnt[2])
    edges.append(elmnt)
  #print(elmnt)
 for i in range(len(nodes)):
  #print nodes[i]
  #print(nodes[i][0])
  coords = nodes[i][2][5:len(nodes[i][2])-2].split(',')
  for k in range(len(coords)):
   coords[k] = float(coords[k])
  node_coords.append(coords)
 for i in range(1,len(edges)):
  edg = []
  #print(edges[i])
  edg.append(int(edges[i][0]))
  edg.append(int(edges[i][2]))
  edge_list.append(edg)
 file.close()

 #print(node_coords)
 #print(edge_list)
 return node_coords, edge_list

def take_input_from_dot(input_file):
 node_coords, edge_list = parse_dot_file(input_file)
 n = len(node_coords)
 return n, node_coords, edge_list 


def take_input_force_directed(file_name):
 file = open(file_name, 'r')
 arr = file.read().split('\n')
 x = []
 y = []
 for i in arr[0].split(','):
  x.append(float(i))
 for i in arr[1].split(','):
  y.append(float(i))
 file.close()
 return x,y

def write_as_txt_random_position(file_name, graph):
 file = open(file_name,"w")
 file.write(str(graph.number_of_nodes())+"\n");
 for j in range(graph.number_of_nodes()):
  file.write(str(random.randint(1,300))+" "+str(random.randint(1,300))+"\n")
 edges = graph.edges()
 for e in edges:
  file.write(str(e[0])+" "+str(e[1])+"\n")
 file.close()

def write_as_txt(file_name, graph, x, y):
 file = open(file_name,"w")
 file.write(str(graph.number_of_nodes())+"\n");
 for j in range(graph.number_of_nodes()):
  file.write(str(x[j])+" "+str(y[j])+"\n")
 edges = graph.edges()
 for e in edges:
  file.write(str(e[0])+" "+str(e[1])+"\n")
 file.close()


