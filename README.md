# Graph Convolutional Neural Network(GraphSAGE) application on Steiner tree

## Install


We use Python 3.7 for this project
please visit https://pytorch.org/ and download suitable pytorch package


```
pip install -r requirements.txt
```

alternativly 
```
conda install -c dglteam dgl 
# please visit https://pytorch.org/ and download suitable pytorch package
# e.g. for linux
conda install pytorch torchvision cpuonly -c pytorch
```


## RUN
```
python connected_graph_task.py
python connected_node_task.py
python steiner_task.py
```
Inside steiner_task.py, I had several task, you can play with different graph set by change "t_task"

## relevant website
https://docs.dgl.ai/en/0.4.x/install/
https://docs.dgl.ai/en/0.4.x/api/python/nn.mxnet.html?highlight=sage#dgl.nn.mxnet.conv.SAGEConv

-----
https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.algorithms.mst.minimum_spanning_tree.html#networkx.algorithms.mst.minimum_spanning_tree
