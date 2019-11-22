# coding: utf-8

from visualize import *
import hiddenlayer as hl

from graphviz import Digraph
import networkx as nx
import re

from allennlp.common.util import import_submodules
from allennlp.predictors import Predictor

from torch._tensor_str import set_printoptions 
import torch.nn as nn

set_printoptions(sci_mode=False)
import_submodules('jdnlp')

model = Predictor.from_path("best/remembert/model.tar.gz", predictor_name="text_classifier", cuda_device=0)
mac = model._model

read_unit = mac.mac.read
ctrl_unit = mac.mac.control


def run_example(sent="Anyway, any real source on what sentence this nigger got?"):
    # register_vis_hooks(mac)
    pred = model.predict(sent)
    # remove_vis_hooks()
    # save_visualization("visuals/mac_viz", format="png")
    
    # inst = model._json_to_instance({"sentence": "Test test test"})     
    # words = inst.fields['tokens'].tokens
    return pred


def attn_heatmaps(read_attn, ctrl_attn, cm=sns.light_palette("navy")):
    read_attn = torch.cat(read_attn).T
    # read_attn[[*range(5), *(-i for i in reversed(range(15)))]]
    hm = sns.heatmap(read_attn.cpu().numpy(), cmap=cm)
    
    # hm = sns.heatmap(read_attn[[*range(5), *(-i for i in reversed(range(15)))]].cpu().numpy(), cmap=cm)
    ticks = hm.yaxis.get_major_ticks()
    hm.set_yticklabels(['topic_{}'.format('n' if (i == len(ticks)-1) else i) for i in range(len(ticks))], rotation=(360))
    for t in ticks[2:-1]: 
        t.set_visible(False)
    
    plt.savefig('visuals/mac-read-heatmap.png'); plt.clf()
    
    ctrl_attn = torch.cat(ctrl_attn).T
    hm = sns.heatmap(ctrl_attn.T.cpu().numpy(), cmap=cm)
    hm.set_yticklabels(['w_{}'.format(i) for (i, w) in enumerate(words)], rotation=(45))
    plt.savefig('visuals/mac-control-heatmap.png'); plt.clf()


def sanitize_node(n):
    n = re.sub(r'/outputs/\d+', '', n)
    if '/' in n:
        #n = '/'.join(n.split('/')[1:])
        n  = n.split('/')[-1]
    #else:
    #    n = re.sub(r'\[.*\]', '', n))
    n = (('/' in n) and '/'.join(n.split('/')[1:])) or n
    return n

def filter_edges(edges):
    cleaned_edges = []
    for e in edges:
        e = tuple(map(sanitize_node, e[:2]))
        if all(e) and e not in cleaned_edges and tuple(reversed(e)) not in cleaned_edges:
            cleaned_edges.append(e)
    
    return cleaned_edges

def build_graph(edges):
    fc = nn.Linear(768, 2)
    net = nn.Sequential(mac, fc) # fails b/c tuple input
    # hl.build_graph(net, tuple(mac.mac.saved_inp)) 
    # graph = hl.build_graph(mac.mac, tuple(mac.mac.saved_inp)) 
    # graph.build_dot().render('visuals/mac-hl', format='png')
    
    g = nx.DiGraph()
    g.add_edges_from(filter_edges(edges))
    g.remove_edges_from(g.selfloop_edges())
    
    # nx.drawing.nx_pydot.write_dot(G, "visuals/mac-nx.dot")
    # !dot -Tpng visuals/mac-nx.dot -o visuals/mac-nx.png 
    
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    graph_attr=dict(size="12,12")
    
    dot = Digraph(node_attr=node_attr, graph_attr=graph_attr)
    dot.edges(g.edges())
    dot.render('visuals/mac-nx', format='png')
    
    return g

"""
 def get_seq_exec_list(model, DUMMY_INPUT): 
   ...:     model.eval() 
   ...:     traced = torch.jit.trace(model, DUMMY_INPUT, check_trace=False) 
   ...:     seq_exec_list = traced.code 
   ...:     seq_exec_list = seq_exec_list.split('\n') 
   ...:     for idx, item in enumerate(seq_exec_list): 
   ...:         print("[{}]: {}".format(idx, item)) 
"""

print(run_example())

X = mac.mac.saved_inp