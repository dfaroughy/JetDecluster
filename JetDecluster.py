import h5py 
import numpy as np 
import pyjet
from pyjet import cluster,DTYPE_PTEPM,DTYPE_EP 
import pandas as pd
from random import randrange,shuffle
from timeit import default_timer as timer
import os
# from functions import *
# from basic_functions import *
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
from scipy.linalg import block_diag



class cluster_events(object):

    def __init__(self, 
                 data,
                 cluster_algorithm='anti_kt', 
                 R=1.0, 
                 ptmin=0.0,
                 stop=None
                 ):
        
        self.data=data
        self.cluster_algorithm=cluster_algorithm
        self.R=R
        self.ptmin=ptmin
        self.stop=stop
        
    def __enter__(self):
        num_const=self.data.shape[1]/3
        truth_flag=False
        if not (num_const)%1==0:
            truth_flag=True
        for N,event in self.data.iterrows():
            if truth_flag:
                truth_label=event[self.data.shape[1]-1]
            pseudojets=np.zeros(int(num_const)*3, dtype=DTYPE_PTEPM)
            cut=int(num_const)
            for j in range(int(num_const)):
                if event[j*3]==0.0:
                    cut=j
                    break
                pseudojets[j]['pT' ] = event[j*3]
                pseudojets[j]['eta'] = event[j*3+1]
                pseudojets[j]['phi'] = event[j*3+2]
            #...cluster jets:  
            pseudojets = np.sort(pseudojets[:cut], order='pT')
            sequence = cluster(pseudojets, R=self.R, p=exponent(self.cluster_algorithm))
            jets = sequence.inclusive_jets(ptmin=self.ptmin)
            label = truth_label
            yield N, jets, label
            
            if N+1==self.stop: break
            
    def __exit__(self, exc_type, exc_value, traceback):
        pass

    
    
def decluster_jet(jets, 
                  node_feature=None, 
                  decluster_algorithm='c/a', 
                  R=1.0,
                  jet_ordering='pt',
                  label=None):
        
    '''
    Reclusters constituents of a jets then extracts then de-clusteres to recover 
    the recombaination binary tree of each jet. Features are computed and extracted
    at each declustering. 
    
    ouput: list of binary trees as networkx graph objects with node attributes.
    
    '''

    jet_trees=[]
    
    for n,jet in enumerate(jets):
        bit='1'
        tree={}
        if len(jet.constituents_array())>1:
            clustered_jet = cluster(jet.constituents_array(), R=R, p=exponent(decluster_algorithm))
            tree[bit]=jet
            jet_tree=nx.Graph()
            jet_tree.add_node(1, njet=n, label=label, pseudojet=jet , primary_branch=True)
            t_max=len(jet)+1
            for t in range(2,t_max):
                subjets_0  = clustered_jet.exclusive_jets(t-1)
                subjets_12 = clustered_jet.exclusive_jets(t)
                d=[]
                for i in subjets_0:
                    if i not in subjets_12:
                        d.append(i)
                        break
                for i in subjets_12:
                    if i not in subjets_0: d.append(i)
                
                if len(d)==3: 
                    mother,hard,soft=d[0],d[1],d[2]
                else: 
                    #print('warning: mother subjet without daughter(s)!')
                    continue
                    
                if len(tree)>0:
                    for i in tree.keys():
                        if tree[i]==d[0]: bit=str(i)
                m=bit
                h=bit+'0'; tree[h]=d[1]
                s=bit+'1'; tree[s]=d[2]
                
                if not int(h[1:]): 
                    primary_branch=True
                else: 
                    primary_branch=False
                    
                if node_feature: 
                    nf=node_feature(mother,hard,soft)
                else: 
                    nf=None
                    
                jet_tree.add_node(int(m,2), pseudojet=mother, feature=nf)
                jet_tree.add_node(int(s,2), pseudojet=soft, feature=None, primary_branch=False)
                jet_tree.add_node(int(h,2), pseudojet=hard, feature=None, primary_branch=primary_branch)
                jet_tree.add_edge(int(m,2), int(s,2), weight=soft.pt/jet.pt)
                jet_tree.add_edge(int(m,2), int(h,2), weight=hard.pt/jet.pt)

            jet_trees.append(jet_tree)                    
        if jet_ordering=='mass': jet_trees.sort(key=lambda x:x.nodes(data=True)[1]['pseudojet'].mass, reverse=True)
        elif jet_ordering=='pt': jet_trees.sort(key=lambda x:x.nodes(data=True)[1]['pseudojet'].pt, reverse=True)
        for n,jet in enumerate(jet_trees): jet.nodes(data=True)[1]['j']=n  # update Njet label to match ordering
            
    return jet_trees


def draw_event_trees(G,cmap=plt.cm.viridis_r):
    plt.figure(figsize=(4*len(G),4), constrained_layout=True)
    for n,g in enumerate(G):
        plt.subplot(int(str(1)+str(len(G))+str(n+1)))
        pos = hierarchy_pos(g,1) 
        edge_color = nx.get_edge_attributes(g,'weight').values()
        node_size=[]
        for n in g.nodes(data=True):
            if len(n[1])==0:
                node_size.append(0)
            else:
                node_size.append(4)
        nx.draw(g,pos=pos,node_size=node_size,node_color='k',alpha=1.0,edge_color=edge_color,width=2,edge_cmap=cmap)


def ep2ptepm(rec):
    """ Convert (E, px, py, pz) into (pT, eta, phi, mass)
    Note that the field names of the input array need not match "E", "px",
    "py", or "pz". This function only assumes that the first four fields
    are those quantities. Garbage in, garbage out.
    """
    E, px, py, pz = rec.dtype.names[:4]
    vects = np.empty(rec.shape[0], dtype=DTYPE_PTEPM)
    ptot = np.sqrt(np.power(rec[px], 2) + np.power(rec[py], 2) + np.power(rec[pz], 2))
    costheta = np.divide(rec[pz], ptot)
    costheta[ptot == 0] = 1.
    good_costheta = np.power(costheta, 2) < 1
    vects['pT'] = np.sqrt(np.power(rec[px], 2) + np.power(rec[py], 2))
    vects['eta'][good_costheta] = -0.5 * np.log(np.divide(1. - costheta, 1. + costheta))
    vects['eta'][~good_costheta & (rec[pz] == 0.)] = 0.
    vects['eta'][~good_costheta & (rec[pz] > 0.)] = 10e10
    vects['eta'][~good_costheta & (rec[pz] < 0.)] = -10e10
    vects['phi'] = np.arctan2(rec[py], rec[px])
    vects['phi'][(rec[py] == 0) & (rec[px] == 0)] = 0
    mass2 = np.power(rec[E], 2) - np.power(ptot, 2)
    neg_mass2 = mass2 < 0
    mass2[neg_mass2] *= -1
    vects['mass'] = np.sqrt(mass2)
    vects['mass'][neg_mass2] *= -1
    return vects



def exponent(jet_algo):
    exponent={}; exponent['anti_kt']=-1; exponent['c/a']=0; exponent['kt']=1
    return exponent[jet_algo]


#........................................................


def dphi(jet1,jet2):
    dphi = abs(jet1.phi - jet2.phi)
    if dphi > np.pi: 
        dphi = 2*np.pi - dphi
    return dphi

#........................................................


def deta(jet1,jet2):
    return jet1.eta - jet2.eta


#........................................................

def deltaR2(jet1,jet2):
    x = deta(jet1,jet2)
    y = dphi(jet1,jet2)
    return x**2 + y**2


#........................................................


def deltaR(jet1,jet2):
    return np.sqrt(deltaR2(jet1,jet2))


#........................................................


def kt_dist(jet1,jet2):
    min_pt = min((jet1.pt)**2,(jet2.pt)**2);
    return min_pt * deltaR2(jet1,jet2)


#........................................................


def inv_M(jet1,jet2):
    px = jet1.px + jet2.px
    py = jet1.py + jet2.py
    pz = jet1.pz + jet2.pz
    e  = jet1.e  + jet2.e
    return np.sqrt(e**2 - px**2 - py**2 - pz**2)
