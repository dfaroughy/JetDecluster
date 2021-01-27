import h5py 
import numpy as np 
import pyjet
from pyjet import cluster,DTYPE_PTEPM,DTYPE_EP 
import pandas as pd
from random import randrange,shuffle
from timeit import default_timer as timer
import os
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from particle import Particle,PDGID
import mplhep as hep 
from numpy.lib.recfunctions import append_fields
plt.style.use(hep.style.ROOT) # For now ROOT defaults to CMS


class get_particles(object):
    def __init__(self,data,stop=None, dtype='EP'):
        self.dtype=dtype
        self.data=data
        self.stop=stop
    def __enter__(self):
        
            # TODO: add hdf5 input type            
            line = HepMCReader(self.data)
            Nev=0
            while True:
                Nev+=1
                evt = line.next()
                if not evt:
                    break 
                final_states=[evt.particles[p][1:] for p in evt.particles if evt.particles[p][0]==1] # extracts only final state particles
                pseudojets=np.zeros(len(final_states),dtype=DTYPE_EP) 
                pseudojets=append_fields(pseudojets, 'pid', data=np.zeros(len(final_states)))
                for j,p in enumerate(final_states): # this is too slow.... need to optimize
                    pseudojets[j]['px']=p[1]
                    pseudojets[j]['py']=p[2]
                    pseudojets[j]['pz']=p[3]
                    pseudojets[j]['E']=p[4]
                    pseudojets[j]['pid']=p[0]
                if self.dtype=='EP':
                    yield pseudojets
                elif self.dtype=='PTEPM':
                    yield ep2ptepm_pid(pseudojets)
                if Nev==self.stop:
                    break
    def __exit__(self, exc_type, exc_value, traceback):
        pass 



def decluster(jet,jet_label=None,node_feature=None,node_label=None,R=1.0,p=0,directed=False):
        
    '''
    Reclusters constituents of a jets then extracts then de-clusteres to recover 
    the recombaination binary tree of each jet. Features are computed and extracted
    at each declustering. 
    
    ouput: list of binary trees as networkx graph objects with node attributes.
    
    '''
    
    def none(*kargs):
        return None
    
    def is_primary(node):    
        if node=='1':
            return True
        elif not int(node[1:]): 
            return True
        else: 
            return False
    
    bit='1'
    tree={}
    if len(jet)>1:
        
        # recluster jet constituents using c/a algorithm:
        
        pseudojets=np.zeros(len(jet),dtype=DTYPE_PTEPM) 
        pseudojets=append_fields(pseudojets, 'pid', data=np.zeros(len(jet)))

        for i,j in enumerate(jet): 
            pseudojets[i]['pid']=j.pid
            pseudojets[i]['pT']=j.pt
            pseudojets[i]['eta']=j.eta
            pseudojets[i]['phi']=j.phi
            pseudojets[i]['mass']=j.mass
            
        clustered_jet = cluster(pseudojets, R=R, p=p)
        
        # define jet tree (as a netwrokx directed or undirected graph):
        
        tree[bit]=jet
        if directed:
            jet_tree=nx.DiGraph() 
        else:
            jet_tree=nx.Graph()    
            
        # root node:    
        jet_tree.add_node(1,jet_label=jet_label)
        
        # decluster jets recursevily. j0 -> j1 j2
        t_max=len(jet)+1
        
        for t in range(2,t_max):
            
            subjets_0  = clustered_jet.exclusive_jets(t-1)
            subjets_12 = clustered_jet.exclusive_jets(t)
            d=[]
            
            # find mother:
            for i in subjets_0:
                if i not in subjets_12:
                    d.append(i)
                    break
                    
            #find daughters:
            for i in subjets_12:
                if i not in subjets_0: d.append(i)
                    
            if len(d)==3: 
                mother,hard,soft=d[0],d[1],d[2]     # fix this!  some mothers have no daughters...
            else:
                continue

            if len(tree)>0:
                for i in tree.keys():
                    if tree[i]==d[0]: 
                        bit=str(i)
            
            # update node labels:
            
            m=bit
            h=bit+'0'; tree[h]=d[1]
            s=bit+'1'; tree[s]=d[2]
            
            # add features/labels/bools to nodes:
            
            if not node_feature:
                node_feature=none
            if not node_label:
                node_label=none
                
            jet_tree.add_node(int(m,2),pseudojet=mother,feature=node_feature(mother,hard,soft),primary_branch=is_primary(m),label=node_label(mother))
            jet_tree.add_node(int(s,2),pseudojet=soft,feature=None,primary_branch=False,label=node_label(soft))
            jet_tree.add_node(int(h,2),pseudojet=hard,feature=None,primary_branch=is_primary(h),label=node_label(hard))
            jet_tree.add_edge(int(m,2),int(s,2),weight=soft.pt/jet.pt)
            jet_tree.add_edge(int(m,2),int(h,2),weight=hard.pt/jet.pt)

        return jet_tree
    else:
        return None






# older function... deprecated
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





class cluster_events(object):

    def __init__(self, 
                 data,
                 p=-1, 
                 R=1.0, 
                 ptmin=20.0,
                 stop=None
                 ):
        
        self.data=data
        self.p=c
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
            sequence = cluster(pseudojets, R=self.R, p=self.p)
            jets = sequence.inclusive_jets(ptmin=self.ptmin)
            label = truth_label
            yield N, jets, label
            
            if N+1==self.stop: break
            
    def __exit__(self, exc_type, exc_value, traceback):
        pass



def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.  
    Licensed under Creative Commons Attribution-Share Alike 
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


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

def ep2ptepm_pid(rec):
    """ Convert (E, px, py, pz, pid) into (pT, eta, phi, mass, pid)
    Modified pyjet function that allows for an additional attribute ('pid') 
    to tag along...
    
    """
    E, px, py, pz , pid = rec.dtype.names[:5]
    vects = np.empty(rec.shape[0], dtype=DTYPE_PTEPM)
    vects=append_fields(vects, 'pid', data=np.zeros(len(rec)))
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
    vects['pid']=rec[pid]
    return vects
  
    
def draw_jet_tree(G,edge_weight='weight',cmap=plt.cm.viridis_r):
    pos = hierarchy_pos(G,1) 
    edge_color = nx.get_edge_attributes(G,edge_weight).values()
    node_size=[]
    nx.draw(G, pos=pos,node_size=5,node_color='k',edge_color=edge_color,width=2,edge_cmap=cmap)
    plt.show()

def draw_trees(G,cmap=plt.cm.viridis_r, jet_label=False):
    plt.figure(figsize=(4*len(G),4), constrained_layout=True)
    for n,g in enumerate(G):
        plt.subplot(int(str(1)+str(len(G))+str(n+1)))
        pos = hierarchy_pos(g,1) 
        edge_color = nx.get_edge_attributes(g,'weight').values()
        jl = nx.get_node_attributes(g,'jet_label')
        nx.draw(g,pos=pos,node_size=4,node_color='k',alpha=1.0,edge_color=edge_color,width=2,edge_cmap=cmap)
        if jet_label:
            nx.draw_networkx_labels(g, pos=pos,labels=jl,verticalalignment='bottom')

        
        
################################################          
# from Python module pyhepmc 1.0.3 

class HepMCReader(object):
    def __init__(self, filename):
        self._file = open(filename)
        self._currentline = None
        self._currentvtx = None
        self.version = None
        ## First non-empty line should be the version info
        while True:
            self._read_next_line()
            if self._currentline.startswith("HepMC::Version"):
                self.version = self._currentline.split()[1]
                break
        ## Read on until we see the START_EVENT_LISTING marker
        while True:
            self._read_next_line()
            if self._currentline == "HepMC::IO_GenEvent-START_EVENT_LISTING":
                break
        ## Read one more line to make the first E line current
        self._read_next_line()

    def _read_next_line(self):
        "Return the next line, stripped of the trailing newline"
        self._currentline = self._file.readline()
        if not self._currentline: # no newline means it's the end of the file
            return False
        if self._currentline.endswith("\n"):
            self._currentline = self._currentline[:-1] # strip the newline
        return True

    def next(self):
        "Return a new event graph"
        evt = Event()
        if not self._currentline or self._currentline == "HepMC::IO_GenEvent-END_EVENT_LISTING":
            return None
        assert self._currentline.startswith("E ")
        vals = self._currentline.split()
        evt.num = int(vals[1])
        evt.weights = [float(vals[-1])] 
        while not self._currentline.startswith("V "):
            self._read_next_line()
            vals = self._currentline.split()
            if vals[0] == "U":
                evt.units = vals[1:3]
            elif vals[0] == "C":
                evt.xsec = [float(x) for x in vals[1:3]]
        # Read the event content lines until an Event line is encountered
        while not self._currentline.startswith("E "):
            vals = self._currentline.split()
            if vals[0] == "P":
                bc = int(vals[1])
                try:
                    mom=[float(x) for x in vals[3:7]]
                    evt.particles[bc]=(int(vals[8]),int(vals[2]),float(vals[3]),float(vals[4]),float(vals[5]),float(vals[6]))
                except:
                    print(vals)
            elif vals[0] == "V":
                bc = int(vals[1])
                self._currentvtx = bc 
                v = Vertex(barcode=bc, pos=[float(x) for x in vals[3:7]], event=evt)
                evt.vertices[bc] = v
            elif not self._currentline or self._currentline == "HepMC::IO_GenEvent-END_EVENT_LISTING":
                break
            self._read_next_line()
        return evt

class Particle(object):
    def __init__(self, pid=0, mom=[0,0,0,0], barcode=0, event=None):
        self.evt = event
        self.barcode = barcode
        self.pid = pid
        self.status = None
        self.mom = list(mom)
        self.nvtx_start = None
        self.nvtx_end = None
        self.mass = None
    def vtx_start(self):
        return self.evt.vertices.get(self.nvtx_start) if self.evt else None
    def vtx_end(self):
        return self.evt.vertices.get(self.nvtx_end) if self.evt else None
    def parents(self):
        return self.vtx_start().parents() if self.vtx_start() else None
    def children(self):
        return self.vtx_end().children() if self.vtx_end() else None
    def __repr__(self):
        return "P" + str(self.barcode)

class Vertex(object):
    def __init__(self, pos=[0,0,0,0], barcode=0, event=None):
        self.evt = event
        self.pos = list(pos)
        self.barcode = barcode
    def parents(self):
        return [p for p in self.evt.particles.values() if p.nvtx_end == self.barcode]
    def children(self):
        return [p for p in self.evt.particles.values() if p.nvtx_start == self.barcode]
    def __repr__(self):
        return "V" + str(self.barcode)

class Event(object):
    def __init__(self):
        self.num = None
        self.weights = None
        self.units = [None, None]
        self.xsec = [None, None]
        self.particles = {}
        self.vertices = {}

    def __repr__(self):
        return "E%d. #p=%d #v=%d, xs=%1.2e+-%1.2e" % \
               (self.num, len(self.particles), len(self.vertices),
                self.xsec[0], self.xsec[1])

    
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

