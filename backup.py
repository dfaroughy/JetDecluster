

class get_particles(object):
    
# this is too slow.... need to optimize

    def __init__(self, 
                 data,
                 nevents=None,
                 dtype='PTEP',
                 sample_label=None
                 ):
        self.data=data
        self.nevents=nevents
        self.DTYPE_PTEP=False
        self.DTYPE_PTEPM=False
        self.HEPMC=False
        self.sample_label=sample_label
        
        if dtype=='PTEP': 
            self.DTYPE_PTEP=True
            self.norm=3
        elif dtype=='PTEPM': 
            self.DTYPE_PTEPM=True
            self.norm=4

        if isinstance(data,list): 
            if data[0].split('.')[-1]=='hepmc': self.HEPMC=True
            elif data.split('.')[-1]==('h5'or 'hdf5'): 
                self.data=pd.read_hdf(self.data)
        
        
    def __enter__(self):
        
        for i,hepmc in enumerate(self.data):
            line = HepMCReader(hepmc)
            N=0
            while True:
                N+=1
                evt = line.next()
                if not evt:
                    break 
                final_states=[evt.particles[p][1:] for p in evt.particles if evt.particles[p][0]==1] # extracts only final state particles
                pseudojets=np.zeros(len(final_states),dtype=DTYPE_EP) 
                for j,p in enumerate(final_states): 
                    pseudojets[j]['px']=p[1]
                    pseudojets[j]['py']=p[2]
                    pseudojets[j]['pz']=p[3]
                    pseudojets[j]['E' ]=p[4]
                yield N, ep2ptepm_pid(pseudojets)
                if N==self.nevents: break
       
        else:
            num_const=self.data.shape[1]/self.norm
            truth_flag=False
            if not (num_const)%1==0:
                truth_flag=True
            for N,event in self.data.iterrows():
                if truth_flag:
                    truth_label=event[self.data.shape[1]-1]
                pseudojets=np.zeros(int(num_const)*self.norm, dtype=DTYPE_PTEPM)
                cut=int(num_const)

                if self.DTYPE_PTEP:
                    for j in range(int(num_const)):
                        if event[j*self.norm]==0.0:
                            cut=j
                            break
                        pseudojets[j]['pT' ] = event[j*self.norm]
                        pseudojets[j]['eta'] = event[j*self.norm+1]
                        pseudojets[j]['phi'] = event[j*self.norm+2]

                elif self.DTYPE_PTEPM:
                    for j in range(int(num_const)):
                        if event[j*self.norm]==0.0:
                            cut=j
                            break
                        pseudojets[j]['pT' ] = event[j*self.norm]
                        pseudojets[j]['eta'] = event[j*self.norm+1]
                        pseudojets[j]['phi'] = event[j*self.norm+2]
                        pseudojets[j]['mass'] = event[j*self.norm+3]
                    
                pseudojets = np.sort(pseudojets[:cut], order='pT')
                yield N, pseudojets[:cut], truth_label
                if N+1==self.nevents: break
            
    def __exit__(self, exc_type, exc_value, traceback):
        pass
    
    
    
class get_jets(object):
    
# still too slow.... need to optimize

    def __init__(self, 
                 data,
                 R=1.0,
                 ptmin=0.0,
                 p=-1,
                 nevents=None,
                 dtype='PTEP',
                 sample_labels=None
                 ):
        self.data=data
        self.nevents=nevents
        self.DTYPE_PTEP=False
        self.DTYPE_PTEPM=False
        self.HEPMC=False
        self.sample_labels=sample_labels
        
        self.R=R
        self.ptmin=ptmin
        self.p=p
        
        if dtype=='PTEP': 
            self.DTYPE_PTEP=True
            self.norm=3
        elif dtype=='PTEPM': 
            self.DTYPE_PTEPM=True
            self.norm=4

        if isinstance(data,list): 
            if data[0].split('.')[-1]=='hepmc': self.HEPMC=True
            elif data.split('.')[-1]==('h5'or 'hdf5'): 
                self.data=pd.read_hdf(self.data)
        

    def __enter__(self):
        
        if self.HEPMC:
            for i,hepmc in enumerate(self.data):
                line = HepMCReader(hepmc)
                N=0
                while True:
                    N+=1
                    evt = line.next()
                    if not evt:
                        break 
                    final_states=[evt.particles[p][1:] for p in evt.particles if evt.particles[p][0]==1]
                    pseudojets=np.zeros(len(final_states),dtype=DTYPE_EP) 
                    for j,p in enumerate(final_states): 
                        pseudojets[j]['px']=p[1]
                        pseudojets[j]['py']=p[2]
                        pseudojets[j]['pz']=p[3]
                        pseudojets[j]['E' ]=p[4]
                    sequence = cluster(pseudojets, R=self.R, p=self.p, ep=True)
                    jets = sequence.inclusive_jets(ptmin=self.ptmin)
                    yield N, jets, self.sample_labels[i]
                    if N==self.nevents[i]: break
       
        else:
            num_const=self.data.shape[1]/self.norm
            truth_flag=False
            if not (num_const)%1==0:
                truth_flag=True
            for N,event in self.data.iterrows():
                if truth_flag:
                    truth_label=event[self.data.shape[1]-1]
                pseudojets=np.zeros(int(num_const)*self.norm, dtype=DTYPE_PTEPM)
                cut=int(num_const)

                if self.DTYPE_PTEP:
                    for j in range(int(num_const)):
                        if event[j*self.norm]==0.0:
                            cut=j
                            break
                        pseudojets[j]['pT' ] = event[j*self.norm]
                        pseudojets[j]['eta'] = event[j*self.norm+1]
                        pseudojets[j]['phi'] = event[j*self.norm+2]

                elif self.DTYPE_PTEPM:
                    for j in range(int(num_const)):
                        if event[j*self.norm]==0.0:
                            cut=j
                            break
                        pseudojets[j]['pT' ] = event[j*self.norm]
                        pseudojets[j]['eta'] = event[j*self.norm+1]
                        pseudojets[j]['phi'] = event[j*self.norm+2]
                        pseudojets[j]['mass'] = event[j*self.norm+3]

                pseudojets = np.sort(pseudojets[:cut], order='pT')
                sequence = cluster(pseudojets, R=self.R, p=self.p)
                jets = sequence.inclusive_jets(ptmin=self.ptmin)
                yield N, jets, truth_label
                if N+1==self.nevents: break
            
    def __exit__(self, exc_type, exc_value, traceback):
        pass


    
def decluster(jet,jet_label=None,node_feature=None,node_label=None,R=1.0, directed=False):
        
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

        for i,j in enumerate(jet): 
            pseudojets[i]['pT']=j.pt
            pseudojets[i]['eta']=j.eta
            pseudojets[i]['phi']=j.phi
            pseudojets[i]['mass']=j.mass
            
        clustered_jet = cluster(pseudojets, R=R, p=0)
        
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

        return Jet_Tree(jet_tree)
    else:
        return None
    
    
class Jet_Tree(object):
    
    def __init__(self,tree):
        
        self.graph=tree
        self.data=tree.nodes(data=True)
        self.root=tree.nodes()[1]
        self.jet=tree.nodes()[1]['pseudojet']
        self.jet_label=tree.nodes()[1]['jet_label']
        self.subjets=nx.get_node_attributes(tree,'pseudojet')
        self.primary_branch=nx.get_node_attributes(tree,'primary_branch')
        self.node_features=nx.get_node_attributes(tree,'feature')
        self.node_labels=nx.get_node_attributes(tree,'label')
        self.node_idx=tree.nodes()
        self.leaf_idx=[n for n in tree.nodes() if not self.node_features[n]]
        self.branch_idx=[n for n in tree.nodes() if self.node_features[n]]

        self.copy=tree.copy()
    
    def draw_tree(self):
        G=[self.graph]
        l=self.jet_label  
        plt.figure(figsize=(4*len(G),4), constrained_layout=True)
        for n,g in enumerate(G):
            plt.subplot(int(str(1)+str(len(G))+str(n+1)))
            pos = hierarchy_pos(g,1) 
            jl = nx.get_node_attributes(g,'jet_label')
            edge_color = nx.get_edge_attributes(g,'weight').values()
            nx.draw(g,pos=pos,node_size=4,node_color='k',alpha=1.0,edge_color=edge_color,width=2,edge_cmap=plt.cm.viridis_r)
            nx.draw_networkx_labels(g, pos=pos,labels=jl,verticalalignment='bottom')







    
def decluster(jet,jet_label=None,node_feature=None,node_label=None, R=1.0, ptmin=20.0, p=0,directed=False):
        
    '''
    Reclusters constituents of a jets then extracts then de-clusteres to recover 
    the recombaination binary tree of each jet. Features are computed and extracted
    at each declustering. 
    
    ouput: list of binary trees as networkx graph objects with node attributes.
    
    '''
    
    def none(*kargs):
        return None
    
    seq=cluster(jet.constituents_array(), R=R, p=p)

    jet_tree=nx.Graph()
    jet_tree.add_node(1,jet_label=jet_label)
    subjets=[]; idx={}; N=0
                
    for t in range(1,seq.n_exclusive_jets(0)+1):
        
        for subjet in seq.exclusive_jets(t):
            if subjet not in subjets:
                N+=1
                subjets.append(subjet)
                idx[subjet.e]=N
        if t>=2:
            j0=[x for x in seq.exclusive_jets(t-1) if x not in seq.exclusive_jets(t)]
            j12=[x for x in seq.exclusive_jets(t) if x not in seq.exclusive_jets(t-1)]
            
            if j0 and j12:
                mother,hard,soft=j0[0],j12[0],j12[1]
                m,h,s=idx[mother.e],idx[hard.e],idx[soft.e]

                if not node_feature:
                    node_feature=none
                if not node_label:
                    node_label=none

                if hard.pt>

                jet_tree.add_node(m,pseudojet=mother,feature=node_feature(mother,hard,soft),label=node_label(mother))
                jet_tree.add_node(s,pseudojet=soft,feature=None,label=node_label(soft))
                jet_tree.add_node(h,pseudojet=hard,feature=None,label=node_label(hard))
                jet_tree.add_edge(m,s,weight=soft.pt/jet.pt)
                jet_tree.add_edge(m,h,weight=hard.pt/jet.pt)
            
    return Jet_Tree(jet_tree)



    







t=timer()
dataset_hepmc=['data/triboson_pythia8_events.hepmc',
               'data/mj2700-3300_tag_1_pythia8_events.hepmc']

with get_jets(data=dataset_hepmc,R=R,ptmin=ptmin,p=-1,nevents=[2000,20000], sample_labels=[1,0]) as event:
    
    for N, jets, truth in event:         
        if (len(jets)>1 and 2700 < inv_M(jets[0],jets[1]) < 3300): 
            for jet in jets:

                tree=decluster(jet, node_feature=lund_triangle,R=R,jet_label=truth)

                if tree:

                    subjets=nx.get_node_attributes(tree,'pseudojet')  
                    primary=nx.get_node_attributes(tree,'primary_branch') 
                    features=nx.get_node_attributes(tree,'feature')  

                    for idx, lund in features.items():
                        if (primary[idx] and lund):

                            Delta=lund[0]
                            kt=lund[1]
                            
                            if truth:
                                lund_0['signal'].append(np.log(R/Delta))
                                lund_1['signal'].append(np.log(kt))
                            else:
                                lund_0['background'].append(np.log(R/Delta))
                                lund_1['background'].append(np.log(kt))
                                
print('...'+ elapsed_time(t))

xedges = np.arange(0.,8,0.1)
yedges = np.arange(-2,8,0.1)

fig = plt.figure(figsize=(13,6))

ax1 = fig.add_subplot(1,2,1)
plt.hist2d(lund_0['background'],lund_1['background'],bins=(xedges, yedges),cmap='gist_heat_r')
plt.title('background',  loc='center')
ax1.set_xlabel('$\log(R/\Delta)$')
ax1.set_ylabel('$\log(k_t)$')

ax2 = fig.add_subplot(1,2,2)
plt.hist2d(lund_0['signal'],lund_1['signal'],bins=(xedges, yedges),cmap='gist_heat_r')
plt.title('signal',  loc='center')
ax2.set_xlabel('$\log(R/\Delta)$')
ax2.set_ylabel('$\log(k_t)$')


plt.tight_layout()
