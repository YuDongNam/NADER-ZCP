import os
import json
import re
import numpy as np
import copy
from matplotlib.colors import LinearSegmentedColormap, rgb2hex
from prettytable import PrettyTable
import graphviz
import networkx as nx

from .utils import *

class BlockGraphManagement:

    def __init__(self,
            name,
            block_txt_dir='',
            inspiration_path='',
            block_anno_path='',
            log_dir='') -> None:
        os.makedirs(log_dir,exist_ok=True)
        self.name = name
        self.block_dir = block_txt_dir
        self.insp_path = inspiration_path
        self.log_dir = log_dir
        self.block_anno_path = block_anno_path
        self.anno_path = os.path.join(log_dir,f'{name}.json')

    def __getitem__(self,i):
        self.load_annos()
        return list(self.annos.keys())[i]
    
    def __contains__(self,name):
        self.load_annos()
        return name in self.annos
    
    def __len__(self):
        self.load_annos()
        return len(self.annos)

    def get_acc(self,name):
        self.load_annos()
        if name in self.annos:
            return self.anno[name]['acc']
        else:
            return None

    def load_annos(self,anno_path=None):
        if not anno_path:
            anno_path = self.anno_path
        if os.path.isfile(anno_path):
            with open(anno_path,'r') as f:
                ds = json.load(f)
        else:
            ds = {}
        self.annos = ds
        return ds

    def load_blocks(self):
        self.load_annos()
        blocks = list(self.annos.keys())
        return blocks

    def get_inspiration(self,id):
        if not id:
            return None
        with open(self.insp_path,'r') as f:
            ds = json.load(f)
            for l in ds:
                if str(l['id'])==str(id):
                    return l['inspiration']
        return None

    def append_inspiration(self,inspiration):
        if not os.path.isfile(self.insp_path):
            ds = []
        else:
            with open(self.insp_path,'r') as f:
                ds = json.load(f)
        id = len(ds)+1
        ds.append({'id':id,'inspiration':inspiration})
        with open(self.insp_path,'w') as f:
            json.dump(ds,f,indent='\t')
        return id

    def get_block_txt(self,block_names,type='all'):
        if isinstance(block_names,list):
            annos = []
            for block_name in block_names:
                path = os.path.join(self.block_dir,f'{block_name}.txt')
                assert os.path.isfile(path)
                anno = {}
                with open(path,'r') as f:
                    txt = f.read()
                    s = re.findall('(##.*?##(.(?!##))*)',txt,re.DOTALL)
                    for name,x in zip(['base','stem','downsample'],s):
                        anno[name] = x[0].strip()
                    annos.append(anno)
        else:
            annos = {}
            path = os.path.join(self.block_dir,f'{block_names}.txt')
            assert os.path.isfile(path),path
            with open(path,'r') as f:
                txt = f.read()
                s = re.findall('(##.*?##(.(?!##))*)',txt,re.DOTALL)
                for name,x in zip(['base','stem','downsample'],s):
                    annos[name] = x[0].strip()
        return annos

    def load_dag(self,annos=None,remove_prefix_tag=''):
        if not annos:
            annos = self.load_annos()
        nodes = {}
        edges = []
        for key,val in annos.items():
            key = key.replace(remove_prefix_tag,'')
            nodes[key] = val
            if val['from_block_name']:
                edges.append([val['from_block_name'].replace(remove_prefix_tag,''),key.replace(remove_prefix_tag,'')])
        dag = {
            'nodes':nodes,
            'edges':edges
        }
        self.dag = dag
        return dag

    def draw_graph(self,annos=None,draw_edge_label=True,remove_prefix_tag='',name=None):
        if not name:
            name = self.name
        dag = self.load_dag(annos=annos,remove_prefix_tag=remove_prefix_tag)
        nodes = dag['nodes']
        edges = dag['edges']
        max_iter=1
        # print(nodes)
        for node,val in nodes.items():
            max_iter=max(max_iter,int(val['iter']))
        max_iter+=1
        cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#CCF2FF" , "#003366"], N=max_iter)
        colors = [rgb2hex(cmap(i/(max_iter-1))) for i in range(max_iter)]
        # draw
        dot = graphviz.Digraph(name, format='png', graph_attr={'rankdir': 'TB'})
        for node,val in nodes.items():
            dot.node(node,label=f"{node.strip('_')}\\nAcc: {val['acc']}",shape='Mrecord',style='filled',fillcolor=colors[int(val['iter'])])
        for edge in edges:
            if draw_edge_label:
                dot.edge(edge[0],edge[1],label='p'+str(nodes[edge[1]]['inspiration_id']))
            else:
                dot.edge(edge[0],edge[1])
        dot.render(name, directory=self.log_dir, format='png', view=False,cleanup=True)

    def get_graph_txt(self):
        """
        #nodes#
        <model_name><block>{block_txt}</block><acc>{acc}</acc></model_name>
        #inspiration#
        {inspiration_name}:{inspiration_txt}
        #edges#
        model_1--inspiration_1-->model_2
        """
        list_nodes = []
        list_inspirations = []
        list_edges = []
        dag = self.load_dag()
        g = nx.DiGraph()
        g.add_edges_from(dag['edges'])
        for node in list(nx.topological_sort(g)):
            node_val = dag['nodes'][node]
            list_nodes.append(f"<{node}><block>{node_val['blocks'][0]}</block><acc>{node_val['acc']}</acc></{node}>")
            if node_val['inspiration_id'] and node_val['from_block_name']:
                list_inspirations.append(f"p{node_val['inspiration_id']}: {node_val['inspiration']}")
                list_edges.append(f"{node_val['from_block_name']}--p{node_val['inspiration_id']}-->{node}")
        nodes = '\n'.join(list_nodes)
        inspirations = '\n'.join(list_inspirations)
        edges = '\n'.join(list_edges)
        txt = f"#nodes#\n{nodes}\n#inspirations#\n{inspirations}\n#edges#\n{edges}"
        return txt
    
    
    def print_table(self):
        with open(self.anno_path,'r') as f:
            ds = json.load(f)
        table = PrettyTable()
        names = ['block_name','params','accuracy']
        table.field_names = names
        table.align = 'l'
        for model_name in ds:
            table.add_row([model_name,ds[model_name]['params'],ds[model_name]['test']['accuracy']])
        print(table)

    def update_train_result(self,train_log_dir,tag_prefix='',filename='test_acc',anno_name=None,flush=False,iter=None):
        if not anno_name:
            anno_name= self.name
        anno_path = os.path.join(self.log_dir,f'{anno_name}.json')
        annos = self.load_annos(anno_path)
        block2anno = {}
        with open(self.block_anno_path,'r') as f:
            for anno in f.readlines():
                anno = json.loads(anno)
                if not iter or (iter and anno['iter']==iter):
                    block2anno[anno['block_name']] = anno
        files = os.listdir(train_log_dir)
        block_list = []
        best_acc = -1
        for block,anno in block2anno.items():
            # Removed verbose per-block print to reduce I/O
            if block in files and ((not flush and block not in annos) or flush):
                res_path = os.path.join(train_log_dir,block,'1',f'{filename}.txt')
                
                # Fallback to val_acc.txt if test_acc.txt is missing
                if not os.path.isfile(res_path) and filename == 'test_acc':
                     res_path = os.path.join(train_log_dir,block,'1','val_acc.txt')
                
                if os.path.isfile(res_path):
                    try:
                        res = np.genfromtxt(res_path,delimiter=',',skip_header=1)
                    except Exception:
                        continue
                    if len(res.shape)==1:
                        res = np.array([res])
                    # print(f'{block} is training {len(res)}/{50}.')
                    maxi = max(res[:,1])
                    acc = round(maxi,2)
                    block_list.append(block)
                    txt_path = os.path.join(self.block_dir,f'{block}.txt')
                    blocks = load_block(txt_path)
                    assert len(blocks)==3
                    annos[block] = {
                        'iter':anno.get('iter', 0), # Default to 0 if 'iter' is missing
                        "from_block_name": anno.get('raw_block_name', 'unknown'),
                        "inspiration_id": anno.get('inspiration_id'),
                        "inspiration":self.get_inspiration(anno.get('inspiration_id')) if 'inspiration' not in anno else anno.get('inspiration'),
                        "acc":acc,
                        "blocks":blocks
                    }
                    best_acc = max(best_acc,acc)
                else:
                    # print(f'{block} not be trained.')
                    pass
            elif block not in files:
                # print(f'{block} not be trained.')
                pass
        save_json(annos,anno_path)
        self.draw_graph(annos=annos,remove_prefix_tag=tag_prefix,name=anno_name)
        return block_list,best_acc


    def search(self,mode='dfs'):
        annos = self.load_annos()
        if mode=='dfs':
            edges1 = {n:[] for n in annos.keys()}
            edges2 = {n:[] for n in annos.keys()}
            root = None
            for key,val in annos.items():
                if val['iter']==0:
                    root = key
                else:
                    parent = val['from_block_name']
                    if parent in edges1:
                        edges1[parent].append(key)
                        if val['acc'] > annos[parent]['acc']:
                           edges2[parent].append(key)
                    else:
                        pass # Ignore orphan blocks or unknown parents to prevent crash
            
            # Fallback if no root found
            if root is None and annos:
                # Find the block with the minimum iteration
                root = min(annos, key=lambda k: annos[k]['iter'])

            for node in edges2:
                edges2[node] = sorted(edges2[node],key=lambda x:annos[x]['acc'],reverse=True)
            def dfs(root):
                if len(edges2[root])>0:
                    for n in edges2[root]:
                        s = dfs(n)
                        if s:
                            return s
                else:
                    if len(edges1[root])!=0:
                        return None
                    else:
                        return root
                return None
            n = dfs(root)
            if not n:
                n = root
            return [n,annos[n]]
        else:
            raise NotImplementedError
        return None

    @staticmethod
    def txt2json(txt):
        s = re.findall('##[^#]*##[^#]*',txt,re.DOTALL)
        res = {
            'base':s[0],
            'stem':s[1],
            'downsample':s[2]
        }
        return res
        

                    
                
