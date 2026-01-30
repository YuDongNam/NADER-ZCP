from datetime import datetime
import numpy as np
import torch
import random
import json
import os
import pdb
import warnings
import asyncio
import logging
import shutil
import copy

from agents.agent_inspiration_retriever import InspirationSampler,InspirationSamplerReflection
from agents.agent_proposer import ResearchProposer,ResearchProposeInspiration,ResearchProposerRandom
from agents.agent_expe_retriever_res import ResearchExperienceRetriever,ResearchExperienceRetrieverRandom
from tools.block_management import BlockGraphManagement
from tools.utils import *


warnings.filterwarnings("ignore")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class TeamResearch:
  
    def __init__(self,
                 base_block=None,
                 insp_table_name='inspirations_040611',
                 db_insp_dir='data/ChromDB/inspirations',
                 retriever_mode='random',
                 candiate_inspiration_num=10,
                 inspirations_path = 'data/inspirations/inspirations_040611.json',
                 use_experience=None,
                 experience_mode='VDB',
                 mode=None,
                 proposer_mode='llm',
                 log_dir = 'logs',
                 tag_prefix='solution1',
                 block_txt_dir=None,
                 block_anno_path=None,
                 train_log_dir=None,
                 logger=None,
                 use_logger=False) -> None:
        self.base_block = base_block
        self.retriever_mode = retriever_mode
        self.candiate_inspiration_num = candiate_inspiration_num
        self.mode = mode
        self.proposer_mode = proposer_mode
        self.use_experience = use_experience
        self.tag_prefix = tag_prefix
        self.train_log_dir = train_log_dir

        # logs
        self.log_dir = log_dir
        if not block_txt_dir:
            block_txt_dir = os.path.join(log_dir,'block_txt')
            os.makedirs(block_txt_dir,exist_ok=True)
        self.block_txt_dir = block_txt_dir
        self.anno_path = os.path.join(log_dir,'anno_research.jsonl')
        
        os.makedirs(self.train_log_dir,exist_ok=True)
        

        # logger
        if not logger and use_logger:
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(tag_prefix)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            file_handler = logging.FileHandler(os.path.join(log_dir,'log_research.txt'))
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
        self.logger = logger

        # block manage
        self.mog_graph_manage = BlockGraphManagement('mog',block_txt_dir,inspirations_path,block_anno_path,log_dir)

        # agents
        llm_log_dir = os.path.join(log_dir,'gpt_response')
        os.makedirs(llm_log_dir,exist_ok=True)
        if retriever_mode=='random':
            self.agent_insp_retriever = InspirationSampler(inspirations_path,llm_log_dir)
        elif retriever_mode=='reflection':
            db_insp_dir = os.path.join(log_dir,'ChromDB','inspirations')
            self.agent_insp_retriever = InspirationSamplerReflection(inspirations_path=inspirations_path,table_name=insp_table_name,db_dir=db_insp_dir,log_dir=log_dir,llm_log_dir=llm_log_dir)
        else:
            raise NotImplementedError
        if self.use_experience:
            if experience_mode=='VDB':
                self.agent_expe_retriever = ResearchExperienceRetriever(self.use_experience)
            elif experience_mode=='random':
                self.agent_expe_retriever = ResearchExperienceRetrieverRandom()
            else:
                raise NotImplementedError
        if proposer_mode == 'llm':
            self.agent_proposer = ResearchProposer(log_dir=llm_log_dir)
        elif proposer_mode == 'random':
            self.agent_proposer = ResearchProposerRandom(log_dir=llm_log_dir)

    def append_anno(self,anno,path):
        with open(path,'a') as f:
            f.write(json.dumps(anno)+'\n') 

    def set_used(self,insp_id):
        self.agent_insp_retriever.set_used(insp_id)

    def __call__(self,iter=None,num=5):
        res = {'iter':iter}
        block_list = self.mog_graph_manage.load_blocks()
        if len(block_list)==0:
            block_list = [self.base_block]
        self.mog_graph_manage.update_train_result(self.train_log_dir,tag_prefix=self.tag_prefix)
        insps = self.agent_insp_retriever(num=self.candiate_inspiration_num)
        
        # ★ 항상 self.base_block을 parent로 사용 (Resume 호환)
        # 이전 iteration에서 best model이 base_block으로 설정되었으므로, 
        # annos에서 greedy 선택 대신 base_block을 직접 사용
        if self.base_block in block_list or len(block_list) <= 1:
            # base_block이 block_list에 있거나, 첫 iteration인 경우
            block_txt = self.mog_graph_manage.get_block_txt(self.base_block)['base']
            block_name = self.base_block
            mode='one'
        elif self.mode=='greedy':
            # base_block이 block_list에 없는 경우 (비정상 상황)
            # 안전을 위해 annos에서 선택하되, 로그 출력
            its = sorted(self.mog_graph_manage.annos.items(),key=lambda x:x[1]['acc'],reverse=True)
            block_txt = its[0][1]['blocks'][0]
            block_name = its[0][0]
            print(f"[Warning] base_block '{self.base_block}' not in block_list. Using greedy selection: {block_name}")
            mode='one'
        elif self.mode=='dfs-one':
            it = self.mog_graph_manage.search(mode='dfs')
            block_txt = it[1]['blocks'][0]
            block_name = it[0]
            mode='one'
        else:
            block_txt = self.mog_graph_manage.get_graph_txt()
            mode='mog'
            block_name = self.base_block  # fallback
        ps = self.agent_proposer(blocks=block_txt,inspirations=insps,mode=mode)
        for key in ['prompt_tokens','completion_tokens']:
            res[key] = ps[key]
        ps_new = []
        for p in ps['proposals']:
            if not p['block_name']:
                if block_name:
                    p['block_name'] = block_name
                else:
                    p['block_name'] = block_list[0]
            if p['block_name'] in block_list:
                p['inspiration'] = insps[p['inspiration_id']]
                p['block'] = self.mog_graph_manage.get_block_txt(p['block_name'])['base']
                if self.use_experience:
                    exps = self.agent_expe_retriever(p['inspiration'])
                    if len(exps)>0:
                        p['experiences'] = exps
                ps_new.append(p)
        res['proposals'] = ps_new
        self.append_anno(res,self.anno_path)
        return res

class TeamResearchNoReader:
  
    def __init__(self,
                 base_block=None,
                 use_experience=None,
                 mode=None,
                 log_dir = 'logs',
                 tag_prefix='solution1',
                 block_txt_dir=None,
                 block_anno_path=None,
                 train_log_dir=None,
                 logger=None,
                 use_logger=False) -> None:
        self.base_block = base_block
        self.mode = mode
        self.use_experience = use_experience
        self.tag_prefix = tag_prefix
        self.train_log_dir = train_log_dir

        # logs
        self.log_dir = log_dir
        if not block_txt_dir:
            block_txt_dir = os.path.join(log_dir,'block_txt')
            os.makedirs(block_txt_dir,exist_ok=True)
        self.block_txt_dir = block_txt_dir
        self.anno_path = os.path.join(log_dir,'anno_research.jsonl')
        
        os.makedirs(self.train_log_dir,exist_ok=True)
        

        # logger
        if not logger and use_logger:
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(tag_prefix)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            file_handler = logging.FileHandler(os.path.join(log_dir,'log_research.txt'))
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
        self.logger = logger

        # block manage
        inspiration_path = os.path.join(log_dir,'inspirations.json')
        self.mog_graph_manage = BlockGraphManagement('mog',block_txt_dir,inspiration_path,block_anno_path,log_dir)

        # agents
        llm_log_dir = os.path.join(log_dir,'gpt_response')
        os.makedirs(llm_log_dir,exist_ok=True)
        if self.use_experience:
            self.agent_expe_retriever = ResearchExperienceRetriever(self.use_experience)
        self.agent_proposer = ResearchProposeInspiration(log_dir=llm_log_dir)

    def append_anno(self,anno,path):
        with open(path,'a') as f:
            f.write(json.dumps(anno)+'\n') 

    def set_used(self,insp_id):
        return None

    def __call__(self,iter=None,num=5):
        res = {'iter':iter}
        block_list = self.mog_graph_manage.load_blocks()
        if len(block_list)==0:
            block_list = [self.base_block]
        self.mog_graph_manage.update_train_result(self.train_log_dir,tag_prefix=self.tag_prefix)
        if len(block_list)==0 or len(block_list)==1:
            if len(block_list)==1:
                assert block_list[0] == self.base_block,block_list
            block_txt = self.mog_graph_manage.get_block_txt([self.base_block])[0]['base']
            mode='one'
            block_name = None
        elif self.mode=='greedy':
            its = sorted(self.mog_graph_manage.annos.items(),key=lambda x:x[1]['acc'],reverse=True)
            block_txt = its[0][1]['blocks'][0]
            block_name = its[0][0]
            mode='one'
        elif self.mode=='dfs-one':
            it = self.mog_graph_manage.search(mode='dfs')
            block_txt = it[1]['blocks'][0]
            block_name = it[0]
            mode='one'
        else:
            raise NotImplementedError
        ps = self.agent_proposer(blocks=block_txt,mode=mode)
        for key in ['prompt_tokens','completion_tokens']:
            res[key] = ps[key]
        ps_new = []
        for p in ps['proposals']:
            if not p['block_name']:
                if block_name:
                    p['block_name'] = block_name
                else:
                    p['block_name'] = block_list[0]
            if p['block_name'] in block_list:
                p['inspiration_id'] = self.mog_graph_manage.append_inspiration(p['inspiration'])
                p['block'] = self.mog_graph_manage.get_block_txt(p['block_name'])['base']
                if self.use_experience:
                    exps = self.agent_expe_retriever(p['inspiration'])
                    if len(exps)>0:
                        p['experiences'] = exps
                ps_new.append(p)
        res['proposals'] = ps_new
        self.append_anno(res,self.anno_path)
        return res

class TeamResearchHandCraft:

    inspirations = [
        'Add convolutional layer.',
        'Add skip connection.',
        'Add dense layer.',
        'Add more kernel.',
        'Add more neurons.',
        'Reduce convolutional layer.',
        'Reduce skip connection.',
        'Reduce dense layer.',
        'Reduce number of kernel.',
        'Reduce neurons.'
    ]
  
    def __init__(self,
                 base_block=None,
                 use_experience=None,
                 mode=None,
                 log_dir = 'logs',
                 tag_prefix='solution1',
                 block_txt_dir=None,
                 block_anno_path=None,
                 train_log_dir=None,
                 logger=None,
                 use_logger=False) -> None:
        self.base_block = base_block
        self.mode = mode
        self.use_experience = use_experience
        self.tag_prefix = tag_prefix
        self.train_log_dir = train_log_dir

        # logs
        self.log_dir = log_dir
        if not block_txt_dir:
            block_txt_dir = os.path.join(log_dir,'block_txt')
            os.makedirs(block_txt_dir,exist_ok=True)
        self.block_txt_dir = block_txt_dir
        self.anno_path = os.path.join(log_dir,'anno_research.jsonl')
        
        os.makedirs(self.train_log_dir,exist_ok=True)
        
        # logger
        if not logger and use_logger:
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(tag_prefix)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            file_handler = logging.FileHandler(os.path.join(log_dir,'log_research.txt'))
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
        self.logger = logger

        # block manage
        inspiration_path = os.path.join(log_dir,'inspirations.json')
        self.mog_graph_manage = BlockGraphManagement('mog',block_txt_dir,inspiration_path,block_anno_path,log_dir)

        # agents
        llm_log_dir = os.path.join(log_dir,'gpt_response')
        os.makedirs(llm_log_dir,exist_ok=True)
        if self.use_experience:
            self.agent_expe_retriever = ResearchExperienceRetriever(self.use_experience)

    def append_anno(self,anno,path):
        with open(path,'a') as f:
            f.write(json.dumps(anno)+'\n') 

    def set_used(self,insp_id):
        return None

    def __call__(self,iter=None,num=5):
        res = {'iter':iter}
        block_list = self.mog_graph_manage.load_blocks()
        if len(block_list)==0:
            block_list = [self.base_block]
        self.mog_graph_manage.update_train_result(self.train_log_dir,tag_prefix=self.tag_prefix)
        if len(block_list)==0 or len(block_list)==1:
            if len(block_list)==1:
                assert block_list[0] == self.base_block,block_list
            block_txt = self.mog_graph_manage.get_block_txt([self.base_block])[0]['base']
            mode='one'
            block_name = block_list[0]
        elif self.mode=='greedy':
            its = sorted(self.mog_graph_manage.annos.items(),key=lambda x:x[1]['acc'],reverse=True)
            block_txt = its[0][1]['blocks'][0]
            block_name = its[0][0]
            mode='one'
        elif self.mode=='dfs-one':
            it = self.mog_graph_manage.search(mode='dfs')
            block_txt = it[1]['blocks'][0]
            block_name = it[0]
            mode='one'
        else:
            raise NotImplementedError
        ins = np.random.choice(range(len(self.inspirations)),10)
        for key in ['prompt_tokens','completion_tokens']:
            res[key] = 0
        ps_new = []
        for inspiration_id in ins:
            p={}
            p['inspiration']=self.inspirations[inspiration_id]
            p['inspiration_id'] = self.mog_graph_manage.append_inspiration(p['inspiration'])
            p['block_name'] = block_name
            p['block'] = self.mog_graph_manage.get_block_txt(p['block_name'])['base']
            if self.use_experience:
                exps = self.agent_expe_retriever(p['inspiration'])
                if len(exps)>0:
                    p['experiences'] = exps
            ps_new.append(p)
        res['proposals'] = ps_new
        self.append_anno(res,self.anno_path)
        return res