from datetime import datetime
import numpy as np
import torch
import random
import json
import os
import pdb
import warnings
import logging
import shutil
import copy
import re


from ModelFactory.block_gen import BlockGen
from ModelFactory.model_gen import ModelGen
from agents import create_agent
from agents.proposal import ProposalSampler
from agents.gpt_generate_block_modify import GPTGenerateBlockModify
from agents.gpt_generate_block_stem import GPTGenerateBlockStemDownsample
from tools.block_management import BlockGraphManagement
from tools.utils import *



warnings.filterwarnings("ignore")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class TeamDevelop:
    """
    base->
    """

    def __init__(self,
                 team_name='try-nfb',
                 model_name='gpt-5-nano',
                 dataset='imagenet-1k',
                 use_experience=None,
                 experience_mode='VDB',
                 database_dir='database',
                 log_dir = 'logs',
                 code_dir='ModelFactory',
                 max_try=5,
                 tag_prefix='trail1',
                 block_txt_dir=None,
                 logger=None,
                 cell_mode='nas-bench',
                 layers_num=20) -> None:
        self.team_name = team_name
        self.database_dir = database_dir
        self.max_try = max_try
        self.tag_prefix = tag_prefix
        self.use_experience = use_experience
        self.cell_mode = cell_mode

        if not code_dir:
            code_dir = os.path.join(log_dir,'codes')
        self.code_dir = code_dir

        # database
        self.database_block_txt_dir = os.path.join(database_dir,'blocks','txts')

        # logs
        # log_dir = os.path.join(log_dir,tag_prefix,datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
        # if tag_prefix:
        #     log_dir = os.path.join(log_dir,tag_prefix)
        self.log_dir = log_dir
        self.anno_path = os.path.join(log_dir,'anno_develop.jsonl')
        if not block_txt_dir:
            block_txt_dir = os.path.join(log_dir,'block_txt')
            os.makedirs(block_txt_dir,exist_ok=True)
        self.block_txt_dir = block_txt_dir
        

        # logger
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger('DevelopTeam')
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            file_handler = logging.FileHandler(os.path.join(log_dir,'log_develop.txt'))
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            # logger.addHandler(console_handler)
            logger.addHandler(file_handler)
            logger.propagate = False
        self.logger = logger


        # agents
        llm_log_dir = os.path.join(log_dir,'gpt_response')
        os.makedirs(llm_log_dir,exist_ok=True)
        self.agent_modify = GPTGenerateBlockModify(model_name=model_name,use_experience=use_experience,experience_mode=experience_mode,log_dir=llm_log_dir,mode=cell_mode)
        self.agent_generate_stem_downample = GPTGenerateBlockStemDownsample(model_name=model_name,block_txts_example_dir=self.database_block_txt_dir,use_experience=use_experience,experience_mode=experience_mode,log_dir=llm_log_dir,dataset=dataset,mode=cell_mode)
        
        # tools
        blocks_dir = os.path.join(code_dir,'blocks')
        models_dir = os.path.join(code_dir,'models')
        self.blocks_dir = blocks_dir
        self.models_dir = models_dir
        if dataset=='imagenet-1k':
            stem_down_scale=4
        elif 'cifar' in dataset.lower() or 'imagenet16-120' in dataset.lower():
            stem_down_scale=1
        else:
            raise NotImplementedError
        self.block_gen = BlockGen(blocks_dir,stem_down_scale=stem_down_scale,mode=cell_mode)
        self.model_gen = ModelGen(blocks_dir,models_dir,dataset=dataset,mode=cell_mode,layers_num=layers_num)
    

    def append_anno(self,anno):
        with open(self.anno_path,'a') as f:
            f.write(json.dumps(anno)+'\n') 

    def __call__(self,inspiration=None,block=None,**kwargs):
        if self.team_name.startswith('try-nfb'):
            return self.chain1(inspiration=inspiration,block=block,**kwargs)
        elif self.team_name.startswith('try-fb'):
            return self.chain2(inspiration=inspiration,block=block,**kwargs)
        else:
            raise NotImplementedError
        return None


    def chain1(self,inspiration=None,inspiration_id=None,block=None,block_name=None,experiences=None):
        """
        retry
        """
        self.logger.info(f"")
        anno = {
            'inspiration_id':inspiration_id,
            'raw_block':block_name,
            'tag_prefix':self.tag_prefix,
            'status':False,
            'existed':False,
            'try':0,
            'fail':False,
            'prompt_tokens':0,
            'completion_tokens':0,
            'base_block':None,
            'stem_block':None,
            'downsample_block':None
        }
        iter = 0
        while iter<self.max_try:
            iter+=1
            anno['try'] = iter
            res = self.agent_modify.run(inspiration,block,experiences)
            new_block = res['list'][0]
            anno['prompt_tokens']+=res['prompt_tokens']
            anno['completion_tokens']+=res['completion_tokens']
            check = self.block_gen.base_block.check(new_block,with_isomorphic=True)
            anno['base_block'] = new_block
            if isinstance(check,dict):
                self.logger.info(f"\tGenerate base_block try {iter}/{self.max_try}, error:{check}")
                ann = copy.deepcopy(anno)
                ann['status'] = False
                ann['fail'] = {
                    'type':'base',
                    'error':check['error']
                }
                self.append_anno(ann)
                continue
            elif check==-1:
                res2 = self.agent_generate_stem_downample.run(block=new_block,proposal=inspiration,example_num=3)
                stem_block,downsample_block = res2['list'][0],res2['list'][1]
                check1 = self.block_gen.stem_block.check(stem_block)
                check2 = self.block_gen.downsample_block.check(downsample_block)
                anno['stem_block'] = stem_block
                anno['downsample_block'] = downsample_block
                if isinstance(check1,dict):
                    self.logger.info(f"\tGenerate stem_block try {iter}/{self.max_try} error:{check1['error']}")
                    ann = copy.deepcopy(anno)
                    ann['status'] = False
                    ann['fail'] = {
                        'type':'stem',
                        'error':check1['error']
                    }
                    self.append_anno(ann)
                    continue
                if isinstance(check2,dict):
                    self.logger.info(f"\tGenerate downsample_block try {iter}/{self.max_try} error:{check2['error']}")
                    ann = copy.deepcopy(anno)
                    ann['status'] = False
                    ann['fail'] = {
                        'type':'downsample',
                        'error':check2['error']
                    }
                    self.append_anno(ann)
                    continue
                if not isinstance(check1,dict) and not isinstance(check2,dict):
                    self.logger.info(f"\tGenerate stem_block and downsample_block try {iter}/{self.max_try}, success")
                    iter=-1
                    break
            else:
                self.logger.info(f"\tGenerate base_block try {iter}/{self.max_try}, base_block existed")
                anno['existed']=True
                pairs = self.block_gen.load_annos()
                stem_block = self.block_gen.stem_block.get_block_txt(pairs[check.replace('_base','')]['stem'])
                downsample_block = self.block_gen.downsample_block.get_block_txt(pairs[check.replace('_base','')]['downsample'])
                iter=-1
                anno['stem_block'] = stem_block
                anno['downsample_block'] = downsample_block
                break
        if iter!=-1:
            return anno

        # save block txt
        block_id = f"{block_name}_p{inspiration_id}"
        tag = re.sub(r"[\d]*$", '', self.tag_prefix)
        block_id = re.sub(rf"{tag}[\d]*_",'',block_id)
        if self.tag_prefix not in block_id:
            block_id = self.tag_prefix+'_'+block_id
        block_txt = new_block+'\n'+stem_block+'\n'+downsample_block
        txt_path = os.path.join(self.block_txt_dir,f'{block_id}.txt')
        with open(txt_path,'w') as f:
            f.write(block_txt)
        anno['new_block'] = block_id
            
        # generate code
        status = self.block_gen.add_blocks_from_txt_path(txt_path)
        if isinstance(status,dict):
            self.logger.info(f"\tGenerate block code error: {status['error']}")
            anno['status']=False
            anno['fail']={
                'type':'block_code',
                'error':status['error']
            }
            self.append_anno(anno)
            return anno
        elif status==True:
            try:
                model_name = self.model_gen.generate_one(block_id)
            except Exception as e:
                model_name = None
                anno['status']=False
                anno['fail']={
                    'type':'model_code',
                    'error':str(e)
                }
                self.append_anno(anno)
                return anno
        assert model_name is not None
        anno['model_name'] = model_name
        anno['status'] = True
        self.append_anno(anno)
        return anno

    def chain2(self,inspiration=None,inspiration_id=None,block=None,block_name=None,experiences=None):
        """
        dialog
        """
        anno = {
            'inspiration_id':inspiration_id,
            'raw_block':block_name,
            'tag_prefix':self.tag_prefix,
            'status':False,
            'existed':False,
            'try':0,
            'fail':False,
            'prompt_tokens':0,
            'completion_tokens':0,
            'base_block':None,
            'stem_block':None,
            'downsample_block':None
        }
        base_error = None
        self.agent_modify.clear_history()
        iter = 0
        while iter<self.max_try:
            iter+=1
            anno['try'] = iter
            if iter==1 or not base_error:
                res = self.agent_modify.run(proposal=inspiration,block=block,res_expe=experiences)
            else:
                assert isinstance(base_error,str),base_error
                res = self.agent_modify.run(feedback=base_error)
            new_block = res['list'][0]
            anno['prompt_tokens']+=res['prompt_tokens']
            anno['completion_tokens']+=res['completion_tokens']
            check = self.block_gen.base_block.check(new_block,with_isomorphic=True)
            anno['base_block'] = new_block
            if isinstance(check,dict):
                self.logger.info(f"\tGenerate base_block try {iter}/{self.max_try}, error:{check}")
                ann = copy.deepcopy(anno)
                ann['status'] = False
                ann['fail'] = {
                    'type':'base',
                    'error':check['error']
                }
                self.append_anno(ann)
                base_error = check['error']
                continue
            elif check==-1:
                stem_error = None
                iiter = 0
                self.agent_generate_stem_downample.clear_history()
                while iiter<self.max_try:
                    iiter+=1
                    anno['try_stem'] = iiter
                    if iiter==1 or not stem_error:
                        res2 = self.agent_generate_stem_downample.run(block=new_block,proposal=inspiration,example_num=3)
                    else:
                        assert isinstance(stem_error,str),stem_error
                        res2 = self.agent_generate_stem_downample.run(feedback=stem_error)
                    if len(res2['list'])<2:
                        ann = copy.deepcopy(anno)
                        ann['fail'] = {
                            'type':'stem',
                            'error':'generat stem and downsample'
                        }
                        self.append_anno(ann)
                        stem_error = None
                        continue
                    stem_block,downsample_block = res2['list'][0],res2['list'][1]
                    check1 = self.block_gen.stem_block.check(stem_block)
                    check2 = self.block_gen.downsample_block.check(downsample_block)
                    anno['stem_block'] = stem_block
                    anno['downsample_block'] = downsample_block
                    if isinstance(check1,dict):
                        self.logger.info(f"\tGenerate stem_block try {iter}/{self.max_try} error:{check1['error']}")
                        ann = copy.deepcopy(anno)
                        ann['fail'] = {
                            'type':'stem',
                            'error':check1['error']
                        }
                        self.append_anno(ann)
                        stem_error = f"stem block error:{check1['error']}"
                        continue
                    if isinstance(check2,dict):
                        self.logger.info(f"\tGenerate downsample_block try {iter}/{self.max_try} error:{check2['error']}")
                        ann = copy.deepcopy(anno)
                        ann['fail'] = {
                            'type':'downsample',
                            'error':check2['error']
                        }
                        self.append_anno(ann)
                        stem_error = f"downsample block error:{check2['error']}"
                        continue
                    if not isinstance(check1,dict) and not isinstance(check2,dict):
                        self.logger.info(f"\tGenerate stem_block and downsample_block try {iter}/{self.max_try}, success")
                        iiter=-1
                        break
                if iiter==-1:
                    iter=-1
                    break
            else:
                self.logger.info(f"\tGenerate base_block try {iter}/{self.max_try}, base_block existed")
                anno['existed']=True
                pairs = self.block_gen.load_annos()
                if check.endswith('_base') and not check.endswith('resnet_base'):
                    check = check[:-5]
                stem_block = self.block_gen.stem_block.get_block_txt(pairs[check]['stem'])
                downsample_block = self.block_gen.downsample_block.get_block_txt(pairs[check]['downsample'])
                anno['stem_block'] = stem_block
                anno['downsample_block'] = downsample_block
                iter=-1
                break
        if iter!=-1:
            return anno

        # save block txt
        block_id = f"{block_name}_p{inspiration_id}"
        # old_tag = re.sub(f"[\d]*$", '', self.tag_prefix)
        # result = re.sub(f"{old_tag}[\d]*_", '', block_id)
        if self.tag_prefix not in block_id:
            block_id = self.tag_prefix+'_'+block_id
        block_txt = new_block+'\n'+stem_block+'\n'+downsample_block
        txt_path = os.path.join(self.block_txt_dir,f'{block_id}.txt')
        with open(txt_path,'w') as f:
            f.write(block_txt)
        anno['new_block'] = block_id
            
        # generate code
        status = self.block_gen.add_blocks_from_txt_path(txt_path)
        if isinstance(status,dict):
            self.logger.info(f"\tGenerate block code error: {status['error']}")
            anno['status']=False
            anno['fail']={
                'type':'block_code',
                'error':status['error']
            }
            self.append_anno(anno)
            return anno
        elif status==True:
            try:
                model_name = self.model_gen.generate_one(block_id)
            except Exception as e:
                model_name = None
                anno['status']=False
                anno['fail']={
                    'type':'model_code',
                    'error':str(e)
                }
                self.append_anno(anno)
                return anno
        assert model_name is not None
        anno['model_name'] = model_name
        anno['status'] = True
        self.append_anno(anno)
        anno['base_block'] = new_block
        anno['stem_block'] = stem_block
        anno['downsample_block'] = downsample_block
        return anno
    


if __name__=='__main__':
    main()