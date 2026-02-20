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
                 layers_num=20,
                 num_stages=None) -> None:
        self.team_name = team_name
        self.database_dir = database_dir
        self.max_try = max_try
        self.tag_prefix = tag_prefix
        self.use_experience = use_experience
        self.cell_mode = cell_mode

        # Determine num_stages from dataset/mode if not explicitly provided
        if num_stages is not None:
            self.num_stages = num_stages
        else:
            if cell_mode == 'nas-bench':
                self.num_stages = 4 if 'imagenet-1k' in dataset else 3
            elif cell_mode == 'darts':
                self.num_stages = 3
            else:
                self.num_stages = 3

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
        retry (with position-aware stage loop)
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

        num_stages = self.num_stages
        stage_base_blocks = {}  # stage_idx -> block text
        all_stages_ok = True

        for stage_idx in range(1, num_stages + 1):
            iter = 0
            stage_ok = False
            while iter < self.max_try:
                iter += 1
                anno['try'] = iter
                res = self.agent_modify.run(inspiration, block, experiences,
                                            stage_idx=stage_idx, num_stages=num_stages)
                new_block = res['list'][0]
                anno['prompt_tokens'] += res['prompt_tokens']
                anno['completion_tokens'] += res['completion_tokens']
                check = self.block_gen.base_block.check(new_block, with_isomorphic=True)
                if isinstance(check, dict):
                    self.logger.info(f"\tGenerate base_block stage {stage_idx}/{num_stages} try {iter}/{self.max_try}, error:{check}")
                    ann = copy.deepcopy(anno)
                    ann['status'] = False
                    ann['fail'] = {'type': 'base', 'stage': stage_idx, 'error': check['error']}
                    self.append_anno(ann)
                    continue
                elif check == -1:
                    # New block, not existed
                    stage_base_blocks[stage_idx] = new_block
                    stage_ok = True
                    break
                else:
                    # Block already existed, reuse
                    self.logger.info(f"\tGenerate base_block stage {stage_idx}/{num_stages} try {iter}/{self.max_try}, base_block existed")
                    stage_base_blocks[stage_idx] = new_block
                    anno['existed'] = True
                    stage_ok = True
                    break
            if not stage_ok:
                all_stages_ok = False
                break

        if not all_stages_ok:
            return anno

        # Use stage 1 block for stem/downsample generation (stage-independent)
        representative_block = stage_base_blocks[1]
        anno['base_block'] = representative_block

        # Generate stem and downsample (unchanged logic)
        iter = 0
        stem_ok = False
        while iter < self.max_try:
            iter += 1
            res2 = self.agent_generate_stem_downample.run(block=representative_block, proposal=inspiration, example_num=3)
            stem_block, downsample_block = res2['list'][0], res2['list'][1]
            check1 = self.block_gen.stem_block.check(stem_block)
            check2 = self.block_gen.downsample_block.check(downsample_block)
            anno['stem_block'] = stem_block
            anno['downsample_block'] = downsample_block
            if isinstance(check1, dict):
                self.logger.info(f"\tGenerate stem_block try {iter}/{self.max_try} error:{check1['error']}")
                ann = copy.deepcopy(anno)
                ann['status'] = False
                ann['fail'] = {'type': 'stem', 'error': check1['error']}
                self.append_anno(ann)
                continue
            if isinstance(check2, dict):
                self.logger.info(f"\tGenerate downsample_block try {iter}/{self.max_try} error:{check2['error']}")
                ann = copy.deepcopy(anno)
                ann['status'] = False
                ann['fail'] = {'type': 'downsample', 'error': check2['error']}
                self.append_anno(ann)
                continue
            if not isinstance(check1, dict) and not isinstance(check2, dict):
                self.logger.info(f"\tGenerate stem_block and downsample_block try {iter}/{self.max_try}, success")
                stem_ok = True
                break

        if not stem_ok:
            return anno

        # Save block txt (all stage bases + stem + downsample)
        block_id = f"{block_name}_p{inspiration_id}"
        tag = re.sub(r"[\d]*$", '', self.tag_prefix)
        block_id = re.sub(rf"{tag}[\d]*_",'',block_id)
        if self.tag_prefix not in block_id:
            block_id = self.tag_prefix+'_'+block_id

        # Compose block text: all stage base blocks + stem + downsample
        base_txts = [stage_base_blocks[s] for s in range(1, num_stages + 1)]
        block_txt = '\n'.join(base_txts + [stem_block, downsample_block])
        txt_path = os.path.join(self.block_txt_dir, f'{block_id}.txt')
        with open(txt_path, 'w') as f:
            f.write(block_txt)
        anno['new_block'] = block_id
        anno['stage_base_blocks'] = {str(k): v for k, v in stage_base_blocks.items()}
            
        # Generate code
        status = self.block_gen.add_blocks_from_txt_path(txt_path, num_stages=num_stages)
        if isinstance(status, dict):
            self.logger.info(f"\tGenerate block code error: {status['error']}")
            anno['status'] = False
            anno['fail'] = {'type': 'block_code', 'error': status['error']}
            self.append_anno(anno)
            return anno
        elif status == True:
            try:
                model_name = self.model_gen.generate_one(block_id)
            except Exception as e:
                model_name = None
                anno['status'] = False
                anno['fail'] = {'type': 'model_code', 'error': str(e)}
                self.append_anno(anno)
                return anno
        assert model_name is not None
        anno['model_name'] = model_name
        anno['status'] = True
        self.append_anno(anno)
        return anno

    def chain2(self,inspiration=None,inspiration_id=None,block=None,block_name=None,experiences=None):
        """
        dialog (with position-aware stage loop)
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

        num_stages = self.num_stages
        stage_base_blocks = {}
        all_stages_ok = True

        for stage_idx in range(1, num_stages + 1):
            base_error = None
            self.agent_modify.clear_history()
            iter = 0
            stage_ok = False
            while iter < self.max_try:
                iter += 1
                anno['try'] = iter
                if iter == 1 or not base_error:
                    res = self.agent_modify.run(proposal=inspiration, block=block, res_expe=experiences,
                                                stage_idx=stage_idx, num_stages=num_stages)
                else:
                    assert isinstance(base_error, str), base_error
                    res = self.agent_modify.run(feedback=base_error)
                new_block = res['list'][0]
                anno['prompt_tokens'] += res['prompt_tokens']
                anno['completion_tokens'] += res['completion_tokens']
                check = self.block_gen.base_block.check(new_block, with_isomorphic=True)
                if isinstance(check, dict):
                    self.logger.info(f"\tGenerate base_block stage {stage_idx}/{num_stages} try {iter}/{self.max_try}, error:{check}")
                    ann = copy.deepcopy(anno)
                    ann['status'] = False
                    ann['fail'] = {'type': 'base', 'stage': stage_idx, 'error': check['error']}
                    self.append_anno(ann)
                    base_error = check['error']
                    continue
                elif check == -1:
                    stage_base_blocks[stage_idx] = new_block
                    stage_ok = True
                    break
                else:
                    self.logger.info(f"\tGenerate base_block stage {stage_idx}/{num_stages} try {iter}/{self.max_try}, base_block existed")
                    stage_base_blocks[stage_idx] = new_block
                    anno['existed'] = True
                    stage_ok = True
                    break
            if not stage_ok:
                all_stages_ok = False
                break

        if not all_stages_ok:
            return anno

        # Use stage 1 block for stem/downsample generation
        representative_block = stage_base_blocks[1]
        anno['base_block'] = representative_block

        # Generate stem and downsample (dialog mode)
        stem_error = None
        iiter = 0
        self.agent_generate_stem_downample.clear_history()
        stem_ok = False
        while iiter < self.max_try:
            iiter += 1
            anno['try_stem'] = iiter
            if iiter == 1 or not stem_error:
                res2 = self.agent_generate_stem_downample.run(block=representative_block, proposal=inspiration, example_num=3)
            else:
                assert isinstance(stem_error, str), stem_error
                res2 = self.agent_generate_stem_downample.run(feedback=stem_error)
            if len(res2['list']) < 2:
                ann = copy.deepcopy(anno)
                ann['fail'] = {'type': 'stem', 'error': 'generate stem and downsample'}
                self.append_anno(ann)
                stem_error = None
                continue
            stem_block, downsample_block = res2['list'][0], res2['list'][1]
            check1 = self.block_gen.stem_block.check(stem_block)
            check2 = self.block_gen.downsample_block.check(downsample_block)
            anno['stem_block'] = stem_block
            anno['downsample_block'] = downsample_block
            if isinstance(check1, dict):
                self.logger.info(f"\tGenerate stem_block try {iiter}/{self.max_try} error:{check1['error']}")
                ann = copy.deepcopy(anno)
                ann['fail'] = {'type': 'stem', 'error': check1['error']}
                self.append_anno(ann)
                stem_error = f"stem block error:{check1['error']}"
                continue
            if isinstance(check2, dict):
                self.logger.info(f"\tGenerate downsample_block try {iiter}/{self.max_try} error:{check2['error']}")
                ann = copy.deepcopy(anno)
                ann['fail'] = {'type': 'downsample', 'error': check2['error']}
                self.append_anno(ann)
                stem_error = f"downsample block error:{check2['error']}"
                continue
            if not isinstance(check1, dict) and not isinstance(check2, dict):
                self.logger.info(f"\tGenerate stem_block and downsample_block try {iiter}/{self.max_try}, success")
                stem_ok = True
                break

        if not stem_ok:
            return anno

        # Save block txt
        block_id = f"{block_name}_p{inspiration_id}"
        if self.tag_prefix not in block_id:
            block_id = self.tag_prefix+'_'+block_id

        base_txts = [stage_base_blocks[s] for s in range(1, num_stages + 1)]
        block_txt = '\n'.join(base_txts + [stem_block, downsample_block])
        txt_path = os.path.join(self.block_txt_dir, f'{block_id}.txt')
        with open(txt_path, 'w') as f:
            f.write(block_txt)
        anno['new_block'] = block_id
        anno['stage_base_blocks'] = {str(k): v for k, v in stage_base_blocks.items()}
            
        # Generate code
        status = self.block_gen.add_blocks_from_txt_path(txt_path, num_stages=num_stages)
        if isinstance(status, dict):
            self.logger.info(f"\tGenerate block code error: {status['error']}")
            anno['status'] = False
            anno['fail'] = {'type': 'block_code', 'error': status['error']}
            self.append_anno(anno)
            return anno
        elif status == True:
            try:
                model_name = self.model_gen.generate_one(block_id)
            except Exception as e:
                model_name = None
                anno['status'] = False
                anno['fail'] = {'type': 'model_code', 'error': str(e)}
                self.append_anno(anno)
                return anno
        assert model_name is not None
        anno['model_name'] = model_name
        anno['status'] = True
        self.append_anno(anno)
        anno['base_block'] = representative_block
        anno['stem_block'] = stem_block
        anno['downsample_block'] = downsample_block
        return anno
    


if __name__=='__main__':
    main()