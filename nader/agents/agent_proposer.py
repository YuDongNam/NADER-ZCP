import json
import numpy as np
from typing import *
import pdb
import re

# from langchain.output_parsers import PydanticOutputParser
# from pydantic import BaseModel, Field

from .base_agent import BaseAgent
from .prompts import PROMPT_PROPOSER_INIT,PROMPT_PROPOSER_MOG,PROMPT_PROPOSE_INSPIRATION_INIT




# class OutProposerInit(BaseModel):
#     inspiration_index:List[str] = Field('List of the inspiration_index of the selected inspirations')
    


class ResearchProposer(BaseAgent):

    PROMPT_TEMPLATES = {
        'one':PROMPT_PROPOSER_INIT,
        'mog':PROMPT_PROPOSER_MOG
    }

    def __init__(self,*args,**kwargs):
        super().__init__(agent_name='proposer',*args,**kwargs)
        # self.output_parser_init = PydanticOutputParser(pydantic_object=OutProposerInit)

    def __call__(self,blocks=None,inspirations=None,mode='one',**kwargs):
        inspiration_txt = '\n'.join([f"{key}:{val}" for key,val in inspirations.items()])
        prompt = self.PROMPT_TEMPLATES[mode].format(block=blocks,inspirations=inspiration_txt)
        message = [{'role':'user','content':prompt}]
        
        res = super().__call__(message,**kwargs)
        output = res['output']
        if mode=='one':
            # Use non-greedy match to capture individual tags if multiple exist
            raw_matches = re.findall('<response>(.*?)</response>', output, re.DOTALL)
            
            # Helper list to hold all potential IDs found
            all_ids = []
            for m in raw_matches:
                # Handle cases where multiple IDs might be comma-separated inside one tag
                all_ids.extend(m.split(','))
            
            output = []
            for p in all_ids:
                p = p.strip()
                if p in inspirations:
                    output.append({'block_name':None,'inspiration_id':p})
        elif mode=='mog':
            ps = re.findall('<proposal>.*?<model>(.*?)</model>.*?<inspiration>(.*?)</inspiration>.*?</proposal>',output,re.DOTALL)
            output = []
            for p in ps:
                id = p[1].strip()
                if id in inspirations:
                    output.append({'block_name':p[0].strip(),'inspiration_id':id})
        else:
            raise NotImplementedError
        res['proposals'] = output
        return res

class ResearchProposerRandom(BaseAgent):


    def __init__(self,*args,**kwargs):
        super().__init__(agent_name='proposer',*args,**kwargs)

    def __call__(self,blocks=None,inspirations=None,**kwargs):
        ids = list(inspirations.keys())
        np.random.shuffle(ids)
        props = []
        for id in ids:
            props.append({'block_name':None,'inspiration_id':id})
        res = {
            'prompt_tokens':0,
            'completion_tokens':0,
            'proposals':props
        }
        return res


class ResearchProposeInspiration(BaseAgent):

    PROMPT_TEMPLATES = {
        'one':PROMPT_PROPOSE_INSPIRATION_INIT
    }

    def __init__(self,*args,**kwargs):
        super().__init__(agent_name='proposer_inspiration',*args,**kwargs)

    def __call__(self,blocks=None,mode='one',**kwargs):
        prompt = self.PROMPT_TEMPLATES[mode].format(block=blocks)
        message = [{'role':'user','content':prompt}]
        res = super().__call__(message,**kwargs)
        output = res['output']
        if mode=='one':
            ps = re.findall('<response>(.*?)</response>',output,re.DOTALL)
            output = []
            if len(ps)>=1:
                for p in ps:
                    p = p.strip()
                    output.append({'block_name':None,'inspiration':p})
        else:
            raise NotImplementedError
        res['proposals'] = output
        return res