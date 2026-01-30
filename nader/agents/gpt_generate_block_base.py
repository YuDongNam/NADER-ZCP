import re
import json
import openai
from openai import OpenAI
import os
import requests
import json
import time

# from .call_gpt import call_gpt
from .base_agent import BaseAgent
from .agent_expe_retriever_dev import DevelopExperienceRetriever,DevelopExperienceRetrieverRandom

class GPTGenerateBlockBase(BaseAgent):

    def __init__(self, agent_name='', model_name='gpt-5-nano', log_dir=None, use_experience=None, experience_mode='VDB') -> None:
        super().__init__(agent_name,model_name,log_dir)
        self.history = []
        self.use_experience = use_experience
        self.experience_mode = experience_mode
        if use_experience:
            if experience_mode=='VDB':
                self.agent_devexpe_retriever = DevelopExperienceRetriever(table_name=use_experience)
            elif experience_mode=='random':
                self.agent_devexpe_retriever = DevelopExperienceRetrieverRandom()
            else:
                raise NotImplementedError

    def clear_history(self):
        self.history = []
    
    
    def call_gpt(self,messages,temp=0.1):
        return super().__call__(messages,temperature=temp)
    
    def parse_result(self,res):
        ret = []
        res = re.sub('\n+','\n',res)
        s = re.findall('(##.*?##((.(?!##))*))',res,re.DOTALL)
        for x in s:
            if 'input' in x[1]:
                ret.append(x[0].strip())
        return ret

