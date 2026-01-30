import re
import json
import openai
from openai import OpenAI
import os
import random
import time
from datetime import datetime
import pdb
import requests
import json
import gc

from .call_llms import call_llm

class BaseAgent():

    def __init__(self, agent_name='', log_dir='logs', output_parser=None) -> None:
        log_dir = os.path.join(log_dir,agent_name)
        os.makedirs(log_dir,exist_ok=True)
        self.log_dir = log_dir
        self.output_parser = output_parser


    def __call__(self, messages, temperature=0.7):
        # Force GC to clear previous iterations garbage
        gc.collect()
        
        start_time = time.time()
        response = call_llm(messages,temperature=temperature)
        if self.log_dir:
            TIME_NOW = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            save_path = os.path.join(self.log_dir,TIME_NOW+'.json')
            with open(save_path,'w') as f:
                json.dump(response,f,indent='\t')
        res = response['content']
        end_time = time.time()
        dur_time = end_time - start_time
        ret = {
            'prompt_tokens':response['prompt_tokens'],
            'completion_tokens':response['completion_tokens'],
            'output':res,
            'time':dur_time
        }
        return ret


