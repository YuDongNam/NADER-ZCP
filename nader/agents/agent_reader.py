import json
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import *
import re

from .base_agent import BaseAgent


class AgentReaderFilterPapers(BaseAgent):

    prompt_template = """- Role: You are a computer vision research specialist.
- Background: You needs to evaluate a given paper to determine if it can inspire to design the better basic blocks architecture of vision models' backbone.
- Profile: You are a researcher with a deep background in the field of computer vision, particularly in deep learning models and visual recognition tasks.
- Skills: Proficient in the latest visual model architectures, understanding the functions of different model building blocks and how they impact model performance, excellent reading ability of paper abstracts.
- Goals: According to the title and abstract of the paper, analyzing whether can get inspiration from the paper to design the better basic backbone architecture of visual model backbone.
- Constraints: The inspiration must be related to the basic block architecture design of the visual model backbone.
- Workflow:
  1. Read and understand the title and abstract of the paper.
  2. Summarizing the innovation and contribution of this paper.
  3. Analyzing whether you can get inspiration from this paper to design a better basic block architecture for visual models
  3. Answer yes or no prefix with ##response## in the end.
- Title: {title}
- Abstract: {abstract}
"""

    def __init__(self,*args,**kwargs):
        super().__init__(agent_name='reader_filter_paper',*args,**kwargs)

    def __call__(self,title=None,abstract=None,**kwargs):
        prompt = self.prompt_template.format(title=title,abstract=abstract)
        message = [{'role':'user','content':prompt}]
        res = super().__call__(message,**kwargs)
        try:
            res['output'] = self.parse_func(res['output'])
        except:
            return None,None
        return res['output'],res['time']
        
    def parse_func(self,s):
        match = re.findall('##response##(.*)',s,re.DOTALL)
        s = match[0].strip().lower()
        if s.startswith('yes'):
            return True
        elif s.startswith('no'):
            return False
        else:
            return None

