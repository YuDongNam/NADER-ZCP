import json
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import *
import pdb

from .base_agent import BaseAgent
from .prompts import definition_BDAG

class OutCheckProposals(BaseModel):
    inds:List[int] = Field('The serial numbers of useful proposals.')


class AgentSelectProposal(BaseAgent):

    prompt_template = f"""###Instruction###
You are an expert who is proficient in various model structures of deep learning. 
Below are the basic block definition of computer vision models, a block to be improved and candidate proposals.
You need to compare the candidate proposals and rank them according their usefulness for guiding the improvement of the block.
{definition_BDAG}
\n###block###
{{block}}
\n###candidate proposals###
{{proposals}}
\n
###output###
Please rank the all candidate proposals in descending order according to their usefulness.
{{format_output}}
"""

    def __init__(self,*args,**kwargs):
        super().__init__(agent_name='check_proposal',*args,**kwargs)
        self.output_parser = PydanticOutputParser(pydantic_object=OutCheckProposals)

    def __call__(self,block=None,proposals=None,**kwargs):
        proposals_txt = '\n'.join([f"{i+1}. {proposal}" for i,proposal in enumerate(proposals)])
        prompt = self.prompt_template.format(block=block,proposals=proposals_txt,format_output=self.output_parser.get_format_instructions())
        message = [{'role':'user','content':prompt}]
        res = super().__call__(message,**kwargs)
        res = res['output']
        try:
            res = self.output_parser.parse(res).inds
        except:
            return []
        res = self.check_out(res)
        return 
        
    def check_out(self,res):
        if not isinstance(res,list):
            return []
        else:
            try:
                res = [int(i)-1 for i in res]
            except:
                return []
        return res

