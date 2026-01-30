import json
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import *
import pdb

from .base_agent import BaseAgent


MOGD = """##Model Optimization Graph Definition##
The optimization path graph of the model is a tree structure.
The node represents the model, describing the structure of the basic block and the accuracy on the test dataset. It is defined in the format of <model_name><block>...</block><acc>...</acc></model_name>. 'model_name' is the name of model. The block is a directed acyclic graph that describes the block calculation process.
Proposals are defined in the form of name:content.
The edge represents the process of obtaining another model from one model through proposal. For example: model_1--proposal_1-->model_2 means that model model_2 was obtained by modifing model model_1 according to the proposal proposal_1.
"""

SELECT_BLOCK_PROPOSAL = """You are a computer vision research expert, and you have deep insights into computer vision models.
You need to select three pairs of model-proposals based on the given model optimization graph and candidate proposals.
The new model modified according to your choice should outperform all existing models in the model optimization graph on the test set.
When you choose the model to be modified and the corresponding proposal, you need to pay attention to the following points:
1. You need to carefully observe and analysis the structure of each model in the model optimization graph.
2. You need to carefully observe and analysis each proposal and the corresponding utility in the model optimization graph.
2. You need to carefully observe and analysis the modification path of the model in the model optimization graph.
3. You should infer the combination of the model to be modified and the corresponding proposal that is most likely to get the best new model based on the modification path in the model optimization graph and candidate proposals.
4. The poorly performing models in the model optimization graph may also achieve the best performance through modification by proposal. You also need to pay attention to and analyze the potential of poorly performing models when making decisions.
The following is the definition of the model optimization graph:
{MOGD}
The following is the current model optimization graph and candidate proposals:
##model optimization graph##
{mog}
##candidate proposals##
{proposals}
##output##
You need to output the name是 of the model to be modified and the names of the corresponding proposal.
{format_output}
"""

# class OutBlockProposal(BaseModel):
#     model_name:str = Field('model name')
#     proposal_name:str = Field('proposal name')

class OutBlockProposal(BaseModel):
    model_proposal_pair:List[Tuple[str,str]] = Field('List of model name and proposal pairs. Each item in the list is model name and proposal name.')
    


class AgentSelectBlockProposal(BaseAgent):

    prompt_template = SELECT_BLOCK_PROPOSAL

    def __init__(self,*args,**kwargs):
        super().__init__(agent_name='select_block_proposal',*args,**kwargs)
        self.output_parser = PydanticOutputParser(pydantic_object=OutBlockProposal)

    def __call__(self,mog=None,proposals=None,**kwargs):
        proposal_txt = '\n'.join([f"{key}:{val}" for key,val in proposals.items()])
        prompt = self.prompt_template.format(MOGD=MOGD,mog=mog,proposals=proposal_txt,format_output=self.output_parser.get_format_instructions())
        message = [{'role':'user','content':prompt}]
        res = super().__call__(message,**kwargs)
        res = res['output']
        try:
            res = self.output_parser.parse(res)
        except:
            return None
        res = res.model_proposal_pair
        ret = []
        if not isinstance(res,list):
            return ret
        for r in res:
            if f"<{r[0]}>" not in mog or f"</{r[0]}>" not in mog or r[1] not in proposals:
                continue
            ret.append(r)
        return ret

if __name__=='__main__':
    path = ''
