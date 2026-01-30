from langchain_community.callbacks import get_openai_callback
from langchain_classic.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import *

from .prompts import prompt_check_proposals,definition_BDAG
from .agent_select_proposal import AgentSelectProposal
from .agent_select_block_proposal import AgentSelectBlockProposal

class OutCheckProposals(BaseModel):
    inds:List[int] = Field('The serial numbers of useful proposals.')


def create_agent(name,llm,log_dir=None):
    if name=='select_proposals':
        # parser = PydanticOutputParser(pydantic_object=OutCheckProposals)
        # prompt = PromptTemplate(
        #     template=prompt_check_proposals,
        #     input_variables=['block','proposals'],
        #     partial_variables={'definition_BDAG':definition_BDAG,'format_output':parser.get_format_instructions()}
        # )
        # agent = LLMChain(llm=llm,prompt=prompt,output_parser=parser,return_final_only=True)
        agent = AgentSelectProposal(log_dir=log_dir)
        return agent
    elif name=='select_block_proposal':
        agent = AgentSelectBlockProposal(log_dir=log_dir)
        return agent
    elif name=='check_block':
        parser = BlockParser()
        prompt = PromptTemplate(
            template=prompt_check_block,
            input_variables=['block'],
            partial_variables={'definition_BDAG':definition_BDAG,'format_output':parser.get_format_instructions()}
        )
        agent = LLMChain(llm=llm,prompt=prompt,output_parser=parser,return_final_only=True)
        return agent
    else:
        raise NotImplementedError

