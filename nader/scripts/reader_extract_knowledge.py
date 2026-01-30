from typing import *
import json
import os
import argparse

from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain_classic.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatMessagePromptTemplate, ChatPromptTemplate
from pydantic import BaseModel, Field

PROMPT_READER_PAPER = """You are a computer vision research expert.
Please respond the keywords and the tasks of the paper.
Please list the inspirations you get from this paper to design the basic block architecture of the visual model backbone.
A paper usually contains several sections: abstract, introduction, related work, methods, experiments and conclusion.
Please focus on the methods of the paper to respond the inspirations.
The following is the content of the paper:
{paper}\n
Inspirations must to be detailed and related to designing the basic block architecture of the visual model backbone.
"""

class Inspirations(BaseModel):
    keyword:List[str] = Field("The keywords of the paper.")
    task:List[str] = Field("The tasks of the paper.")
    inspirations:Optional[List[str]] = Field('Inspirations that related to designing the basic block architecture of the visual model backbone.')


def read_paper(file_path):
    with open(file_path,'r') as f:
        txt = f.read()
    parser = PydanticOutputParser(pydantic_object=Inspirations)
    prompt = ChatPromptTemplate(
        messages=[
            ChatMessagePromptTemplate.from_template(
                role='user',
                template=PROMPT_READER_PAPER+'\n{format_output}',
                input_variables=['paper'],
                partial_variables={'format_output':parser.get_format_instructions()}
            )
        ]
    )
    if 'deepseek' in os.environ['LLM_MODEL_NAME']:
        llm = ChatOpenAI(
            model=os.environ['LLM_MODEL_NAME'],
            openai_api_key=os.environ['API_KEY_DEEPSEEK'],
            base_url="https://api.deepseek.com/v1",
            temperature=0.7
        )
    elif 'gpt' in os.environ['LLM_MODEL_NAME']:
        llm = ChatOpenAI(
            model=os.environ['LLM_MODEL_NAME'],
            temperature=0.7,
            openai_api_key=os.environ['OPENAI_API_KEY']
        )
    else:
        raise NotImplementedError
    chain = LLMChain(llm=llm,prompt=prompt,output_parser=parser,return_final_only=True)
    with get_openai_callback() as cb:
        for i in range(10):
            try:
                out = chain({'paper':txt},return_only_outputs=True)
                break
            except Exception as e:
                if i==10-1:
                    raise
                if e.body['code']=='RateLimitReached':
                    continue
        out=out['text']
        cost = cb.total_cost
        res = {
            "tasks":out.task,
            "keywords":out.keyword,
            "inspirations":out.inspirations,
            "cost":cost
        }
    return res
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno-path', default='data/papers/cvpr-2023/annotations_filted.json')
    parser.add_argument('--txt-dir', default='data/papers/cvpr-2023/txts')
    parser.add_argument('--out-dir', default='data/papers/cvpr-2023/txts_inspirations')
    parser.add_argument('--llm-log-dir',default='logs/llm_response/reader-extract-knowledge')
    args = parser.parse_args()

    os.makedirs(args.out_dir,exist_ok=True)
    with open(args.anno_path,'r') as f:
        annos = json.load(f)
    num = 0
    for i,anno in enumerate(annos):
        if anno['tag']:
            num+=1
            res = anno
            res.update(read_paper(os.path.join(args.txt_dir,f"paper{anno['id']}.txt")))
            with open(os.path.join(args.out_dir,f"paper{anno['id']}.json"),'w') as f:
                json.dump(res,f,indent='\t')
            print(f"{i+1}/{len(annos)}")