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

from .prompts import prompt_modify_block,prompt_modify_block_darts,prompt_modify_block2,prompt_develop_experience,prompt_research_experience,prompt_modify_block_staged,STAGE_CONTEXTS
from .gpt_generate_block_base import GPTGenerateBlockBase


class GPTGenerateBlockModify(GPTGenerateBlockBase):
    """
    Modify block, following proposal.
    """

    def __init__(self, agent_name='modify_base', prompt_template=prompt_modify_block, mode='nas-bench', **kwargs) -> None:
        super().__init__(agent_name, **kwargs)
        self.mode = mode
        if mode=='nas-bench':
            self.prompt_template = prompt_modify_block
        elif mode=='darts':
            self.prompt_template = prompt_modify_block_darts
        else:
            raise NotImplementedError
    
    def run(self, proposal=None, block=None, res_expe=None, feedback=None, temperature=0.1, stage_idx=None, num_stages=None):
        """
        multi round conversation
        stage_idx: 1-indexed stage position (e.g., 1, 2, 3)
        num_stages: total number of stages (e.g., 3 for NAS-Bench-201)
        """
        if proposal is not None and block is not None:
            if res_expe:
                expe_s = '\n'.join([f"{i+1}. {expe}" for i, expe in enumerate(res_expe)])
                res_expe = prompt_research_experience.format(experience=expe_s)+"\n"
            else:
                res_expe = ""
            if self.use_experience:
                expes = self.agent_devexpe_retriever(proposal,mode='base')
                expe_s = '\n'.join([f"{i+1}. {expe}" for i, expe in enumerate(expes)])
                dev_expe = prompt_develop_experience.format(experience=expe_s)+"\n"
            else:
                dev_expe = ""
            # Use staged prompt if stage info is provided
            if stage_idx is not None and num_stages is not None:
                stage_context = STAGE_CONTEXTS.get(num_stages, {}).get(stage_idx, "")
                prompt = prompt_modify_block_staged.format(
                    proposal=proposal, res_expe=res_expe, dev_expe=dev_expe, block=block,
                    stage_idx=stage_idx, num_stages=num_stages, stage_context=stage_context
                )
            else:
                prompt = self.prompt_template.format(proposal=proposal,res_expe=res_expe,dev_expe=dev_expe,block=block)
            self.history = [{'role':'user','content':prompt}]
        elif feedback is not None:
            assert len(self.history)>=1
            self.history.append({'role':'user','content':f"The block you generate has following error:{feedback},please fix it and generate a new one."})
        else:
            raise NotImplementedError
        # print(self.history)
        response = self.call_gpt(self.history,temperature)
        res = response['output']
        ret = {
            'output':res,
            'prompt_tokens':response['prompt_tokens'],
            'completion_tokens':response['completion_tokens'],
            'list':[] if 'yes' in res.lower() else self.parse_result(res),
            'time':response['time']
        }
        self.history.append({'role':'assistant','content':ret['list'][0]})
        return ret
    
if __name__=='__main__':
    p = {
        # "name": "squeeze_and_excitation",
        # "operation": "Add a SE block contains two main steps: squeeze and excitation. If the input feature map is represented as X, the squeezed signal S is obtained through global average pooling, S = GlobalAvgPool(X). Pay attention to changing the shape of S to (B,C). The excitation operation applies a transformation F to S, F(S) = sigmoid(W2 * ReLU(W1 * S)), where W1 and W2 are learned parameters. Then reshape S back to (B,C,1,1) and the output is the rescaled feature map, R = F(S) * X, where * denotes channel-wise multiplication."
        # "name":'bottleneck',
        # "operation": "A typical bottleneck block consists of a series of convolutional layers: first, a 1x1 convolution is applied to reduce the dimensionality (depth) of the input feature maps, followed by a computationally expensive operation (like a 3x3 convolution) performed on the reduced representation, and finally, another 1x1 convolution is used to expand the feature maps back to a higher dimension. This sequence allows the network to learn more complex features with fewer resources. Mathematically, if the input is represented as x, and we apply a dimensionality-reduction convolution F1(x), followed by a transformation F2(F1(x)), and finally a dimensionality-expansion convolution F3(F2(F1(x))), the output of the bottleneck block is F3(F2(F1(x)))."
        # "name":"parallel",
        # "operation": "The two blocks receive input from the same input, process it separately and then merge the features by concat or Add. Pay attention to renumbering the nodes to prevent duplication of numbers. Pay attention to processing the width of the feature at the appropriate position to make it possible to merge the features. Carry out the corresponding operation, and the final output size is the same as the original one. Finally, give the new block a new name."
        # "name":"invert_bottleneck",
        # "operation":"The inverted bottleneck first expand the number of channels, then reduce the number of channels before outputting after processing."
        # "name":"bottleneck",
        # "operation":"The bottleneck first reduce the number of channels, then expand the number of channels before outputting after processing."
        # "name":"grouped_convolution",
        # "operation":"Specify the groups parameter of the convolution operation to set the number of groups. Please be careful to specify a reasonable number of groups, and ensure that the number of input and output channels of the convolution can be evenly divided by the number of groups."
        # "name":"depthwise_separable_convolution",
        # "operation":"Select one or more appropriate convolution operation nodes and increase their convolution kernel size."
        # "name":"swap_activation_normalization",
        # "operation":"Swap the order of calculation of all activation functions and normalization operations."

    }
    b = [
#         """##ResNextBasicBlock##
# 0:input
# 1:output
# 2:Conv2d(out_channels=C/4,kernel_size=1,stride=1)
# 3:BN
# 4:ReLU
# 5:Conv2d(out_channels=C/4,kernel_size=3,stride=1,groups=32)
# 6:BN
# 7:ReLU
# 8:Conv2d(out_channels=C,kernel_size=1,stride=1)
# 9:BN
# 10:Add
# 11:ReLU
# 0->2
# 2->3
# 3->4
# 4->5
# 5->6
# 6->7
# 7->8
# 8->9
# 9->10
# 0->10
# 10->11
# 11->1""",
"""##ResNetBottleBlock##
0:input
1:output
2:Conv2d(out_channels=C/4,kernel_size=1,stride=1)
3:BN
4:ReLU
5:Conv2d(out_channels=C/4,kernel_size=3,stride=1)
6:BN
7:ReLU
8:Conv2d(out_channels=C,kernel_size=1,stride=1)
9:BN
10:Add
11:ReLU
0->2
2->3
3->4
4->5
5->6
6->7
7->8
8->9
9->10
0->10
10->11
11->1"""
    ]
    print(b)
    agent = GPTGenerateBlock3(block_txts_dir='/data3/yangzekang/ModelGen/ModelGen/block_txts',
                              log_dir='/data3/yangzekang/ModelGen/ModelGen/logs/gpt_response/generate_block3',
                              prompt_template=prompt_modify_block2)
    p = """1.Change the calculation ratio of features with different resolutions to (3,3,9,3)
2. Change the kernel size of the convolution in the stem cell to 4 and the step size to 4
3. Convert the convolution at the bottleneck position in each basic block into a depthwise convolution
4. Turn each basic block into an inverted bottleneck structure
5. Move up depthwise convolution and change the kernel size to 7
6. Change the activation function to GELU and reduce the number of activation functions in each block to one
7. Change the activation function to Layer Normalization and reduce the number of Normalization in each block to one
8. A separate downsampling layer is used. The convolution kernel size in the downsampling layer is 2 and the step size is 2 followed by Layer Normalization"""
    res = agent.gen(proposal=p,blocks=b)
    print(res['output'])
    block_factory = BlockFactory()
    check = block_factory.check(res['output'])
    print(check)
    print('finish!')


