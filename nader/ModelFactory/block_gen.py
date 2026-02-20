import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from block_factory import BlockFactory, DAGError
import json
import os
import re
import pdb

class BlockGen:

    def __init__(self,blocks_dir='/data3/yangzekang/ModelGen/ModelGen/ModelFactory/blocks',register_path='ModelFactory.register',stem_down_scale=1,mode='nas-bench') -> None:
        self.base_block = BlockFactory(os.path.join(blocks_dir,'base'),type='base',register_path=register_path,mode=mode)
        self.stem_block = BlockFactory(os.path.join(blocks_dir,'stem'),type='stem',register_path=register_path,stem_down_scale=stem_down_scale,mode=mode)
        self.downsample_block = BlockFactory(os.path.join(blocks_dir,'downsample'),type='downsample',register_path=register_path,mode=mode)
        self.anno_path = os.path.join(blocks_dir,'anno_pairs.json')
        self.annos = self.load_annos()

    def load_annos(self):
        if os.path.isfile(self.anno_path):
            with open(self.anno_path,'r') as f:
                ds = json.load(f)
        else:
            ds = {}
        return ds

    def add_pair(self,name,base,stem,downsample):
        """base can be a single string or a list of strings for staged blocks"""
        self.annos[name] = {
            'base':base,
            'stem':stem,
            'downsample':downsample
        }
        with open(self.anno_path,'w') as f:
            json.dump(self.annos,f,indent='\t')
    
    def delete_pair(self,name):
        del self.annos[name]
        with open(self.anno_path,'w') as f:
            json.dump(self.annos,f,indent='\t')


    def add_blocks_from_txt_path(self,path,with_isomorphic=True,num_stages=1):
        """
        -1: existed
        {'error':...} error
        True created
        
        num_stages: number of base blocks in the txt file (1 = legacy, >1 = staged)
        File format: [base_1] [base_2] ... [base_N] [stem] [downsample]
        """
        id,_ = os.path.splitext(os.path.basename(path))
        if id in self.annos:
            return {'error':f'{id} existed'}
        with open(path,'r') as f:
            blocks = f.read()
        pattern = '(##(.*?)##(.(?!##))*)'
        matches = re.findall(pattern, blocks, flags=re.MULTILINE|re.DOTALL)

        expected_count = num_stages + 2  # N bases + stem + downsample
        if len(matches) < expected_count:
            # Fallback to legacy 3-block format
            num_stages = 1
            expected_count = 3
        assert len(matches) >= expected_count, f"Expected {expected_count} blocks but found {len(matches)}"

        # Parse blocks: first num_stages are bases, then stem, then downsample
        base_texts = []
        stem_text = None
        downsample_text = None

        for i, match in enumerate(matches[:expected_count]):
            name = match[1]
            s = match[0].strip('\n')
            if i < num_stages:
                # This is a base block
                base_texts.append(s)
            elif 'stem' in name or i == num_stages:
                stem_text = s
            elif 'downsample' in name or i == num_stages + 1:
                downsample_text = s

        if stem_text is None or downsample_text is None:
            return {'error': 'Could not identify stem and downsample blocks'}

        # Validate and add base blocks
        base_ids = []
        for si, base_s in enumerate(base_texts):
            out = self.base_block.check(base_s, with_isomorphic=with_isomorphic)
            if isinstance(out, dict):
                return out
            elif out == -1:
                stage_suffix = f'_s{si+1}' if num_stages > 1 else ''
                base_id = self.base_block.add_block(base_s, f'{id}{stage_suffix}')
                if isinstance(base_id, dict):
                    return base_id
                base_ids.append(base_id)
            else:
                # Block already existed, reuse its id
                base_id = self.base_block.add_block(base_s, f'{id}_s{si+1}' if num_stages > 1 else id)
                if isinstance(base_id, dict):
                    base_ids.append(out)  # Use the existing id
                else:
                    base_ids.append(base_id)

        # Validate and add stem
        out2 = self.stem_block.check(stem_text, with_isomorphic=with_isomorphic)
        if isinstance(out2, dict):
            return out2
        if out2 == -1:
            stem_id = self.stem_block.add_block(stem_text, id)
            if isinstance(stem_id, dict):
                return stem_id
        else:
            stem_id = out2

        # Validate and add downsample
        out3 = self.downsample_block.check(downsample_text, with_isomorphic=with_isomorphic)
        if isinstance(out3, dict):
            return out3
        if out3 == -1:
            downsample_id = self.downsample_block.add_block(downsample_text, id)
            if isinstance(downsample_id, dict):
                return downsample_id
        else:
            downsample_id = out3

        # Store base as list if multi-stage, string if single
        base_val = base_ids if num_stages > 1 else base_ids[0]
        self.add_pair(id, base_val, stem_id, downsample_id)
        return True

    def delete_blocks(self,ids):
        for id in ids:
            self.delete_pair(id)
            self.base_block.delete_block(id)
    
    def add_blocks_from_txt_dir(self,txt_dir):
        for file in os.listdir(txt_dir):
            path = os.path.join(txt_dir,file)
            if os.path.isfile(path):
                print(path)
                status = self.add_blocks_from_txt_path(path)
                if status!=True:
                    print(status)

if __name__=='__main__':
    gen = BlockGen(block_dir='/home/SENSETIME/yangzekang/ModelGen/ModelFactoryV1/blocks',blocks_path='ModelFactoryV1')
    gen.add_blocks_from_txt_path('/data3/yangzekang/ModelGen/ModelGen/block_txts/block_2.txt')

        