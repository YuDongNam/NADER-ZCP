import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import json
import logging
import torch
import torch.nn as nn
from thop import profile
from ModelFactory.register import Registers, import_one_modules_for_register,import_module_from_path,import_all_modules_for_register2
import pdb


MODEL_CODE_DARTS_CIFAR = """import torch.nn as nn
from {register_path} import Registers

class AuxiliaryHeadCIFAR(nn.Module):

  def __init__(self, C, num_classes):
    super(AuxiliaryHeadCIFAR, self).__init__()
    self.features = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False), # image size = 2 x 2
      nn.Conv2d(C, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=False),
      nn.Conv2d(128, 768, 2, bias=False),
      nn.BatchNorm2d(768),
      nn.ReLU(inplace=False)
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0),-1))
    return x

@Registers.model
class {model_name}(nn.Module):

    def __init__(self,num_classes=100,auxiliary=False) -> None:
        super().__init__()
        self.auxiliary = auxiliary
        self.drop_path_prob = 0.0
        base = Registers.block['{block}']
        stem = Registers.block['{stem}']
        downsample = Registers.block['{downsample}']
        self.stem = stem(3,{widths[0]})
        cells_type = [0]*{depths[0]}+[1]+[0]*{depths[1]}+[1]+[0]*{depths[2]}
        cell_widths = [{widths[0]}]*{depths[0]}+[{widths[1]}]+[{widths[1]}]*{depths[1]}+[{widths[2]}]+[{widths[2]}]*{depths[2]}
        self.layer_num = sum({depths})+2
        self.cells = nn.ModuleList()
        reduction_pre,reduction_pre_pre = False, False
        width_pre,width_pre_pre = {widths[0]},{widths[0]}
        for i in range(self.layer_num):
            if i in [self.layer_num//3,2*self.layer_num//3]:
                cell = downsample(width_pre,cell_widths[i],width_pre_pre,reduction_pre_pre)
                reduction_pre = True
            else:
                cell = base(width_pre,cell_widths[i],width_pre_pre,reduction_pre_pre)
                reduction_pre = False
            width_pre_pre = width_pre
            width_pre = cell_widths[i]
            self.cells += [cell]
            reduction_pre_pre = reduction_pre
        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR({widths[2]}, num_classes)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear({widths[2]},num_classes)

    def forward(self,x):
        logits_aux = None
        x1 = x2 = self.stem(x)
        for i,cell in enumerate(self.cells):
            x1,x2 = x2,cell(x1,x2,self.drop_path_prob)
            if i == 2*self.layer_num//3:
                if self.auxiliary and self.training:
                    logits_aux = self.auxiliary_head(x2)
        x2 = self.avg_pool(x2)
        x2 = x2.view(x2.size(0), -1)
        logits = self.fc(x2)
        return logits,logits_aux
        
"""

class ModelGen():

    def __init__(self,blocks_dir,models_dir,dataset='cifar10',register_path = 'ModelFactory.register',mode='nas-bench',layers_num=8) -> None:
        self.models_dir = models_dir
        self.dataset = dataset
        self.package = 'codes.model'
        self.register_path = register_path
        self.mode = mode
        self.layers_num = layers_num
        self.model_dir = os.path.join(models_dir,'code')
        os.makedirs(self.model_dir,exist_ok=True)
        self.anno_path = os.path.join(models_dir,'anno.json')
        self.pairs_path = os.path.join(blocks_dir,'anno_pairs.json')

        import_all_modules_for_register2(blocks_dir)
        import_all_modules_for_register2(models_dir)

    def load_annos(self):
        if os.path.isfile(self.anno_path):
           with open(self.anno_path,'r') as f:
               ds = json.load(f)
        else:
            ds = {}
        return ds

    def update_anno(self,model_name,val):
        if os.path.isfile(self.anno_path):
           with open(self.anno_path,'r') as f:
               ds = json.load(f)
        else:
            ds = {}
        ds[model_name] = val
        with open(self.anno_path,'w') as f:
            json.dump(ds,f,indent='\t') 

    def get_width_and_depth(self,block,stem,downsample,tolerance=0.05,max_iter=10,groups=0,return_pf=False):
        """
        Fix depth, width align to ResNet50
        """
        # width = [256,512,1024,2048]
        # depths = [2,3,5,2]
        # depths = [3,4,6,3]
        if self.mode=='nas-bench':
            if self.dataset=='imagenet-1k':
                widths_scale = [1,2,4,8]
                x1_min,x1_max=16,512
                depths = [3,3,9,3]
                param_target = 30.0
                flops_target = 5.4
                input_shape = (1,3,224,224)
                num_classes = 1000
            elif self.dataset.startswith('nas-bench-201'):
                widths_scale = [1,2,4]
                x1_min,x1_max=4,256
                depths = [5,5,5]
                param_target = 1.5
                if self.dataset.endswith('cifar10') or self.dataset.endswith('cifar100'):
                    flops_target = 0.2
                    input_shape = (1,3,32,32)
                    num_classes = 100
                elif self.dataset.endswith('imagenet16-120'):
                    flops_target = 0.05
                    input_shape = (1,3,16,16)
                    num_classes = 120
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        elif self.mode=='darts':
            if self.dataset in ['cifar10','cifar100']:
                widths_scale = [1,2,4]
                x1_min,x1_max=4,64
                if self.layers_num:
                    assert (self.layers_num-2)%3==0
                    depths = [(self.layers_num-2)//3]*3
                    layer_num = self.layers_num
                else:
                    raise NotImplementedError
                # depths = [5,5,8]
                # param_target = 3.4
                if self.layers_num==20:
                    param_target = 3.84
                elif self.layers_num==8:
                    param_target = 1.2
                else:
                    raise NotImplementedError
                if self.dataset.endswith('cifar10') or self.dataset.endswith('cifar100'):
                    # flops_target = 0.64
                    flops_target = 5
                    input_shape = (1,3,32,32)
                    num_classes = 10
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        tolerance_param = 3*param_target/100.0
        tolerance_flops = 3*flops_target/100.0

        res = None
        i = 0
        c_last = -1
        while x1_min <= x1_max:
            x1_mid = int((x1_min + x1_max) / 2)
            if x1_mid==c_last:
                res = x1_mid
                break
            if x1_mid==64:
                x1_max=128
                continue
            c_last = x1_mid
            if groups!=0:
                x1_mid = x1_mid + (groups-x1_mid%groups)
            x1_mid=int(x1_mid)
            if self.mode == 'nas-bench':
                block_stem = Registers.block[stem]
                block_downsample = Registers.block[downsample]
                # Support staged blocks: use first block for profiling
                if isinstance(block, list):
                    block_base = Registers.block[block[0]]
                else:
                    block_base = Registers.block[block]
                layers = []
                layers.append(block_stem(3,x1_mid))
                for width,depth in zip(widths_scale,depths):
                    for _ in range(depth):
                        layers.append(block_base(in_channels=int(x1_mid*width)))
                    if width!=widths_scale[-1]:
                        layers.append(block_downsample(int(x1_mid*width),int(x1_mid*width*2)))
                layers.append(nn.AdaptiveAvgPool2d((1, 1)))
                layers.append(nn.Flatten())
                m = nn.Linear(x1_mid*widths_scale[-1],num_classes)
                layers.append(m)
                m = nn.Sequential(*layers)
            elif self.mode == 'darts':
                widths = [int(x1_mid*w) for w in widths_scale]
                model_code = MODEL_CODE_DARTS_CIFAR.format(register_path=self.register_path,model_name='temp',block=block,stem=stem,downsample=downsample,widths=widths,depths=depths)
                code_path = os.path.join(self.model_dir,f'temp.py')
                with open(os.path.join(self.model_dir,f'temp.py'),'w') as f:
                    f.write(model_code)
                # import_one_modules_for_register('temp',package=self.package)
                module = import_module_from_path('temp',code_path)
                m = module.temp(num_classes)
            flops,current_value = profile(m,(torch.randn(input_shape),),verbose=False)
            current_value=current_value/(1000**2)
            flops=flops/(1000**3)
            # print(x1_mid,current_value)
            if abs(current_value - param_target) < tolerance_param and (flops - flops_target) < tolerance_flops:
                res = x1_mid
                break
            elif current_value < param_target and flops<flops_target:
                x1_min = x1_mid
            else:
                x1_max = x1_mid
            i+=1
            if i==max_iter:
                res = x1_mid
                break
        if i==max_iter and (current_value>param_target or flops>flops_target):
            if groups!=0:
                res-=groups
            else:
                res-=1
        widths = [int(res*w) for w in widths_scale]
        if return_pf:
            return widths,depths,current_value,flops
        return widths,depths


    def _generate_one(self,model_name,block,stem,downsample,save_file_path):
        # widths = [36,72,144]
        # depths = [5,5,8]
        # depths = [6,6,6]
        # depths = [2,2,2]

        try:
            widths,depths = self.get_width_and_depth(block,stem,downsample)
        except Exception as e:
            if 'group' in str(e):
                try:
                    widths,depths = self.get_width_and_depth(block,stem,downsample,groups=8)
                except Exception as e:
                    try:
                        widths,depths = self.get_width_and_depth(block,stem,downsample,groups=16)
                    except:
                        try:
                            widths,depths = self.get_width_and_depth(block,stem,downsample,groups=32)
                        except:
                            try:
                                widths,depths = self.get_width_and_depth(block,stem,downsample,groups=64)
                            except:
                                widths,depths = self.get_width_and_depth(block,stem,downsample,groups=128)
            else:
                raise e
        if self.mode == 'nas-bench':
            if self.dataset=='imagenet-1k':
                model_code = f"""import torch.nn as nn
from {self.register_path} import Registers

@Registers.model
class {model_name}(nn.Module):

    def __init__(self,num_classes=100) -> None:
        super().__init__()
        block = Registers.block['{block}']
        stem = Registers.block['{stem}']
        downsample = Registers.block['{downsample}']
        self.stem = stem(in_channels=3,out_channels={widths[0]})
        layers = []
        for _ in range({depths[0]}):
            layers.append(block(in_channels={widths[0]},out_channels={widths[0]}))
        self.layer1 = nn.Sequential(*layers)
        self.downsample1 = downsample(in_channels={widths[0]},out_channels={widths[1]})
        layers = []
        for _ in range({depths[1]}):
            layers.append(block(in_channels={widths[1]},out_channels={widths[1]}))
        self.layer2 = nn.Sequential(*layers)
        self.downsample2 = downsample(in_channels={widths[1]},out_channels={widths[2]})
        layers = []
        for _ in range({depths[2]}):
            layers.append(block(in_channels={widths[2]},out_channels={widths[2]}))
        self.layer3 = nn.Sequential(*layers)
        self.downsample3 = downsample(in_channels={widths[2]},out_channels={widths[3]})
        layers = []
        for _ in range({depths[3]}):
            layers.append(block(in_channels={widths[3]},out_channels={widths[3]}))
        self.layer4 = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear({widths[3]},num_classes)

    def forward(self,x):
        h = self.stem(x)
        h = self.layer1(h)
        h = self.downsample1(h)
        h = self.layer2(h)
        h = self.downsample2(h)
        h = self.layer3(h)
        h = self.downsample3(h)
        h = self.layer4(h)
        h = self.avg_pool(h)
        h = h.view(h.size(0), -1)
        h = self.fc(h)
        return h

"""
            elif self.dataset.startswith('nas-bench-201'):
                model_code = f"""import torch.nn as nn
from {self.register_path} import Registers

@Registers.model
class {model_name}(nn.Module):

    def __init__(self,num_classes=100) -> None:
        super().__init__()
        block = Registers.block['{block}']
        stem = Registers.block['{stem}']
        downsample = Registers.block['{downsample}']
        self.stem = stem(in_channels=3,out_channels={widths[0]})
        layers = []
        for _ in range({depths[0]}):
            layers.append(block(in_channels={widths[0]},out_channels={widths[0]}))
        self.layer1 = nn.Sequential(*layers)
        self.downsample1 = downsample(in_channels={widths[0]},out_channels={widths[1]})
        layers = []
        for _ in range({depths[1]}):
            layers.append(block(in_channels={widths[1]},out_channels={widths[1]}))
        self.layer2 = nn.Sequential(*layers)
        self.downsample2 = downsample(in_channels={widths[1]},out_channels={widths[2]})
        layers = []
        for _ in range({depths[2]}):
            layers.append(block(in_channels={widths[2]},out_channels={widths[2]}))
        self.layer3 = nn.Sequential(*layers)
        layers = []
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear({widths[2]},num_classes)

    def forward(self,x):
        h = self.stem(x)
        h = self.layer1(h)
        h = self.downsample1(h)
        h = self.layer2(h)
        h = self.downsample2(h)
        h = self.layer3(h)
        h = self.avg_pool(h)
        h = h.view(h.size(0), -1)
        h = self.fc(h)
        return h

"""
        elif self.mode == 'darts':
            if self.dataset in ['cifar10','cifar100']:
                model_code = MODEL_CODE_DARTS_CIFAR.format(register_path=self.register_path,model_name=model_name,block=block,stem=stem,downsample=downsample,widths=widths,depths=depths)
        with open(save_file_path,'w') as f:
            f.write(model_code)
        return widths,depths

    def cal_params_flops(self,block_name,input_shape=(1,3,224,224)):
        if self.dataset=='imagenet-1k':
            input_shape=(1,3,224,224)
        elif self.dataset in ['cifar10','cifar100','nas-bench-201-cifar10','nas-bench-201-cifar100']:
            input_shape=(1,3,32,32)
        elif self.dataset=='nas-bench-201-imagenet16-120':
            input_shape=(1,3,16,16)
        else:
            raise NotImplementedError
        import_one_modules_for_register(block_name,package=self.package)
        model = Registers.model[block_name]()
        flops, params = profile(model,(torch.randn(input_shape),),verbose=False)
        flops = '{:.4f}'.format(flops/(1000**3))
        params = '{:.4f}'.format(params/(1000**2))
        return params,flops

    def _generate_one_staged(self,model_name,base_blocks,stem,downsample,save_file_path):
        """
        Generate model code with stage-specific base blocks.
        base_blocks: list of block registry names, one per stage
        """
        # Use first block for width/depth profiling
        try:
            widths,depths = self.get_width_and_depth(base_blocks,stem,downsample)
        except Exception as e:
            if 'group' in str(e):
                for g in [8,16,32,64,128]:
                    try:
                        widths,depths = self.get_width_and_depth(base_blocks,stem,downsample,groups=g)
                        break
                    except:
                        continue
            else:
                raise e

        num_stages = len(base_blocks)
        if self.mode == 'nas-bench':
            if self.dataset.startswith('nas-bench-201') and num_stages == 3:
                # Generate block assignment lines
                block_assigns = '\n'.join(
                    [f"        block_s{i+1} = Registers.block['{base_blocks[i]}']" for i in range(num_stages)]
                )
                model_code = f"""import torch.nn as nn
from {self.register_path} import Registers

@Registers.model
class {model_name}(nn.Module):

    def __init__(self,num_classes=100) -> None:
        super().__init__()
{block_assigns}
        stem = Registers.block['{stem}']
        downsample = Registers.block['{downsample}']
        self.stem = stem(in_channels=3,out_channels={widths[0]})
        layers = []
        for _ in range({depths[0]}):
            layers.append(block_s1(in_channels={widths[0]},out_channels={widths[0]}))
        self.layer1 = nn.Sequential(*layers)
        self.downsample1 = downsample(in_channels={widths[0]},out_channels={widths[1]})
        layers = []
        for _ in range({depths[1]}):
            layers.append(block_s2(in_channels={widths[1]},out_channels={widths[1]}))
        self.layer2 = nn.Sequential(*layers)
        self.downsample2 = downsample(in_channels={widths[1]},out_channels={widths[2]})
        layers = []
        for _ in range({depths[2]}):
            layers.append(block_s3(in_channels={widths[2]},out_channels={widths[2]}))
        self.layer3 = nn.Sequential(*layers)
        layers = []
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear({widths[2]},num_classes)

    def forward(self,x):
        h = self.stem(x)
        h = self.layer1(h)
        h = self.downsample1(h)
        h = self.layer2(h)
        h = self.downsample2(h)
        h = self.layer3(h)
        h = self.avg_pool(h)
        h = h.view(h.size(0), -1)
        h = self.fc(h)
        return h

"""
            elif self.dataset == 'imagenet-1k' and num_stages == 4:
                block_assigns = '\n'.join(
                    [f"        block_s{i+1} = Registers.block['{base_blocks[i]}']" for i in range(num_stages)]
                )
                model_code = f"""import torch.nn as nn
from {self.register_path} import Registers

@Registers.model
class {model_name}(nn.Module):

    def __init__(self,num_classes=100) -> None:
        super().__init__()
{block_assigns}
        stem = Registers.block['{stem}']
        downsample = Registers.block['{downsample}']
        self.stem = stem(in_channels=3,out_channels={widths[0]})
        layers = []
        for _ in range({depths[0]}):
            layers.append(block_s1(in_channels={widths[0]},out_channels={widths[0]}))
        self.layer1 = nn.Sequential(*layers)
        self.downsample1 = downsample(in_channels={widths[0]},out_channels={widths[1]})
        layers = []
        for _ in range({depths[1]}):
            layers.append(block_s2(in_channels={widths[1]},out_channels={widths[1]}))
        self.layer2 = nn.Sequential(*layers)
        self.downsample2 = downsample(in_channels={widths[1]},out_channels={widths[2]})
        layers = []
        for _ in range({depths[2]}):
            layers.append(block_s3(in_channels={widths[2]},out_channels={widths[2]}))
        self.layer3 = nn.Sequential(*layers)
        self.downsample3 = downsample(in_channels={widths[2]},out_channels={widths[3]})
        layers = []
        for _ in range({depths[3]}):
            layers.append(block_s4(in_channels={widths[3]},out_channels={widths[3]}))
        self.layer4 = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear({widths[3]},num_classes)

    def forward(self,x):
        h = self.stem(x)
        h = self.layer1(h)
        h = self.downsample1(h)
        h = self.layer2(h)
        h = self.downsample2(h)
        h = self.layer3(h)
        h = self.downsample3(h)
        h = self.layer4(h)
        h = self.avg_pool(h)
        h = h.view(h.size(0), -1)
        h = self.fc(h)
        return h

"""
            else:
                raise NotImplementedError(f"Staged generation not supported for {self.dataset} with {num_stages} stages")
        else:
            raise NotImplementedError(f"Staged generation not supported for mode={self.mode}")

        with open(save_file_path,'w') as f:
            f.write(model_code)
        return widths,depths

    def generate_one(self,id,mode='nas-bench'):
        with open(self.pairs_path,'r') as f:
            ds = json.load(f)
        model_name = id.replace('block','model')
        if model_name in self.load_annos():
            return model_name
        save_file_path = os.path.join(self.model_dir,f'{model_name}.py')
        block_info = ds[id]
        base = block_info['base']

        if isinstance(base, list):
            # Staged: multiple base blocks
            widths,depths = self._generate_one_staged(model_name,base,block_info['stem'],block_info['downsample'],save_file_path)
        else:
            # Legacy: single base block
            widths,depths = self._generate_one(model_name,base,block_info['stem'],block_info['downsample'],save_file_path)

        params,flops = self.cal_params_flops(model_name)
        d = {
            'blocks':{'base':base,'stem':block_info['stem'],
            'downsample':block_info['downsample']},
            'widths':widths,
            'depths':depths,
            'params':float(params),
            'flops':float(flops)
        }
        self.update_anno(model_name,d)
        return model_name
        

    def generate_all(self):
        with open(self.pairs_path,'r') as f:
            ds = json.load(f)
        for id in ds:
            self.generate_one(id)