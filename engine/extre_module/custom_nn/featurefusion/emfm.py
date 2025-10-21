import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')

import warnings     
warnings.filterwarnings('ignore')
from calflops import calculate_flops  

import torch   
import torch.nn as nn
import torch.nn.functional as F  
 
from engine.extre_module.ultralytics_nn.conv import Conv

class EMFM(nn.Module):
    def __init__(self, inc, dim, reduction=8): 
        super(EMFM, self).__init__()   

        self.height = len(inc)   
        d = max(int(dim/reduction), 4) 
        e_channel = self.height * dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),   
            nn.ReLU(), 
            nn.Conv2d(d, dim * self.height, 1, bias=False)
        ) 
 
        self.softmax = nn.Softmax(dim=1)  
        self.econv = nn.ModuleList([
            Conv(e_channel, e_channel //2),
            Conv(e_channel // 2, dim)
        ])
        self.conv = Conv(2 * dim,  dim)
        self.conv1x1 = nn.ModuleList([])
        for i in inc:     
            if i != dim:
                self.conv1x1.append(Conv(i, dim, 1))
            else:
                self.conv1x1.append(nn.Identity())

    def forward(self, in_feats_):
        in_feats = []
        for idx, layer in enumerate(self.conv1x1):
            in_feats.append(layer(in_feats_[idx])) 
    
        B, C, H, W = in_feats[0].shape 

        e_feat = in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)    
    
        feats_sum = torch.sum(in_feats, dim=1)     
        attn = self.mlp(self.avg_pool(feats_sum))  
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))
     
        out = torch.sum(in_feats*attn, dim=1) 
        for layer in self.econv:
            e_feat = layer(e_feat)
        out = torch.cat((out, e_feat), dim=1) 
        out = self.conv(out)  


        return out     
 
if __name__ == '__main__':
    RED, GREEN, BLUE, YELLOW, ORANGE, RESET = "\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[38;5;208m", "\033[0m"   
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')     
    batch_size, channel_1, channel_2, height, width = 1, 32, 16, 32, 32   
    ouc_channel = 64   
    inputs_1 = torch.randn((batch_size, channel_1, height, width)).to(device)     
    inputs_2 = torch.randn((batch_size, channel_2, height, width)).to(device)
    
    module = EMFM([channel_1, channel_2], ouc_channel).to(device)

    outputs = module([inputs_1, inputs_2])
    print(GREEN + f'inputs1.size:{inputs_1.size()} inputs2.size:{inputs_2.size()} outputs.size:{outputs.size()}' + RESET)     

    print(ORANGE)   
    flops, macs, _ = calculate_flops(model=module,
                                     args=[[inputs_1, inputs_2]],
                                     output_as_string=True,    
                                     output_precision=4,     
                                     print_detailed=True) 
    print(RESET) 