from torch.autograd import Variable
import torch
from torch import nn
from collections import OrderedDict
import math
import numpy as np

def compute_integral_part(input, overflow_rate):
    abs_value = input.abs().view(-1)
    sorted_value = abs_value.sort(dim=0, descending=True)[0]
    split_idx = int(overflow_rate * (len(sorted_value)/2))
    v = sorted_value[split_idx]
    if isinstance(v, Variable):
        v = v.data.cpu().numpy()
    #sf = math.ceil(math.log2(v+1e-12))
    sf=v
    return sf

def linear_quantize(input, sf, bits):
    assert bits >= 1, bits
    if bits == 1:
        return (torch.sign(input) - 0.5)*sf
    #delta = math.pow(2.0, -sf)
    bound = math.pow(2.0, bits-1)
    delta = sf/bound
    min_val = - bound
    max_val = bound - 1
    rounded = torch.floor(input / delta + 0.5)

    clipped_value = torch.clamp(rounded, min_val, max_val) * delta
    return clipped_value

def quant_model_bit(model,Bits):
    state_dict_quant = model.state_dict()
    sf_list=[]
    snr_dict={}
    kk=0  
    for name, module in model.named_modules():
      if isinstance(module, nn.Conv2d):
        if isinstance(Bits,int):
            bits=Bits
        else:
            bits=Bits[kk]
        kk=kk+1
        v=state_dict_quant[name+'.weight']
        nornow=1e10
        vq = v
        for clip in [0, 0.005,0.01,0.02]:
            sf =  compute_integral_part(v, overflow_rate=clip)
            v_quant  = linear_quantize(v, sf, bits=bits)
            norv=(v_quant-v).norm()
            if nornow > norv:
                vq = v_quant
                nornow = norv
                sf_t=sf
        sf_list.append(sf_t)
        state_dict_quant[name+'.weight'] = vq
        error_noise=torch.norm(vq-v)
        par_power=torch.norm(v)
        snr=error_noise/par_power
        snr_dict[name]=snr
    model.load_state_dict(state_dict_quant)
    return model, sf_list, snr_dict

def changemodelbit_fast(Bits,model,sf_list):
    state_dict_quant = model.state_dict()
    kk=0    
    for name, module in model.named_modules():
      if isinstance(module, nn.Conv2d):
        v=state_dict_quant[name+'.weight']
        if isinstance(Bits,int):
            bits=Bits
        else:
            bits=Bits[kk]
        sf = 6. if sf_list is None else sf_list[kk]  #quant.compute_integral_part(v, overflow_rate=clip)
        v_quant  = linear_quantize(v, sf, bits=bits)
        kk=kk+1     
        state_dict_quant[name+'.weight'] = v_quant     
    model.load_state_dict(state_dict_quant)

#########################################Activation
def __PintQuantize__(num_bit, tensor, clamp_v):
    qmin = 0.
    qmax = 2. ** num_bit - 1.
    scale = clamp_v / (qmax - qmin)
    scale = torch.max(scale.to(tensor.device), torch.tensor([1e-8]).to(tensor.device))
    output = tensor.detach()
    output = output.div(scale)
    output = output.clamp(qmin, qmax).round()  # quantize
    output = output.mul(scale)  # dequantize

    return output.view(tensor.shape)


class LinearQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bit, clamp):

        output = __PintQuantize__(bit, input, clamp)
        #ctx.mark_dirty(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        return grad_output, None, None


class QuantReLU(nn.ReLU):
    def __init__(self, bit=8, channelnum=1):
        super(QuantReLU, self).__init__()
        self.quant_bit = bit
        self.register_buffer('statistic_max', torch.ones(1))
        self.momentum = 0.1
        self.running = False
        self.numact = 0

    def forward(self, input):
        out = nn.functional.relu(input)
        if self.running and (not self.training):
            size = input.size()
            self.numact = input.numel()
            Max = out.max()
            self.statistic_max.data = torch.max(self.statistic_max.data, Max)
        else:
            out = LinearQuantizeSTE.apply(out, self.quant_bit, self.statistic_max.data)

        return out

class DynamicQuantReLU(nn.ReLU):
    def __init__(self, bit=8, channelnum=1):
        super(DynamicQuantReLU, self).__init__()
        self.quant_bit = bit
        self.clamp = torch.FloatTensor([6.0])
        self.numact = 0

    def forward(self, input):
        out = nn.functional.relu(input)
        self.clamp = out.max()
        self.numact = max(input.numel(),self.numact)
        out = LinearQuantizeSTE.apply(out, self.quant_bit, self.clamp)

        return out


import math

def quant_relu_module(m,n_dict,pre=''):
    children = list(m.named_children())
    c = None
    cn = None
    if  pre=='module':
        pre=''
    if pre:
        pre=pre+'.'

    for name, child in children:
        if isinstance(child, nn.ReLU) or isinstance(child, nn.ReLU6):
            if c == None:
                assert False
                continue

            noir=n_dict[pre+cn]

            bit = round(math.log2(3.)-math.log2(noir)) #8# if 'extr' in pre else round(math.log2(3)-math.log2(noir))
            bit = min(8,bit)
            m._modules[name] = QuantReLU(bit,c.out_channels)
            print((pre+name, bit))
            c = None
        elif isinstance(child, nn.Conv2d):
            c = child
            cn = name
        else:
            quant_relu_module(child,n_dict,pre+name)

def quant_relu_module_bit(m,bit,pre=''):
    children = list(m.named_children())
    c = None
    if pre:
        pre=pre+'.'

    for name, child in children:
        if (isinstance(child, nn.ReLU) or isinstance(child, nn.ReLU6)) and 'layer' in pre:
            if c == None:
                assert False
                continue

            m._modules[name] = QuantReLU(bit,c.out_channels)#DynamicQuantReLU
            print((pre+name, bit))
            c = None
        elif isinstance(child, nn.Conv2d):
            c = child
        else:
            quant_relu_module_bit(child,bit,pre+name)


def dequant_relu_module(m):
    children = list(m.named_children())

    for name, child in children:
        if isinstance(child, QuantReLU) or isinstance(child, DynamicQuantReLU):
            m._modules[name] = nn.ReLU()
        elif isinstance(child, nn.Conv2d):
            continue
        else:
            dequant_relu_module(child)


def running_module(m):
    children = list(m.named_modules())

    for name, mc in children:
        if isinstance(mc, QuantReLU):
            mc.running = True
            mc.training = False


def test_module(m):
    children = list(m.named_modules())

    for name, mc in children:
        if isinstance(mc, QuantReLU):
            mc.running = False

def act_averagebit(model):
    children = list(model.named_modules())
    totalbit=0
    totalact=0

    for name, mc in children:
        if isinstance(mc, QuantReLU) or isinstance(mc, DynamicQuantReLU):
            totalact += mc.numact
            totalbit += mc.numact*mc.quant_bit
    return totalbit/totalact

