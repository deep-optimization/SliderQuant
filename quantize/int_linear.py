import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize.catq import CATQQuantizer
from quantize.quantizer import UniformAffineQuantizer





class QuantLinear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(
        self,
        org_module: nn.Linear,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
        disable_input_quant=False,
    ):
        super().__init__()
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.register_buffer('weight',org_module.weight)
        if hasattr(org_module,"bias") and  org_module.bias is not None:
            self.register_buffer('bias',org_module.bias)
        else:
            self.bias = None
            
        if hasattr(org_module,"in_features"):
            self.in_features = org_module.in_features
            self.out_features = org_module.out_features
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        self.quant_rate = 1.0
        quant_mode = weight_quant_params.get("quant_mode")
        if quant_mode == "catq":
            self.weight_quantizer = CATQQuantizer(
                org_module.weight,
                group_size=weight_quant_params["group_size"],
                init_round_thd=weight_quant_params["init_round_thd"],
                progressive_ratio=weight_quant_params["progressive_ratio"],
                s0=weight_quant_params["s0"],
            )
            if not disable_input_quant and act_quant_params["n_bits"] < 16:
                self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
            else:
                self.act_quantizer = None
        elif weight_quant_params["n_bits"] > 1:
            self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params,shape=org_module.weight.shape,is_weight_quant=True)
            if not disable_input_quant:
                self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
            else:
                self.act_quantizer = None

        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False
    
    def forward(self, input: torch.Tensor):
        if self.use_temporary_parameter:
            weight = self.weight_quantizer(self.temp_weight,self.quant_rate)
            bias = self.temp_bias
        elif self.use_weight_quant:
            weight = self.weight_quantizer(self.weight,self.quant_rate)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias

        if self.use_act_quant and not self.disable_input_quant:
            input = self.act_quantizer(input,self.quant_rate)
            
        
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)


        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False, quant_rate:float = 1.0):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        self.quant_rate = quant_rate

    def set_catq_progress(self, progress: float):
        if isinstance(self.weight_quantizer, CATQQuantizer):
            self.weight_quantizer.set_progress(progress)

    @torch.no_grad()
    def materialize_weight(self):
        if isinstance(self.weight_quantizer, CATQQuantizer):
            self.weight.copy_(self.weight_quantizer.materialize(self.weight))

