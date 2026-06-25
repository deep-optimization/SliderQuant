import torch
import typing
from quantize import utils
import transformers
import tqdm, math
from models.hadamard_utils import random_hadamard_matrix, matmul_hadU, is_pow2
from dataclasses import dataclass

OPT_MODEL = transformers.models.opt.modeling_opt.OPTForCausalLM
OPT_LAYER = transformers.models.opt.modeling_opt.OPTDecoderLayer
LLAMA_MODEL = transformers.models.llama.modeling_llama.LlamaForCausalLM
LLAMA_LAYER = transformers.models.llama.modeling_llama.LlamaDecoderLayer
QWEN_MODEL =  transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM
QWEN_LATER = transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer


DEV = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

class RMSN(torch.nn.Module):
    """
    This class implements the Root Mean Square Normalization (RMSN) layer.
    We use the implementation from LLAMARMSNorm here:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L75
    """

    def __init__(self, mean_dim: int,ori_norm, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mean_dim = mean_dim
        if ori_norm:
            self.weight = torch.nn.Parameter(torch.ones_like(ori_norm.weight))
            if hasattr(ori_norm,"bias") and ori_norm.bias is not None:
                self.bias = torch.nn.Parameter(torch.zeros_like(ori_norm.bias))
        else:
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)



def replace_modules(
    root: torch.nn.Module,
    type_to_replace,
    new_module_factory,
    replace_layers: bool,
) -> None:
    """Replace modules of given type using the supplied module factory.

    Perform a depth-first search of a module hierarchy starting at root
    and replace all instances of type_to_replace with modules created by
    new_module_factory. Children of replaced modules are not processed.

    Args:
        root: the root of the module hierarchy where modules should be replaced
        type_to_replace: a type instances of which will be replaced
        new_module_factory: a function that given a module that should be replaced
            produces a module to replace it with.
    """
    for name, module in root.named_children():
        new_module = None
        if isinstance(module, type_to_replace):
            if replace_layers:  # layernorm_fusion.replace_layers case where transformer layers are replaced
                new_module = new_module_factory(module, int(name))
            else:  # layernorm_fusion.fuse_modules case where layernorms are fused
                new_module = new_module_factory(module)
        elif len(list(module.children())) > 0:
            replace_modules(module, type_to_replace, new_module_factory, replace_layers)

        if new_module is not None:
            setattr(root, name, new_module)

def get_model_type(model):
    if isinstance(model, OPT_MODEL):
        model_type = OPT_MODEL
    elif isinstance(model, LLAMA_MODEL):
        model_type = LLAMA_MODEL
    elif isinstance(model, QWEN_MODEL):
        model_type = QWEN_MODEL
    else:
        raise ValueError(f'Unknown model type {model}')
    return model_type


def model_type_extractor(model):
    if isinstance(model, LLAMA_MODEL):
        return LLAMA_MODEL
    elif isinstance(model, QWEN_MODEL):
        return QWEN_MODEL
    elif isinstance(model, OPT_MODEL):
        return OPT_MODEL
    else:
        raise ValueError(f'Unknown model type {model}')

def get_embeddings(model, model_type) -> list[torch.nn.Module]:
    if model_type == LLAMA_MODEL or  model_type == QWEN_MODEL :
        return [model.model.embed_tokens]
    elif model_type == OPT_MODEL:
        return [model.model.decoder.embed_tokens, model.model.decoder.embed_positions]
    else:
        raise ValueError(f'Unknown model type {model_type}')


def get_transformer_layers(model, model_type):
    if model_type == LLAMA_MODEL  or  model_type == QWEN_MODEL:
        return [layer for layer in model.model.layers]
    elif model_type == OPT_MODEL:
        return [layer for layer in model.model.decoder.layers]
    else:
        raise ValueError(f'Unknown model type {model_type}')


def get_lm_head(model, model_type):
    if model_type == LLAMA_MODEL or  model_type == QWEN_MODEL:
        return model.lm_head
    elif model_type == OPT_MODEL:
        return model.lm_head
    else:
        raise ValueError(f'Unknown model type {model_type}')




def get_pre_head_layernorm(model, model_type):
    if model_type == LLAMA_MODEL  or  model_type == QWEN_MODEL :
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm,
                          (transformers.models.llama.modeling_llama.LlamaRMSNorm,transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm))
    elif model_type == OPT_MODEL:
        pre_head_layernorm = model.model.decoder.final_layer_norm
        assert pre_head_layernorm is not None
    else:
        raise ValueError(f'Unknown model type {model_type}')
    return pre_head_layernorm

def fuse_ln_linear(layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, 'bias'):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
            linear.bias.data = linear.bias.data.double() + torch.matmul(W_, layernorm.bias.double())
            linear.bias.data = linear.bias.data.to(linear_dtype)
            
def bake_mean_into_linear(linear: torch.nn.Linear) -> None:
    """
    This function takes a linear layer and subtracts the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution which is usually done inside layernorm.
    """
    linear_dtype = linear.weight.dtype
    W_ = linear.weight.data.double()
    linear.weight.data = W_ - W_.mean(dim=-2, keepdim=True)
    linear.weight.data = linear.weight.data.to(linear_dtype)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        linear.bias.data = b_ - b_.mean()
        linear.bias.data = linear.bias.data.to(linear_dtype)

         
            
def fuse_layer_norms(model):
    
    model_type = get_model_type(model)
    
    kwargs = {'model': model, 'model_type': model_type}
    
    # Embedding fusion
    for W in get_embeddings(**kwargs):
        W_ = W.weight.data.double()
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)
        
    layers = get_transformer_layers(**kwargs)
    
    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in layers:
        
        # fuse the input layernorms into the linear layers
        if model_type == LLAMA_MODEL  or  model_type == QWEN_MODEL :
            fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])    
            fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
        elif model_type == OPT_MODEL:
            fuse_ln_linear(layer.self_attn_layer_norm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
            fuse_ln_linear(layer.final_layer_norm, [layer.fc1])
        else:
            raise ValueError(f'Unknown model type {model_type}')
            
            
    
        if model_type == OPT_MODEL:
            bake_mean_into_linear(layer.self_attn.out_proj)
            bake_mean_into_linear(layer.fc2)
                    
    
    fuse_ln_linear(get_pre_head_layernorm(**kwargs), [get_lm_head(**kwargs)])
    
    model_type2_NORM = {
        LLAMA_MODEL: transformers.models.llama.modeling_llama.LlamaRMSNorm,
        QWEN_MODEL: transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm,
        OPT_MODEL: torch.nn.LayerNorm
    }
    
    replace_modules(
        model,
        model_type2_NORM[model_type],
        lambda ori_norm: RMSN(model.config.hidden_size,ori_norm),
        replace_layers=False,
    )
    


def get_orthogonal_matrix(size, mode, device=DEV):
    return random_hadamard_matrix(size, device)
    

def rotate_embeddings(model, Q: torch.Tensor) -> None:
    # Rotate the embeddings.
    model_type = model_type_extractor(model)
    for W in get_embeddings(model, model_type):
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

    
def rotate_attention_inputs(layer, Q, model_type) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device=DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

def rotate_attention_output(layer, Q, model_type) -> None:
    # Rotate output matrix of the self-attention layer.
    if model_type == LLAMA_MODEL or  model_type == QWEN_MODEL:
        W = layer.self_attn.o_proj
    elif model_type == OPT_MODEL:
        W = layer.self_attn.out_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=DEV, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def apply_exact_had_to_linear(module, had_dim=-1, output=False):
    """把(可能分块的) Hadamard 旋转吸收进 nn.Linear 权重 (QuaRot 风格)。
    had_dim=-1: 对整维做 Hadamard; had_dim=K: 按 K 维分块做 (如 head_dim)。
    output=False: 作用于输入维(weight 的列); output=True: 作用于输出维(weight 的行)。
    """
    assert isinstance(module, torch.nn.Linear)
    dtype = module.weight.dtype
    W = module.weight.data.to(device=DEV, dtype=torch.float64)
    if had_dim == -1:
        if output:
            W = matmul_hadU(W.t()).t().contiguous()      # Hadamard on out_features
        else:
            W = matmul_hadU(W)                            # Hadamard on in_features
    else:
        assert is_pow2(had_dim), "分块 Hadamard 要求块大小为 2 的幂"
        if output:
            out_f, in_f = W.shape
            W = W.t().contiguous().reshape(in_f, out_f // had_dim, had_dim)   # 按输出维分块
            W = matmul_hadU(W).reshape(in_f, out_f).t().contiguous()
        else:
            out_f, in_f = W.shape
            W = W.contiguous().reshape(out_f, in_f // had_dim, had_dim)       # 按输入维分块
            W = matmul_hadU(W).reshape(out_f, in_f).contiguous()
    module.weight.data = W.to(device="cpu", dtype=dtype)


def rotate_mlp_input(layer, Q, model_type):
    # Rotate the MLP input weights.
    if model_type == LLAMA_MODEL or  model_type == QWEN_MODEL:
        mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    elif model_type == OPT_MODEL:
        mlp_inputs = [layer.fc1]
    else:
        raise ValueError(f'Unknown model type {model_type}')
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
    
def rotate_mlp_output(layer, Q, model_type,add_online_rotate=True):
    # Rotate the MLP output weights and bias.
    if model_type == LLAMA_MODEL or  model_type == QWEN_MODEL:
        W = layer.mlp.down_proj
    elif model_type == OPT_MODEL:
        W = layer.fc2
    else:
        raise ValueError(f'Unknown model type {model_type}')
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=DEV, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)
    if add_online_rotate:
        # SliderQuant+: down_proj 输入维 (intermediate) 的在线 Hadamard，权重侧吸收逆变换。
        # 配套 int_llama_layer MLP forward 中对激活做 matmul_hadU。
        apply_exact_had_to_linear(W, had_dim=-1, output=False)


def rotate_attention_v_o(layer, head_dim, model_type):
    # SliderQuant+: v-o 的 head_dim Hadamard，可完全吸收 (Hadamard 穿过 attention)，forward 无需改动。
    if model_type == LLAMA_MODEL or model_type == QWEN_MODEL:
        v_proj, o_proj = layer.self_attn.v_proj, layer.self_attn.o_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')
    apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)   # v 输出维分块 Hadamard
    apply_exact_had_to_linear(o_proj, had_dim=head_dim, output=False)  # o 输入维分块 Hadamard (抵消)


def rotate_head(model, Q: torch.Tensor) -> None:
    # Rotate the head.
    W = get_lm_head(model, model_type=model_type_extractor(model))
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

@torch.inference_mode()
def rotate_model(model, args,add_online_rotate=True):
    Q = get_orthogonal_matrix(model.config.hidden_size,
                                                args.rotate_mode)
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads


    model_type = model_type_extractor(model)
    rotate_embeddings(model, Q)
    rotate_head(model, Q)
    utils.cleanup_memory()
    layers = get_transformer_layers(model, model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputs(layers[idx], Q, model_type)
        rotate_attention_output(layers[idx], Q, model_type)
        rotate_mlp_input(layers[idx], Q, model_type)
        rotate_mlp_output(layers[idx], Q, model_type,add_online_rotate)
        if add_online_rotate:
            rotate_attention_v_o(layers[idx], head_dim, model_type)



def get_rotate_model(model,save_path,add_online_rotate=False):
    @dataclass
    class parm():
        rotate_mode:str
        save_qmodel_path:str
        fp32_had:bool
    args = parm("hadamard",save_path,False)
    model.to("cpu")
    model.eval()
    fuse_layer_norms(model)
    rotate_model(model, args,add_online_rotate=add_online_rotate)
    save_dict = {}
    save_dict["model"] = model.state_dict()
    torch.save(save_dict, args.save_qmodel_path)
    return save_dict