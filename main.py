import os
import sys
import random
import numpy as np
from models.LMClass import LMClass
import torch
import time
from datautils import get_loaders
import multiprocessing

from parallel_utils import get_lowest_occupied_gpu
import torch.nn as nn
from quantize.sliderquant import sliderquant
from tqdm import tqdm
import utils
from pathlib import Path
import accelerate
from accelerate import Accelerator

from models.rotation_utils import get_rotate_model

import pdb
import argparse
import yaml

import torch.distributed as dist
from quantize.utils import evaluate

torch.backends.cudnn.benchmark = True


gpu_counter = 0

net_choices = [
    "llama-7b",
    "llama-13b",
    "llama-30b",
    "llama-65b",
    "Llama-2-7b",
    "Llama-2-13b",
    "Llama-2-70b",
    "Llama-3-8B",
    "Qwen2.5-7B",
]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,help="use json to config")
    parser.add_argument("--quant_gate", default=False, action="store_true", help="quantize MoE router gates")
    parser.add_argument("--update_gate", default=False, action="store_true", help="use 16-bit gate weights/activations when updating MoE router gates")
    parser.add_argument("--model", type=str, help="path or name of the base model to load")
    parser.add_argument("--teach_model", type=str, default=None, help="path or name of the teacher model used to generate distillation targets")
    parser.add_argument("--cache_dir", default="./cache", type=str, help="cache dir of dataset, leading to faster debug")
    parser.add_argument("--output_dir", default="../log/", type=str, help="direction of logging file")
    parser.add_argument("--export_model_path", default=None, type=str, help="direction for saving fp16 model which is more friendly to quantize.")
    parser.add_argument("--export_model_mode", default="fp16", type=str, choices=["fp16", "quant"], help="export the fp16 model or the quantized model")
    parser.add_argument("--resume", type=str, default=None, help="path to a checkpoint used to resume quantization or evaluation")
    parser.add_argument("--quant_step", type=int, default=2, help="number of progressive quantization stages")
    parser.add_argument("--train_resume", type=str, default=None, help="path to a slider_parameters.pth checkpoint for training resume")
    parser.add_argument("--gqa_scales", choices=["copy", "mean"], default="copy", help="how to initialize grouped-query attention scales")
    parser.add_argument("--layers_assigned_gpu", type=str, default=None, help="comma-separated layer-to-GPU assignment for training")
    parser.add_argument("--test_datasets", type=str, default="wikitext2,c4", help="comma-separated evaluation datasets")
    parser.add_argument("--circular_aug", default=False, action="store_true", help="reuse the final quantization stage as an extra augmentation round")
    parser.add_argument("--loss_type", default="mean", type=str, choices=["mean", "add"], help="loss aggregation mode across windows")
    parser.add_argument("--use_bfloat16", default=False, action="store_true", help="run the pipeline in bfloat16")
    parser.add_argument("--fp16_act", default=False, action="store_true", help="cache activations in fp16 to reduce CPU memory")
    parser.add_argument("--debug", default=False, action="store_true", help="enable debug mode")
    parser.add_argument("--low_memory", default=True, action="store_true", help="reduce GPU memory use by staging tensors")
    parser.add_argument("--low_cpu_memory", default=False, action="store_true", help="reduce CPU memory use when caching calibration data")
    parser.add_argument('--lm_eval_batch_size', type=str, default="64", help='batch size used by lm-eval-harness')
    parser.add_argument("--num_layer", type=int, default=1, help="number of consecutive layers in each sliding window")
    parser.add_argument("--sliding_layer", type=int, default=None, help="sliding stride between neighboring windows")
    parser.add_argument("--load_rotate_model_path", type=str, default=None, help="path to the rotated fp16 checkpoint used for initialization")
    parser.add_argument("--lora_rank", type=int, default=4, help="LoRA rank for adapted layers")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument("--quant_layer_list", type=str, default=None, help="automatically generated list of quantized layers; manual input is not supported")
    parser.add_argument("--use_lora", default=False, action="store_true", help="enable LoRA adaptation during quantization training")
    parser.add_argument("--use_ddp", default=False, action="store_true", help="run training with DistributedDataParallel")
    parser.add_argument("--use_down_scale", default=True, action="store_true", help="apply down-projection scaling smoothing when supported")
    parser.add_argument("--seqlen", type=int, default=2048, help="maximum sequence length for calibration and evaluation")
    parser.add_argument("--wo_lwc", default=False, action="store_true", help="disable learnable weight clipping")
    parser.add_argument("--lora_layer_list", type=str, default=None, help="automatically generated list of LoRA layers; manual input is not supported")
    parser.add_argument("--lora_iter_num_list", type=str, default=None, help="automatically generated per-layer LoRA iteration counts; manual input is not supported")
    parser.add_argument("--lora_r_list", type=str, default=None, help="automatically generated per-layer LoRA ranks; manual input is not supported")
    parser.add_argument("--lora_quant", default=True, action="store_true", help="quantize LoRA weights during training")
    parser.add_argument("--use_lr_scheduler", default=True, action="store_true", help="enable the learning-rate scheduler")
    parser.add_argument("--quant_rate_list", type=float, nargs='+', default=None, help="quantization ratios to sweep across stages")
    parser.add_argument("--quant_rate", type=float, default=1.0, help="current quantization ratio for the active stage")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="fraction of steps used for LR warmup")
    parser.add_argument("--quant_mode_layer_list", type=str, default=None, help="automatically generated per-layer quantization mode map; manual input is not supported")
    parser.add_argument("--quant_mode", type=str, default="slider", help="layer quantization mode name, such as slider, fp16, lora_only, or weight_merge")
    parser.add_argument("--layer_windows_scheduler", type=str, default=None, help="explicit layer-window schedule, e.g. 0-3,2-5")
    parser.add_argument("--littlt_bs_round", type=str, default=None, help="comma-separated round indices that use batch size 1")
    parser.add_argument("--calib_dataset",type=str,default="wikitext2",
        # choices=["wikitext2", "ptb", "c4", "mix","pile"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument("--weight_merge", default=False, action="store_true", help="merge weights before evaluation or export")
    parser.add_argument("--resume_layers_num", type=int, default=None, help="number of layers to restore when resuming from a checkpoint")
    parser.add_argument("--start_round", type=int, default=0, help="starting round index when resuming training")
    parser.add_argument("--nsamples", type=int, default=128, help="number of calibration samples")
    parser.add_argument("--batch_size", type=int, default=1, help="training batch size for quantization optimization")
    parser.add_argument("--inference_batch_size", type=int, default=1, help="batch size used to generate cached activations")
    parser.add_argument("--seed", type=int, default=2, help="random seed for calibration and training")
    parser.add_argument("--tasks", default="", help="comma-separated downstream evaluation tasks")
    parser.add_argument("--test_mode", default=False, action="store_true", help="skip training and only run evaluation/export flow")
    parser.add_argument("--eval_ppl", default=False, action="store_true", help="evaluate perplexity after quantization")
    parser.add_argument("--wbits", type=int, default=4, help="weight bit-width")
    parser.add_argument("--abits", type=int, default=16, help="activation bit-width")
    parser.add_argument("--group_size", type=int, default=None, help="weight quantization group size; None means per-channel")
    parser.add_argument("--fill_window_size", type=int, default=None, help="window size used for progressive layer scheduling")
    parser.add_argument("--grad_clip", default=None, type=float, help="maximum gradient norm for clipping; None disables clipping")
    parser.add_argument("--loss_function", default="mse", type=str, choices=["mse", "huber"], help="loss function used for training rounds")
    parser.add_argument("--alpha", type=float, default=0.5, help="mixing factor for the current loss setup")
    parser.add_argument("--auto_lr_scale", default=True, action="store_true", help="automatically scale learning rates by the active quantization ratio")
    parser.add_argument("--scale_lr", type=float, default=5e-3, help="learning rate for learned scale parameters")
    parser.add_argument("--lora_lr", type=float, default=0, help="learning rate for LoRA parameters")
    parser.add_argument("--lwc_lr", type=float, default=1e-2, help="learning rate for learnable weight clipping parameters")
    parser.add_argument("--epochs", type=int, default=10, help="number of training epochs per quantization stage")
    parser.add_argument("--let", default=True, action="store_true", help="enable learnable equivalent transformation")
    parser.add_argument("--lwc", default=True, action="store_true", help="enable learnable weight clipping")
    parser.add_argument("--quant_warp", default=False, action="store_true", help="force the quantization workflow even when wbits and abits are both 16")
    parser.add_argument("--train_last_round", default=False, action="store_true", help="only train the final quantization round")
    parser.add_argument("--use_fp_inp_loss", default=False, action="store_true", help="include the fp16-input auxiliary loss")
    parser.add_argument("--use_base_loss", default="last", type=str, choices=["none", "last", "all"], help="which base-loss windows to include")
    parser.add_argument("--use_quant_tar_loss", default=False, action="store_true", help="include the quant-target auxiliary loss")
    parser.add_argument("--last_round_inp_num", default=1, type=int, help="number of previous-round activations cached for alignment losses")
    parser.add_argument("--symmetric", default=False, action="store_true", help="use symmetric weight quantization")
    parser.add_argument("--disable_zero_point", default=False, action="store_true", help="disable zero-point in quantization")
    parser.add_argument("--a_dynamic_method", type=str, default="per_token", choices=["per_token"], help="activation quantization dynamic method")
    parser.add_argument("--w_dynamic_method", type=str, default="per_channel", choices=["per_channel"], help="weight quantization dynamic method")
    parser.add_argument("--limit", type=int, default=-1, help="limit the number of evaluation samples; -1 means no limit")
    parser.add_argument("--multigpu", default=False, action="store_true", help="map the model across multiple GPUs for evaluation")
    parser.add_argument("--parallelize", default=False, action="store_true", help="enable model parallel evaluation")
    parser.add_argument("--deactive_amp", default=False, action="store_true", help="disable AMP when 8-bit quantization is used")
    parser.add_argument(
        "--attn_implementation",
        type=str, required=False, default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="attention implementation that the model works with",
    )
    parser.add_argument("--net", type=str, default=None, choices=net_choices, help="model family name used for layer handling and cache naming")
    parser.add_argument("--act_symmetric", default=False, action="store_true", help="use symmetric activation quantization")

    args = parser.parse_args()

    if args.config is not None:
        # 使用config文件来保存
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
            for key, value in config.items():
                if key not in ["debug","output_dir","weight_merge","test_mode","tasks","eval_ppl"]:
                    parser.set_defaults(**{key: value})
        args = parser.parse_args()
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        args_dict = vars(args)
        args_yaml = yaml.dump(args_dict)
        with open( output_dir/'args.yaml', 'w') as f:
            f.write(args_yaml)
    return args

def setup_ddp():
    dist.init_process_group(
        backend="nccl",
        init_method="env://",  # 使用 torchrun 自动设置的环境变量
    )

def get_quant_model(args):
   
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    if args.epochs > 0:
        assert args.lwc or args.let
        
    if (args.wbits<16 and args.wbits>=8) or (args.abits<16 and args.abits>=8):
        args.deactive_amp = True

    # init logger
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)

    
    if args.use_ddp:
        rank = dist.get_rank()
    else:
        rank = 0
    logger = utils.create_logger(output_dir,dist_rank=rank)
    logger.info(args)
    
    if args.net is None:
        args.net = args.model.split('/')[-1]
    args.model_family = args.net.split('-')[0]
    lm = LMClass(args)
    if args.load_rotate_model_path is not None:
        if rank == 0:
            rotate_model_dir = os.path.dirname(args.load_rotate_model_path)
            if rotate_model_dir:
                os.makedirs(rotate_model_dir, exist_ok=True)
            if os.path.exists(args.load_rotate_model_path) is False:
                save_dict = get_rotate_model(lm.model,args.load_rotate_model_path)
                lm = LMClass(args)
                del save_dict
        if args.use_ddp:
            dist.barrier()
        save_dict = torch.load(args.load_rotate_model_path, map_location="cpu")
        lm.model.load_state_dict(save_dict["model"])
        logger.info(f"load fp16 model from {args.load_rotate_model_path}")
        del save_dict
    lm.seqlen = args.seqlen
    lm.model.eval()
    for param in lm.model.parameters():
        param.requires_grad = False
    args.weight_quant_params = {
        "n_bits": args.wbits,
        "per_channel_axes": [0],
        "symmetric": args.symmetric,
        "dynamic_method": args.w_dynamic_method,
        "group_size": args.group_size,
        "lwc":args.lwc,
        "disable_zero_point": args.disable_zero_point,
    }
    args.act_quant_params = {
        "n_bits":  args.abits,
        "per_channel_axes": [],
        "symmetric": args.act_symmetric,
        "dynamic_method": args.a_dynamic_method,
    }
    args.q_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.k_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.v_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.p_quant_params = {
        "n_bits": 16,
        "metric": "fix0to1",
    }

    if args.multigpu:
        gpu_id = get_lowest_occupied_gpu(wait_memory=5000)
        lm._device = f"cuda:{gpu_id}"
        logger.info(f"set quantization in gpu {gpu_id}")



    if args.wbits < 16 or args.abits <16 or args.quant_warp:
        logger.info("=== start quantization ===")
        tick = time.time()     
        if args.seqlen != 2048:
            cache_dataloader = f'{args.cache_dir}/dataloader_{args.net}_{args.calib_dataset}_{args.nsamples}_{args.seqlen}.cache'
        else:
            cache_dataloader = f'{args.cache_dir}/dataloader_{args.net}_{args.calib_dataset}_{args.nsamples}.cache'
        if os.path.exists(cache_dataloader):
            dataloader = torch.load(cache_dataloader)
            logger.info(f"load calibration from {cache_dataloader}")
        else:
            if "," in args.calib_dataset:
                calib_dataset_list = args.calib_dataset.split(",")
            else:
                calib_dataset_list = [args.calib_dataset]
            dataloader_list = []
            
            calib_nsamples_list = [ args.nsamples // len(calib_dataset_list) for _ in range(len(calib_dataset_list))] 

            for calib_dataset,calib_nsamples in zip(calib_dataset_list,calib_nsamples_list):
                _dataloader, _ = get_loaders(
                    calib_dataset,
                    nsamples=calib_nsamples,
                    seed=args.seed,
                    model=args.model,
                    seqlen=lm.seqlen,
                    args=args,
                )
                dataloader_list.append(_dataloader)
            dataloader = []
            for _dataloader in dataloader_list:
                dataloader.extend(_dataloader)
            random.shuffle(dataloader)
            torch.save(dataloader, cache_dataloader)    

        if args.teach_model is not None:
            teach_lm = LMClass(args,args.teach_model)
        else:
            teach_lm = None
        if args.quant_mode in ["fp16"]:
            args.resume = None

        _,inps,infer_attention_mask,position_ids = sliderquant(
            lm,
            args,
            dataloader,
            logger,
            teach_lm,
        )
        logger.info(f"total use time: {(time.time() - tick)/3600:.2f}h")
    else:
        inps,infer_attention_mask,position_ids= None,None,None
    
    return lm,logger,inps,infer_attention_mask,position_ids
    
def main():
    multiprocessing.set_start_method('spawn')
    
    args = parse_arguments()
    if args.use_ddp is True:
        setup_ddp()
    lm,logger,_,_,_ = get_quant_model(args=args)
        
    if args.export_model_path:
        logger.info("export model")
        ori_lm = LMClass(args)
        new_model_weights = lm.model.state_dict()
        hf_model_path = os.path.join(args.export_model_path,f"{args.net}-SliderQuant")
        ori_lm.model.load_state_dict(new_model_weights,strict=False)
        ori_lm.model.save_pretrained(hf_model_path)
        ori_lm.tokenizer.save_pretrained(hf_model_path)

    evaluate(lm, args,logger)

    


if __name__ == "__main__":
    print(sys.argv)
    main()
