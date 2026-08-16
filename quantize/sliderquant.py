import torch
import torch.nn as nn
from models.int_llama_layer import QuantLlamaDecoderLayer
import copy
import hashlib
import json
import math
import os
from tqdm import tqdm
from train_utils import to_float,to_half
from quantize.utils import (
    get_catq_parameters,
    get_slider_parameters,
    get_lwc_parameters,
    slider_state_dict,
)

from train_utils import to_dev,obtain_teacher_output,obtain_studnet_output,replace_ori_layer,init_model,model_to_inference_mode,SubLayer
import time
from transformers import get_scheduler
from quantize.utils import cleanup_memory
from quantize.checkpoint import atomic_torch_save, window_checkpoint
from torch.utils.data import DataLoader,Dataset
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from tqdm import tqdm
from quantize.utils import evaluate

def setup_ddp():
    dist.init_process_group(
        backend="nccl",
        init_method="env://",  # 使用 torchrun 自动设置的环境变量
    )

def cleanup_ddp():
    dist.destroy_process_group()


class Quant_dataset(Dataset):
    def __init__(self,aug_quant_inps=None,aug_fp_inps=None, aug_quant_targets=None,aug_fp_targets=None,attention_masks=None,samples_num=512,windows_num=1,args=None):
        """
        In i-th round
        quant_inps: the output from (i-1)th quant model using quant_inps.
        aug_fp_inps: the output from (i-1)th fp16 model using fp_inps.
        fp_target: the output from i-th fp16 model using fp_inps.
        aug_quant_target: the output from i-th fp16 model using quant_inps.

        fp_inps --------->  [ fp16 model] ------------> fp_target
        quant_inps --------->  [ fp16 model] ------------> quant_target

        quant_inps ----->   [quant model] -------------> out1 <-> [fp_target,quant_target]
        fp_inps ----->   [quant model] -------------> out2 <->fp_target
        """
        # self.quant_inps = quant_inps
        self.samples_num = samples_num
        self.windows_num = windows_num
        assert self.windows_num == aug_quant_inps.shape[1]
        assert self.samples_num == len(aug_quant_inps)

        self.aug_quant_inps = aug_quant_inps
        self.aug_fp_targets = aug_fp_targets
        self.attention_masks = attention_masks

        if aug_fp_inps is not None:
            self.aug_fp_inps = aug_fp_inps
        else:
            self.aug_fp_inps = torch.ones(self.samples_num,self.windows_num)

        if aug_quant_targets is not None:
            self.aug_quant_targets = aug_quant_targets
        else:
            self.aug_quant_targets = torch.ones(self.samples_num,self.windows_num)

    def __len__(self):
        return self.samples_num

    def __getitem__(self, idx):
        return self.aug_quant_inps[idx],self.aug_fp_inps[idx],self.aug_quant_targets[idx],self.aug_fp_targets[idx],self.attention_masks[idx]



MB = 1024.0 * 1024.0


def masked_reconstruction_loss(output, target, token_mask, loss_func):
    target = target.to(output.device).float()
    elementwise_loss = loss_func(output.float(), target)
    valid = token_mask.to(
        output.device,
        dtype=elementwise_loss.dtype,
    ).unsqueeze(-1)
    return (elementwise_loss * valid).sum() / (
        valid.sum() * output.shape[-1]
    )


def train_one_round(r,epochs,sub_layers,layer_id_list,qdataset,cur_epochs,optimizer,lr_scheduler,attention_mask_batch,position_ids,position_embeddings,devs,args,logger,max_train_steps,init_quant_rate,fp16_type,global_start_time,total_epochs,loss_func,acts_round_idx):
    optimized_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]

    if args.use_ddp is True:
        rank = dist.get_rank()
        sub_layers = DDP(sub_layers,device_ids=[rank])
        sampler = DistributedSampler(qdataset, shuffle=True, seed=args.seed)
        shuffle = None
    else:
        rank = 0
        sampler = None
        shuffle = True
    qdataloader = DataLoader(qdataset,batch_size=args.batch_size,shuffle=shuffle,num_workers=0,pin_memory=True,sampler=sampler)



    if args.loss_type == "mean":
        if args.use_base_loss == "none":
            base_loss_num = 0
        elif args.use_base_loss == "last":
            base_loss_num = 1
        elif args.use_base_loss == "all":
            base_loss_num =  args.last_round_inp_num
        else:
            raise NotImplementedError()
        Accumulated_loss_num = (int(args.use_fp_inp_loss) + int(args.use_quant_tar_loss) ) *  args.last_round_inp_num + base_loss_num
    elif args.loss_type == "add":
        Accumulated_loss_num = 1
    else:
        raise NotImplementedError("noly support mean and add!")

    logger.info(f"Accumulated_loss_num is {Accumulated_loss_num}")

    train_layers = sub_layers.module.module if args.use_ddp else sub_layers
    train_layers.train()
    total_updates = max(epochs * len(qdataloader), 1)
    update_index = 0
    hard_stage_started = False

    for e in range(epochs):
        if args.use_ddp is True:
            sampler.set_epoch(e)
        start_time = time.time()
        epoch_losses = []
        epoch_norms = []
        epoch_losses_fp = []
        epoch_losses_quant = []
        epoch_losses_base = []
        # import ipdb;ipdb.set_trace()


        for quant_input_list,fp_input_list,quant_tar_list,fp_tar_list,token_mask in qdataloader:
            if args.quant_mode == "catq":
                progress = (update_index + 1) / total_updates
                if (
                    progress > args.progressive_ratio
                    and not hard_stage_started
                    and rank == 0
                ):
                    logger.info(
                        "CAT-Q hard stage starts at epoch %s update %s/%s",
                        e,
                        update_index,
                        total_updates,
                    )
                    hard_stage_started = True
                for layer in train_layers:
                    layer.set_catq_progress(progress)
            batch_loss = [0.0 for _ in range(quant_input_list.shape[1])]
            batch_base_loss = [0.0 for _ in range(quant_input_list.shape[1])]
            batch_fp_loss = [0.0 for _ in range(quant_input_list.shape[1])]
            batch_quant_loss = [0.0 for _ in range(quant_input_list.shape[1])]

            train_layers.zero_grad(set_to_none=True)

            for w_idx in range(quant_input_list.shape[1]):
                quant_input,fp_input,quant_tar,fp_tar = quant_input_list[:,w_idx],fp_input_list[:,w_idx],quant_tar_list[:,w_idx],fp_tar_list[:,w_idx]
                inp_list = [quant_input] if args.use_fp_inp_loss is False else [quant_input,fp_input]
                for inp_idx,inp in enumerate(inp_list):
                    if inp_idx == 0 and (args.use_base_loss == "none" or args.use_base_loss == "last" and w_idx != quant_input_list.shape[1] -1) and args.use_quant_tar_loss is False:
                        continue
                    loss_list = []

                    train_context = torch.cuda.amp.autocast(dtype=torch.bfloat16)

                    with train_context:
                        if args.use_ddp:
                            inp = inp.to(rank, non_blocking=True)
                            batch_token_mask = token_mask.to(rank, non_blocking=True)
                            out = sub_layers(inp, batch_token_mask)
                        else:
                            out = obtain_studnet_output(
                                sub_layers,[args.quant_mode_layer_list[i] for i in layer_id_list],
                                inp, token_mask, position_ids,position_embeddings, args,
                                devs=devs,return_gpu=True
                            )
                            batch_token_mask = token_mask.to(out.device)
                        def reconstruction_loss(target):
                            if args.quant_mode != "catq":
                                return loss_func(
                                    out.float(),
                                    target.to(out.device).float(),
                                )
                            return masked_reconstruction_loss(
                                out,
                                target,
                                batch_token_mask,
                                loss_func,
                            )
                        if inp_idx == 0:
                            # out is quant_out
                            if args.use_base_loss == "all" or  (w_idx == quant_input_list.shape[1] -1 and args.use_base_loss == "last"):
                                loss_base = reconstruction_loss(fp_tar)
                                loss_list.append(loss_base)
                                batch_base_loss[w_idx] += loss_base.item()
                            else:
                                loss_base = torch.tensor(0.0)

                            if args.use_quant_tar_loss is True:
                                loss_quant = reconstruction_loss(quant_tar)
                                loss_list.append(loss_quant)
                                batch_quant_loss[w_idx] += loss_quant.item()
                            else:
                                loss_quant = torch.tensor(0.0)

                        else:
                            # out is fp_out
                            if args.use_fp_inp_loss is True:
                                loss_fp = reconstruction_loss(fp_tar)
                                loss_list.append(loss_fp)
                                batch_fp_loss[w_idx] += loss_fp.item()
                            else:
                                loss_fp = torch.tensor(0.0)
                        loss = sum(loss_list) / Accumulated_loss_num

                    loss.backward()
                    batch_loss[w_idx] += loss.detach().item()


                if args.debug is True and not math.isfinite(loss.detach().item()):
                    logger.info("Loss is NAN, stopping training")
                    import ipdb;ipdb.set_trace()
                assert math.isfinite(loss.item()),"Loss is NAN, stopping training!"
            if args.grad_clip is not None:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    optimized_parameters,
                    max_norm=args.grad_clip,
                )
                # logger.info(f"Gradient norm: {total_norm:.4f} Max norm: {args.grad_clip}")

            optimizer.step()
            update_index += 1
            epoch_norms.append(0.0)
            epoch_losses.append(batch_loss)
            epoch_losses_base.append(batch_base_loss)
            epoch_losses_fp.append(batch_fp_loss)
            epoch_losses_quant.append(batch_quant_loss)

            if args.use_lr_scheduler is True:
                lr_scheduler.step()

            current_memory = torch.cuda.memory_allocated() / MB
            max_memory = torch.cuda.max_memory_allocated() / MB


        cur_epochs += 1
        epoch_mean_loss = torch.tensor(epoch_losses).mean(dim=0)
        # epoch_mean_norms = sum(epoch_norms) / len(epoch_norms)
        epoch_mean_loss_base = torch.tensor(epoch_losses_base).mean(dim=0)
        epoch_mean_loss_quant = torch.tensor(epoch_losses_quant).mean(dim=0)
        epoch_mean_loss_fp = torch.tensor(epoch_losses_fp).mean(dim=0)
        loss_str = ""
        for r_idx in range(quant_input_list.shape[1]):
            loss_str += f" loss r{acts_round_idx[r_idx]}:{epoch_mean_loss[r_idx]} "

        for r_idx in range(quant_input_list.shape[1]):
            loss_str += f" loss base r{acts_round_idx[r_idx]}:{epoch_mean_loss_base[r_idx]} "

        for r_idx in range(quant_input_list.shape[1]):
            loss_str += f" loss quant r{acts_round_idx[r_idx]}:{epoch_mean_loss_quant[r_idx]} "

        for r_idx in range(quant_input_list.shape[1]):
            loss_str += f" loss fp r{acts_round_idx[r_idx]}:{epoch_mean_loss_fp[r_idx]} "

        if rank == 0:
            logger.info(
                f"Round {r} epoch {e} {loss_str} lr:{optimizer.param_groups[0]['lr']:.8g} max memory_allocated: {max_memory}MB current memory_allocated: {current_memory}MB epoch_time: {time.time() - start_time:.2f}s use_time:{(time.time() - global_start_time)/3600:.2f}h ETA: {(time.time() - global_start_time) / cur_epochs * (total_epochs-cur_epochs) / 3600:.2f}h"
            )


    return sub_layers,cur_epochs


def sliderquant(
    lm,
    args,
    dataloader,
    logger=None,
    teach_lm=None,
):
    logger.info("Starting ...")

    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False

    if args.use_ddp:
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
    else:
        rank = 0


    if "llama" in args.net.lower() or "vicuna" in args.net.lower() or "qwen" in args.net.lower():
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1",
        }
        if args.use_down_scale is True:
            pairs["down_proj"] = "fc2"
        layer_name_prefix = "model.layers"
    else:
        raise NotImplementedError("Only llama/qwen/vicuna are kept in this open-source snapshot.")



    # import ipdb;ipdb.set_trace()
    layers[0] = layers[0].to(dev)
    fp32_type = torch.float
    fp16_type =  torch.bfloat16 if args.use_bfloat16 is True else torch.float16
    act_dtype =  fp16_type if args.fp16_act is True else fp32_type
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=act_dtype, device="cpu"
    )
    token_masks = torch.ones((args.nsamples, lm.seqlen), dtype=torch.bool)
    # import ipdb;ipdb.set_trace()
    cache = {"i": 0}

    # catch the first layer input
    class StopCapture(Exception):
        pass

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp.cpu()
            cache["i"] += 1
            if "position_embeddings" not in cache:
                cache["position_embeddings"] = kwargs["position_embeddings"]
            if self.is_llama:
                cache.setdefault("position_ids", kwargs["position_ids"])
            raise StopCapture

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            sample_index = cache["i"]
            attention_mask = (
                batch[2]
                if len(batch) > 2
                else torch.ones_like(batch[0], dtype=torch.bool)
            )
            token_masks[sample_index] = attention_mask[0].bool()
            try:
                model(batch[0].to(dev), attention_mask=attention_mask.to(dev))
            except StopCapture:
                pass

    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "llama" in args.net.lower() or "vicuna" in args.net.lower() or "qwen" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    else:
        raise NotImplementedError("Only llama/qwen/vicuna are kept in this open-source snapshot.")


    # import ipdb;ipdb.set_trace()
    inps = inps.to("cpu")
    # same input of first layer for fp model and quant model

    cleanup_memory(logger=logger)

    # import ipdb;ipdb.set_trace()
    if is_llama:
        position_ids = cache["position_ids"].cpu()
        position_embeddings = tuple(value.cpu() for value in cache["position_embeddings"])
    else:
        position_ids = None
        position_embeddings = None


    if  args.quant_mode in ["fp16"]:
        args.resume = None
    if args.resume:
        slider_parameters = torch.load(args.resume, map_location="cpu", weights_only=False)
    else:
        slider_parameters = {}

    if args.train_resume is not None and args.test_mode is False:
        training_state = torch.load(
            args.train_resume, map_location="cpu", weights_only=False
        )
        if training_state.get("__format__") == "scaleq_window_v1":
            with open(args.calib_manifest, "rb") as handle:
                calibration_hash = hashlib.sha256(handle.read()).hexdigest()
            assert training_state["model"] == args.model
            assert training_state["model_revision"] == args.model_revision
            assert (
                training_state["calib_manifest_sha256"] == calibration_hash
            ), "checkpoint AYOT manifest does not match this run"
            source_sha = os.environ.get("SCALEQ_SHA")
            if source_sha is not None:
                assert (
                    training_state["source_sha"] == source_sha
                ), "checkpoint source SHA does not match the staged repository"
            code_sha = os.environ.get("SCALEQ_CODE_SHA")
            if code_sha is not None:
                assert (
                    training_state.get("code_sha") == code_sha
                ), "checkpoint code SHA does not match the staged repository"
            slider_parameters = training_state["layers"]
            args.start_round = training_state["next_round"]
            args.resume_layers_num = max(slider_parameters) + 1
        else:
            slider_parameters = training_state
        args.resume = args.train_resume


    args.quant_layer_list = [int(layer_id) for layer_id in range(len(layers))]
    logger.info(f"these layer will quant:{args.quant_layer_list}")

    if args.use_lora is True:
        args.lora_layer_list = args.quant_layer_list  # only when quant use lora
    else:
        args.lora_layer_list = []
    logger.info(f"these layer will refine with lora:{args.lora_layer_list}")


    args.lora_iter_num_list = {layer_id:1 for layer_id in range(len(layers))}
    logger.info(f"each layer will refine with lora num iter:{args.lora_iter_num_list}")


    args.lora_r_list = {layer_id:args.lora_rank for layer_id in range(len(layers))}
    logger.info(f"each layer lora's r:{args.lora_r_list}")


    args.quant_mode_layer_list = { layer_id:(args.quant_mode if layer_id in args.quant_layer_list else "fp16") for layer_id in range(len(layers)) }



    logger.info(f"each layer quant mode:{args.quant_mode_layer_list}")

    if args.sliding_layer is None:
        args.sliding_layer = args.num_layer
    logger.info(f"sliding_layer:{args.sliding_layer}")


    init_quant_rate = args.quant_rate

    if args.quant_rate_list is None:
        args.quant_rate_list = (
            [1.0]
            if args.quant_mode == "catq"
            else np.linspace(0, 1, args.quant_step + 1).tolist()[1:]
        )
    if args.quant_mode == "catq":
        args.quant_step = 1

    logger.info(f"quant_step:{args.quant_step} quant_rate_list:{args.quant_rate_list} lora_quant:{args.lora_quant}")

    if args.test_mode is True:
        args.quant_rate = 1.0

    if  (args.resume is not None or args.train_resume is not None) and  args.resume_layers_num is None:
         args.resume_layers_num = len(layers)


    # 模型初始化
    model_attr = dict(
        is_llama=is_llama,
        pairs=pairs,
        layer_name_prefix=layer_name_prefix,
        slider_parameters=slider_parameters,
        dtype=fp32_type,
    )

    init_model(config=lm.model.config,layers=layers,args=args,DecoderLayer=DecoderLayer,model_attr=model_attr,logger=logger,dev="cpu")
    logger.info("Model Initialized")

    samples_per_rank = (
        math.ceil(args.nsamples / dist.get_world_size())
        if args.use_ddp
        else args.nsamples
    )
    num_update_steps_per_epoch = math.ceil(samples_per_rank / args.batch_size)
    global_start_time = time.time()


    cur_epochs = 0 # 已经训练的epochs

    if args.fill_window_size is not None:
        args.fill_start_window_size = args.fill_window_size
        args.fill_end_window_size = args.fill_window_size


    if args.layer_windows_scheduler is not None:
        layer_windows_scheduler = []
        for window_str in args.layer_windows_scheduler.split(","):
            layer_windows_scheduler.append([int(s) for s in window_str.split("-")])
        num_round = len(layer_windows_scheduler)
        # import ipdb;ipdb.set_trace()
        assert layer_windows_scheduler[-1][-1] == len(layers) - 1

    elif args.fill_window_size is not None:
        total_num_layers = len(layers)
        if args.fill_start_window_size is not None:
            start_layer_windows_scheduler = [list(range(i+1))  for i in range(args.fill_start_window_size)]
            start_len = (args.fill_start_window_size -  args.sliding_layer)
        else:
            start_layer_windows_scheduler = []
            start_len = 0

        if args.fill_end_window_size is not None:
            end_start_layer_windows_scheduler = [list(range(total_num_layers-args.fill_end_window_size +i,total_num_layers))  for i in range(args.fill_end_window_size)]
            end_len = (args.fill_end_window_size -  args.sliding_layer)
        else:
            end_start_layer_windows_scheduler = []
            end_len = 0
        mid_len = total_num_layers - start_len - end_len

        mid_round = math.ceil((mid_len - args.num_layer) / args.sliding_layer) + 1
        mid_layer_windows_scheduler = [
            [i for i in range(r * args.sliding_layer + start_len , min(r * args.sliding_layer + args.num_layer + start_len ,len(layers)))]
            for r in range(mid_round)
        ]
        layer_windows_scheduler = start_layer_windows_scheduler + mid_layer_windows_scheduler + end_start_layer_windows_scheduler
        num_round = len(layer_windows_scheduler)
    else:
        num_round = math.ceil((len(layers) - args.num_layer) / args.sliding_layer) + 1
        layer_windows_scheduler = [
            [i for i in range(r * args.sliding_layer, min(r * args.sliding_layer + args.num_layer,len(layers)))]
            for r in range(num_round)
        ]

    logger.info(f"layer_windows_scheduler is  {layer_windows_scheduler}")
    if args.quant_mode == "catq":
        assert layer_windows_scheduler[:4] == [
            [0],
            [0, 1],
            [0, 1, 2],
            [0, 1, 2, 3],
        ]
        assert [len(window) for window in layer_windows_scheduler[-4:]] == [4, 3, 2, 1]
        middle_windows = layer_windows_scheduler[4:-4]
        assert all(len(window) == 4 for window in middle_windows)
        assert all(
            right[0] - left[0] == 2
            for left, right in zip(middle_windows, middle_windows[1:])
        )

    if teach_lm is not None:
        teach_model = teach_lm.model
        teach_model.config.use_cache = False
        teach_layers = teach_model.model.layers
        teach_layers = teach_layers.to(dev)
        logger.info(f"Teacher model Initialized from {args.teach_model}")
    else:
        teach_layers = layers


    cleanup_memory(logger=logger)
    assert args.quant_step == len(args.quant_rate_list)


    if args.circular_aug:
        assert len(args.quant_rate_list) > 1

    if args.littlt_bs_round is not None:
        littlt_bs_round = [int(i) for i in args.littlt_bs_round.split(",")]
        littlt_bs_round = [i if i >=0 else num_round+i for i in littlt_bs_round]
    else:
        littlt_bs_round = []
    global_batch_size = args.batch_size
    for step,quant_rate in enumerate(args.quant_rate_list):
        if args.circular_aug is True and  step+1 == len(args.quant_rate_list):
            windows_quant_inps[:,-1] =    copy.deepcopy(inps)
            windows_fp_inps[:,-1]    =    copy.deepcopy(inps)
        else:
            windows_quant_inps =    copy.deepcopy(inps).unsqueeze(1).repeat(1,args.last_round_inp_num,1, 1) # if None, not need cache. if True, need but had not been cached
            windows_fp_inps    =    copy.deepcopy(windows_quant_inps) # if None, not need cache. if True, need but had not been cached

        if step+1 == args.quant_step and args.debug is False:
            inps = None
            print("delete inps!")
        cleanup_memory(logger=logger)

        if args.use_quant_tar_loss:
            windows_quant_targets = copy.deepcopy(windows_quant_inps) # if None, not need cache. if True, need but had not been cached
        else:
            windows_quant_targets = None


        windows_fp_targets =  copy.deepcopy(windows_fp_inps)

        args.quant_rate = quant_rate
        if args.low_memory is False:
            windows_quant_inps = windows_quant_inps.to(dev)
            windows_fp_inps = windows_fp_inps.to(dev)


        cleanup_memory(logger=logger)

        for r in range(num_round):

            if r in littlt_bs_round:
                args.batch_size = 1
            else:
                args.batch_size = global_batch_size
            if args.test_mode is True:
                break
            layer_id_list = layer_windows_scheduler[r]
            logger.info(f"=== Step: {step+1}/{args.quant_step}   Round: {r+1}/{num_round} ===")
            logger.info(
                f"=== Start quantize layer{layer_id_list[0]}-layer{layer_id_list[-1]} ==="
            )

            if args.loss_function == "mse":
                loss_func = torch.nn.MSELoss(
                    reduction="none" if args.quant_mode == "catq" else "mean"
                )
            elif args.loss_function == "huber":
                delta = 0.1 + r/num_round * args.huber_loss_max
                loss_func = torch.nn.HuberLoss(
                    delta=delta,
                    reduction="none" if args.quant_mode == "catq" else "mean",
                )
            else:
                raise NotImplementedError("only support mse and huber loss function")

            # del finished layers
            if args.low_cpu_memory is True and step+1 == len(args.quant_rate_list):
                for l_idx in range(layer_id_list[0]):
                    if lm.model.model.layers[l_idx] is not None:
                        lm.model.model.layers[l_idx] = torch.nn.Identity()
                # import ipdb;ipdb.set_trace()
                logger.info(f"del layer 0-{layer_id_list[0]-1}")


            sub_layers = layers[layer_id_list[0]:layer_id_list[-1]+1]
            teach_sub_layers = teach_layers[layer_id_list[0]:layer_id_list[-1]+1]


            cleanup_memory(logger=logger)
            logger.info(f"layer_id_list: {layer_id_list}")

            sub_layers = to_float(sub_layers,dtype=fp32_type)
            sub_layers = to_dev(sub_layers, [dev] * len(sub_layers))  #single gpu

            teach_sub_layers = to_float(teach_sub_layers,dtype=fp32_type)
            teach_sub_layers = to_dev(teach_sub_layers, [dev] * len(teach_sub_layers))  #single gpu


            acts_round_idx = list(range(r+1-args.last_round_inp_num,r+1))
            logger.info(f"act_round_idx is {acts_round_idx}")


            # import ipdb;ipdb.set_trace()
            if  r >= args.start_round:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(dtype=fp16_type):
                        for r_idx in range(args.last_round_inp_num):
                            # get quant_target
                            if  args.use_quant_tar_loss:
                                for j in tqdm(range(0,args.nsamples,args.inference_batch_size)):
                                    bs_local = min(args.inference_batch_size,args.nsamples-j)
                                    windows_quant_targets[j:j+bs_local,r_idx] = obtain_teacher_output(
                                        teach_sub_layers,
                                        windows_quant_inps[j:j+bs_local,r_idx],
                                            token_masks[j:j+bs_local],
                                        position_ids,
                                        position_embeddings=position_embeddings,
                                        args=args,
                                        devs=[dev] * len(teach_sub_layers),
                                    )
                                logger.info(f"finish to obtain quant_target round {acts_round_idx[r_idx]} of full-precision model!")
                            # get fp_target
                            for j in tqdm(range(0,args.nsamples,args.inference_batch_size)):
                                bs_local = min(args.inference_batch_size,args.nsamples-j)
                                windows_fp_targets[j:j+bs_local,r_idx] = obtain_teacher_output(
                                    teach_sub_layers,
                                    windows_fp_inps[j:j+bs_local,r_idx],
                                    token_masks[j:j+bs_local],
                                    position_ids,
                                    position_embeddings=position_embeddings,
                                    args=args,
                                    devs=[dev] * len(teach_sub_layers),
                                )
                            logger.info(f"finish to obtain fp_target round {acts_round_idx[r_idx]} of full-precision model!")

            cleanup_memory(logger=logger)


            epochs = args.epochs if args.quant_mode == "catq" else args.epochs // args.quant_step
            total_epochs = args.epochs*num_round


            if args.layers_assigned_gpu is not None:
                devs = [torch.device(f"cuda:{gpu}") for gpu in args.layers_assigned_gpu.split(",")]
                assert len(devs) == len(sub_layers), "layers_assigned_gpu number is not equal to layer number!"
                sub_layers = to_dev(sub_layers, devs)  #mutil-gpu
            else:
                devs = [dev] * len(sub_layers)

            max_train_steps = epochs * num_update_steps_per_epoch

            lr_factor = args.batch_size
            effective_lora_lr = (
                args.lora_lr
                if args.quant_mode == "catq"
                else args.lora_lr * lr_factor
            )
            logger.info(
                "auto lr scale is %s lora_lr is %s scale_lr is %s lwc_lr is %s",
                args.auto_lr_scale,
                effective_lora_lr,
                args.scale_lr * lr_factor,
                args.lwc_lr * lr_factor,
            )


            params = []

            if args.quant_mode == "catq":
                params.append(
                    {
                        "params": get_catq_parameters(sub_layers),
                        "lr": args.learnable_factor_lr,
                        "weight_decay": 0.0,
                    }
                )
                if args.use_lora:
                    params.append(
                        {
                            "params": get_slider_parameters(
                                sub_layers,
                                ["lora_"],
                            ),
                            "lr": args.lora_lr,
                            "weight_decay": 0.0,
                        }
                    )
            elif args.use_lora is True and (r not in littlt_bs_round or args.quant_mode == "lora_only"):
                params.append({"params":get_slider_parameters(sub_layers, ["lora_"]),"lr":args.lora_lr*lr_factor,"weight_decay":0.0})
            if args.quant_mode != "catq" and args.scale_lr > 0:
                params.append({"params":get_slider_parameters(sub_layers, ["scale"]),"lr":args.scale_lr*lr_factor,"weight_decay":0.0})
            if args.quant_mode != "catq" and args.lwc_lr > 0:
                params.append({"params":get_lwc_parameters(sub_layers),"lr":args.lwc_lr*lr_factor,"weight_decay":0.0})



            optimizer = torch.optim.AdamW(params)

            lr_scheduler = get_scheduler(
                name="linear",
                optimizer=optimizer,
                num_warmup_steps=max_train_steps*args.warmup_ratio,
                num_training_steps=max_train_steps,
            )


            # import ipdb;ipdb.set_trace()
            if args.use_ddp and r >= args.start_round:
                sub_layers = SubLayer(sub_layers,quant_mode_sub_layer_list=[args.quant_mode_layer_list[i] for i in layer_id_list],
                            position_ids=position_ids,position_embeddings=position_embeddings,args=args)


            cleanup_memory(logger=logger)


            if args.use_ddp and r >= args.start_round:
                sub_layers = sub_layers.cuda()

            if args.use_ddp:
                dist.barrier()

            # train loop
            if r < args.start_round:
                logger.info(f"round {r} skip because resume from disk!")
                qdataset = None
            else:
                qdataset = Quant_dataset(aug_quant_inps=windows_quant_inps,aug_fp_inps=windows_fp_inps,aug_quant_targets=windows_quant_targets,aug_fp_targets=windows_fp_targets,attention_masks=token_masks,samples_num=args.nsamples,windows_num=args.last_round_inp_num,args=args)
                sub_layers,cur_epochs = train_one_round(r=r,epochs=epochs,sub_layers=sub_layers,layer_id_list=layer_id_list,attention_mask_batch=None,cur_epochs=cur_epochs,
                                position_ids=position_ids,position_embeddings=position_embeddings,devs=devs,args=args,logger=logger,max_train_steps=max_train_steps,optimizer=optimizer,lr_scheduler=lr_scheduler,qdataset=qdataset,
                                init_quant_rate=init_quant_rate,fp16_type=fp16_type,global_start_time=global_start_time,total_epochs=total_epochs,loss_func=loss_func,acts_round_idx=acts_round_idx)

            if args.use_ddp and r >= args.start_round:
                sub_layers = sub_layers.module.module
                sub_layers = sub_layers.to("cpu")




            if args.use_ddp:
                dist.barrier()

            del optimizer,qdataset,lr_scheduler
            for r_idx in range(args.last_round_inp_num-1):
                windows_quant_inps[:,r_idx] = windows_quant_inps[:,r_idx+1]
                windows_fp_inps[:,r_idx] = windows_fp_inps[:,r_idx+1]



            cleanup_memory(logger=logger)


            sliding_layer = layer_windows_scheduler[min(r+1,num_round-1)][0] - layer_windows_scheduler[r][0]
            if args.quant_mode == "catq":
                for layer in sub_layers:
                    layer.set_catq_progress(1.0)
            if r < num_round-1 and sliding_layer>0:
                sub_layers = to_dev(sub_layers, [dev] * len(sub_layers))  #single gpu
                with torch.no_grad():
                    with torch.cuda.amp.autocast(dtype=fp16_type):
                        # get next fp_16
                        for j in tqdm(range(0,args.nsamples,args.inference_batch_size)):
                            bs_local = min(args.inference_batch_size,args.nsamples-j)
                            windows_fp_inps[j:j+bs_local,-1] = obtain_teacher_output(
                                teach_sub_layers[:sliding_layer],
                                windows_fp_inps[j:j+bs_local,-1],
                                token_masks[j:j+bs_local],
                                position_ids,
                                position_embeddings=position_embeddings,
                                args=args,
                                devs=[dev] * len(teach_sub_layers),
                            )
                        logger.info(f"finish to obtain round {r+1} inps of full-precision model!")

                        for j in tqdm(range(0,args.nsamples,args.inference_batch_size)):
                            bs_local = min(args.inference_batch_size,args.nsamples-j)
                            windows_quant_inps[j:j+bs_local,-1] = obtain_studnet_output(
                                sub_layers[:sliding_layer],
                                [args.quant_mode_layer_list[i] for i in layer_id_list[:sliding_layer]],
                                windows_quant_inps[j:j+bs_local,-1],
                                token_masks[j:j+bs_local],
                                position_ids,
                                position_embeddings=position_embeddings,
                                args=args,
                                devs=[dev] * len(sub_layers),
                            )
                        logger.info(f"finish to obtain round {r+1} inps of quant model!")

            with torch.no_grad():
                for idx,i in enumerate(layer_id_list):
                    qlayer = sub_layers[idx]
                    qlayer.clear_temp_variable()
                    if args.quant_mode == "catq" and (
                        not args.use_ddp or dist.get_rank() == 0
                    ):
                        logger.info(
                            "CAT-Q layer %s statistics: %s",
                            i,
                            json.dumps(qlayer.catq_statistics(), sort_keys=True),
                        )
                    if epochs>0:
                        sub_layers[idx] = qlayer.to("cpu")
                        slider_parameters[i] = slider_state_dict(qlayer)
                    else:
                        sub_layers[idx] = qlayer.to("cpu")

                if (
                    epochs > 0
                    and r >= args.start_round
                    and (not args.use_ddp or dist.get_rank() == 0)
                ):
                    atomic_torch_save(
                        slider_parameters,
                        os.path.join(args.output_dir, "slider_parameters.pth"),
                    )
                    atomic_torch_save(
                        window_checkpoint(slider_parameters, r + 1, args),
                        os.path.join(args.output_dir, "training_state.pt"),
                    )
                    logger.info(
                        "saved window %s checkpoint after layers %s",
                        r,
                        layer_id_list,
                    )

                del qlayer
                sub_layers = to_half(sub_layers,dtype=fp16_type)
                sub_layers = to_dev(sub_layers, ["cpu"] * len(sub_layers))  #single gpu
                teach_sub_layers = to_half(teach_sub_layers,dtype=fp16_type)
                teach_sub_layers = to_dev(teach_sub_layers, ["cpu"] * len(teach_sub_layers))  #single gpu
                replace_ori_layer(layers, sub_layers, layer_id_list, args)


            del sub_layers,teach_sub_layers
            cleanup_memory(logger=logger)

    if args.use_ddp:  # 保证参数保存完整
        dist.barrier()

    logger.info("Model quantization finished! start change model to inference mode!")
    args.quant_rate = 1.0
    model_to_inference_mode(layers=layers,args=args,dtype=fp16_type,dev=dev)
    model.to(fp16_type)


    try:
        del quant_inps
        del fp_inps
        # del tmp_fp_inps
    except Exception as e:
        logger.info(f"del tensor occurs {e}, skip!")
    cleanup_memory(logger=logger)

    logger.info("Model is changed to inderence mode!")

    cleanup_memory(logger=logger)

    model.config.use_cache = use_cache

    return model,inps,token_masks,position_ids
