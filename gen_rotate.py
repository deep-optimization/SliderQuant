import os, argparse, torch
from models.LMClass import LMClass
from models.rotation_utils import get_rotate_model

# Minimal args needed by LMClass to load the base model.
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--net", type=str, default=None)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--seqlen", type=int, default=2048)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--cache_dir", type=str, default="./cache")
parser.add_argument("--attn_implementation", type=str, default="eager")
parser.add_argument("--use_bfloat16", action="store_true", default=True)
parser.add_argument("--online_had", action="store_true", default=False, help="SliderQuant+: absorb online Hadamard (down_proj input + v-o) into weights")
args = parser.parse_args()
if args.net is None:
    args.net = args.model.split("/")[-1]
args.model_family = args.net.split("-")[0]

if os.path.exists(args.save_path):
    print(f"rotate model already exists at {args.save_path}, skip.")
    raise SystemExit(0)

os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
lm = LMClass(args)
save_dict = get_rotate_model(lm.model, args.save_path, add_online_rotate=args.online_had)
print(f"saved rotated model to {args.save_path} (online_had={args.online_had})")
