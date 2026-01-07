# extract_spm_feats_and_annotate.py
import os, json, argparse
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from main_calib_train import (
    load_backbone_cfg, build_backbone_from_cfg, load_backbone_weights
)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone_cfg', required=True)
    ap.add_argument('--backbone_ckpt', required=True)
    ap.add_argument('--image_root', default='')  # 用来补全 images 里的相对路径/文件名
    ap.add_argument('--sft_path', required=True, help='train.json / train.jsonl')
    ap.add_argument('--output_feat_dir', required=True)
    ap.add_argument('--out_path', default='', help='可选：输出新文件；不填则原地覆盖')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--img_size', type=int, default=576)

    # 路径写回策略
    ap.add_argument('--feat_path_mode', choices=['abs', 'rel_to_sft', 'rel_to_root'],
                    default='rel_to_sft',
                    help='abs: 写绝对路径；rel_to_sft: 相对sft文件目录；rel_to_root: 相对image_root')
    ap.add_argument('--skip_existing', action='store_true',
                    help='若特征文件已存在则跳过计算（强烈建议开）')
    return ap.parse_args()

def is_jsonl(path: str) -> bool:
    return path.lower().endswith('.jsonl')

def read_sft(path: str):
    if is_jsonl(path):
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
        return data, 'jsonl'
    else:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f), 'json'

def write_sft(path: str, data, fmt: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if fmt == 'jsonl':
        with open(path, 'w', encoding='utf-8') as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    else:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

def load_img(abs_path: str, img_size: int):
    img = Image.open(abs_path).convert('RGB')
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])
    return tfm(img)

def resolve_image_path(img_item: str, image_root: str) -> str:
    # img_item 可能是文件名、相对路径、绝对路径
    if os.path.isabs(img_item):
        return img_item
    if image_root:
        return os.path.normpath(os.path.join(image_root, img_item))
    return os.path.normpath(img_item)

def make_feat_path(img_abs_path: str, output_feat_dir: str) -> str:
    base = os.path.splitext(os.path.basename(img_abs_path))[0]
    return os.path.join(output_feat_dir, base + '.npy')

def normalize_writeback_path(feat_abs_path: str, args, sft_dir: str) -> str:
    if args.feat_path_mode == 'abs':
        return os.path.abspath(feat_abs_path)
    if args.feat_path_mode == 'rel_to_root':
        root = os.path.abspath(args.image_root) if args.image_root else os.getcwd()
        return os.path.relpath(os.path.abspath(feat_abs_path), root)
    # rel_to_sft
    return os.path.relpath(os.path.abspath(feat_abs_path), sft_dir)

def main():
    args = parse_args()
    os.makedirs(args.output_feat_dir, exist_ok=True)

    data, fmt = read_sft(args.sft_path)
    assert isinstance(data, list), 'SFT 数据应为 list（jsonl读出来也是list）'

    # build backbone
    cfg = load_backbone_cfg(args.backbone_cfg)
    backbone = build_backbone_from_cfg(cfg, num_classes=27)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    backbone = load_backbone_weights(backbone, args.backbone_ckpt, device)
    backbone.eval()

    # 探测维度（用第一张图）
    first_img_item = data[0]['images'][0]
    first_abs = resolve_image_path(first_img_item, args.image_root)
    x0 = load_img(first_abs, args.img_size).unsqueeze(0).to(device)
    with torch.inference_mode():
        feat0 = backbone.forward_features(x0)
    d_feat = int(feat0.shape[-1])
    print(f'[extract] feature dim = {d_feat}')

    sft_dir = os.path.dirname(os.path.abspath(args.sft_path))

    # cache: 同一张图在SFT里重复出现时，避免重复算
    cache = {}  # img_abs -> feat_abs

    new_data = []
    for i, ex in enumerate(data):
        img_item = ex['images'][0]
        img_abs = resolve_image_path(img_item, args.image_root)

        if img_abs in cache:
            feat_abs = cache[img_abs]
        else:
            feat_abs = make_feat_path(img_abs, args.output_feat_dir)
            if args.skip_existing and os.path.exists(feat_abs):
                cache[img_abs] = feat_abs
            else:
                x = load_img(img_abs, args.img_size).unsqueeze(0).to(device)
                with torch.inference_mode():
                    feat = backbone.forward_features(x).squeeze(0).detach().cpu().float().numpy()
                # 强制保存为 1D float32，避免后续 collate 形状坑
                feat = np.asarray(feat, dtype=np.float32).reshape(-1)
                np.save(feat_abs, feat)
                cache[img_abs] = feat_abs

        ex['spm_feat_path'] = normalize_writeback_path(feat_abs, args, sft_dir)
        ex['spm_feat_dim'] = d_feat
        new_data.append(ex)

        if (i + 1) % 100 == 0:
            print(f'[extract] done {i+1}/{len(data)}')

    out_path = args.out_path.strip() or args.sft_path
    write_sft(out_path, new_data, fmt)
    print(f'[extract] all done. wrote to: {out_path}')

if __name__ == '__main__':
    main()
