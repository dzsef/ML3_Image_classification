import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageChops, ImageOps
import torch
import torch.backends.cudnn as cudnn
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRCNN_DIR = ROOT / 'third_party' / 'srcnn_pytorch'
sys.path.insert(0, str(SRCNN_DIR))

from models import SRCNN  
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr  


def load_model(weights_file, device):
    model = SRCNN().to(device)
    state_dict = model.state_dict()
    for name, param in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
        if name in state_dict:
            state_dict[name].copy_(param)
        else:
            raise KeyError(name)
    model.eval()
    return model


def prepare_images(image_path, scale):
    hr = Image.open(image_path).convert('RGB')
    hr_width = (hr.width // scale) * scale
    hr_height = (hr.height // scale) * scale
    hr = hr.resize((hr_width, hr_height), resample=Image.BICUBIC)
    lr = hr.resize((hr_width // scale, hr_height // scale), resample=Image.BICUBIC)
    bicubic = lr.resize((lr.width * scale, lr.height * scale), resample=Image.BICUBIC)
    return hr, bicubic, lr


def to_y_channels(hr, bicubic):
    hr_np = np.array(hr).astype(np.float32)
    bicubic_np = np.array(bicubic).astype(np.float32)
    hr_ycbcr = convert_rgb_to_ycbcr(hr_np)
    bicubic_ycbcr = convert_rgb_to_ycbcr(bicubic_np)
    hr_y = hr_ycbcr[..., 0] / 255.0
    bicubic_y = bicubic_ycbcr[..., 0] / 255.0
    return hr_y, bicubic_y, bicubic_ycbcr


def run_model(model, device, bicubic_y):
    y = torch.from_numpy(bicubic_y).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(y).clamp(0.0, 1.0)
    return preds


def make_srcnn_rgb(preds, bicubic_ycbcr):
    preds_y = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
    output = np.array([preds_y, bicubic_ycbcr[..., 1], bicubic_ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(output)


def compute_metrics(hr_y, bicubic_y, preds, device, with_ssim):
    hr_t = torch.from_numpy(hr_y).unsqueeze(0).unsqueeze(0).to(device)
    bicubic_t = torch.from_numpy(bicubic_y).unsqueeze(0).unsqueeze(0).to(device)
    psnr_bicubic = calc_psnr(bicubic_t, hr_t).item()
    psnr_srcnn = calc_psnr(preds, hr_t).item()
    ssim_bicubic = None
    ssim_srcnn = None
    if with_ssim:
        preds_np = preds.squeeze(0).squeeze(0).cpu().numpy()
        ssim_bicubic = ssim(hr_y, bicubic_y, data_range=1.0)
        ssim_srcnn = ssim(hr_y, preds_np, data_range=1.0)
    return psnr_bicubic, psnr_srcnn, ssim_bicubic, ssim_srcnn


def save_montage(records, out_path, scale, model, device):
    if not records:
        return
    samples = []
    for record in records:
        hr, bicubic, lr = prepare_images(record['path'], scale)
        hr_y, bicubic_y, bicubic_ycbcr = to_y_channels(hr, bicubic)
        preds = run_model(model, device, bicubic_y)
        srcnn = make_srcnn_rgb(preds, bicubic_ycbcr)
        
        diff = ImageChops.difference(srcnn, bicubic)
        diff = diff.convert('L')
        diff = diff.point(lambda p: p * 15)
        diff_map = ImageOps.colorize(diff, black="black", mid="red", white="yellow")
        
        lr_display = lr.resize(hr.size, resample = Image.NEAREST)
        samples.append((lr_display, hr, bicubic, srcnn))

    width, height = samples[0][0].size
    header_h = 24
    montage = Image.new('RGB', (width * 5, header_h + height * len(samples)), color=(20, 20, 20))
    draw = ImageDraw.Draw(montage)
    draw.text((width * 0 + 6, 4), 'Input (LR)', fill=(230, 230, 230))
    draw.text((width * 1 + 6, 4), 'HR', fill=(230, 230, 230))
    draw.text((width * 2 + 6, 4), 'Bicubic', fill=(230, 230, 230))
    draw.text((width * 3 + 6, 4), 'SRCNN', fill=(230, 230, 230))
    draw.text((width * 4 + 6, 4), 'Diff (x15)', fill=(230, 230, 230)) 

    for idx, (lr_disp, hr, bicubic, srcnn) in enumerate(samples):
        y = header_h + idx * height
        montage.paste(lr_disp, (0, y))
        montage.paste(hr,      (width * 1, y))
        montage.paste(bicubic, (width * 2, y))
        montage.paste(srcnn,   (width * 3, y))
        montage.paste(diff_map,(width * 4, y)) 

    montage.save(out_path)


def summarize(values):
    arr = np.array(values, dtype=np.float32)
    return {
        'mean': float(arr.mean()),
        'std': float(arr.std()),
        'min': float(arr.min()),
        'max': float(arr.max()),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--hr-dir', type=str, required=True)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--ssim', action='store_true')
    parser.add_argument('--topk', type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'qualitative').mkdir(parents=True, exist_ok=True)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.weights, device)

    image_paths = sorted(Path(args.hr_dir).glob('*.png'))
    if not image_paths:
        raise FileNotFoundError(f'No PNG images found in {args.hr_dir}')

    rows = []
    for image_path in tqdm(image_paths, desc='Evaluating', unit='img'):
        hr, bicubic, _ = prepare_images(image_path, args.scale)
        hr_y, bicubic_y, bicubic_ycbcr = to_y_channels(hr, bicubic)
        preds = run_model(model, device, bicubic_y)
        psnr_bicubic, psnr_srcnn, ssim_bicubic, ssim_srcnn = compute_metrics(
            hr_y, bicubic_y, preds, device, args.ssim
        )
        rows.append({
            'name': image_path.name,
            'path': str(image_path),
            'psnr_bicubic': psnr_bicubic,
            'psnr_srcnn': psnr_srcnn,
            'ssim_bicubic': ssim_bicubic,
            'ssim_srcnn': ssim_srcnn,
        })

    rows_sorted = sorted(rows, key=lambda r: r['psnr_srcnn'], reverse=True)
    topk = max(1, min(args.topk, len(rows_sorted)))
    best = rows_sorted[:topk]
    worst = list(reversed(rows_sorted[-topk:]))

    save_montage(best, out_dir / 'qualitative' / 'best.png', args.scale, model, device)
    save_montage(worst, out_dir / 'qualitative' / 'worst.png', args.scale, model, device)

    csv_path = out_dir / 'metrics.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.writer(f)
        header = ['image', 'psnr_bicubic', 'psnr_srcnn']
        if args.ssim:
            header += ['ssim_bicubic', 'ssim_srcnn']
        writer.writerow(header)
        for row in rows:
            record = [row['name'], f"{row['psnr_bicubic']:.4f}", f"{row['psnr_srcnn']:.4f}"]
            if args.ssim:
                record += [f"{row['ssim_bicubic']:.6f}", f"{row['ssim_srcnn']:.6f}"]
            writer.writerow(record)

    psnr_bicubic_stats = summarize([r['psnr_bicubic'] for r in rows])
    psnr_srcnn_stats = summarize([r['psnr_srcnn'] for r in rows])
    delta_psnr = psnr_srcnn_stats['mean'] - psnr_bicubic_stats['mean']

    summary_lines = [
        f"Images: {len(rows)}",
        f"PSNR Bicubic mean/std/min/max: {psnr_bicubic_stats['mean']:.4f} / "
        f"{psnr_bicubic_stats['std']:.4f} / {psnr_bicubic_stats['min']:.4f} / "
        f"{psnr_bicubic_stats['max']:.4f}",
        f"PSNR SRCNN   mean/std/min/max: {psnr_srcnn_stats['mean']:.4f} / "
        f"{psnr_srcnn_stats['std']:.4f} / {psnr_srcnn_stats['min']:.4f} / "
        f"{psnr_srcnn_stats['max']:.4f}",
        f"Delta PSNR (SRCNN - Bicubic): {delta_psnr:.4f}",
    ]

    if args.ssim:
        ssim_bicubic_stats = summarize([r['ssim_bicubic'] for r in rows])
        ssim_srcnn_stats = summarize([r['ssim_srcnn'] for r in rows])
        delta_ssim = ssim_srcnn_stats['mean'] - ssim_bicubic_stats['mean']
        summary_lines += [
            f"SSIM Bicubic mean/std/min/max: {ssim_bicubic_stats['mean']:.6f} / "
            f"{ssim_bicubic_stats['std']:.6f} / {ssim_bicubic_stats['min']:.6f} / "
            f"{ssim_bicubic_stats['max']:.6f}",
            f"SSIM SRCNN   mean/std/min/max: {ssim_srcnn_stats['mean']:.6f} / "
            f"{ssim_srcnn_stats['std']:.6f} / {ssim_srcnn_stats['min']:.6f} / "
            f"{ssim_srcnn_stats['max']:.6f}",
            f"Delta SSIM (SRCNN - Bicubic): {delta_ssim:.6f}",
        ]

    summary_path = out_dir / 'summary.txt'
    summary_path.write_text('\n'.join(summary_lines))

    for line in summary_lines:
        print(line)
