import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    subprocess.run(cmd, check=True)


def parse_simple_yaml(path):
    config = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        if ':' not in stripped:
            raise ValueError(f'Invalid config line: {line}')
        key, value = stripped.split(':', 1)
        key = key.strip()
        value = value.strip()
        if value == '':
            config[key] = None
            continue
        lower = value.lower()
        if lower in ('true', 'false'):
            config[key] = lower == 'true'
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            config[key] = value[1:-1]
            continue
        try:
            if any(ch in value for ch in ('.', 'e', 'E')):
                config[key] = float(value)
            else:
                config[key] = int(value)
            continue
        except ValueError:
            config[key] = value
    return config


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--patch-size', type=int, default=33)
    parser.add_argument('--stride', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--data-dir', type=str, default='data/div2k')
    parser.add_argument('--artifacts-dir', type=str, default='artifacts')
    parser.add_argument('--outputs-dir', type=str, default='outputs')
    parser.add_argument('--skip-prepare', action='store_true')
    parser.add_argument('--skip-train', action='store_true')
    parser.add_argument('--skip-eval', action='store_true')
    parser.add_argument('--eval-ssim', action='store_true')
    args = parser.parse_args()

    defaults = {
        action.dest: parser.get_default(action.dest)
        for action in parser._actions
        if action.dest != 'help'
    }
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = root / config_path
        config = parse_simple_yaml(config_path)
        for key, value in config.items():
            if key not in defaults:
                raise KeyError(f'Unknown config key: {key}')
            if getattr(args, key) == defaults[key]:
                setattr(args, key, value)

    py = sys.executable

    data_dir = root / args.data_dir
    artifacts_dir = root / args.artifacts_dir
    outputs_dir = root / args.outputs_dir

    h5_dir = artifacts_dir / 'h5'
    h5_dir.mkdir(parents=True, exist_ok=True)
    train_h5 = h5_dir / f'div2k_train_x{args.scale}.h5'
    val_h5 = h5_dir / f'div2k_val_x{args.scale}.h5'

    model_out_dir = outputs_dir / f'srcnn_div2k_x{args.scale}'
    weights_path = model_out_dir / f'x{args.scale}' / 'best.pth'
    eval_out_dir = outputs_dir / f'eval_x{args.scale}'

    if not args.skip_prepare:
        run([
            py, str(root / 'third_party' / 'srcnn_pytorch' / 'prepare.py'),
            '--images-dir', str(data_dir / 'HR' / 'train'),
            '--output-path', str(train_h5),
            '--scale', str(args.scale),
            '--patch-size', str(args.patch_size),
            '--stride', str(args.stride),
        ])
        run([
            py, str(root / 'third_party' / 'srcnn_pytorch' / 'prepare.py'),
            '--images-dir', str(data_dir / 'HR' / 'val'),
            '--output-path', str(val_h5),
            '--scale', str(args.scale),
            '--eval',
        ])

    if not args.skip_train:
        run([
            py, str(root / 'third_party' / 'srcnn_pytorch' / 'train.py'),
            '--train-file', str(train_h5),
            '--eval-file', str(val_h5),
            '--outputs-dir', str(model_out_dir),
            '--scale', str(args.scale),
            '--lr', str(args.lr),
            '--batch-size', str(args.batch_size),
            '--num-epochs', str(args.epochs),
            '--num-workers', str(args.num_workers),
            '--seed', str(args.seed),
        ])

    if not args.skip_eval:
        if not weights_path.exists():
            raise FileNotFoundError(f'Weights not found at {weights_path}')
        eval_cmd = [
            py, str(root / 'scripts' / 'eval_div2k.py'),
            '--weights', str(weights_path),
            '--hr-dir', str(data_dir / 'HR' / 'val'),
            '--scale', str(args.scale),
            '--out-dir', str(eval_out_dir),
        ]
        if args.eval_ssim:
            eval_cmd.append('--ssim')
        run(eval_cmd)
