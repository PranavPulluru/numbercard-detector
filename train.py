"""Train the single-class card detector and copy it to models/card_detector.pt.

Splits are made by recording session (subfolders of datasets/cards), so the
validation and test scores reflect days the model has never seen.

Examples:
    python train.py --val-sessions 2026-10-29_pm --test-sessions 2026-10-30_am
    python train.py --eval-only --split test     # re-score the saved model
"""
import argparse
import glob
import os
import shutil

import yaml
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', default=os.path.join(HERE, 'data', 'card.yaml'))
    p.add_argument('--val-sessions', nargs='*', default=[], help='sessions held out for validation')
    p.add_argument('--test-sessions', nargs='*', default=[], help='sessions held out for final testing')
    p.add_argument('--model', default='yolov8n.pt', help='starting weights')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--imgsz', type=int, default=1280, help='cards are small, so train large')
    p.add_argument('--batch', type=int, default=-1, help='-1 picks the largest batch that fits the GPU')
    p.add_argument('--out', default=os.path.join(HERE, 'models', 'card_detector.pt'))
    p.add_argument('--eval-only', action='store_true', help='skip training, evaluate --out')
    p.add_argument('--split', default='val', choices=['val', 'test'])
    return p.parse_args()


def resolve_data(data_yaml):
    """Return (dataset_root, resolved_yaml_path).

    card.yaml's `path` is relative to the yaml file, but ultralytics resolves
    relative paths against its own global datasets folder. Write a copy with
    an absolute path next to the data and train from that.
    """
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(data_yaml)), cfg['path']))
    os.makedirs(root, exist_ok=True)
    cfg['path'] = root
    resolved = os.path.join(root, 'card_resolved.yaml')
    with open(resolved, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return root, resolved


def write_splits(root, val_sessions, test_sessions):
    """Write train.txt / val.txt / test.txt listing images by session."""
    sessions = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d, 'images')))
    if not sessions:
        raise SystemExit(f'No sessions with an images/ folder under {root}. Run autolabel.py first.')
    unknown = set(val_sessions + test_sessions) - set(sessions)
    if unknown:
        raise SystemExit(f'Unknown sessions {sorted(unknown)}; found {sessions}')
    if not val_sessions:
        val_sessions = sessions[-1:]
        print(f'No --val-sessions given; holding out the last session: {val_sessions}')

    splits = {'train': [], 'val': [], 'test': []}
    for s in sessions:
        split = 'val' if s in val_sessions else 'test' if s in test_sessions else 'train'
        for img in sorted(glob.glob(os.path.join(root, s, 'images', '*.jpg'))):
            splits[split].append('./' + os.path.relpath(img, root).replace(os.sep, '/'))
    if not splits['train']:
        raise SystemExit('Every session is held out; leave at least one for training.')
    if not splits['test']:
        splits['test'] = splits['val']  # keeps the yaml valid; say so in the report
    for name, lines in splits.items():
        with open(os.path.join(root, f'{name}.txt'), 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f'{name}: {len(lines)} images')


def main():
    args = parse_args()
    root, data = resolve_data(args.data)
    if not args.eval_only:
        write_splits(root, args.val_sessions, args.test_sessions)
        model = YOLO(args.model)
        model.train(
            data=data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
            project=os.path.join(HERE, 'runs'), name='card', exist_ok=False,
            patience=30,
            # Lighting behind a windshield swings a lot; text must not be mirrored.
            hsv_v=0.5, hsv_s=0.5, degrees=5, perspective=0.0005, fliplr=0.0, scale=0.5,
        )
        best = os.path.join(model.trainer.save_dir, 'weights', 'best.pt')
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        shutil.copy(best, args.out)
        print(f'Saved {best} -> {args.out}')

    metrics = YOLO(args.out).val(data=data, imgsz=args.imgsz, split=args.split, verbose=False)
    print(f'{args.split}: mAP50 {metrics.box.map50:.3f}  mAP50-95 {metrics.box.map:.3f}  '
          f'precision {metrics.box.mp:.3f}  recall {metrics.box.mr:.3f}')


if __name__ == '__main__':
    main()
