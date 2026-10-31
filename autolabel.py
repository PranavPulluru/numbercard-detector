"""Draft YOLO labels for the "card" class from recorded pickup-line footage.

Two label sources are combined:
  1. OCR: EasyOCR finds 4-digit text, and the box is grown outward to the
     bright card around it.
  2. YOLO-World zero-shot detection with prompts like "white paper sign".
     A YOLO-World box that contains an OCR hit replaces the grown box. If the
     card's edge can't be found, the source column says 'ocr-guess(review)'.
     YOLO-World boxes without an OCR hit are kept only above --world-conf.

Every label is a draft. Review and fix them in Label Studio, CVAT or Roboflow
(all import YOLO format) before training.

Output layout, one folder per recording session so train.py can split by day:
    datasets/cards/<session>/images/*.jpg
    datasets/cards/<session>/labels/*.txt     (YOLO: 0 cx cy w h, normalised)
    datasets/cards/<session>/unlabeled/*.jpg  (no card found: check by hand)
    datasets/cards/<session>/autolabel.csv    (where each box came from)

Examples:
    python autolabel.py --video recordings/2026-10-27_pm.mp4 --session 2026-10-27_pm
    python autolabel.py --images frames/ --session test --no-world
"""
import argparse
import csv
import glob
import os

import cv2
import numpy as np

import util

WORLD_PROMPTS = ['white paper card', 'paper sign on car dashboard', 'white placard with number']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--video', help='recording to sample frames from')
    src.add_argument('--images', help='folder of frames that are already extracted')
    p.add_argument('--session', required=True, help='name for this recording, e.g. 2026-10-27_pm')
    p.add_argument('--out', default='datasets/cards')
    p.add_argument('--fps', type=float, default=1.0, help='frames per second to sample from --video')
    p.add_argument('--min-conf', type=float, default=0.3, help='EasyOCR confidence for a hit')
    p.add_argument('--no-world', action='store_true', help='skip YOLO-World, use OCR only')
    p.add_argument('--world-model', default='yolov8s-worldv2.pt')
    p.add_argument('--world-conf', type=float, default=0.25,
                   help='confidence for YOLO-World boxes that have no OCR hit')
    return p.parse_args()


def sample_video(path, fps):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f'Could not open video: {path}')
    step = max(1, round((cap.get(cv2.CAP_PROP_FPS) or 30) / fps))
    stem = os.path.splitext(os.path.basename(path))[0]
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            yield f'{stem}_{idx:06d}', frame
        idx += 1
    cap.release()


def read_folder(path):
    for f in sorted(glob.glob(os.path.join(path, '*'))):
        if util.is_image(f):
            frame = cv2.imread(f)
            if frame is not None:
                yield os.path.splitext(os.path.basename(f))[0], frame


def grow_to_card(frame, text_box, color_tol=30.0):
    """Expand a digit text box to the card around it. Returns (box, found).

    Samples the card's paper colour (Lab) just outside the digits, keeps pixels
    close to that colour, and takes the connected blob that covers that ring.
    Matching colour rather than brightness separates a yellow tag from sky
    reflections and a white card from a grey seat. Card layouts differ (the
    number can be at the top or bottom), so the search window is symmetric.

    If no plausible blob is found, returns a symmetric guess with found=False
    so the label is flagged for review instead of trusted.
    """
    x1, y1, x2, y2 = text_box
    tw, th = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    guess = util.clip_box((cx - 0.8 * tw, cy - 2.5 * th, cx + 0.8 * tw, cy + 2.5 * th), frame.shape)
    wx1, wy1, wx2, wy2 = util.clip_box((cx - 2 * tw, cy - 6 * th, cx + 2 * tw, cy + 6 * th), frame.shape)
    win = frame[wy1:wy2, wx1:wx2]
    if win.size == 0:
        return guess, False
    lab = cv2.cvtColor(cv2.GaussianBlur(win, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)

    # Ring just outside the digits; its lighter half is the paper, not ink.
    pad = 0.5 * th
    ring = np.zeros(lab.shape[:2], np.uint8)
    cv2.rectangle(ring, (int(x1 - wx1 - pad), int(y1 - wy1 - pad)),
                  (int(x2 - wx1 + pad), int(y2 - wy1 + pad)), 1, -1)
    cv2.rectangle(ring, (int(x1 - wx1), int(y1 - wy1)), (int(x2 - wx1), int(y2 - wy1)), 0, -1)
    px = lab[ring == 1]
    if len(px) < 10:
        return guess, False
    paper = np.median(px[px[:, 0] >= np.median(px[:, 0])], axis=0)

    dist = np.linalg.norm(lab - paper, axis=2)
    k = max(3, int(th * 0.8)) | 1
    win_h, win_w = dist.shape
    img_h, img_w = frame.shape[:2]
    # Start loose; if the blob runs into the search window's edge it has leaked
    # into a similar-coloured hand, seat or reflection, so tighten and retry.
    for tol in (color_tol, color_tol * 0.75, color_tol * 0.5):
        mask = (dist < tol).astype(np.uint8)
        # Fill the ink (digits, name, logo) so the card is one blob, then cut thin bridges.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if n < 2:
            break
        # The card is the blob that covers the most of the ring.
        cover = np.bincount(labels[ring == 1], minlength=n)
        cover[0] = 0
        best = int(cover.argmax())
        if cover[best] < 0.3 * ring.sum():
            break
        x, y, w, h, _ = stats[best]
        if w < 1.05 * tw or h < 1.5 * th:
            break  # too small to hold the digits; tighter only shrinks it
        # Touching the photo's own border is fine (card cut off); touching the
        # window's border inside the photo means a leak.
        leaked = ((x == 0 and wx1 > 0) or (y == 0 and wy1 > 0) or
                  (x + w >= win_w and wx2 < img_w) or (y + h >= win_h and wy2 < img_h))
        if not leaked:
            return (int(wx1 + x), int(wy1 + y), int(wx1 + x + w), int(wy1 + y + h)), True
    return guess, False


def contains(box, point):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def to_yolo(box, shape):
    h, w = shape[:2]
    x1, y1, x2, y2 = box
    return f'0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}'


def load_world(args):
    if args.no_world:
        return None
    try:
        from ultralytics import YOLOWorld
        model = YOLOWorld(args.world_model)
        model.set_classes(WORLD_PROMPTS)
        return model
    except Exception as e:  # missing CLIP dependency, no network, etc.
        print(f'YOLO-World unavailable ({e}); continuing with OCR only.')
        return None


def label_frame(frame, world, args):
    """Return [(box, source, number), ...] for one frame."""
    hits = util.find_card_numbers(frame, args.min_conf)
    world_boxes = []
    if world is not None:
        res = world(frame, conf=0.05, verbose=False)[0]
        world_boxes = [b[:5] for b in res.boxes.data.tolist()]

    labels, used = [], set()
    for number, _, tb in hits:
        centre = ((tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2)
        grown, found = grow_to_card(frame, tb)
        grown_area = (grown[2] - grown[0]) * (grown[3] - grown[1])
        match = None
        for i, wb in enumerate(world_boxes):
            area = (wb[2] - wb[0]) * (wb[3] - wb[1])
            if contains(wb, centre) and area < 6 * grown_area and (match is None or wb[4] > world_boxes[match][4]):
                match = i
        if match is not None:
            used.add(match)
            labels.append((tuple(world_boxes[match][:4]), 'ocr+world', number))
        else:
            labels.append((grown, 'ocr' if found else 'ocr-guess(review)', number))
    for i, wb in enumerate(world_boxes):
        if i not in used and wb[4] >= args.world_conf:
            labels.append((tuple(wb[:4]), f'world({wb[4]:.2f})', ''))
    return labels


def main():
    args = parse_args()
    root = os.path.join(args.out, args.session)
    dirs = {d: os.path.join(root, d) for d in ('images', 'labels', 'unlabeled')}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    world = load_world(args)
    frames = sample_video(args.video, args.fps) if args.video else read_folder(args.images)

    n_frames = n_labeled = n_boxes = 0
    with open(os.path.join(root, 'autolabel.csv'), 'a', newline='') as f:
        log = csv.writer(f)
        for name, frame in frames:
            n_frames += 1
            labels = label_frame(frame, world, args)
            if not labels:
                cv2.imwrite(os.path.join(dirs['unlabeled'], name + '.jpg'), frame)
                continue
            n_labeled += 1
            n_boxes += len(labels)
            cv2.imwrite(os.path.join(dirs['images'], name + '.jpg'), frame)
            with open(os.path.join(dirs['labels'], name + '.txt'), 'w') as lf:
                lf.write('\n'.join(to_yolo(b, frame.shape) for b, _, _ in labels) + '\n')
            for b, source, number in labels:
                log.writerow([name, source, number] + [round(v) for v in b])
            if n_frames % 25 == 0:
                print(f'{n_frames} frames, {n_labeled} labeled', flush=True)

    with open(os.path.join(args.out, 'classes.txt'), 'w') as f:
        f.write('card\n')
    print(f'Done: {n_frames} frames, {n_labeled} with draft labels ({n_boxes} boxes), '
          f'{n_frames - n_labeled} in {dirs["unlabeled"]} for manual review.')


if __name__ == '__main__':
    main()
