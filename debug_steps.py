"""Save an image of every pipeline step for one photo, to see where reading fails.

Steps 01-05 follow baseline.py (find car -> windshield crop -> OCR).
Steps 06-09 follow main.py (find card -> card crop -> OCR). Until a card
detector is trained, step 06 uses autolabel.py's OCR-grown card box instead,
and says so.

    python debug_steps.py --source path/to/photo.jpg
    python debug_steps.py --source photo.jpg --card-model models/card_detector.pt

Output goes to debug_steps/<photo name>/ with a steps.txt summary.
Output is git-ignored because photos show students' names.
"""
import argparse
import os

import cv2
from ultralytics import YOLO

import autolabel
import util

GREEN, RED, BLUE, ORANGE = (0, 200, 0), (0, 0, 255), (255, 120, 0), (0, 165, 255)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', required=True, help='photo to trace')
    p.add_argument('--out', default='debug_steps')
    p.add_argument('--vehicle-model', default='yolov8n.pt')
    p.add_argument('--card-model', default='models/card_detector.pt',
                   help='used for step 06 if the file exists')
    p.add_argument('--roi', type=float, nargs=2, default=(0.0, 0.7), metavar=('TOP', 'BOTTOM'))
    p.add_argument('--min-conf', type=float, default=0.4)
    return p.parse_args()


class Steps:
    def __init__(self, folder):
        self.folder = folder
        self.n = 0
        self.notes = []
        os.makedirs(folder, exist_ok=True)

    def save(self, name, img, note):
        self.n += 1
        fname = f'{self.n:02d}_{name}.jpg'
        cv2.imwrite(os.path.join(self.folder, fname), img)
        self.notes.append(f'{fname}\n    {note}')
        print(f'{fname}: {note}')

    def finish(self):
        with open(os.path.join(self.folder, 'steps.txt'), 'w') as f:
            f.write('\n\n'.join(self.notes) + '\n')


def thick(img):
    return max(2, round(max(img.shape[:2]) / 400))


def label(img, text, org, color):
    s = max(0.5, max(img.shape[:2]) / 1600)
    x, y = int(org[0]), int(max(org[1], 20 * s))
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), thick(img) + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, s, color, thick(img), cv2.LINE_AA)


def draw_ocr(img, detections):
    """Green = accepted 4-digit number, red = rejected (reason shown)."""
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    # Enlarge small crops (display only) so the labels fit.
    f = max(1.0, 700 / max(vis.shape[:2]))
    if f > 1:
        vis = cv2.resize(vis, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    for d in detections:
        x1, y1, x2, y2 = (int(v * f) for v in d['box'])
        color = GREEN if d['number'] else RED
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick(vis))
        tag = f"{d['number']} ({d['score']:.2f})" if d['number'] else f"'{d['raw']}' x {d['reason']}"
        label(vis, tag, (x1, y1 - 6), color)
    return vis


def describe(detections):
    if not detections:
        return 'OCR found no text.'
    parts = [f"'{d['raw']}' -> " + (f"ACCEPT {d['number']} ({d['score']:.2f})" if d['number']
                                    else f"reject: {d['reason']}") for d in detections]
    return 'OCR found: ' + '; '.join(parts)


def winner(detections):
    hits = [d for d in detections if d['number']]
    return max(hits, key=lambda d: d['box'][3] - d['box'][1]) if hits else None


def main():
    args = parse_args()
    frame = cv2.imread(args.source)
    if frame is None:
        raise SystemExit(f'Could not read image: {args.source}')
    stem = os.path.splitext(os.path.basename(args.source))[0]
    steps = Steps(os.path.join(args.out, stem))
    h, w = frame.shape[:2]
    steps.save('input', frame, f'Original photo, {w}x{h}.')

    # ---- Path A: baseline.py (car -> windshield crop -> OCR) ----
    cars = util.detect_vehicles(YOLO(args.vehicle_model), frame)
    vis = frame.copy()
    for c in cars:
        cv2.rectangle(vis, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), BLUE, thick(vis))
        label(vis, f'vehicle {c[4]:.2f}', (c[0] + 10, c[1] + 40), BLUE)
    steps.save('vehicles', vis, f'COCO YOLOv8 found {len(cars)} vehicle(s): '
               + ', '.join(f'conf {c[4]:.2f} box {[round(v) for v in c[:4]]}' for c in cars)
               if cars else 'COCO YOLOv8 found no vehicle, so baseline.py would skip this frame.')

    baseline_number = None
    if cars:
        car = max(cars, key=lambda c: (c[2] - c[0]) * (c[3] - c[1]))
        ch = car[3] - car[1]
        x1, y1, x2, y2 = util.clip_box((car[0], car[1] + args.roi[0] * ch, car[2], car[1] + args.roi[1] * ch),
                                       frame.shape)
        roi = frame[y1:y2, x1:x2]
        steps.save('car_roi', roi, f'Top {args.roi[1]:.0%} of the largest vehicle box, where the windshield '
                   f'and dashboard should be. Crop {x2 - x1}x{y2 - y1}.')
        roi_pre = util.preprocess(roi)
        steps.save('car_roi_preprocessed', roi_pre, 'Grayscale + CLAHE contrast boost, the input EasyOCR sees.')
        dets = util.ocr_all(roi_pre, args.min_conf)
        win = winner(dets)
        baseline_number = win['number'] if win else None
        steps.save('car_roi_ocr', draw_ocr(roi_pre, dets), describe(dets) + (
            f' -> largest accepted: {baseline_number}' if win else ' -> no 4-digit number.'))

    # ---- Path B: main.py (card -> card crop -> OCR) ----
    use_model = os.path.exists(args.card_model)
    if use_model:
        res = YOLO(args.card_model)(frame, imgsz=1280, verbose=False)[0]
        cards = [b[:5] for b in res.boxes.data.tolist()]
        source = f'card detector {args.card_model}'
    else:
        hits = util.find_card_numbers(util.preprocess(frame, min_height=0), args.min_conf)
        cards = []
        if hits:
            box, found = autolabel.grow_to_card(frame, hits[0][2])
            cards = [list(box) + [1.0 if found else 0.0]]
        source = 'STAND-IN: no trained card detector yet, so this box is autolabel.py\'s OCR-grown card box'
    vis = frame.copy()
    for c in cards:
        cv2.rectangle(vis, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), ORANGE, thick(vis))
    steps.save('card_box', vis, f'{source}. Found {len(cards)} card(s)'
               + (': ' + ', '.join(str([round(v) for v in c[:4]]) for c in cards) if cards else '.'))

    main_number = None
    if cards:
        card = max(cards, key=lambda c: c[4])
        x1, y1, x2, y2 = util.clip_box(card, frame.shape, pad=util.CARD_PAD)
        crop = frame[y1:y2, x1:x2]
        steps.save('card_crop', crop, f'Card box plus {util.CARD_PAD:.0%} padding. Crop {x2 - x1}x{y2 - y1}.')
        pre = util.preprocess(crop)
        steps.save('card_preprocessed', pre, f'Grayscale + CLAHE, upscaled to at least 96 px tall. '
                   f'Now {pre.shape[1]}x{pre.shape[0]}.')
        dets = util.ocr_all(pre, args.min_conf)
        win = winner(dets)
        main_number = win['number'] if win else None
        steps.save('card_ocr', draw_ocr(pre, dets), describe(dets) + (
            f' -> largest accepted wins: {main_number}' if win else ' -> no 4-digit number.'))

    # ---- Result ----
    vis = frame.copy()
    for c in cards:
        cv2.rectangle(vis, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), GREEN if main_number else RED,
                      thick(vis))
    s = max(1.0, w / 400)
    text = main_number or baseline_number or 'no read'
    cv2.putText(vis, text, (20, int(60 * s)), cv2.FONT_HERSHEY_SIMPLEX, 2 * s, (0, 0, 0), int(6 * s), cv2.LINE_AA)
    cv2.putText(vis, text, (20, int(60 * s)), cv2.FONT_HERSHEY_SIMPLEX, 2 * s, GREEN, int(3 * s), cv2.LINE_AA)
    steps.save('result', vis, f'Car-crop path read: {baseline_number}. Card-crop path read: {main_number}. '
               'In video, each read is one vote; a number is logged after 3 agreeing votes.')
    steps.finish()
    print(f'Saved to {steps.folder}')


if __name__ == '__main__':
    main()
