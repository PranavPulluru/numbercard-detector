"""Shared helpers for reading 4-digit pickup-line number cards.

Adapted from the ANPR repo's util.py: the plate-format logic is replaced by a
4-digit card format, and live-stream pieces (frame reader, per-track voting,
append-only CSV logging, overlay drawing) are added.
"""
import csv
import os
import re
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime

import cv2
import numpy as np

# COCO class ids for vehicles that have a dashboard: car, bus, truck.
VEHICLE_CLASSES = [2, 5, 7]

CARD_DIGITS = 4
CARD_PAD = 0.08     # fraction added around a detected card before OCR
DIGITS = '0123456789'

# Letters OCR commonly confuses with digits. Extends the repo's dict_char_to_int.
dict_char_to_int = {'O': '0', 'D': '0', 'Q': '0',
                    'I': '1', 'L': '1', '|': '1',
                    'Z': '2',
                    'J': '3',
                    'A': '4',
                    'S': '5',
                    'G': '6', 'b': '6',
                    'T': '7',
                    'B': '8'}

_reader = None


def get_reader():
    """Create the EasyOCR reader once, on the GPU when available."""
    global _reader
    if _reader is None:
        import easyocr
        import torch
        if torch.cuda.is_available():
            limit_gpu_cache(torch)
        _reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    return _reader


def limit_gpu_cache(torch, fraction=None):
    """Cap PyTorch's GPU memory cache.

    EasyOCR sees a different crop size almost every frame, and PyTorch keeps
    cached blocks for each one; in a long video run the cache grew to 10 GB
    while only ~1 GB was in use. On Windows that cache also counts against
    system commit, and allocations started failing once it filled. With a cap,
    PyTorch frees cached blocks and reuses them instead of reserving more.
    Override with the CARD_READER_GPU_FRACTION environment variable.
    """
    fraction = fraction or float(os.environ.get('CARD_READER_GPU_FRACTION', '0.35'))
    torch.cuda.set_per_process_memory_fraction(fraction)


# --------------------------------------------------------------------------- #
# Card format
# --------------------------------------------------------------------------- #
def card_complies_format(text, min_real_digits=3):
    """True if text is 4 characters that are digits or digit look-alikes.

    At least `min_real_digits` must already be digits. This stops words such
    as "SOLO" or parts of a student's name from being mapped into a number.
    """
    if len(text) != CARD_DIGITS:
        return False
    real = sum(c in DIGITS for c in text)
    return real >= min_real_digits and all(c in DIGITS or c in dict_char_to_int for c in text)


def format_card(text):
    """Map look-alike letters to digits, e.g. '6O93' -> '6093'."""
    return ''.join(c if c in DIGITS else dict_char_to_int[c] for c in text)


def _clean(text):
    # Keep case for the 'b' -> '6' mapping, upper-case everything else.
    # Card edges and corners often come through as brackets or slashes: '[3605/'.
    text = re.sub(r'[\s.,\-_:;\'"()\[\]{}/\\]', '', text)
    return ''.join(c if c == 'b' else c.upper() for c in text)


# --------------------------------------------------------------------------- #
# OCR
# --------------------------------------------------------------------------- #
def preprocess(crop, min_height=96):
    """Grayscale + CLAHE, upscaling small crops so digits are big enough for OCR.

    The repo's fixed binary threshold at 64 is dropped: behind a windshield the
    card's brightness varies too much for a fixed threshold.
    """
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h = crop.shape[0]
    if 0 < h < min_height:
        scale = min_height / h
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(crop)


def ocr_all(img, min_conf=0.4):
    """Run EasyOCR and explain every piece of text it found.

    Returns a list of dicts with raw text, score, box (x1, y1, x2, y2), the
    formatted number if accepted, and the reason if rejected.
    """
    out = []
    for bbox, raw, score in get_reader().readtext(img):
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        text = _clean(raw)
        if score < min_conf:
            reason = f'confidence {score:.2f} < {min_conf}'
        elif not card_complies_format(text):
            reason = 'not 4 digits'
        else:
            reason = None
        out.append({'raw': raw, 'score': float(score),
                    'box': (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))),
                    'number': None if reason else format_card(text), 'reason': reason})
    return out


def find_card_numbers(img, min_conf=0.4):
    """Run EasyOCR text detection + recognition and return every 4-digit hit.

    Returns a list of (number, score, (x1, y1, x2, y2)) sorted by text height,
    largest first. Coordinates are in `img`'s pixel space.
    """
    hits = [(d['number'], d['score'], d['box']) for d in ocr_all(img, min_conf) if d['number']]
    hits.sort(key=lambda h: h[2][3] - h[2][1], reverse=True)
    return hits


def read_card(crop, min_conf=0.4):
    """Read the card number from a crop. Replaces the repo's read_license_plate.

    The card also carries a student name and a small extra digit, so the
    largest text that matches the 4-digit format wins.
    Returns (number, score, text_height_px) with the height measured in the
    original crop's pixels, or (None, None, None).
    """
    if crop is None or crop.size == 0:
        return None, None, None
    img = preprocess(crop)
    hits = find_card_numbers(img, min_conf)
    if not hits:
        return None, None, None
    number, score, box = hits[0]
    scale = img.shape[0] / crop.shape[0]
    return number, score, (box[3] - box[1]) / scale


def read_card_number(crop, min_conf=0.4):
    """(number, score) only; see read_card."""
    number, score, _ = read_card(crop, min_conf)
    return number, score


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def get_car(box, vehicle_track_ids):
    """Return (x1, y1, x2, y2, car_id) of the tracked vehicle containing box.

    Same contract as the repo's get_car, but uses the box centre instead of
    full containment, so a card touching the edge of a car's box still counts.
    """
    x1, y1, x2, y2 = box[:4]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    for xcar1, ycar1, xcar2, ycar2, car_id in vehicle_track_ids:
        if xcar1 < cx < xcar2 and ycar1 < cy < ycar2:
            return xcar1, ycar1, xcar2, ycar2, car_id
    return -1, -1, -1, -1, -1


def clip_box(box, shape, pad=0.0):
    """Pad a box by a fraction of its size and clip it to the frame."""
    h, w = shape[:2]
    x1, y1, x2, y2 = box[:4]
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    return (int(max(0, x1 - px)), int(max(0, y1 - py)),
            int(min(w, x2 + px)), int(min(h, y2 + py)))


# --------------------------------------------------------------------------- #
# Video input
# --------------------------------------------------------------------------- #
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def is_image(source):
    return isinstance(source, str) and os.path.splitext(source)[1].lower() in IMAGE_EXTS


def is_live(source):
    """Camera index or network stream, as opposed to a video file."""
    return source.isdigit() or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))


class FrameSource:
    """Iterate (frame_idx, timestamp_seconds, frame) over a camera, stream or file.

    For live sources a background thread keeps only the newest frame, so a slow
    model drops frames instead of falling further and further behind the lane.
    Video files are read frame by frame and timestamped by video time;
    `start` and `duration` (seconds) limit a file run to one section.
    """

    def __init__(self, source, start=0.0, duration=None):
        self.live = is_live(source)
        self.cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not self.cap.isOpened():
            raise RuntimeError(f'Could not open video source: {source}')
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.first = 0
        self.last = None
        if not self.live:
            self.first = int(round(start * self.fps))
            if self.first:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.first)
            if duration:
                self.last = self.first + int(round(duration * self.fps)) - 1
        self._latest = None
        self._seq = 0
        self._stopped = False
        self._lock = threading.Lock()
        if self.live:
            self._thread = threading.Thread(target=self._grab, daemon=True)
            self._thread.start()

    def _grab(self):
        while not self._stopped:
            ok, frame = self.cap.read()
            if not ok:
                self._stopped = True
                break
            with self._lock:
                self._latest = frame
                self._seq += 1

    def __iter__(self):
        idx = -1
        if not self.live:
            idx = self.first - 1
            while self.last is None or idx < self.last:
                ok, frame = self.cap.read()
                if not ok:
                    return
                idx += 1
                yield idx, idx / self.fps, frame
            return
        last_seq = 0
        while not self._stopped or self._seq != last_seq:
            with self._lock:
                frame, seq = self._latest, self._seq
            if frame is None or seq == last_seq:
                time.sleep(0.002)
                continue
            last_seq = seq
            idx += 1
            yield idx, time.time(), frame

    def release(self):
        self._stopped = True
        self.cap.release()


# --------------------------------------------------------------------------- #
# Tracking
# --------------------------------------------------------------------------- #
class Tracker:
    """SORT wrapper that tolerates frames with no detections.

    The original main.py passed an empty list to SORT, which crashes on frames
    with no vehicles; SORT needs an empty (0, 5) array.
    """

    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.2):
        from sort.sort import Sort
        self.sort = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)

    def update(self, boxes):
        dets = np.asarray(boxes, dtype=float).reshape(-1, 5) if len(boxes) else np.empty((0, 5))
        return self.sort.update(dets).tolist()


def detect_vehicles(model, frame, conf=0.4, imgsz=640):
    """Run the COCO model and return [[x1, y1, x2, y2, score], ...] for vehicles."""
    res = model(frame, classes=VEHICLE_CLASSES, conf=conf, imgsz=imgsz, verbose=False)[0]
    return [b[:5] for b in res.boxes.data.tolist()]


# --------------------------------------------------------------------------- #
# Per-track voting
# --------------------------------------------------------------------------- #
class CardVoter:
    """Collect OCR reads per track and decide when a car's number is certain.

    Each read is weighted by OCR confidence x text height, because reads get
    more reliable as the car gets closer. In the pickup-line footage a tilted
    card flipped between 3605 and 5605 at middling confidence while still
    small, and only the close-up reads were reliably right.

    A number is decided early when it has `min_reads` reads and `early_margin`
    times the weight of any rival. Otherwise the decision waits until the car
    leaves view, and needs `min_reads_on_loss` reads and `final_margin` times
    the rival's weight. Anything less is dropped: logging nothing beats
    logging a wrong number.
    The same number is not logged again within `dedupe_seconds`, which covers
    a car whose track ID changes mid-lane.
    """

    def __init__(self, min_reads=3, min_reads_on_loss=2, lost_after=60, dedupe_seconds=120,
                 early_margin=3.0, final_margin=2.0):
        self.min_reads = min_reads
        self.min_reads_on_loss = min_reads_on_loss
        self.early_margin = early_margin
        self.final_margin = final_margin
        self.lost_after = lost_after
        self.dedupe_seconds = dedupe_seconds
        self.votes = defaultdict(lambda: defaultdict(list))  # track -> number -> [(score, weight)]
        self.last_seen = {}
        self.emitted = {}         # track -> number already logged
        self.recent = {}          # number -> time last logged

    def add(self, track_id, number, score, frame_idx, text_height=1.0):
        self.last_seen[track_id] = frame_idx
        if number is not None and track_id not in self.emitted:
            self.votes[track_id][number].append((score, score * (text_height or 1.0)))

    def seen(self, track_id, frame_idx):
        self.last_seen[track_id] = frame_idx

    def _leader(self, track_id, needed, margin):
        weights = Counter({n: sum(w for _, w in v) for n, v in self.votes[track_id].items()})
        if not weights:
            return None
        ranked = weights.most_common(2)
        top, top_w = ranked[0]
        second_w = ranked[1][1] if len(ranked) > 1 else 0.0
        if len(self.votes[track_id][top]) >= needed and top_w >= margin * second_w:
            return top
        return None

    def _emit(self, track_id, number, now, frame_idx, reason):
        self.emitted[track_id] = number
        if now - self.recent.get(number, -1e18) < self.dedupe_seconds:
            return None
        self.recent[number] = now
        scores = [s for s, _ in self.votes[track_id][number]]
        return {'time': now, 'frame': frame_idx, 'track_id': int(track_id), 'number': number,
                'confidence': round(float(np.mean(scores)), 3), 'reads': len(scores), 'reason': reason}

    def step(self, now, frame_idx):
        """Return the list of newly decided rows for this frame."""
        rows = []
        for tid in list(self.votes):
            if tid in self.emitted:
                continue
            number = self._leader(tid, self.min_reads, self.early_margin)
            if number:
                row = self._emit(tid, number, now, frame_idx, 'confirmed')
                if row:
                    rows.append(row)
        for tid, last in list(self.last_seen.items()):
            if frame_idx - last <= self.lost_after:
                continue
            if tid not in self.emitted:
                number = self._leader(tid, self.min_reads_on_loss, self.final_margin)
                if number:
                    row = self._emit(tid, number, now, frame_idx, 'track_lost')
                    if row:
                        rows.append(row)
            self.votes.pop(tid, None)
            self.last_seen.pop(tid, None)
            self.emitted.pop(tid, None)
        return rows

    def flush(self, now, frame_idx):
        """Decide every remaining track, e.g. at the end of a video file."""
        for tid in self.last_seen:
            self.last_seen[tid] = -10 ** 9
        return self.step(now, frame_idx)

    def best_guess(self, track_id):
        """(number, reads) for the overlay, whether or not it is confirmed yet."""
        if track_id in self.emitted:
            return self.emitted[track_id], None
        votes = self.votes.get(track_id, {})
        if not votes:
            return None, 0
        number = max(votes, key=lambda n: sum(w for _, w in votes[n]))
        return number, len(votes[number])


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
class CsvLogger:
    """Append-only CSV writer; replaces the repo's write-everything-at-the-end write_csv."""

    FIELDS = ['timestamp', 'frame', 'track_id', 'number', 'confidence', 'reads', 'reason']

    def __init__(self, path, live):
        self.path = path
        self.live = live
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        self.f = open(path, 'a', newline='')
        self.w = csv.DictWriter(self.f, fieldnames=self.FIELDS)
        if new:
            self.w.writeheader()
            self.f.flush()

    def write(self, row):
        t = row['time']
        ts = datetime.fromtimestamp(t).isoformat(timespec='seconds') if self.live else f'{t:.2f}s'
        self.w.writerow({'timestamp': ts, **{k: row[k] for k in self.FIELDS[1:]}})
        self.f.flush()
        print(f"[{ts}] car {row['track_id']}: {row['number']} "
              f"(conf {row['confidence']}, {row['reads']} reads, {row['reason']})", flush=True)

    def close(self):
        self.f.close()


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=4, line_length=40):
    """Corner-bracket box, from the repo's visualize.py."""
    x1, y1 = top_left
    x2, y2 = bottom_right
    for (x, y, dx, dy) in [(x1, y1, 1, 1), (x1, y2, 1, -1), (x2, y1, -1, 1), (x2, y2, -1, -1)]:
        cv2.line(img, (x, y), (x, y + dy * line_length), color, thickness)
        cv2.line(img, (x, y), (x + dx * line_length, y), color, thickness)
    return img


def _label(frame, text, org, color, scale=0.9):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def draw_overlay(frame, tracks, voter, card_boxes=(), fps=None, reads=(), logged=()):
    """Draw tracked cars with their current or confirmed number, plus card boxes.

    reads: this frame's OCR attempts as (box, number or None, score); the box
    is where OCR looked. logged: numbers written to the CSV so far.
    """
    for (x1, y1, x2, y2), number, score in reads:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 1)
        text = f'read {number} ({score:.2f})' if number else 'read: no 4-digit text'
        _label(frame, text, (int(x1) + 8, int(y2) - 12), (255, 255, 0), 0.7)
    if logged:
        _label(frame, 'Logged: ' + '  '.join(logged[-6:]), (10, frame.shape[0] - 20), (0, 255, 0), 1.0)
    for x1, y1, x2, y2, tid in tracks:
        number, reads = voter.best_guess(tid)
        confirmed = reads is None
        color = (0, 255, 0) if confirmed else (0, 200, 255)
        draw_border(frame, (int(x1), int(y1)), (int(x2), int(y2)), color)
        label = f'#{int(tid)}'
        if number:
            label += f'  {number}' + ('' if confirmed else f' ({reads})')
        cv2.putText(frame, label, (int(x1), max(30, int(y1) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    for x1, y1, x2, y2 in card_boxes:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    if fps:
        cv2.putText(frame, f'{fps:.1f} FPS', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return frame


class VideoSaver:
    """Write annotated frames to an .mp4, opened on the first frame so size is known."""

    def __init__(self, path, fps):
        self.path, self.fps, self.writer = path, fps, None

    def write(self, frame):
        if not self.path:
            return
        if self.writer is None:
            h, w = frame.shape[:2]
            self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w, h))
        self.writer.write(frame)

    def close(self):
        if self.writer is not None:
            self.writer.release()
