"""Zero-training baseline: find 4-digit cards with OCR alone, no card detector.

Vehicles are detected with COCO YOLOv8 and tracked with SORT. EasyOCR runs
text detection + recognition on the windshield/dashboard region of each car
and keeps only 4-digit reads. Slower and noisier than main.py, but works on
day one, and its hits seed autolabel.py.

Examples:
    python baseline.py --source samples/example_card.jpg
    python baseline.py --source pickup_clip.mp4 --show
    python baseline.py --source 0 --show            # webcam / USB camera
"""
import argparse
import time

import cv2
from ultralytics import YOLO

import util


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', default='0', help='image, video file, camera index or RTSP URL')
    p.add_argument('--out', default='baseline.csv', help='CSV log of decided numbers')
    p.add_argument('--vehicle-model', default='yolov8n.pt')
    p.add_argument('--roi', type=float, nargs=2, default=(0.0, 0.7), metavar=('TOP', 'BOTTOM'),
                   help='vertical slice of each car box to OCR, as fractions (default: top 70%%, '
                        'which skips the licence plate)')
    p.add_argument('--ocr-every', type=int, default=3, help='OCR each car every N frames')
    p.add_argument('--min-car-width', type=int, default=160,
                   help='skip OCR on cars narrower than this many pixels (card too small to read)')
    p.add_argument('--start', type=float, default=0.0, help='video files: start at this second')
    p.add_argument('--duration', type=float, default=None, help='video files: stop after this many seconds')
    p.add_argument('--save', default=None, help='write the annotated video to this .mp4')
    p.add_argument('--min-conf', type=float, default=0.4, help='minimum EasyOCR confidence')
    p.add_argument('--min-reads', type=int, default=3, help='agreeing reads needed to log a number')
    p.add_argument('--show', action='store_true', help='display a live overlay window')
    return p.parse_args()


def run_image(path, min_conf):
    """Read every 4-digit number in a still image, no tracking."""
    frame = cv2.imread(path)
    if frame is None:
        raise SystemExit(f'Could not read image: {path}')
    hits = util.find_card_numbers(util.preprocess(frame, min_height=0), min_conf)
    if not hits:
        print('No 4-digit card number found.')
    for number, score, box in hits:
        print(f'{number}  (conf {score:.2f}, box {[round(v) for v in box]})')
    return hits


def main():
    args = parse_args()
    if util.is_image(args.source):
        run_image(args.source, args.min_conf)
        return

    vehicle_model = YOLO(args.vehicle_model)
    tracker = util.Tracker()
    src = util.FrameSource(args.source, args.start, args.duration)
    voter = util.CardVoter(min_reads=args.min_reads, lost_after=int(src.fps * 2))
    log = util.CsvLogger(args.out, live=src.live)
    saver = util.VideoSaver(args.save, src.fps)
    util.get_reader()  # load OCR weights before the first frame

    t_prev, fps = time.time(), None
    now, idx = 0.0, 0
    logged = []
    try:
        for idx, now, frame in src:
            tracks = tracker.update(util.detect_vehicles(vehicle_model, frame))
            reads = []
            for x1, y1, x2, y2, tid in tracks:
                voter.seen(tid, idx)
                if (idx + int(tid)) % args.ocr_every or x2 - x1 < args.min_car_width:
                    continue
                h = y2 - y1
                roi = util.clip_box((x1, y1 + args.roi[0] * h, x2, y1 + args.roi[1] * h), frame.shape)
                crop = frame[roi[1]:roi[3], roi[0]:roi[2]]
                number, score, height = util.read_card(crop, args.min_conf)
                voter.add(tid, number, score, idx, height)
                reads.append((roi, number, score))

            for row in voter.step(now, idx):
                log.write(row)
                logged.append(row['number'])

            if args.show or args.save:
                t = time.time()
                fps = 1 / max(t - t_prev, 1e-6) if fps is None else 0.9 * fps + 0.1 / max(t - t_prev, 1e-6)
                t_prev = t
                overlay = util.draw_overlay(frame, tracks, voter, fps=fps if args.show else None,
                                            reads=reads, logged=logged)
                saver.write(overlay)
                if args.show:
                    cv2.imshow('baseline', overlay)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
    except KeyboardInterrupt:
        pass
    finally:
        for row in voter.flush(now, idx):
            log.write(row)
        src.release()
        saver.close()
        log.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
