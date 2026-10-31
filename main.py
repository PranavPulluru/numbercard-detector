<<<<<<< HEAD
"""Live pickup-line card reader.

Pipeline per frame:
    vehicles (COCO YOLOv8) -> SORT track IDs
    cards (models/card_detector.pt, trained with train.py)
    card -> car assignment -> crop -> EasyOCR -> 4-digit check
    per-car vote -> one CSV row per car once the number is certain

Examples:
    python main.py --source 0 --show                      
    python main.py --source rtsp://user:pass@cam/stream   
    python main.py --source held_out_clip.mp4 --out eval.csv
    python main.py --source samples/example_card.jpg      
    python main.py --source 0 --track-cards              
"""
=======
>>>>>>> 19a131e4ae369f583778cf6cf60ead877916e38c
import argparse
import time

import cv2
from ultralytics import YOLO

import util


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', default='0', help='camera index, RTSP URL, video file or image')
    p.add_argument('--out', default='cards.csv', help='CSV log, appended to across runs')
    p.add_argument('--card-model', default='models/card_detector.pt',
                   help='card detector (.pt, or a TensorRT .engine exported from it)')
    p.add_argument('--vehicle-model', default='yolov8n.pt')
    p.add_argument('--imgsz', type=int, default=1280, help='card detector input size; match training')
    p.add_argument('--card-conf', type=float, default=0.4)
    p.add_argument('--min-conf', type=float, default=0.4, help='minimum EasyOCR confidence')
    p.add_argument('--min-reads', type=int, default=3, help='agreeing reads needed to log a number')
    p.add_argument('--dedupe-seconds', type=float, default=120,
                   help='ignore the same number again within this window')
    p.add_argument('--track-cards', action='store_true',
                   help='track card boxes directly instead of cars (close-up cameras)')
    p.add_argument('--show', action='store_true', help='display a live overlay window')
    p.add_argument('--start', type=float, default=0.0, help='video files: start at this second')
    p.add_argument('--duration', type=float, default=None, help='video files: stop after this many seconds')
    p.add_argument('--save', default=None, help='write the annotated video to this .mp4')
    return p.parse_args()


def detect_cards(model, frame, imgsz, conf):
    res = model(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
    return [b[:5] for b in res.boxes.data.tolist()]


def read_crop(frame, box, min_conf):
    """(number, score, text height) for one card box."""
    x1, y1, x2, y2 = util.clip_box(box, frame.shape, pad=util.CARD_PAD)
    return util.read_card(frame[y1:y2, x1:x2], min_conf)


def run_image(args, card_model):
    frame = cv2.imread(args.source)
    if frame is None:
        raise SystemExit(f'Could not read image: {args.source}')
    cards = detect_cards(card_model, frame, args.imgsz, args.card_conf)
    if not cards:
        print('No card detected.')
    for box in cards:
        number, score, _ = read_crop(frame, box, args.min_conf)
        print(f'card {[round(v) for v in box[:4]]} (det {box[4]:.2f}): '
              + (f'{number} (ocr {score:.2f})' if number else 'unreadable'))


def main():
    args = parse_args()
    card_model = YOLO(args.card_model)
    if util.is_image(args.source):
        run_image(args, card_model)
        return

    vehicle_model = None if args.track_cards else YOLO(args.vehicle_model)
    tracker = util.Tracker()
    src = util.FrameSource(args.source, args.start, args.duration)
    voter = util.CardVoter(min_reads=args.min_reads, lost_after=int(src.fps * 2),
                           dedupe_seconds=args.dedupe_seconds)
    log = util.CsvLogger(args.out, live=src.live)
    saver = util.VideoSaver(args.save, src.fps)
    util.get_reader()

    t_prev, fps = time.time(), None
    now, idx = 0.0, 0
    try:
        for idx, now, frame in src:
            cards = detect_cards(card_model, frame, args.imgsz, args.card_conf)

            if args.track_cards:
                tracks = tracker.update(cards)
                assigned = [(t[:4], t[4]) for t in tracks]
            else:
                tracks = tracker.update(util.detect_vehicles(vehicle_model, frame))
                assigned = []
                for card in cards:
                    *_, car_id = util.get_car(card, tracks)
                    if car_id != -1:
                        assigned.append((card[:4], car_id))

            for *_, tid in tracks:
                voter.seen(tid, idx)
            for box, tid in assigned:
                number, score, height = read_crop(frame, box, args.min_conf)
                voter.add(tid, number, score, idx, height)

            for row in voter.step(now, idx):
                log.write(row)

            if args.show or args.save:
                t = time.time()
                inst = 1 / max(t - t_prev, 1e-6)
                fps = inst if fps is None else 0.9 * fps + 0.1 * inst
                t_prev = t
                overlay = util.draw_overlay(frame, tracks, voter, [c[:4] for c in cards],
                                            fps if args.show else None)
                saver.write(overlay)
                if args.show:
                    cv2.imshow('card reader', overlay)
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