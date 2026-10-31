# Pickup-line number-card reader

Reads the 4-digit number cards on car dashboards at a school pickup line from a live camera, and logs one CSV row per car.

Adapted from [computervisioneng/automatic-number-plate-recognition-python-yolov8](https://github.com/computervisioneng/automatic-number-plate-recognition-python-yolov8). The vehicle tracking (YOLOv8 + SORT) is kept. The licence-plate detector is replaced by a card detector, the plate format by a 4-digit check, and the offline video flow by a live stream with per-car voting.

```
camera -> YOLOv8n (cars) -> SORT track IDs
       -> card_detector.pt -> card inside which car?
       -> crop -> CLAHE -> EasyOCR -> 4-digit check (largest text wins)
       -> vote across frames -> one CSV row per car
```

## Setup

```bash
git clone https://github.com/abewley/sort.git          # tracker, into ./sort
python -m venv ~/.venvs/card-reader                     # keep the venv out of OneDrive
~/.venvs/card-reader/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
~/.venvs/card-reader/Scripts/pip install -r requirements.txt
```

Activate the environment before running anything below (`source ~/.venvs/card-reader/Scripts/activate` in Git Bash, or `~\.venvs\card-reader\Scripts\Activate.ps1` in PowerShell). Model weights for YOLOv8, YOLO-World and EasyOCR download automatically on first run.

## Workflow

**1. Try the zero-training baseline.** It OCRs the windshield area of every tracked car. It is slower and noisier than the final pipeline, but needs no data.

```bash
python baseline.py --source samples/example_card.jpg
python baseline.py --source pickup_clip.mp4 --show
```

**2. Record and auto-label.** Record 2–3 pickup sessions from the final camera spot, ideally on different days and in different light. Then draft labels:

```bash
python autolabel.py --video recordings/2026-10-27_pm.mp4 --session 2026-10-27_pm
```

This writes `datasets/cards/<session>/images` and `labels` in YOLO format. Frames where nothing was found go to `unlabeled/`. Review every label in Label Studio, CVAT or Roboflow, fix bad boxes, and label the missed cards in `unlabeled/`. Aim for 300–500 frames that contain a card.

**3. Train the card detector.** Hold out whole sessions, not random frames.

```bash
python train.py --val-sessions 2026-10-29_pm --test-sessions 2026-10-30_am
```

The best weights are copied to `models/card_detector.pt`. Training uses image size 1280 because cards are small in a full frame.

**4. Run it live.**

```bash
python main.py --source 0 --show                         # USB camera
python main.py --source rtsp://user:pass@camera/stream   # IP camera
```

Rows are appended to `cards.csv` with columns timestamp, frame, track_id, number, confidence, reads, reason. Each read is weighted by OCR confidence times text height, because reads get more reliable as a car gets closer. A number is logged early once it has 3 reads and three times the weight of any rival. Otherwise the decision waits until the car leaves view, and needs 2 reads and twice the rival's weight. Anything less is dropped, because a wrong number is worse than a missed one.

To process part of a recording and save the annotated video, use `--start`, `--duration` and `--save`:

```bash
python baseline.py --source recording.mp4 --start 60 --duration 10 --save clip.mp4
``` If the camera is so close that whole cars aren't visible, add `--track-cards`.

If FPS is too low, export the detector to TensorRT and pass the `.engine` file:

```bash
yolo export model=models/card_detector.pt format=engine imgsz=1280 half=True
python main.py --source 0 --card-model models/card_detector.engine
```

**5. Measure accuracy on a held-out clip.** Write the true numbers, one per car, in a text file.

```bash
python main.py --source held_out_clip.mp4 --out eval.csv
python evaluate.py --pred eval.csv --truth held_out_truth.txt
```

## Files

| File | Purpose |
|---|---|
| `main.py` | Live pipeline with the trained card detector |
| `baseline.py` | No-training OCR pipeline, also useful as a fallback |
| `autolabel.py` | Draft YOLO labels from recordings via OCR and YOLO-World |
| `train.py` | Session-based split, training, and evaluation |
| `evaluate.py` | Correct, wrong and missed counts against ground truth |
| `debug_steps.py` | Saves an image of every pipeline step for one photo, to see where reading fails |
| `util.py` | OCR, 4-digit format, tracking, voting, CSV and overlay helpers |
| `data/card.yaml` | Dataset definition, one class `card` |

## Practical notes

- **Glare** is the biggest risk. A circular polarizing filter, a slightly downward camera angle, and a spot where cars slow down matter more than the model.
- **Privacy.** Only numbers and timestamps are logged. Recordings and datasets are git-ignored. Delete raw footage once labeling is done, since it shows students and families.
