"""

Score a run's CSV against the numbers that actually came through.

    python main.py --source held_out_clip.mp4 --out eval.csv
    python evaluate.py --pred eval.csv --truth held_out_truth.txt

"""
import argparse
import csv
from collections import Counter


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pred', required=True, help='CSV from main.py or baseline.py')
    p.add_argument('--truth', required=True, help='one true card number per line')
    args = p.parse_args()

    with open(args.pred, newline='') as f:
        pred = Counter(row['number'].zfill(4) for row in csv.DictReader(f))
    with open(args.truth) as f:
        truth = Counter(line.strip().zfill(4) for line in f if line.strip())

    correct = sum((pred & truth).values())
    wrong = sum((pred - truth).values())
    missed = sum((truth - pred).values())
    cars = sum(truth.values())
    
    print(f'cars in clip : {cars}')
    print(f'correct      : {correct} ({correct / max(cars, 1):.0%})')
    print(f'wrong        : {wrong}   <- target 0')
    print(f'missed       : {missed} ({missed / max(cars, 1):.0%})')
    if wrong:
        print('logged but not in truth:', ', '.join(sorted((pred - truth).elements())))
    if missed:
        print('in truth but not logged:', ', '.join(sorted((truth - pred).elements())))


if __name__ == '__main__':
    main()
