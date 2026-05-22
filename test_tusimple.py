import argparse
import json
import numpy as np
import os

from hungarian_matching import caculate_tp_fp_fn


def load_tusimple_gt(json_path):
    """
    Konwersja TuSimple GT do listy linii:
    Z lane'ów (lista x-ów dla ustalonych y) generujemy segmenty (x1,y1,x2,y2).
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    lanes = data["lanes"]
    h_samples = data["h_samples"]

    gt_lines = []

    for lane in lanes:
        points = []
        for x, y in zip(lane, h_samples):
            if x >= 0:  # -2 oznacza brak punktu
                points.append((x, y))

        # zamiana punktów lane’a na segmenty
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            gt_lines.append([x1, y1, x2, y2])

    return gt_lines


def main():
    parser = argparse.ArgumentParser(description="Evaluate TuSimple predictions")
    parser.add_argument("--pred", type=str, required=True)
    parser.add_argument("--gt", type=str, required=True)
    parser.add_argument("--align", default=False, action="store_true")
    args = parser.parse_args()

    pred_path = args.pred
    gt_path = args.gt

    filenames = sorted(os.listdir(pred_path))

    total_tp = np.zeros(99)
    total_fp = np.zeros(99)
    total_fn = np.zeros(99)

    total_tp_align = np.zeros(99)
    total_fp_align = np.zeros(99)
    total_fn_align = np.zeros(99)

    for filename in filenames:
        if not filename.endswith(".npy"):
            continue
        if "align" in filename:
            continue

        pred = np.load(os.path.join(pred_path, filename))

        if args.align:
            pred_align = np.load(
                os.path.join(pred_path, filename.split(".")[0] + "_align.npy")
            )

        gt_file = os.path.join(gt_path, filename)

        if not os.path.exists(gt_file):
            print(f"Missing GT for {filename}, skipping...")
            continue

        gt = np.load(gt_file, allow_pickle=True).tolist()
        gt = gt["coords"].astype(np.float32)

        for i in range(1, 100):
            thresh = i * 0.01

            tp, fp, fn = caculate_tp_fp_fn(pred.tolist(), gt, thresh=thresh)
            total_tp[i - 1] += tp
            total_fp[i - 1] += fp
            total_fn[i - 1] += fn

            if args.align:
                tp, fp, fn = caculate_tp_fp_fn(pred_align.tolist(), gt, thresh=thresh)
                total_tp_align[i - 1] += tp
                total_fp_align[i - 1] += fp
                total_fn_align[i - 1] += fn

    total_recall = total_tp / (total_tp + total_fn + 1e-6)
    total_precision = total_tp / (total_tp + total_fp + 1e-6)
    f = 2 * total_recall * total_precision / (total_recall + total_precision + 1e-6)

    print("Mean P:", total_precision.mean())
    print("Mean R:", total_recall.mean())
    print("Mean F:", f.mean())
    print("F@0.95:", f[94])

    if args.align:
        total_recall_align = total_tp_align / (total_tp_align + total_fn_align + 1e-6)
        total_precision_align = total_tp_align / (
            total_tp_align + total_fp_align + 1e-6
        )
        f_align = (
            2
            * total_recall_align
            * total_precision_align
            / (total_recall_align + total_precision_align + 1e-6)
        )

        print("Mean P_align:", total_precision_align.mean())
        print("Mean R_align:", total_recall_align.mean())
        print("Mean F_align:", f_align.mean())
        print("F_align@0.95:", f_align[94])


if __name__ == "__main__":
    main()
