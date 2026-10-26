# Assignment 2 Report

- **Name**: Gonzalo Angulo
- **Student ID**: 18620
- **Email**: gangulo.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Business problem, and the cost of being wrong

An inspection train photographs the track and produces around 20,000 images a
day. Today a dispatcher decides, image by image, whether a stretch of track
needs a maintenance crew. The model supports that decision: for each image it
says **pass** (nothing to do), **fail** (send a crew) or **review** (an
inspector looks first). The dispatcher acts on fail and review; nobody looks at
a pass.

The decision unit is one track image. It is positive if it contains at least
one defect instance (`Missing Fastener`, `Defective Joint` or
`Missing Dogscrew`); the model's score for the image is the highest confidence
among its defect detections.

The two mistakes cost very different amounts:

- **False negative** (defect present, image passed): the fastener stays missing
  until the next inspection cycle. I cost this at **€2,000**: an emergency
  repair plus a speed restriction is in the order of €20,000 once it is
  discovered, and I assume roughly one missed fastener in ten gets to that point
  before the next scheduled pass. The remaining nine are caught later at normal
  repair cost, which is the same cost the crew would have incurred anyway.
- **False positive** (no defect, image failed): a two-person crew drives to the
  location, finds nothing and comes back. Around two hours of crew time and a
  vehicle: **€150**.
- **Human review**: an inspector at ~€30/h needs about two minutes per image:
  **€1** per reviewed image. A reviewed image is assumed to be resolved
  correctly.
- True positives and true negatives carry **€0** extra: the repair after a
  correct fail happens whatever the model says, and a correct pass costs nothing.

So a miss is about 13× worse than a false alarm, and about 2,000× worse than a
review. Everything downstream follows from that ratio: the threshold sits low,
the review band is used to absorb false alarms rather than misses, and the
metric I care about per class is recall on the defect classes, not precision.

## Dataset

**Roboflow Universe:** workspace `railway-defects-osolv`, project
`fastener-detection-7a3vn`, **version 1**.
https://universe.roboflow.com/railway-defects-osolv/fastener-detection-7a3vn

One image is a close-up of a stretch of rail with its fastening hardware. Five
classes, object detection: `Good Fasteners`, `Good Joint` (the two negative
classes) and `Missing Fastener`, `Defective Joint`, `Missing Dogscrew` (the
three defect classes). CC BY 4.0, used as published, not forked.

It serves the problem rather than resembling it because the labels encode the
exact decision: the same hardware appears labelled as good or as defective, so
the model has to learn the difference that matters to the dispatcher, not just
"there is a fastener here". I rejected the PCB-defect datasets I looked at
first precisely because they contained only defective boards: with no negative
images there is no false-alarm rate to measure and the costing half of the
assignment disappears.

## What the dataset's composition implies

Computed from the label files (`artifacts/dataset_stats.json`):

| Split | Images | Images with a defect | Defective Joint | Good Fasteners | Good Joint | Missing Dogscrew | Missing Fastener |
|---|---|---|---|---|---|---|---|
| train | 2,057 | 903 (43.9%) | 299 | 1,433 | 385 | 68 | 727 |
| valid | 266 | 105 (39.5%) | 41 | 177 | 52 | 15 | 71 |
| test | 248 | 110 (44.4%) | 31 | 174 | 52 | 8 | 96 |

Median box covers 9% of the image (p10 1%, p90 45%). All annotations are
polygons rather than boxes, which I convert to their bounding box, the same
thing Ultralytics does at train time.

What this implies:

- **`Missing Dogscrew` cannot be trusted.** 68 training and 8 test instances.
  Any per-class number on it is noise; I still count it as a defect for the
  decision, but I do not report it as a result of its own.
- **`Defective Joint` is thin too** (31 test instances), and it is the class
  the model confuses most, so its precision has a wide margin of error.
- **The test split does not look like production.** 44% of test images contain
  a defect; a maintained line is nowhere near that. Nothing I measure on the
  test split transfers to "cost per 1,000 real images" without re-weighting by
  an assumed base rate, which I do in the costing section.
- **The images were augmented before upload.** Roboflow reports no
  augmentation (2,571 images split 2,057/266/248), but every file is named
  `aug_prefix_0_<n>…`, so the uploader generated augmented copies offline and
  Roboflow then split them randomly. Near-duplicates of training images are
  therefore almost certainly in the test split, and every test number below is
  an upper bound on what a genuinely unseen stretch of track would give.
- **Objects are large**, so a nano model at 640px is adequate; small-object
  recall is not the limiting factor here.
- The dataset is one uploader's images with no camera or line metadata, so I
  cannot say whether the test images are from the same track sections as the
  training images. If they are, every number below is optimistic.

## Finetuning and baselines

Object detection with `yolo11n`, finetuned from the COCO checkpoint for 60
epochs at 640px, batch 16, seed 42, default Ultralytics augmentation, on an
Apple M4 Max (MPS) in 54 minutes. Nano rather than small because the objects
are large and the detector was never going to be the bottleneck; the decision
layer is.

All three rows are evaluated on the same 248 test images, defect classes only,
IoU 0.5, confidence 0.25 (`artifacts/baselines.json`):

| Model | mAP50 | mAP50-95 | Precision (defects) | Recall (defects) | TP / FP / FN |
|---|---|---|---|---|---|
| Finetuned yolo11n | **0.880** | **0.578** | 0.829 | 0.682 | 92 / 19 / 43 |
| COCO yolo11n, zero-shot | 0 | 0 | 0 | 0 | 0 / 24 / 135 |
| Predict nothing | 0 | 0 | — | 0 | 0 / 0 / 135 |

The COCO model has no fastener classes, so its AP is zero by construction; the
fairest thing I can report is class-agnostic overlap, and even then none of its
24 boxes overlaps a defect. The gap is as large as expected: this is a domain
COCO has never seen. The informative comparison is against predict-nothing,
which the finetuned model beats on recall but which costs the same as "do
nothing" in the next section — that is the number the model has to justify
itself against.

## Per-class evaluation and failure modes

Test split, Ultralytics `val` (`artifacts/metrics_finetuned.json`):

| Class | Role | Test instances | Precision | Recall | AP50 | AP50-95 |
|---|---|---|---|---|---|---|
| Good Joint | negative | 52 | 0.908 | 0.904 | **0.967** | 0.750 |
| Missing Fastener | **defect** | 96 | 0.873 | 0.896 | 0.917 | 0.436 |
| Good Fasteners | negative | 174 | 0.888 | 0.869 | 0.867 | 0.561 |
| Defective Joint | **defect** | 31 | 0.782 | 0.903 | 0.852 | 0.573 |
| Missing Dogscrew | defect | 8 | 1.000 | 0.681 | 0.800 | 0.572 |
| all | | 361 | 0.889 | 0.850 | 0.880 | 0.578 |

The headline mAP50 is carried by `Good Joint` and `Missing Fastener`.
`Defective Joint` drags: at conf 0.25 the confusion matrix shows it being
called `Good Joint` and `Good Joint` being called `Defective Joint` in both
directions, roughly a sixth of each class. That is the one confusion that
costs money, because a defective joint labelled "good" produces an image with
no defect score at all, which no threshold can recover. `Missing Fastener` is
mostly confused with `Good Fasteners` rather than missed outright.
`Missing Dogscrew`'s row is eight instances and should be read as "unknown".

At the image level the failure is different from what the confusion matrix
suggests, and worse: at the chosen edges **29 of the 110 defective test images
pass**, and in **26 of those 29 the model produces no box of any class above
0.25** — it does not call the defect "good", it does not see the hardware at
all. Those images contain 19 `Missing Fastener`, 11 `Defective Joint` and 5
`Missing Dogscrew` instances; five of the eight test Dogscrews are lost this
way. The joint confusion costs precision; the blind images cost recall, and
recall is what the cost matrix punishes. No threshold reaches an image whose
highest defect score is below 0.07, so the 26% floor on missed defective images
is a model problem, not an operating-point problem.

## Cost matrix and operating point

Costs per image, from the first section: FN €2,000, FP €150, review €1, TP and
TN €0. **Assumed base rate: 2%** of track images contain a defect. The test
split's 44% is a sampling artefact of how the dataset was built; a maintained
line with a regular inspection cycle runs at a few percent at most, and 2% is
my central estimate. All costs below are per 1,000 production images,
re-weighted from the test split's per-class rates to that base rate
(`decision.py`, `artifacts/cost_sweep.csv`, `artifacts/cost_vs_threshold.png`).

Sweeping a single threshold from 0.01 to 0.99: the minimum is at **t = 0.07**,
€15,872 per 1,000 images (5.3 missed defects, 35.5 false dispatches per 1,000).
The curve is flat between 0.07 and 0.30 and rises steeply above 0.5, where
recall on `Defective Joint` collapses. Below 0.07 false dispatches explode
because almost every negative image has some low-confidence defect box.

With a review band the optimum becomes **pass below 0.07, review in
[0.07, 0.58), fail at 0.58 and above**: **€13,758 per 1,000 images**, 5.3 missed
defects, 21.3 false dispatches, 1.7% of images reviewed. The band removes
false alarms, not misses: every defective image that passes at 0.07 still
passes, because its score is below the band. Images in the band are mostly
negatives with a medium-confidence box, which an inspector dismisses in two
minutes instead of a crew in two hours.

The assumption that would most change the answer is the **base rate**. Missed
defects per 1,000 scale linearly with it: at 5% the same operating point costs
€29,567 and misses 13 defects per 1,000; at 1% it costs about €8,500. The
second is the **FN cost**: the model beats doing nothing as long as a missed
fastener costs more than about €220, so the conclusion is robust to it being
wrong by a factor of five, but the review band stops paying for itself if FN
drops below the FP cost. The ratio, not the absolute number, is what matters.

## What a person still has to check

Edges 0.07 and 0.58 send **1.7%** of images to review. At 20,000 images a day
that is **342 images, 11.4 inspector-hours, 1.6 inspectors** at two minutes
per image. Two inspectors on one operations desk is realistic; that is the
capacity I set, and the optimiser chose the band under that cap. Without the
cap it would rather review 5.8% of images (€10,239 per 1,000) and need 5.5
inspectors, which is a hiring decision, not a threshold.

Priced against the alternatives, per 1,000 images at 2% base rate:

| Policy | Cost | Missed defects | False dispatches | People |
|---|---|---|---|---|
| Do nothing (pass everything) | €40,000 | 20 | 0 | 0 |
| Human checks everything | €1,000 | 0 | 0 | **95 inspectors** (667 h/day) |
| Model alone, t = 0.07 | €15,872 | 5.3 | 35.5 | 0 |
| **Model + review, [0.07, 0.58)** | **€13,758** | 5.3 | 21.3 | 1.6 |
| Model + review, no capacity cap | €10,239 | 5.1 | 0 | 5.5 |

"Human checks everything" is the cheapest line on paper and the one I am not
choosing: at €1 an image it only looks cheap because 20,000 images a day is
667 hours of looking, which nobody is staffing for an image stream where 98%
of frames are fine. The model's job is to turn 20,000 images into 342 an
inspector can actually look at. What the model does not do is reduce missed
defects below 5 per 1,000 — that floor is the 26% of defective images it does
not score at all, and it is the thing to fix next.

## Limitations & next steps

- **Blind images are the cost driver.** 26 defective test images get no
  detection of any class. The next concrete step is to look at those 26 and
  find what they share (lighting, angle, scale, an augmentation artefact from
  the uploader's pipeline) and then either add that condition to training or,
  cheaper, add a fourth rule: an image with *no* box of any class above 0.25
  goes to review, since a track image with no visible fastener is itself
  suspicious. At 2% base rate and €1 a review that is nearly free, and it is
  the only lever that moves the 5.3 missed defects per 1,000.
- **The joint confusion costs precision.** `Defective Joint` vs `Good Joint`
  in both directions. More joint examples (31 test, 299 train is thin for the
  hardest class) or a two-stage classifier on joint crops.
- **Base rate and FN cost are assumptions, not measurements.** The first thing
  to do with a real operator is replace 2% with their defect rate per segment
  and €2,000 with their incident and repair accounting. The sweep re-runs in
  seconds (`SWEEP_ONLY=1`).
- **Test split provenance.** The `aug_prefix` file names mean augmented copies
  of the same source photos are spread across train and test, so 0.88 mAP is
  an upper bound. With more time I would de-duplicate by source image before
  splitting; with a real operator I would hold out a line, or at least a date.
- **`Missing Dogscrew` is unmeasured.** Eight instances. Either collect more or
  merge it into `Missing Fastener` for the decision, which is how it is treated
  here anyway.
- **Review assumes the inspector is right.** The costing treats a reviewed
  image as resolved correctly. Inspector error at 342 images a day is small but
  not zero; it would add a few percent to the missed-defect line.

## Generative AI use disclosure

I used Claude (Anthropic) as a coding and writing assistant for this
assignment. It wrote the boilerplate: the Roboflow download, the Ultralytics
training and validation calls, the prediction dump, the threshold sweep
implementation, the Gradio dashboard, and the plots. It also formatted this
report from my numbers and notes and helped me phrase sections. The decisions
graded in this assignment are mine: the choice of problem and of this dataset
over the PCB sets, the per-image decision unit, every euro figure and the
base rate, the inspector capacity, and the operating point. Claude proposed
candidate values for some of these; I checked each one, changed the review
cost and the base rate, and can defend them without the code in front of me.
