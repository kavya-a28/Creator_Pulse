# Synthetic training data: effect on real comments

Generated 2026-10-26T04:34:50+00:00. All test comments are real. Labels on real comments are AI-generated (Claude) and not fully human-verified. Synthetic rows are training-only.

| Training data | Video acc | Video macro F1 | Random macro F1 | CV macro F1 (899) | CV macro F1, unseen (821) |
|---|---|---|---|---|---|
| real only | 0.667 | 0.427 [0.372, 0.474] | 0.492 | 0.456 | 0.452 |
| real + 250/class synthetic | 0.716 | 0.661 [0.475, 0.768] | 0.619 | 0.596 | 0.572 |
| real + 500/class synthetic | 0.721 | 0.661 [0.479, 0.765] | 0.571 | 0.606 | 0.573 |
| real + 1000/class synthetic | 0.721 | 0.682 [0.496, 0.786] | 0.583 | 0.618 | 0.581 |
| real + all synthetic (10002) | 0.716 | 0.649 [0.492, 0.759] | 0.652 | 0.622 | 0.578 |
| synthetic only | 0.508 | 0.462 [0.366, 0.548] | 0.448 | 0.439 | 0.396 |

Per-class recall, grouped CV (all 899 real comments):

| Training data | Question | Request | Confusion/Problem | Praise | Spam | Other |
|---|---|---|---|---|---|---|
| real only | 71/117 | 99/144 | 0/19 | 104/165 | 0/10 | 354/444 |
| real + 250/class synthetic | 80/117 | 108/144 | 8/19 | 111/165 | 3/10 | 338/444 |
| real + 500/class synthetic | 83/117 | 108/144 | 7/19 | 112/165 | 4/10 | 337/444 |
| real + 1000/class synthetic | 84/117 | 111/144 | 8/19 | 111/165 | 4/10 | 340/444 |
| real + all synthetic (10002) | 84/117 | 111/144 | 9/19 | 111/165 | 4/10 | 338/444 |
| synthetic only | 92/117 | 106/144 | 10/19 | 112/165 | 6/10 | 119/444 |

Per-class recall, grouped CV, unseen comments only:

| Training data | Question | Request | Confusion/Problem | Praise | Spam | Other |
|---|---|---|---|---|---|---|
| real only | 62/103 | 88/130 | 0/5 | 91/148 | 0/6 | 340/429 |
| real + 250/class synthetic | 71/103 | 96/130 | 2/5 | 97/148 | 3/6 | 325/429 |
| real + 500/class synthetic | 72/103 | 96/130 | 2/5 | 97/148 | 3/6 | 324/429 |
| real + 1000/class synthetic | 73/103 | 99/130 | 2/5 | 97/148 | 3/6 | 327/429 |
| real + all synthetic (10002) | 73/103 | 99/130 | 2/5 | 97/148 | 3/6 | 325/429 |
| synthetic only | 79/103 | 95/130 | 3/5 | 99/148 | 3/6 | 115/429 |

Paired bootstrap, video test macro F1 (all synthetic minus real only): +0.200, 95% CI [0.07, 0.334], synthetic better in 100% of resamples.

Trained and tested on synthetic data: accuracy 1.000, macro F1 1.000. This says nothing about real comments and must not be reported as model performance.
