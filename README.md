# Left vs right fist decoding from EEG

Predicts whether a person moved their **left** or **right** fist from 64-channel EEG data, using the PhysioNet [EEG Motor Movement/Imagery Database](https://physionet.org/content/eegmmidb/1.0.0/) (EEGMMIDB). Filter-bank CSP + shrinkage LDA is the final model used, trained on 101 subjects. On a person the model has never seen, it is right **62.6%** of the time (chance = 50%).

Full pipeline, reasoning and evaluation: `notebooks/executed_lr.ipynb`.

## Running the prediction script

```bash
python -m venv .venv && source .venv/bin/activate #may need to use python3 if on mac
pip install -r requirements.txt

python predict.py path/to/S019R03.edf     # add --compare to show the annotated labels
```

One prediction per cue in the file:

```
S019R03.edf  (run 3, executed, 15 cues)
 onset_s  pred   p_right
    4.10  left     0.317
   12.30  left     0.406
   20.50  right    0.713
```

```
S019R03.edf  (run 3, executed, 15 cues)
 onset_s  pred   p_right  annotated
    4.10  left     0.317  right
   12.30  left     0.406  left
   20.50  right    0.713  left
   28.70  left     0.338  right
   36.90  right    0.631  left
   45.10  right    0.581  right
   53.30  left     0.327  left
   61.50  right    0.710  right
   69.70  left     0.453  left
   77.90  right    0.537  right
   86.10  left     0.288  left
   94.30  right    0.733  right
  102.50  right    0.555  left
  110.70  right    0.703  right
  118.90  left     0.452  left
agreement: 0.667 (15 cues; with this few trials, expect wide swings from chance)

```

Accepts left/right fist runs — 3, 7, 11 (executed) or 4, 8, 12 (imagined) — with all 64 channels. Refuses fists/feet and baseline runs since the model was only trained for binary classification of left/right.

**Subjects the model never saw: 3, 19, 39, 68, 87.** These subjects were randomly selected and left out of training for use in the prediction script.

Note: Refer to comments at the top of predict.py for more details on running the prediction script if needed
