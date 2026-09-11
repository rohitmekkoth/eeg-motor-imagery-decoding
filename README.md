# Left vs right fist decoding from EEG

Predicts whether a person moved their **left** or **right** fist from 3 seconds of 64-channel EEG, using the PhysioNet [EEG Motor Movement/Imagery Database](https://physionet.org/content/eegmmidb/1.0.0/) (EEGMMIDB). Filter-bank CSP + shrinkage LDA, trained on 101 subjects. On a person the model has never seen, it is right **62.6%** of the time (chance = 50%).

Full pipeline, reasoning and evaluation: `notebooks/executed_lr.ipynb`.

## Running the prediction script

```bash
python -m venv .venv && source .venv/bin/activate
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

Accepts left/right fist runs — 3, 7, 11 (executed) or 4, 8, 12 (imagined) — with all 64 channels. Refuses fists/feet and baseline runs since the model was only trained for binary classification of left/right.

**Subjects the model never saw: 3, 19, 39, 68, 87.** These subjects were randomly selected and left out of training for use in the prediction script.
