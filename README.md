# Reframe - AI Journaling App and Cognitive Distortion Analyzer

1 in 5 adults experience a mental health condition each year. Therapy helps, but it's expensive, inaccessible, and happens once a week. The other 167 hours each week are unexamined.

**Reframe** brings a core CBT technique into your daily writing: automatically detecting the negative thought patterns (cognitive distortions) that fuel anxiety and depression, sentence by sentence, as you write.

---

## What it does

Paste or type a journal entry. The app splits it into sentences and classifies each one against 10 clinically-recognized cognitive distortions from CBT:

| Distortion | Example |
|---|---|
| All-or-Nothing Thinking | "I always mess everything up" |
| Overgeneralization | "Nobody ever listens to me" |
| Mind Reading | "She probably thinks I'm incompetent" |
| Fortune-telling | "This is never going to work out" |
| Emotional Reasoning | "I feel like a failure, so I must be one" |
| Labeling | "I'm such an idiot" |
| Should Statements | "I should be further along by now" |
| Mental Filtering | "The whole day was ruined" |
| Magnification | "This mistake is going to ruin my entire career" |
| Personalization | "It's my fault they're upset" |

Each sentence is highlighted and color-coded by distortion type with confidence scores. Hover to see definitions.

---

## Why it matters

Cognitive distortions are automatic. That's the problem; they don't feel irrational, they feel like facts. This tool makes the invisible *visible*, creating the kind of self-awareness that CBT therapists spend sessions trying to build.

Built at the intersection of NLP and clinical psychology, this project is an exploration of what accessible, AI-assisted mental health tooling could look like.

---

## Tech stack

**Frontend:** Next.js 16 (App Router), React 19, Tailwind CSS v4
**Backend:** Flask, Hugging Face `transformers`, PyTorch
**Model:** fine-tuned `distilbert-base-uncased`, trained end-to-end on ~3,500 labeled sentence-level examples across 11 classes (10 distortions + "No Distortion")
**Dataset:** [Cognitive Distortion Detection Dataset](https://www.kaggle.com/datasets/sagarikashreevastava/cognitive-distortion-detetction-dataset) (Kaggle) + LLM-augmented samples for underrepresented classes, grouped 80/10/10 train/validation/test split

---

## Getting started

```bash
# Clone the repo
git clone https://github.com/shubhiupa19/reframe.git
cd reframe

# Backend setup
cd backend
pip install -r requirements.txt
python app.py            # runs on http://127.0.0.1:5001
                          # fetches the fine-tuned DistilBERT weights from the
                          # public Hugging Face Hub model repo on first run

# Frontend setup (new terminal)
npm install
npm run dev               # runs on http://localhost:3000
```

---

## Roadmap

This is an actively developed project. Upcoming:

- **Multi-label classification** — letting a sentence carry more than one distortion at once, instead of forcing a single label (the dataset's own secondary-distortion annotations show real sentences often exhibit more than one)
- **Session history** — persistent journaling with longitudinal pattern analysis
- **Agentic reframing** — tool-use layer that generates CBT-style reframes for flagged sentences using therapist response data from the original dataset

See `backend/ML_EXPERIMENTS.md` for the full experiment log behind the model.

---

## Current limitations

Classification accuracy is 68.27% (69.2% macro F1) on a held-out, paragraph-grouped test set, best treated as a journaling aid that surfaces patterns for reflection, not a clinical diagnostic tool. Some distortion types (Magnification, Emotional Reasoning) are harder to distinguish from each other and score lower individually. The model currently assigns exactly one distortion per sentence, even though some sentences genuinely exhibit more than one.

---

## Background

Cognitive distortions were first described by psychiatrist Aaron Beck in the 1960s and are a cornerstone of Cognitive Behavioral Therapy. The 10 distortion types detected here are based on Beck's original framework as expanded by David Burns in *Feeling Good* (1980).

---

*Built by [Shubhi Upadhyay](https://github.com/shubhiupa19) — CS + Psychology, NYU '26*
