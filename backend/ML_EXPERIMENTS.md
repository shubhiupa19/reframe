# ML Model Experiments Log

A record of everything tried to improve the cognitive distortion classifier's accuracy,
in chronological order, with results and takeaways. Useful for talking through the
iteration process in interviews — what was tried, why, what happened, and what it implies.

## 1. Baseline: TF-IDF + Logistic Regression

- **Setup**: `TfidfVectorizer` (bigrams, 5000 features) → `LogisticRegression`
- **Result**: ~34% accuracy, 2,530 samples, 11 classes
- Class sizes were imbalanced: "No Distortion" had 933 samples, smallest class
  ("All-or-nothing thinking") had 100.

## 2. Data augmentation attempts

- **Attempt 1 — downsample "No Distortion" to 200**: accuracy _dropped_ to 18%
  (1,797 total samples). Lesson: can't fix imbalance by just removing data when
  already data-starved overall — it just makes the dataset smaller.
- **Fix — augment first, then downsample**: added AI-generated sentence-level
  samples (1,529 rows in `augmented_data.csv`) to bulk up small classes, _then_
  downsampled "No Distortion" to 400. This became the standing data pipeline for
  every experiment after.
- Also switched training examples from full paragraphs to individual sentences
  (matches how the model is actually used at inference time — sentence by
  sentence). Concretely: paragraphs _with_ a distortion contribute one row each
  (just the `Distorted part` span — a column the Kaggle dataset itself provides,
  not something this code derives) → 1,597 rows. Paragraphs with _no_ distortion
  get exploded into individual sentences (6,536 of them), all labeled "No
  Distortion," then downsampled to 400. Final training set:
  1,597 + 400 + 1,529 (augmented) = **3,526 rows**, plus whatever user feedback
  has accumulated at retrain time. (The raw "2,530" figure from the baseline
  above is paragraph count, not final row count — the two are easy to conflate
  since sentence-splitting wasn't introduced until this step.)
- **Data leakage risk — fixed (2026-07-26)**: the paragraph→sentence split
  happens _before_ the train/test split, and a "No Distortion" paragraph can
  contribute several sentences (same writer, same story, overlapping
  vocabulary/tone). A plain random split could land some of those sentences in
  train and others in test purely by chance — not the model overfitting, but
  the _test set_ not being fully independent of train for that one class.
  Rows with a distortion label were never at risk (each paragraph contributes
  exactly one row, the `Distorted part` span), so only the ~400 "No
  Distortion" sentence rows were exposed.
  - **Fix**: both `train_embeddings_lr.py` and `train_distilbert.py` now tag every row
    with a `group` (the source paragraph id for "No Distortion" sentences;
    a unique id per row for everything else, since those rows were never at
    risk) and split with `GroupShuffleSplit` keyed on that group, instead of a
    plain random/stratified split. Verified directly: after the split, the set
    of groups in train and the set of groups in test/holdout have zero
    overlap.
  - **Tradeoff**: `GroupShuffleSplit` doesn't support `stratify`, so the
    resulting test/validation sets no longer have exactly even class counts
    (`train_distilbert.py`'s post-fix test set ranged from 18 to 44 examples
    per class, vs. a roughly even ~32 before). This means run-to-run accuracy
    is noisier than before — see section 7 for what this looked like in
    practice.
  - **Did it actually change "No Distortion" accuracy?** Not meaningfully.
    Comparing the affected class specifically: 0.69 f1 before the fix vs. 0.70
    f1 after (`train_distilbert.py`, section 7). If leakage had been
    materially propping that number up, fixing it should have dropped it
    noticeably — it didn't. The fix was still worth doing (it closes a real
    methodological gap and the risk was structural, not something you'd want
    to just eyeball away), but the leakage doesn't appear to have been a big
    driver of the reported numbers in practice.

## 3. Tried SVM

- Swapped `LogisticRegression` for an SVM classifier on the TF-IDF features.
- **Result**: no improvement over the TF-IDF + LR baseline. Abandoned in favor of
  changing the feature representation instead (see next).

## 4. Switched TF-IDF → SentenceTransformer embeddings

- **Setup**: `all-MiniLM-L6-v2` (384-dim frozen embeddings) → `LogisticRegression`
  (`class_weight="balanced"`, `max_iter=1000`, `C=1.0`, `solver="lbfgs"`)
- **Result**: ~49% accuracy — the single biggest jump in the whole project (34% → 49%).
  Semantic embeddings capture meaning similarity that raw word-frequency vectors can't
  (e.g. "I always ruin everything" and "Nothing I do ever works out" land close together
  even with zero shared words).
- This was the model deployed to production for a while (`train_embeddings_lr.py` /
  `distortion_model.pkl`), until the fine-tuned DistilBERT model (section 7) replaced
  it in `app.py` on 2026-07-26.
- **Re-run after the `GroupShuffleSplit` leakage fix (2026-07-26)**: 49.64% accuracy,
  699 test rows — consistent with the pre-fix ~49% number, same conclusion as section 2:
  the leakage wasn't a major driver of the reported accuracy.
- **Mental model worth keeping**: MiniLM "manicures" the text into a shaped,
  comparable form (the 384-dim vector) as a separate, one-time step, _before_
  LogisticRegression ever sees it or the 11 labels. LogisticRegression can only
  draw dividing lines through whatever shape it's handed — it has no way to
  ask for the data to be reshaped. That framing is why the next several
  experiments (tuning, bigger encoder) all plateau in the same band: none of
  them can touch the manicuring step itself.

## 5. Hyperparameter tuning (GridSearchCV) — three rounds, all near-identical results

| Round | What was swept                                  | Best params                       | Best score      |
| ----- | ----------------------------------------------- | --------------------------------- | --------------- |
| 1     | `C` only, no scaling                            | `C=10.0`                          | 48.39% macro F1 |
| 2     | `C` only, with `StandardScaler` added before LR | `C=0.01`                          | 49.18% macro F1 |
| 3     | `C` × `penalty` (l1/l2) × `solver` (lbfgs/saga) | `C=0.01, penalty=l2, solver=saga` | 49.22% macro F1 |

- Notable: adding `StandardScaler` flipped the best `C` from 10.0 to 0.01 — a good
  illustration of why L2 regularization strength is scale-dependent (unscaled
  high-variance embedding dimensions need less penalty per unit of `C`; scaled
  features need more).
- `l1` penalty never won — `l2` was consistently at least as good.
- **Takeaway**: three independent hyperparameter searches all topped out in the same
  48–49% band. The classifier isn't leaving performance on the table — regularization
  strength, penalty type, and solver choice aren't the bottleneck.

## 6. Embedding model swap — MiniLM vs mpnet

- Tried `all-mpnet-base-v2` (768-dim, larger/more accurate pretrained encoder) as a
  drop-in replacement for `all-MiniLM-L6-v2`, same classifier config, as an isolated
  experiment (not deployed — `app.py` still uses the committed local MiniLM encoder).
- **Result**: 49.72% accuracy vs MiniLM's 49.01% — about a 0.7-point bump. Per-class
  results were a mixed bag (some classes improved, e.g. All-or-nothing thinking; others
  got worse, e.g. Magnification) rather than a clean win across the board.
- **Takeaway**: a meaningfully bigger/better frozen embedding model still landed in the
  same ~48–50% band as every other experiment. This is the fourth independent lever
  (classifier tuning ×3, embedding model ×1) that produced only marginal movement —
  strong evidence the ceiling here is about **data quality/quantity or task difficulty**
  (genuinely overlapping categories like "Overgeneralization" vs "All-or-nothing
  thinking"), not about squeezing more out of a frozen-embeddings + linear-classifier
  approach.

## 7. Fine-tuned DistilBERT

- **Implementation**: `train_distilbert.py` reproduces this experiment and saves the
  fine-tuned model, tokenizer, and label mappings under
  `models/distilbert-cognitive-distortions-improved/` (generated artifacts are
  gitignored).
- Instead of frozen embeddings + a separate classifier, fine-tuned DistilBERT
  end-to-end on the same dataset.
- **Reproduced locally to verify the claim**: re-implemented the same approach
  (`distilbert-base-uncased`, `Trainer` API, 3 epochs, `lr=2e-5`, same train/test split
  as every other experiment) and fine-tuned on Apple Silicon (MPS). **Result: 63.88%
  accuracy** — confirms the original claim was real and reproducible, not a stale/
  inflated note.
- **Improved reproducible run**: the committed script now deduplicates exact examples,
  uses stratified 80/10/10 train/validation/test splits, trains for up to 6 epochs with
  warmup, weight decay, label smoothing, and early stopping, and selects checkpoints by
  validation macro F1. The best checkpoint was epoch 4. **Untouched test result: 65.62%
  accuracy and 65.82% macro F1** (352 examples). Validation peaked at 66.48% accuracy
  and 66.42% macro F1. This is the preferred result because the final test set is no
  longer also used for checkpoint selection.
- **Re-run after the `GroupShuffleSplit` leakage fix (2026-07-26)**: swapped the
  stratified `train_test_split` above for a group-aware split keyed on source paragraph
  (see section 2). **Test result: 68.27% accuracy, 69.20% macro F1** (353 examples, best
  checkpoint at step 528/1056). This is *higher* than the pre-fix run, not lower —
  worth being precise about why: `GroupShuffleSplit` doesn't support `stratify`, so this
  test set has uneven class counts (18–44 examples per class) instead of the previous
  even split, and a smaller/uneven test set carries more run-to-run variance on its own.
  The one number that actually isolates the leakage effect — "No Distortion" f1, the
  only class that was ever at risk — barely moved (0.69 → 0.70), which suggests the
  leakage wasn't a meaningful driver of the reported score either way. Both numbers
  above are legitimate; the post-fix one is simply the more methodologically sound of
  the two.
- **By far the best result across every experiment** — every class landed at f1 ≥ 0.48
  (vs. several classes stuck around 0.3 with frozen embeddings). In the post-fix run,
  strongest: Should statements (0.81), Labeling (0.75), All-or-nothing thinking (0.75),
  Mental filter (0.74). Weakest: Magnification (0.55), Emotional Reasoning (0.61) —
  still among the hardest classes, but meaningfully improved from the frozen-embedding
  approach either way.
- **Deployed to production (2026-07-26)** — `app.py` now loads this model via a
  Hugging Face `pipeline("text-classification", ...)`, replacing the SentenceTransformer
  + LogisticRegression path from section 4. `backend/.gitignore` was narrowed so the
  deploy workflow (which does a fresh `git init` + `git add .` inside `backend/` and
  pushes to the HF Space) actually includes `config.json`, `model.safetensors`,
  `tokenizer.json`, and `tokenizer_config.json` — it previously blanket-ignored the
  whole model folder, which would have silently kept the fine-tuned model out of every
  deploy. Training checkpoints (`checkpoint-*/`, up to 767MB each) stay excluded, since
  only the final `save_model()` output is needed for inference.
- **Why this jump makes sense**: end-to-end fine-tuning lets the transformer's own
  representations adapt specifically to this classification task, instead of relying on
  generic sentence-similarity embeddings that were never trained with these 11 labels
  in mind. It also comes with real tradeoffs frozen-embedding approaches don't have —
  meaningfully higher overfitting risk on a ~3,500-example dataset, needing careful
  regularization (small learning rate, few epochs, early stopping), and a heavier
  model to deploy.
- **Manicuring vs. learning, collapsed into one loop**: with frozen MiniLM,
  "shape the data" and "learn from the shaped data" are two separate,
  sequential steps done by two separate things — only the second one ever
  sees the 11 labels. Fine-tuning DistilBERT merges them: gradients flow all
  the way back through every transformer layer, not just a final classifier
  head, so the layers doing the "manicuring" get the same feedback signal as
  the classifier and can reshape themselves around what actually separates
  the 11 classes. That's the real mechanism behind breaking the frozen-embedding
  ceiling — not "a better classifier," but "the representation itself is no
  longer stuck being generic."
- Note DistilBERT's own final layer is structurally the same thing as
  LogisticRegression (a dot product per class + softmax) — fine-tuning didn't
  replace LogisticRegression, it just stopped freezing everything upstream of it.

## 8. Planned: recall-weighted scoring metric (not yet run)

- **Decision (2026-07-15)**: `tune_embeddings_lr.py`'s `GridSearchCV` currently scores on
  `f1_macro`, which weighs precision and recall equally. For this app, missing a
  real distortion (false negative) is worse than flagging a sentence that isn't
  actually distorted (false positive) — the whole point of the tool is
  surfacing patterns for the user to reflect on, and a missed one means that
  chance never happens at all.
- **But not an aggressive recall lean**: false positives aren't free here either
  — this is a mental-health-adjacent tool, and wrongly telling someone their
  healthy thought is "distorted" can plant self-doubt or erode trust in the
  tool, which would hurt real recall in practice if users start ignoring
  flagged sentences altogether.
- **Planned change**: swap `scoring='f1_macro'` for a recall-weighted F-beta
  score (`fbeta_score` with `beta≈1.2–1.5`, via `sklearn.metrics.make_scorer`)
  — a mild lean toward recall, not the aggressive `beta=2` convention.
- Not yet run. Once done, this section should be updated with the actual best
  params and score, same as sections 5 and 6.

## 9. Fixed: tuning script had drifted out of sync with the leakage fix

- **Found (2026-07-28)**: while extracting the data-loading logic shared by
  `train_embeddings_lr.py` and `tune_embeddings_lr.py` into a common
  `data_prep.py`, noticed `tune_embeddings_lr.py` had never been updated
  alongside the group-aware leakage fix in section 2 — it still tagged no
  rows with a `group` and used a plain `train_test_split` plus a plain
  integer `cv` (regular `KFold`) inside `GridSearchCV`.
- **Fix**: `tune_embeddings_lr.py` now uses the same `GroupShuffleSplit` for
  its outer train/test split, and `GroupKFold` (instead of a bare integer)
  for `GridSearchCV`'s internal cross-validation, so a paragraph's sentences
  can no longer end up split across CV folds either.
- **Re-run with the fix (2026-07-28)**: `train_embeddings_lr.py` reproduced its
  documented post-leakage-fix result exactly (49.64% accuracy, 699 test rows),
  confirming the refactor didn't change its behavior. `tune_embeddings_lr.py`'s
  `GridSearchCV` with the new `GroupShuffleSplit`/`GroupKFold` came back with
  **best parameters `C=0.01, penalty=l2, solver=saga`, F1 49.14%** — the same
  best combination as the pre-fix round 3 result in section 5 (49.22%), a
  ~0.08-point difference well within normal run-to-run noise. As expected given
  how little the leakage fix moved anything else in this project, fixing the
  tuning script's methodology didn't change its conclusion either.

## Overall narrative

1. Representation matters far more than classifier tuning: switching TF-IDF →
   embeddings (34% → 49%) dwarfs every hyperparameter search that followed (all within
   ~1 point of each other).
2. Once on frozen embeddings, four different levers (regularization strength, penalty/
   solver, feature scaling, embedding model size) all independently converged on the
   same ~48–50% ceiling — a clear signal that further gains need a different kind of
   change, not more tuning of the same setup.
3. Fine-tuning the transformer itself broke through that ceiling (63.88%, later
   68.27% after fixing the paragraph-leakage split — see section 7), consistent with
   the idea that generic frozen embeddings, however good, aren't as informative as
   representations learned specifically for this task. This model is what's actually
   deployed now.
4. Next real lever isn't more GridSearchCV — it's targeted data collection/
   augmentation for the worst-performing classes (Magnification, Emotional Reasoning).
