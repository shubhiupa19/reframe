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

- **Attempt 1 — downsample "No Distortion" to 200**: accuracy *dropped* to 18%
  (1,797 total samples). Lesson: can't fix imbalance by just removing data when
  already data-starved overall — it just makes the dataset smaller.
- **Fix — augment first, then downsample**: added AI-generated sentence-level
  samples (1,529 rows in `augmented_data.csv`) to bulk up small classes, *then*
  downsampled "No Distortion" to 400. This became the standing data pipeline for
  every experiment after.
- Also switched training examples from full paragraphs to individual sentences
  (matches how the model is actually used at inference time — sentence by
  sentence). Concretely: paragraphs *with* a distortion contribute one row each
  (just the `Distorted part` span — a column the Kaggle dataset itself provides,
  not something this code derives) → 1,597 rows. Paragraphs with *no* distortion
  get exploded into individual sentences (6,536 of them), all labeled "No
  Distortion," then downsampled to 400. Final training set:
  1,597 + 400 + 1,529 (augmented) = **3,526 rows**, plus whatever user feedback
  has accumulated at retrain time. (The raw "2,530" figure from the baseline
  above is paragraph count, not final row count — the two are easy to conflate
  since sentence-splitting wasn't introduced until this step.)
- **Known caveat — data leakage risk (not fixed as of this writing)**: the
  paragraph→sentence split happens *before* `train_test_split`, and that split
  is random at the sentence level, not grouped by source paragraph. A "No
  Distortion" paragraph can contribute several sentences (same writer, same
  story, overlapping vocabulary/tone) — some of those can land in train, others
  in test, purely by chance. This isn't the model overfitting; it's the *test
  set* not being fully independent of train for that one class, which likely
  makes "No Distortion" accuracy look somewhat better than it would on truly
  unseen writing. Rows with a distortion label don't have this problem — each
  paragraph contributes exactly one row (the `Distorted part` span), so no
  group can span both train and test. Fix would be a paragraph-grouped split
  (`GroupShuffleSplit` keyed on a paragraph id) instead of a random one —
  deferred for now since the affected slice (400 of 3,526 rows) is a minority
  of the data.

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
- This became the model actually deployed to production (`train_model.py` /
  `distortion_model.pkl`).
- **Mental model worth keeping**: MiniLM "manicures" the text into a shaped,
  comparable form (the 384-dim vector) as a separate, one-time step, *before*
  LogisticRegression ever sees it or the 11 labels. LogisticRegression can only
  draw dividing lines through whatever shape it's handed — it has no way to
  ask for the data to be reshaped. That framing is why the next several
  experiments (tuning, bigger encoder) all plateau in the same band: none of
  them can touch the manicuring step itself.

## 5. Hyperparameter tuning (GridSearchCV) — three rounds, all near-identical results

| Round | What was swept | Best params | Best score |
|---|---|---|---|
| 1 | `C` only, no scaling | `C=10.0` | 48.39% macro F1 |
| 2 | `C` only, with `StandardScaler` added before LR | `C=0.01` | 49.18% macro F1 |
| 3 | `C` × `penalty` (l1/l2) × `solver` (lbfgs/saga) | `C=0.01, penalty=l2, solver=saga` | 49.22% macro F1 |

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

- Instead of frozen embeddings + a separate classifier, fine-tuned DistilBERT
  end-to-end on the same dataset.
- **Reproduced locally to verify the claim**: re-implemented the same approach
  (`distilbert-base-uncased`, `Trainer` API, 3 epochs, `lr=2e-5`, same train/test split
  as every other experiment) and fine-tuned on Apple Silicon (MPS). **Result: 63.88%
  accuracy** — confirms the original claim was real and reproducible, not a stale/
  inflated note.
- **By far the best result across every experiment** — every class landed at f1 ≥ 0.48
  (vs. several classes stuck around 0.3 with frozen embeddings). Strongest: Should
  statements (0.82), Fortune-telling (0.69), No Distortion (0.69), Mental filter (0.71).
  Weakest: Magnification (0.48), Personalization (0.51) — still the two hardest classes,
  but meaningfully improved from the frozen-embedding approach.
- **Not yet in production** — would require swapping `app.py`'s inference path and
  handling a much larger model (~260MB) at deploy time.
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

- **Decision (2026-07-15)**: `tune_model.py`'s `GridSearchCV` currently scores on
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

## Overall narrative

1. Representation matters far more than classifier tuning: switching TF-IDF →
   embeddings (34% → 49%) dwarfs every hyperparameter search that followed (all within
   ~1 point of each other).
2. Once on frozen embeddings, four different levers (regularization strength, penalty/
   solver, feature scaling, embedding model size) all independently converged on the
   same ~48–50% ceiling — a clear signal that further gains need a different kind of
   change, not more tuning of the same setup.
3. Fine-tuning the transformer itself broke through that ceiling (63.88%),
   consistent with the idea that generic frozen embeddings, however good, aren't as
   informative as representations learned specifically for this task.
4. Next real lever isn't more GridSearchCV — it's either deploying the fine-tuned
   DistilBERT model, or further targeted data collection/augmentation for the
   worst-performing classes (Magnification, Emotional Reasoning, Overgeneralization).
