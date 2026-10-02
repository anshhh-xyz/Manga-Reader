# How `score.py` Works — Manga Task

## 1. What is `score.py`?

`score.py` is the evaluation/judging system for the manga task.

It compares:

- **Ground-truth/reference JSONL** — what the manga annotations say.
- **Prediction JSONL** — what our pipeline predicted.

Its main purpose is to measure two broad things:

1. Did we recover the correct story text and preserve its order?
2. Did we assign the correct speaker to the recovered text?

The scorer does not compare raw images directly. It evaluates the structured JSONL output.

---

# 2. Overall Flow

The scorer roughly works like this:

```text
Ground truth JSONL
        +
Prediction JSONL
        |
        v
   Validate JSON
        |
        v
   Normalize text
        |
        v
     Tokenize
        |
        v
 Fuzzy token matching
        |
        v
 Token sequence alignment
        |
        v
 Match speaker labels
        |
        +------------------------+
        |                        |
        v                        v
 Text metrics             Speaker metrics
        |                        |
        +-----------+------------+
                    |
                    v
              Final report
```

---

# 3. Step 1 — Input Validation

`load_jsonl()` reads both the reference and prediction JSONL files.

Each sequence must have:

```json
{
  "sequence_id": "...",
  "pages": [
    [...],
    [...],
    [...]
  ]
}
```

For this project, there are normally three pages per sequence.

Each text item must contain exactly:

```json
{
  "speaker": "...",
  "text": "..."
}
```

The scorer also checks things such as:

- duplicate sequence IDs
- duplicate JSON keys
- exactly three page lists
- valid speaker values
- non-empty text
- control characters

For reference data, `UNKNOWN` speakers can be treated as unresolved speaker identities.

---

# 4. Step 2 — Text Normalization

Before comparing text, `score.py` runs the `norm()` function.

It performs things such as:

- Unicode NFKC normalization
- converts curly quotes to normal quotes
- converts `…` to `...`
- normalizes different dash characters
- case-folds text (effectively making comparison case-insensitive)
- removes certain invisible Unicode characters
- normalizes whitespace
- cleans spaces around punctuation

For example:

```text
Gold:
“Hello… World!”

Prediction:
"hello... world!"
```

These are normalized so harmless formatting differences do not unnecessarily hurt the score.

---

# 5. Step 3 — Tokenization

The scorer breaks text into tokens.

For example:

```text
Where are you going?
```

becomes roughly:

```text
Where
are
you
going
?
```

Each token keeps its speaker information.

Conceptually:

```text
("Where", "char1")
("are", "char1")
("you", "char1")
("going", "char1")
("?", "char1")
```

This lets the scorer evaluate text and speaker information at token level.

---

# 6. Step 4 — Fuzzy Token Matching

The scorer does not require every token to be character-for-character identical.

It calculates edit distance between tokens.

The similarity is:

```text
similarity =
1 - edit_distance(a, b) / max(len(a), len(b))
```

A token can match when:

- both tokens have enough characters
- similarity is at least `0.6`

Examples conceptually:

```text
hello  vs hello  -> 1.0
hello  vs helo   -> partial match
hello  vs world  -> no match
```

This is useful for OCR because OCR can produce small spelling mistakes.

---

# 7. Step 5 — Token Sequence Alignment

`align_tokens()` finds the best monotonic alignment between the ground-truth tokens and predicted tokens.

It uses dynamic programming.

The important idea is:

> The scorer tries to determine which predicted tokens correspond to which ground-truth tokens while preserving their order.

For example:

```text
GOLD:
I am going home now

PRED:
I am goin home
```

The predicted tokens can still be aligned to the appropriate ground-truth tokens.

The implementation uses an `O(n*m)` dynamic-programming table for each page.

The resulting matches also carry a weight representing how similar the matched tokens were.

---

# 8. Step 6 — Speaker Label Mapping

A major issue in this task is that speaker labels don't necessarily have to use the same names as the development annotations.

For example, our prediction might use:

```text
char1
char2
```

while the reference uses:

```text
char3
char1
```

The scorer therefore finds the best one-to-one mapping between predicted speaker labels and ground-truth speaker labels.

It uses a maximum-weight assignment, implemented with the Hungarian algorithm.

Conceptually:

```text
Predicted char1 -> Gold char3
Predicted char2 -> Gold char1
```

The scorer finds the mapping that maximizes the total matched-token weight.

`NARRATION` remains a fixed identity.

---

# 9. The Five Main Macro Metrics

The `evaluate()` function calculates macro averages for five main metrics:

```text
1. text_order_score
2. balanced_joint_f1
3. speaker_accuracy_on_matched
4. joint_f1
5. matched_token_coverage
```

These are the main metrics you should understand.

---

# 10. `text_order_score`

## What does it measure?

It measures how similar the predicted page text is to the ground-truth page text.

For each page, it calculates:

```text
1 - edit_distance(gold_text, predicted_text)
    / max(lengths of the two texts)
```

The scores of active pages are then averaged.

A perfect page gets:

```text
1.0
```

A very different prediction approaches:

```text
0
```

## What does it tell us?

Primarily:

> Did our OCR/filtering/ordering pipeline recover the correct story text?

This is especially useful for Phases 1–4.

---

# 11. `matched_token_coverage`

## What does it measure?

It measures how much of the ground-truth token content was successfully matched.

Conceptually:

```text
matched token weight
--------------------
gold token count
```

Example:

```text
Gold tokens:       100
Matched tokens:     80

Coverage:          0.80
```

## Why is it useful?

It helps reveal recall problems.

For example, a system could recover a small amount of text very accurately while completely missing a lot of the actual dialogue.

`matched_token_coverage` helps show how much of the reference text was actually recovered.

---

# 12. `speaker_accuracy_on_matched`

## What does it measure?

Among the successfully matched tokens:

> How often did we assign the correct speaker?

Conceptually:

```text
correct speaker-token weight
----------------------------
matched speaker-token weight
```

Example:

```text
Matched speaker tokens: 100
Correct speaker:          85

Speaker accuracy:       0.85
```

This metric is only about matched text.

That means it should NOT be used by itself to judge the whole system.

A system could recover very little text and still have high speaker accuracy on the small amount it did recover.

---

# 13. `joint_f1`

This combines text matching and speaker correctness.

A token contributes as a correct joint prediction only when:

```text
the text token matches
        AND
the speaker matches
```

The scorer uses an F1-style formula:

```text
2 * correct
------------------------------
known_gold + predicted_speaker
```

This is therefore closer to:

> Did we correctly recover the text AND who said it?

It is stricter than simply measuring OCR/text correctness.

---

# 14. `balanced_joint_f1`

This is especially important for the manga task.

The scorer intentionally prevents a character with a huge amount of dialogue from dominating the result.

For example:

```text
Character A -> 100 tokens
Character B -> 10 tokens
Character C -> 10 tokens
```

A simple token-weighted score could be dominated by Character A.

`balanced_joint_f1` instead gives the different speaker identities a more balanced contribution.

The code explicitly describes the purpose as preventing:

> a long monologue from outweighing several missed characters.

So this metric is particularly relevant to Phase 5, where speaker assignment and character consistency become important.

---

# 15. Supporting Diagnostics

Besides the five main macro metrics, the scorer outputs useful diagnostics.

These include:

```text
active_text_pages
gold_tokens
predicted_tokens
scored_gold_speaker_tokens
scored_predicted_speaker_tokens
matched_scored_token_weight
correct_speaker_token_weight
speaker_mapping
balanced_speaker_mapping
page_text_scores
missing_sequence
```

These are useful for understanding WHY a score is good or bad.

For example:

```text
text_order_score = 0.80
matched_token_coverage = 0.45
```

might indicate that the recovered text is fairly similar where it exists, but a lot of the ground-truth text is missing.

---

# 16. How the Scores Relate to the Project Phases

| Phase | Useful metrics | Main question |
|---|---|---|
| Phase 1 — OCR baseline | `text_order_score`, `matched_token_coverage` | Are we recovering the text? |
| Phase 2 — OCR comparison | Same + `joint_f1` | Which baseline behaves better? |
| Phase 3 — Alignment | Coverage + diagnostics | Is our OCR/label alignment reliable? |
| Phase 4 — Filter + ordering | `text_order_score` + coverage | Are we keeping story text and putting it in the correct order? |
| Phase 5 — Speakers | `balanced_joint_f1`, `joint_f1`, speaker accuracy | Are text and speakers both correct? |
| Phase 6–7 | All main metrics | Did adaptation/changes actually improve the system? |

---

# 17. Which Metrics Matter Most?

The project roadmap identifies these as the two main metrics:

```text
text_order_score
balanced_joint_f1
```

The roadmap also specifically warns that:

```text
speaker_accuracy_on_matched
```

can be misleading if very little text is recovered.

So don't think:

```text
speaker accuracy = 90%
        ↓
system is excellent
```

Instead, look at the combination:

```text
How much text did we recover?
        +
Was it in the correct order?
        +
Did we assign speakers correctly?
        +
Are different characters being handled fairly?
```

---

# 18. Simple Mental Model

Remember the scorer like this:

```text
text_order_score
    =
"Did we get the story text right and in the right sequence?"

matched_token_coverage
    =
"How much of the actual text did we recover?"

speaker_accuracy_on_matched
    =
"When text matched, did we assign its speaker correctly?"

joint_f1
    =
"Did we get text + speaker correct together?"

balanced_joint_f1
    =
"Did we get text + speaker correct together,
while preventing long speeches from dominating the result?"
```

---

# 19. One Complete Example

Imagine the ground truth is:

```text
char1: Where are you going?
char2: Home.
```

Our prediction is:

```text
char7: Where are you goin?
char9: Home.
```

The scorer will roughly:

```text
1. Normalize both
        ↓
2. Tokenize
        ↓
3. Fuzzy-match "goin" with "going"
        ↓
4. Align the tokens in order
        ↓
5. Discover:
       char7 -> char1
       char9 -> char2
        ↓
6. Calculate text metrics
        ↓
7. Calculate speaker metrics
        ↓
8. Produce the final report
```

The exact score depends on all matched token weights and the complete sequence, but the important point is that the scorer is designed to tolerate small OCR differences while still requiring the overall text and speaker assignments to be correct.

---

# 20. Bottom Line

`score.py` is the project's **automatic judge**.

It does not simply calculate one generic "accuracy".

It uses:

```text
Text normalization
       ↓
Tokenization
       ↓
Edit-distance fuzzy matching
       ↓
Dynamic-programming sequence alignment
       ↓
Hungarian speaker-label assignment
       ↓
Multiple text + speaker metrics
```

The two headline metrics are:

```text
text_order_score
balanced_joint_f1
```

And the other important metrics help diagnose:

```text
matched_token_coverage
speaker_accuracy_on_matched
joint_f1
```

So whenever we change something in the pipeline, the important question is not just:

> "Did the code run?"

It is:

> "Did the change improve the relevant evaluation metrics on the same validation split?"
