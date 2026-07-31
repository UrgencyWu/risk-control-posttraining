# Evaluation Protocol

This document defines the evaluation contract for all reported German Credit
results. It is intentionally separate from model training so that operating
point selection is auditable and cannot leak test labels.

## Split and decision protocol

The German Credit split is frozen at 700 training, 100 validation, and 200 test
records. Model weights and checkpoints may be selected using the training and
validation splits only. The test split is used once for final metrics.

For every model, the threshold is selected by minimising:

```text
Cost = 5 * false negatives + 1 * false positives
```

on the validation prediction artifact over the fixed grid
`{0.05, 0.10, ..., 0.95}`. The chosen threshold is then frozen and applied to
the corresponding test prediction artifact. Test labels are used only to
calculate final metrics after the threshold is fixed.

`python -m src.evaluation.c7_final` implements this protocol. It is
evaluation-only: it reads committed JSONL prediction artifacts, never loads a
model, and never reruns inference. The generated
`outputs/c7_final_metrics.json` records both the threshold and its
`threshold_source`.

## LLM class-score definition

Zero-shot and SFT inference use a **first-token class score**. Given prompt
`x`, let `t_low` and `t_high` be the first tokenizer tokens produced by the
strings `low` and `high`. The reported high-risk score is:

```text
s_low  = log P(t_low  | x)
s_high = log P(t_high | x)
p_high = exp(s_high) / (exp(s_low) + exp(s_high))
```

This is a two-class normalisation over first continuation tokens. It is not the
full conditional sequence likelihood of `"low risk"` or `"high risk"`; any
comparison or reproduction should preserve this exact score definition.

## Reproducing published metrics

Install the lightweight evaluation environment and run:

```bash
python -m pip install -r requirements-eval.txt
python -m unittest discover -s tests -v
python -m src.evaluation.c7_final
```

The final command regenerates `outputs/c7_final_metrics.json` from the frozen
validation and test prediction artifacts without a GPU or model weights. To
avoid overwriting the tracked artifact during an independent check, pass
`--output /tmp/c7_final_metrics.json`.
