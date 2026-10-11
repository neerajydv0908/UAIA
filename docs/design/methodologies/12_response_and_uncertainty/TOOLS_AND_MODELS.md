# 12. Return the answer, citations, and uncertainty — tools and models

Status: proposed options · Updated: 2026-09-28.

[All steps](../README.md) · [Explanation](README.md) · [Methods](METHODS.md) · [Tools/models](TOOLS_AND_MODELS.md) · [Selection](SELECTION.md)

## Concrete options

| Tool/approach | Use |
| --- | --- |
| Python/Markdown template | First result display, with safe escaping and valid source links |
| Structured citation records | Link final statements to stored source passages |
| Ordinary web interface later | Expandable sources and optional feedback once the pipeline works |
| scikit-learn calibration tools | Later fit a probability mapping from scores using separate data |
| Reliability plots and error/coverage tables | Check whether displayed certainty is justified |

[scikit-learn's calibration guide](https://scikit-learn.org/stable/modules/calibration.html) documents probability-calibration approaches. Methods such as sigmoid or isotonic mapping fit a score-to-probability relationship; choosing and validating that relationship is our responsibility.

No generative model is required to show a checked answer. If used for presentation, it adds a new generation step that needs verification.

