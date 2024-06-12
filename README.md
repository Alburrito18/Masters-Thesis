# Large Language Models For Patient Document Summarization

##  A case study in applying large language models for patient document summarization conducted at Sahlgrenska University Hospital

## Authors: [Albert Lund](https://github.com/Alburrito18/) & [Felix Nilsson](https://github.com/Felix-Nilsson)

---
This repository contains the code, report and slides for our masters thesis:

* [Report](Masters_Thesis.pdf)
* [Slides](slides.pdf)

--- 
## Abstract

Reading patient documents is a time-consuming but necessary part of a doctor’s duties, which is often further slowed down by poorly designed software systems. This,
in turn, contributes to the already psychologically stressful environment of being
a doctor. However, large language models (LLMs) have recently shown excellent
results on many downstream tasks, including summarization. Moreover, performance on such tasks shows little degradation when transferred to a language other
than English, despite relatively limited exposure to the target language. In this
thesis, we show how LLMs can save time in healthcare by generating automatic
summaries over patient document. In particular, we closely examine the potential
of open-source LLMs, which allow for more control, in contrast to proprietary LLMs,
which currently represent the state of the art. To this end, we design an automatic
evaluation procedure that compares a given model’s summarization capabilities to
that of a clinician. We then optimize an open-source LLM via finetuning to show
performance comparable to GPT-4 on the said procedure. Finally, we conduct a
small-scale study in which doctors compare summaries produced by our LLM solution to those of a rule-based summarizer and a doctor. We find that while doctors
prefer the human summary, the LLM outperformed the rule-based summarizer. Interpreting these results, we see the future of automatic medical summarization as
promising. However, in our view, the use of a novel technology such as LLMs needs
to be navigated carefully to avoid harming patients. The thesis was conducted at
Sahlgrenska University Hospital (SU), where it was part of a larger project looking
at AI in healthcare, and it was organized by SU’s AI Competence Center (AICC)