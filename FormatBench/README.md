# FormatBench Collection and Annotation

## Multi-Choice Questions (MCQ)

Adapted from Text TRtrieval Conference (TREC) question classification dataset (Li and Roth 2002; Hovy et al. 2001).

It is a task that, given a question, maps it to one of the given classes, which provides a semantic constraint on the sought-after answer.

## Extractive Question Answering (EQA)

Adapted from Stanford Question Answering Dataset (SQuAD) (Rajpurkar et al. 2016).

It is a reading comprehension dataset, where the answer to every question is a segment of text, or span, from the corresponding reading passage. We use a copied format in this task, i.e., requiring LLMs to directly copy the span of the passage without modification.

## Named Entity Recognition (NER)

Adapted from CoNLL-2003 Task (Sang and De Meulder 2003).

It is a named entity recognition (NER) task to detect and categorize named entities.

## Constituency Parsing (Parse)

Adapted from the open source subset of the Penn Treebank (PTB) (Marcus, Santorini, and Marcinkiewicz 1993).

We use the bracket sequence representation of a constituency tree.

## Caption Segmentation (CapSeg)

We adapt MuST-Cinema (Karakanta, Negri, and Turchi 2020) dataset, a multilingual speech translation corpus built from TED subtitles, to construct the caption segmentation.

CapSeg involves inserting end-of-block and end-of-line tags in the raw English text to represent the split of captions in videos, thus simulating the generation of English video captions.

## Terminology Machine Translation (MTT)

Adapted from the WMT 2023 Terminology Shared Task (Semenov et al. 2023).

It is a Germany-English machine terminology translation (MTT) task that challenges machine translation systems to accurately and effectively translate technical terms and specialized vocabulary.

## Acrostic Writing (AcroW)

We collect the acrostic poem dataset from [Poem Hunter](https://www.poemhunter.com/), focusing on the acrostic category to gather 927 acrostic poems via web scraping. We additionally combine [Kaggle Poems Dataset](https://www.kaggle.com/datasets/michaelarman/poemsdataset) acrostic poems with these data.

To ensure data quality and consistency, we eliminate redundant punctuation and standardize poem lines, retaining only the acrostic portion. Moreover, we filter out poems that do not meet acrostic requirements, such as initial letters failing to form coherent words or lacking relevance, to maintain data accuracy.

In this task, an LLM is challenged to compose an acrostic poem, adhering to the format of having the first letter of each line spell out the intended message.

## Formatted Time Generation (FTime)

Combining template filling and manual composition, we generate 5036 pieces of formatted time generation instructions, and annotate the corresponding results manually.

After annotation, we randomly sample 3% data to conduct a cross-validation involving re-annotation by a different annotator. The consistency between the initial annotation and the re-annotation is found to be 98.01%, thereby confirming the trustworthiness of our annotated data.

## Text Game Agent (Agent)

Adapted from the First TextWorld Problems (FTWP) (Adam, Marc-Alexandre, and Pedro 2019).

The task is to build an AI agent that moves automatically and efficiently in a text simulated world according to text feedback, and finally completes the given goal.

## XDL Generation (XDL)

XDL(ChemicalDescription Language) (Seifrid et al. 2022) is an XML-based programming language used in chemical synthesis specification and experimental procedure transfer among robots and laboratories.

Following previous work (Skreta et al. 2023), we conduct XDL Generation task by examining the ability of LLMs to generate compilable XDL programs given the description of XDL.