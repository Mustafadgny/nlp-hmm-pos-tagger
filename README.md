# NLP Hidden Markov Model (HMM) POS Tagger 🏷️

This repository demonstrates a foundational Natural Language Processing (NLP) technique: **Part-of-Speech (POS) Tagging** using a **Hidden Markov Model (HMM)**. 

## 🧠 Core Concept
Part-of-Speech tagging is the process of categorizing words in a text corresponding to a particular part of speech (e.g., Nouns, Verbs, Pronouns) based on both its definition and its context.

A **Hidden Markov Model (HMM)** is a statistical Markov model in which the system being modeled is assumed to be a Markov process with unobserved (hidden) states. In this NLP context:
- **Observations:** The actual words in the sentence (e.g., "I", "am", "a", "teacher").
- **Hidden States:** The grammatical tags we want to predict (e.g., `PRP` for Pronoun, `VBP` for Verb, `NN` for Noun).

The HMM learns two key metrics from the training data:
1. **Transition Probabilities:** How likely is it that a Noun follows a Determiner?
2. **Emission Probabilities:** How likely is it that the word "teacher" acts as a Noun?

## 🚀 Features
- **Custom Training:** Trains an HMM tagger from scratch using a labeled list of word-tag tuples.
- **Sequence Prediction:** Evaluates and predicts the most probable sequence of POS tags for completely unseen sentences.
- **NLTK Integration:** Utilizes the robust `nltk.tag.hmm` module for model training and inference.

## 🛠️ Tech Stack
- `Python`
- `NLTK` (Natural Language Toolkit)

## 💻 How to Run
1. Ensure NLTK is installed: 
   ```bash
   pip install nltk
