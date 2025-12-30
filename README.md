# Pantry2Plate - Recipe Generation

This project focuses on **recipe generation** using GPT-2, exploring how dataset size and decoding strategies affect the quality and coherence of generated cooking instructions.

---

## Features
- Fine-tuned **GPT-2** on recipe subsets (5k, 10k, 20k, 30k samples).  
- Evaluated **decoding strategies**: Greedy, Beam Search (beam=5), and Top-K (k=50).  
- Preprocessed recipes into a normalized template with `<|RECIPE|>` and `<|END|>` tokens, removed malformed entries, and concatenated instructions for training.  

---

## Evaluation
- **ROUGE-1**: Word overlap with reference recipes.  
- **ROUGE-L**: Measures sequence and structural logic of instructions.  
- **Cosine Similarity**: Semantic similarity between generated and reference recipes using MiniLM embeddings.  
- Evaluated on a 100-recipe test set.  

---

## Key Findings
- Dataset size improves quality up to ~20k samples; diminishing returns after that.  
- **Greedy decoding** produces the most stable, reference-aligned recipes.  
- Beam Search adds detail but can be redundant; Top-K sampling is more creative but lowers similarity metrics.  
- Recommended: **Greedy decoding with 20k+ samples** for a good balance of quality and efficiency.  

---

## Tech Stack
- **Python, HuggingFace Transformers (GPT-2), PyTorch**  
- **Evaluation:** ROUGE, Cosine Similarity (MiniLM)  
- **Dataset:** Preprocessed recipe subsets (5k–30k samples)  

