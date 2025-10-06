# 🧠 Small Language Model (SLM) – CPU-Only Build from Scratch

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License: CC](https://img.shields.io/badge/License-Creative%20Commons-lightgrey.svg)
![Status](https://img.shields.io/badge/Status-Experimental-orange.svg)

A **Small Language Model (SLM)** implementation designed to run **entirely on CPUs** with no GPU dependency.  
This project explores how lightweight language models can be built from scratch, trained on domain-specific datasets, and deployed for practical use cases such as retrieval-augmented generation (RAG), knowledge base search, and lightweight automation.

---

## 🌍 What is an SLM?

- **SLM (Small Language Model):**  
  A compact model trained with fewer parameters (typically millions to low billions) optimized for **efficiency, privacy, and CPU-only environments**.  
  Best for **edge devices, internal company servers, and specialized use cases**.

- **LLM (Large Language Model):**  
  Massive models (tens to hundreds of billions of parameters) trained on huge datasets.  
  Require **GPUs/TPUs, large-scale infrastructure, and higher cost** but achieve state-of-the-art generalization.

---

## ⚖️ LLM vs SLM – Pros & Cons

| Feature            | LLM (Large)                                        | SLM (Small)                                        |
|--------------------|----------------------------------------------------|---------------------------------------------------|
| **Performance**    | High accuracy, broad generalization, multilingual  | Moderate accuracy, domain-specific is stronger     |
| **Hardware Needs** | Requires GPUs/TPUs, lots of VRAM, cluster setups   | Runs on CPUs, lightweight servers, or even laptops |
| **Cost**           | Expensive to train & host                          | Cheap to run, easy to maintain                     |
| **Use Cases**      | Chatbots, reasoning, coding assistants, open Q&A   | Internal bots, RAG with docs, automation, edge AI |
| **Privacy**        | Data often processed on external infra             | Can run fully on-prem / air-gapped                 |
| **Custom Training**| Expensive & resource-heavy                         | Easy to fine-tune on niche tasks                   |

---

## 🛠️ Features

- ✅ CPU-only training and inference  
- ✅ Tokenizer balanced for **English + Japanese**  
- ✅ Retrieval-Augmented Generation (RAG) integration (FAISS backend)  
- ✅ Instruction-tuning dataset templates
- ✅ Educational build — from **tokenizer → training → inference pipeline**  

---

## 📂 Project Structure

```
slm-cpu/
├── data/             # Training datasets (plain text or JSONL)
│   ├── en.txt        # Tiny English corpus
│   ├── ja.txt        # Tiny Japanese corpus
│   └── instruct.jsonl # Example instruction-tuning dataset
├── adapters/         # LoRA adapters saved here after training
├── exports/          # Final merged/exported models
├── scripts/          # Python scripts for training, generation, merging
│   ├── train_lora_causal.py
│   ├── train_sft_lora.py
│   ├── generate.py
│   ├── merge_lora.py
│   └── router.py
└── README.md
```

---

## 🚀 Quick Start
LoRA stands for Low-Rank Adaptation of large language models.
It’s a method to fine-tune big models efficiently without retraining all their parameters.

# create venv
```
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip wheel setuptools
pip install transformers datasets peft torch sentencepiece
```

# train LoRA (English)
```python
python scripts/train_lora_causal.py --model_path ~/models/distilgpt2 \
  --train_file data/en.txt --output_dir adapters/en_lora
```

# train LoRA (Japanese)
```python
python scripts/train_lora_causal.py \
  --model_path ~/models/rinna-jgpt2-small \
  --train_file data/ja.txt \
  --output_dir adapters/ja_lora
```
# generate
```python
python scripts/generate.py --base ~/models/distilgpt2 \
  --adapter adapters/en_lora --prompt "Write a haiku about teamwork"

```

## 🔍 Example: RAG with Confluence Docs



---

## 📊 Roadmap

- [ ] Add multilingual tokenizer for EN/JA  
- [ ] Pre-train SLM on domain-specific text  
- [ ] Create Docker image for easy deployment  
- [ ] Release pretrained checkpoints  

---

## 📜 License

This project is licensed under the **Creative Commons Legal Code License** – see the [LICENSE](LICENSE) file for details.  

---

## 🤝 Contributing

Contributions, issues, and pull requests are welcome!  
If you’d like to help, please open an issue first to discuss the scope.

---

## 🙏 Acknowledgments

- Inspired by the open-source ML community (Hugging Face, FAISS, SentenceTransformers)  
- Built for research, learning, and CPU-only environments  
- Developed with ❤️ by [ROB ASHAD UR]
- Inspired by NVIDIA Research 
- https://research.nvidia.com/labs/lpr/slm-agents/

