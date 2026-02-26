# 🇳🇵 Yamraj Khadka — ML Engineer Portfolio

> **Live Site:** [parallax1235.github.io/Yamraj-portpolio-projects--](https://parallax1235.github.io/Yamraj-portpolio-projects--/)

---

## About

This is the source for my personal ML engineering portfolio — a hand-coded, single-file HTML site built without templates or frameworks.

I'm a **Computer Engineering undergraduate** at Tribhuvan University, IOE Purwanchal Campus, Biratnagar, Nepal. I build production-grade AI systems: fine-tuned LLMs, RAG pipelines, computer vision models, multi-agent frameworks, and full-stack deployments.

---

## 🚀 Featured Projects

### 🏛️ Nepal Legal AI System — [e-wakil/Fine-tuning-penal-code-Nepal](https://github.com/e-wakil/Fine-tuning-penal-code-Nepal)
End-to-end legal LLM pipeline for Nepal's National Penal Code 2017.
- PyMuPDF extraction → hierarchical chunking → instruction dataset
- **Mistral-7B** fine-tuned (13.5 GB FP16) → GGUF quantized (4 GB)
- FastAPI backend + React Native Android app
- Community-adopted within **24 hours** — re-quantized into Q2–Q8 variants by `mradermacher`
- **Stack:** Mistral-7B · FAISS · SentenceTransformers · FastAPI · React Native · llama.cpp

### 🛰️ U-Net Land Cover Segmentation — [ICRTAI 2025](https://github.com/yamrajkhadka/project_on_img_seg_using_unet_archi)
Research paper presented at the 1st International Conference on Recent Trends in AI.
- Custom composite loss: 60% Focal Tversky + 40% Weighted CCE
- **Mean IoU: 0.674** across 7 land classes (DeepGlobe dataset)
- **Stack:** TensorFlow · Keras · Albumentations · OpenCV · Streamlit

### 🤖 HerAI — Multi-Agent System (LangGraph) — [github.com/yamrajkhadka/Agentic-AI](https://github.com/yamrajkhadka/Agentic-AI)
5-agent orchestration: Mood → Memory → Content → Safety pipeline.
- **Stack:** LangGraph · LangChain · FAISS · Streamlit · Ollama

### ⚡ GGUF + RAG Legal Chatbot — [e-wakil/gguf-with-rag](https://github.com/e-wakil/gguf-with-rag)
Real-time legal assistant with WebSocket streaming, cross-encoder reranking, and LRU cache.
- **Stack:** FastAPI · WebSocket · FAISS IVF · ctransformers

### 📱 Nepal Legal Android App
Native Android app powered by the GGUF API. Includes chat history, auth, and document scanner.
- **Stack:** React Native · Expo · TypeScript

### ⚙️ MLOps — Trip Duration Prediction — [github.com/yamrajkhadka/mloops_self_paced](https://github.com/yamrajkhadka/mloops_self_paced)
Full reproducible pipeline with MLflow, Optuna, and model registry on NYC taxi data.

---

## 🤗 Models on Hugging Face

| Model | ID | Size |
|---|---|---|
| Nepal Legal Mistral-7B (FP16) | [yamraj047/nepal-legal-mistral-7b](https://huggingface.co/yamraj047/nepal-legal-mistral-7b) | 13.5 GB |
| Nepal Legal Mistral-7B GGUF (Q4_K_M) | [yamraj047/nepal-legal-mistral-7b-GGUF](https://huggingface.co/yamraj047/nepal-legal-mistral-7b-GGUF) | 4.07 GB |

---

## 🧠 Skills

**Deep Learning:** TensorFlow · Keras · PyTorch · U-Net · CNN · LSTM/GRU/RNN  
**LLMs & NLP:** Mistral-7B · LLM Fine-tuning · GGUF Quantization · FAISS · SentenceTransformers · RAG  
**Agents:** LangChain · LangGraph · Multi-agent Orchestration  
**MLOps & Deployment:** FastAPI · Streamlit · MLflow · Docker · WebSocket · React Native

---

## 📄 Research

**"Land Cover Segmentation from Satellite Imagery Using U-Net with Custom Loss and Morphological Postprocessing"**  
*ICRTAI 2025 · June 28–29 · Nepal*  
Mentored by Prof. Dr. Sudan Jha (AI & IoT) · mIoU: 0.674

---

## 📊 Stats

| Metric | Value |
|---|---|
| LLM Fine-tuned | 13.5 GB (Mistral-7B FP16) |
| Community Adoption | < 24 hours |
| U-Net Mean IoU | 0.674 |
| Live Deployments | 4+ |
| Deep Learning Days in Public | 17+ |
| LinkedIn Followers | 509+ |

---

## 🔗 Connect

- **GitHub:** [github.com/yamrajkhadka](https://github.com/yamrajkhadka)
- **Hugging Face:** [huggingface.co/yamraj047](https://huggingface.co/yamraj047)
- **LinkedIn:** [linkedin.com/in/yamraj-khadka-7806b833b](https://www.linkedin.com/in/yamraj-khadka-7806b833b)
- **Location:** Biratnagar, Nepal 🇳🇵

---

## 🏗️ About This Site

- Pure HTML/CSS/JS — no frameworks, no build tools
- Animated neural network canvas background
- Terminal boot sequence
- 3D card tilt effects
- Scroll-reveal animations
- Responsive design

> *"Build things that matter for the place you're from."*

---

© 2025 Yamraj Khadka · Computer Engineering · TU IOE Purwanchal Campus · Built by hand, not by template
