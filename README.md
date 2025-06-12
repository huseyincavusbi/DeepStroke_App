# 🧠 DeepStroke AI - Brain CT Stroke Detection

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-red.svg)](https://pytorch.org)
[![Gradio](https://img.shields.io/badge/Gradio-5.33-orange.svg)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue.svg)](https://huggingface.co/spaces/huseyincavus/DeepStroke/)

## Overview

DeepStroke AI is an advanced deep learning system for automated stroke detection in brain CT scans. Built with SE-ResNeXt50 architecture, it provides rapid, accurate stroke probability assessment to assist medical professionals in clinical decision-making.

## 🚀 Features

- **High Performance**: ROC-AUC 0.98+ on validation datasets
- **Real-time Analysis**: <1 second inference time
- **Interactive Dashboard**: Clean, medical-grade interface
- **Clinical Integration**: Optimized threshold (49.02%) for balanced sensitivity/specificity
- **Example Images**: 7 built-in test cases for demonstration

## 🏗️ Model Architecture

- **Network**: SE-ResNeXt50 with Squeeze-and-Excitation attention
- **Input**: 224×224 RGB images (non-contrast brain CT only)
- **Training Data**: 10,000+ validated brain CT scans
- **Optimization**: Youden's Index for optimal clinical threshold

## ⚠️ Important Limitations

**CRITICAL**: This model is trained exclusively on **non-contrast brain CT scans** and will fail on:
- Contrast-enhanced CT scans
- MRI images
- Other imaging modalities
- Non-brain anatomical regions

## 🔬 Clinical Usage

This AI system is designed for **research and educational purposes only**. Always:
- Consult qualified medical professionals for diagnosis
- Follow standard emergency protocols for suspected acute stroke
- Use as diagnostic support, not replacement for clinical judgment
- Validate AI findings with radiological assessment

## 🛠️ Technical Specifications

- **Framework**: PyTorch 2.7.1
- **Interface**: Gradio 5.33.1
- **Visualization**: Plotly interactive charts
- **Deployment**: Hugging Face Spaces compatible
- **Hardware**: CUDA GPU accelerated (CPU fallback available)

## 📊 Performance Metrics

- **Sensitivity**: High stroke detection rate
- **Specificity**: Low false positive rate
- **Balanced Accuracy**: Optimized for clinical workflow
- **Inference Speed**: Sub-second processing
- **Model Size**: ~25M parameters

## 🚀 Quick Start

### 🌐 Try the Live Demo
**👉 [Try DeepStroke AI on Hugging Face Spaces](https://huggingface.co/spaces/huseyincavus/DeepStroke/)**

### 💻 Local Usage
1. Upload a non-contrast brain CT image
2. Click "Analyze CT Scan" or try example images
3. Review the probability gauge and clinical recommendations
4. Always correlate with clinical findings

## 📝 Citation

If you use DeepStroke AI in your research, please cite appropriately and note that this is for research purposes only.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**⚕️ Medical Disclaimer**: This AI tool is for research and educational purposes only. It does not provide medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical decisions.
