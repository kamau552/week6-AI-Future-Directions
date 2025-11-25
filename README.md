# Week 6 Emerging AI Technologies – Practical & Theoretical Exploration

This project is part of my assignment on exploring **emerging AI trends**, including Edge AI, AI–IoT systems, and the differences between Quantum AI and classical AI.  
It contains both **theoretical analysis** and **hands-on implementation** using Python, TensorFlow Lite, and a simulated IoT workflow.

---

## Part 1: Theoretical Analysis

### **Q1: How Edge AI Reduces Latency and Enhances Privacy**
Edge AI performs computation directly on local devices (e.g., smartphones, Raspberry Pi, drones) instead of sending data to the cloud.  
Because processing happens near the data source:

- **Latency decreases** → no long round-trip to cloud servers.  
- **Privacy improves** → sensitive data doesn't leave the device.  
- **Bandwidth costs reduce** → only small predictions are transmitted.

**Real-World Example:**  
**Autonomous drones** use Edge AI to detect obstacles and navigate in real time. If the drone had to send images to the cloud for processing, delays could cause crashes. Edge AI lets the drone react in milliseconds, even without internet.

---

### **Q2: Quantum AI vs Classical AI in Optimization**
Classical AI uses traditional hardware and algorithms for optimization, but struggles with extremely large search spaces.

Quantum AI uses:

- **Superposition** → exploring many states at once  
- **Quantum tunneling** → escaping local minima  
- **Entanglement** → solving certain problems exponentially faster

**Industries that benefit the most:**

- **Drug discovery** – faster molecule optimization  
- **Finance** – portfolio optimization  
- **Logistics & supply chains** – route optimization  
- **Energy** – smart grid optimization  
- **Cybersecurity** – faster pattern search  

Quantum AI is still emerging, but its future impact is huge.

---

# Part 2: Practical Implementation

## Task 1: Edge AI Prototype – Image Classification

I trained a lightweight CNN model to classify simple recyclable items (plastic vs paper).  
The model is converted to **TensorFlow Lite**, which makes it suitable for running on edge devices such as Raspberry Pi.

### Steps Completed:
1. Trained a small CNN on a sample dataset  
2. Evaluated accuracy  
3. Converted the `.h5` model to `.tflite`  
4. Tested inference using a sample image  

You can view all the code inside the **Jupyter Notebook** in this repo.

### Accuracy Results (Example)
- Training Accuracy: **92%**
- Validation Accuracy: **88%**
- TFLite model works successfully with <1MB size

### Why This Helps Edge Applications  
Since the model is so lightweight, it can:
- Run offline  
- Respond instantly (useful for smart bins, drones, or sorting robots)  
- Protect privacy (images stay local)

---

## Task 2: AI-Driven IoT Smart Agriculture System

I designed a conceptual IoT system for smart farming.

### **Sensors Needed**
- Soil moisture sensor  
- Temperature sensor  
- Humidity sensor  
- Light intensity sensor  
- pH sensor  

### **AI Model**
I propose a **simple regression model** that predicts **crop yield** based on sensor trends and environmental conditions.

### **Data Flow Diagram**
[Sensors]
    ↓
[Microcontroller / IoT Node]
    ↓ (raw data)
[Edge Processing / Preprocessing Layer]
    ↓
[AI Model – Crop Prediction]
    ↓
[Farmer Dashboard / Alerts]

# Tools Used  
- Python  
- TensorFlow & TensorFlow Lite  
- Google Colab  
- Matplotlib + NumPy  
- Simple IoT architecture design  

---

# 🙋 Author  
This project was completed individually as part of my Emerging AI Technologies coursework.
