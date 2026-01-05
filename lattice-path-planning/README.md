# 🌊 Wavy Lattice Slicer 🧶
> **Make your 3D prints dance with bio-inspired internal structures!** 💃✨

![Python](https://img.shields.io/badge/Made%20with-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Love](https://img.shields.io/badge/Built%20with-💖-ff69b4?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Experimental-orange?style=for-the-badge)

Hi there! 👋 Welcome to **Wavy Slicer**. This isn't your boring old grid infill generator. This is a research-grade slicing engine that uses **Graph Theory** and **Sine Waves** to create organic, strong, and beautiful internal structures for FDM 3D printing.

---

## 🎨 What makes it special? (The Magic)

### 1. 🌊 Bio-Mimetic Wavy Infill
Instead of rigid straight lines, we use **sinusoidal modulation** ($x' = x + A \sin(ky)$) to wiggle the path.
* **Why?** It looks super cool AND it increases the contact area between layers for better adhesion! 

### 2. 🧠 Smart Hybrid Brain
The slicer is "self-aware"! It analyzes the geometry of every layer:
* **Big Area?** -> Generates the Wavy Lattice for speed and strength. 🚀
* **Tiny/Broken Area?** -> Automatically switches to **Contour Mode** (concentric rings) to ensure the print doesn't fail. 🛡️

### 3. 🧭 Graph-Based Path Planning
We treat the printing path as a **Graph Traversal Problem** ($G(V,E)$).
* Uses **DFS (Depth First Search)** and **Beam Search** to find the most efficient route.
* Minimizes "Travel Moves" (jumping around without printing). 🏃‍♂️💨

---

## 🛠️ Tech Stack (The Ingredients)

We baked this project using these awesome libraries:

* 🐍 **Python** (The core logic)
* 📐 **Trimesh** (For slicing the STL)
* 🔵 **Shapely** (For polygon math & clipping)
* 🕸️ **NetworkX** (For graph building & traversal)
* 🔢 **NumPy/SciPy** (For the mathy stuff)

---

## 🚀 Getting Started

Want to try it out? Follow these simple steps!

### 1. Clone the repo
```bash
git clone [https://github.com/your-username/wavy-slicer.git](https://github.com/your-username/wavy-slicer.git)
cd wavy-slicer
