# 🎓 Mentor–Student Matching System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Notebook-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An ML-powered recommendation system that intelligently matches CSE students to faculty mentors based on subject preference alignment — with hard capacity constraints and zero interaction history.**

[📖 Report](#-report) · [🚀 Quick Start](#-quick-start) · [📊 Results](#-results) · [🔬 Methodology](#-methodology)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [The Problem](#-the-problem)
- [Key Features](#-key-features)
- [Methodology](#-methodology)
- [Dataset](#-dataset)
- [Models](#-models)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Output Files](#-output-files)
- [Statistical Analysis](#-statistical-analysis)
- [Limitations](#-limitations)
- [Future Scope](#-future-scope)
- [Team](#-team)

---

## 🔍 Overview

Manual mentor assignment in engineering colleges is **arbitrary and academically misaligned** — students are paired with mentors based on roll numbers or alphabetical order, with zero consideration of subject compatibility.

This project builds an automated matching pipeline that:

- **Encodes** mentor expertise and student preferences as mathematical vectors in a shared 28-subject space
- **Scores** every student–mentor pair using cosine similarity (1,200 pairs simultaneously)
- **Assigns** each student to their best available mentor with a hard cap of **15 students per mentor**
- **Cascades** overflow students to their next best available choice
- **Compares** two ML models (Content-Based vs Cascade Hybrid) using rigorous statistical tests
- **Exports** results as a downloadable 4-sheet Excel file

> **Institution:** IIIT Bhopal · Department of Computer Science and Engineering
> **Project Type:** Minor Project · Group 41

---

## ❗ The Problem

```
120 students  ×  10 mentors  ×  28 technical subjects  ×  0 historical match records
```

| Challenge | Description |
|---|---|
| 🎲 **Manual Assignment** | Roll-number based — ignores academic compatibility entirely |
| ⚖️ **No Capacity Control** | One mentor could get 40 students, another gets 2 |
| ❄️ **System Cold Start** | Zero interaction history → pure Collaborative Filtering impossible |
| 🔄 **Overflow Handling** | When best mentor is full, must cascade to next best automatically |

### Why Not Collaborative Filtering?

> Pure CF requires a **User × Item interaction matrix** built from real historical behaviour — ratings, sessions, feedback. Our system is brand new. Every cell of this matrix is empty. This is the **System Cold Start Problem** — the most severe form of cold start. CF is **structurally impossible**, not just impractical.

---

## ✨ Key Features

- ✅ **Cold-start aware** — designed from the ground up for zero historical data
- ✅ **Two ML models** — Content-Based Filtering and Cascade Hybrid compared
- ✅ **Hard capacity enforcement** — 15-student cap with automatic overflow cascade
- ✅ **Statistically validated** — Wilcoxon test + Spearman correlation
- ✅ **6 visualisation graphs** — distribution, load, overflow, scatter analysis
- ✅ **Excel output** — 4-sheet downloadable result with scores and mentor details
- ✅ **100% assignment** — all 120 students assigned, no mentor exceeds cap
- ✅ **No pip installs** — all libraries pre-installed in Google Colab

---

## 🔬 Methodology

### Step 1 — Vectorisation

**Mentors** → Binary subject vector via `MultiLabelBinarizer`

```python
# Dr. Anil Sharma teaches: ML, Deep Learning, Data Science
# Result across 28 subjects:
[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, ...]
#                         ^ML  ^DL                ^DS
```

**Students** → Rank-weighted vector via custom `rank_weights()`

```python
# Rank 1 → weight 7,  Rank 2 → weight 6,  ...,  Rank 7 → weight 1
# Aarav ranks: ML(R1), DL(R2), DS(R3), AI(R4), CV(R5), Algo(R6), SysD(R7)
# Result:
[7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, ...]
# ^ML ^DL ^DS ^AI ^CV ^Al ^SD
```

> **Why weighted (not binary)?** Binary treats Rank 1 = Rank 7. Weighted encoding preserves the *strength* of preference — the most informative signal in the data.

### Step 2 — Cosine Similarity

```
cos(θ) = (A · B) / (‖A‖ × ‖B‖)    →    Result ∈ [0, 1]
```

All 1,200 student–mentor pairs computed **simultaneously** as a matrix operation:

```python
cb_sim_df = pd.DataFrame(
    cosine_similarity(student_df.values, mentor_df.values),  # 120×28 × 28×10
    index=student_df.index,
    columns=mentor_df.index
)
# Output: 120×10 similarity matrix
```

> **Why cosine (not Euclidean)?** Cosine is magnitude-independent — it measures directional alignment regardless of how many subjects a student ranked. Euclidean would unfairly penalise students who ranked fewer subjects.

### Step 3 — Capacity-Constrained Assignment

```python
def assign_with_capacity(sim_df, max_load=15):
    load = {mentor: 0 for mentor in sim_df.columns}
    for student in sim_df.index:
        ranked = sim_df.loc[student].sort_values(ascending=False)
        for rank_i, (mentor, score) in enumerate(ranked.items(), 1):
            if load[mentor] < max_load:   # capacity gate
                load[mentor] += 1
                results.append({...})
                break                      # assigned — stop
```

**If Rank-1 mentor is full → try Rank-2 → Rank-3 → ... until slot found.**

---

## 📂 Dataset

| File | Rows | Columns | Description |
|---|---|---|---|
| `mentors.csv` | 10 | 4 | Mentor Name, Subject1, Subject2, Subject3 |
| `students.csv` | 120 | 8 | Student Name, Rank1 through Rank7 |

### Mentors

| Mentor | Subject 1 | Subject 2 | Subject 3 |
|---|---|---|---|
| Dr. Anil Sharma | Machine Learning | Deep Learning | Data Science |
| Prof. Sunita Gupta | Artificial Intelligence | NLP | Computer Vision |
| Dr. Ramesh Verma | Blockchain | Web3 | Cryptography |
| Ms. Priya Nair | Web Development | Cloud Computing | DevOps |
| Mr. Vikram Singh | Data Science | Big Data | Data Engineering |
| Dr. Kavita Reddy | Cybersecurity | Network Security | Ethical Hacking |
| Prof. Suresh Iyer | Internet of Things | Embedded Systems | Robotics |
| Ms. Anjali Mehta | Full Stack Development | Mobile App Dev | UI/UX Design |
| Dr. Deepak Joshi | Computer Vision | Image Processing | Augmented Reality |
| Mr. Manoj Pillai | Competitive Programming | Algorithms | System Design |

### Shared Subject Vocabulary — 28 Technical Domains

```
Machine Learning · Deep Learning · Artificial Intelligence · NLP · Computer Vision
Data Science · Image Processing · Augmented Reality · Blockchain · Web3 · Cryptography
Cybersecurity · Ethical Hacking · Network Security · IoT · Embedded Systems · Robotics
Web Development · Full Stack Dev · Cloud Computing · DevOps · Mobile App Dev · UI/UX
Big Data · Data Engineering · Competitive Programming · Algorithms · System Design
```

---

## 🤖 Models

### Model 1 — Content-Based Filtering

```
Student profile vector  ──cosine similarity──▶  Mentor subject vector
                                                  ↓
                                         Match score ∈ [0,1]
                                                  ↓
                                    assign_with_capacity(max=15)
```

- Profile-to-profile comparison only
- No historical data needed
- Fully valid under System Cold Start

### Model 2 — Cascade Hybrid

```
For student i:
  ├─ Find most similar peer in students[0…i-1]
  │     ├─ peer sim ≥ 0.80?  → Inherit peer's mentor  [CF-style signal]
  │     └─ peer sim < 0.80?  → Use CB best match       [cold start fallback]
  └─ Record assignment in interaction matrix → next student

Final score = 0.6 × CB_normalised + 0.4 × peer_signal_normalised
                         ↓
              assign_with_capacity(max=15)
```

- Sequential processing builds peer pool progressively
- 60/40 weighted blend after min-max normalisation
- Cascade hybrid architecture (not pure CF)

> **Note:** The Cascade Hybrid is *not* pure Collaborative Filtering. The peer signal uses declared preference similarity as a proxy — the closest valid approximation under System Cold Start.

---

## 📊 Results

### Summary

| Metric | Content-Based | Hybrid |
|---|---|---|
| **Mean Match Score** | 0.8196 | **0.8614** ✓ |
| Median Score | 0.831 | **0.872** ✓ |
| Std Deviation | **0.097** | 0.172 |
| Rank-1 Assigned | **112 / 120 (93.3%)** ✓ | 108 / 120 (90%) |
| Overflow to Rank-2 | **8 students** ✓ | 12 students |
| Rank-3+ Overflow | **0** | **0** |
| All Students Assigned | ✅ 120/120 | ✅ 120/120 |
| Mentor Cap Respected | ✅ All ≤ 15 | ✅ All ≤ 15 |

### 🏆 Winner: Hybrid Model
Higher mean score confirmed statistically significant by Wilcoxon test (p = 0.0000)

### Mentor Load Distribution

```
Dr. Anil Sharma      ██████████████████  15  (at capacity — ML most popular)
Prof. Sunita Gupta   ██████████████████  15  (at capacity — AI/NLP popular)
Ms. Anjali Mehta     ██████████████████  15  (at capacity — Full Stack popular)
Dr. Ramesh Verma     █████████████████   14
Mr. Vikram Singh     █████████████       11
Prof. Suresh Iyer    █████████████       11
Dr. Deepak Joshi     █████████████       11
Mr. Manoj Pillai     █████████████       11
Dr. Kavita Reddy     ███████████          9
Ms. Priya Nair       █████████            8
                     ─────────────────── 15 (cap)
```

---

## 📈 Statistical Analysis

### Wilcoxon Signed-Rank Test

| Parameter | Value |
|---|---|
| Null Hypothesis H₀ | No significant difference between CB and Hybrid scores |
| p-value | **0.0000** |
| Decision | **Reject H₀** — difference is statistically real |
| Why not t-test? | No normality assumption required for Wilcoxon |

### Spearman Rank Correlation

| Parameter | Value |
|---|---|
| ρ (rho) | **0.7161** |
| Interpretation | Strong agreement on which students get well-matched |
| Implication | 21% of students get different mentor — that's the Hybrid's unique contribution |

### Agreement Rate

```
79% of students → Same mentor from both models
21% of students → Different mentor (Hybrid's peer signal changed recommendation)
```

---

## 📁 Project Structure

```
mentor-student-matching/
│
├── 📓 mentor_matching_colab.py     # Main Google Colab notebook (14 sections)
│
├── 📊 Data/
│   ├── mentors.csv                 # 10 faculty mentors with 3 subjects each
│   └── students.csv                # 120 students with 7 ranked preferences
│
├── 📈 Graphs/                      # Auto-generated by running notebook
│   ├── graph1_box_plot.png
│   ├── graph2_mean_scores.png
│   ├── graph3_histogram.png
│   ├── graph4_mentor_load.png
│   ├── graph5_overflow.png
│   └── graph6_scatter.png
│
└── 📋 Output/                      # Auto-generated by running notebook
    ├── matching_results.csv
    └── mentor_student_matching.xlsx  # 4-sheet Excel download
```

### Notebook Sections

| Section | Purpose |
|---|---|
| 1. Import Libraries | Load pandas, numpy, sklearn, scipy, matplotlib |
| 2. Load Data | Read mentors.csv and students.csv |
| 3. Configuration | MAX_LOAD=15, SIM_THRESHOLD=0.80, CB/CF weights |
| 4. Preprocessing | Build mentor binary + student weight matrices |
| 5. Why No CF | Comment block — System Cold Start explanation |
| 6. Assignment Engine | `assign_with_capacity()` shared function |
| 7. Model 1 — CB | Cosine similarity + capacity assignment |
| 8. Model 2 — Hybrid | Cascade loop + score blending + assignment |
| 9. Mentor Load Table | Students per mentor for both models |
| 10. Statistical Analysis | Wilcoxon, Spearman, agreement rate, verdict |
| 11. Visualisations | 6 graphs saved as PNG files |
| 12. Combined Results | Merge CB and Hybrid results side by side |
| 13. Excel Download | 4-sheet file + `files.download()` trigger |
| 14. Final Summary | Printed summary box with all key metrics |

---

## 🚀 Quick Start

### Prerequisites

All libraries are **pre-installed** in Google Colab. No pip installs required.

```
pandas · numpy · scikit-learn · scipy · matplotlib · openpyxl
```

### Steps

**1. Clone the repository**

```bash
git clone https://github.com/your-username/mentor-student-matching.git
cd mentor-student-matching
```

**2. Open in Google Colab**

Upload `mentor_matching_colab.py` to Google Colab, or click the badge below:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

**3. Upload data files**

In the Colab Files panel (left sidebar), upload:
- `mentors.csv`
- `students.csv`

**4. Run all cells**

`Runtime → Run All` or press `Shift+Enter` on each cell sequentially.

**5. Download results**

Section 13 automatically triggers download of `mentor_student_matching.xlsx`

### Configuration (Optional)

Tune these parameters in **Section 3** before running:

```python
MAX_LOAD      = 15    # max students per mentor (increase to relax constraint)
SIM_THRESHOLD = 0.80  # peer similarity threshold (lower = more CF decisions)
CB_WEIGHT     = 0.60  # content-based weight in hybrid blend
CF_WEIGHT     = 0.40  # peer-signal weight (CB_WEIGHT + CF_WEIGHT must = 1.0)
```

---

## 📤 Output Files

### `mentor_student_matching.xlsx`

| Sheet | Contents | Rows |
|---|---|---|
| **Combined Results** | CB + Hybrid assignments side by side, Same Mentor flag | 120 |
| **Content-Based Detail** | Assigned mentor + their 3 subjects + score | 120 |
| **Hybrid Detail** | Assigned mentor + their 3 subjects + score | 120 |
| **Mentor Load Summary** | Students per mentor, remaining capacity, at-cap flag | 10 |

### Sample Output (Combined Results)

| Student Name | Mentor (CB) | Score (CB) | Rank Used (CB) | Mentor (Hybrid) | Score (Hybrid) | Same Mentor |
|---|---|---|---|---|---|---|
| Aarav Sharma | Dr. Anil Sharma | 0.9910 | 1 | Dr. Anil Sharma | 0.9945 | True |
| Aditya Verma | Dr. Ramesh Verma | 0.9732 | 1 | Dr. Ramesh Verma | 0.9801 | True |
| Ananya Singh | Ms. Anjali Mehta | 0.9145 | 1 | Ms. Priya Nair | 0.8823 | False |

---

## ⚠️ Limitations

| Limitation | Description |
|---|---|
| **No ground truth** | No labels/ratings to evaluate actual match quality — only score comparison |
| **Proxy CF** | Hybrid's collaborative signal uses preference similarity, not real interactions |
| **Greedy assignment** | Not globally optimal (Hungarian Algorithm would be, at O(n³) cost) |
| **Static system** | No real-time updates — re-run required if preferences change |
| **Small dataset** | 10 mentors / 120 students — sufficient for demo, small for deep CF |
| **Self-reported data** | Student preferences may not accurately reflect actual academic needs |

---

## 🔮 Future Scope

```
1. 📝 Collect post-session feedback  →  Enable genuine Collaborative Filtering
2. 🧮 Hungarian Algorithm            →  Globally optimal assignment (O(n³))
3. 🌐 Streamlit web app              →  Interactive UI for students and admins
4. ⚖️ Dynamic weight adjustment      →  CB-heavy early, CF-heavy as data grows
5. 🎯 Enriched features              →  Learning style, career goals, availability
6. 🔄 Longitudinal tracking          →  Compare ML vs manual outcomes over semesters
7. 🛡️ Fairness constraints           →  Minimum load floor + diversity enforcement
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.x |
| Data Processing | pandas, numpy |
| ML / Similarity | scikit-learn (cosine_similarity, MultiLabelBinarizer) |
| Statistical Tests | scipy (wilcoxon, spearmanr) |
| Visualisation | matplotlib |
| Excel Export | openpyxl |
| Platform | Google Colab |

---

## 👨‍💻 Team

| Name | Scholar Number | Role |
|---|---|---|
| **Shreyas Shinde** | 23U02100 | ML Pipeline, CB Model, Statistical Analysis |
| **Shivanshu Mishra** | 23U02041 | Hybrid Model, Visualisations, Report |

**Supervisor:** Dr. Sreemoyee Biswas, Assistant Professor, IIIT Bhopal

**Institution:** Indian Institute of Information Technology, Bhopal
**Department:** Computer Science and Engineering
**Group:** 41 · Minor Project

---

## 📖 Report

The complete project report covers:

- Literature Review (Resnick & Varian 1997 through Spearman 1904)
- Problem Definition with formal bipartite matching formulation
- Methodology — mathematical foundations of cosine similarity, vectorisation, and hybrid blending
- Result Discussion with all 6 graphs and statistical test interpretations
- Conclusion & Future Scope
- IEEE-format references



