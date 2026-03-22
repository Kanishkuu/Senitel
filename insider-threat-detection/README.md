# CERT Insider Threat Detection System

**AI-Powered Insider Threat Detection using Graph Neural Networks, Temporal Transformers, and Ensemble Anomaly Detection**

---

## 🎯 Quick Start

### Prerequisites
- **Python 3.11+**
- **16 GB RAM** (minimum; 32 GB recommended)
- **NVIDIA GPU with CUDA** (RTX 3060+ recommended; RTX 4060 ✅)
- **~30 GB disk space** for processed data

### Setup

```bash
# 1. Navigate to project directory
cd C:/Darsh/NCPI/insider-threat-detection

# 2. Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment file
copy .env.example .env

# 5. Check GPU is available
python -m src.pipeline.stage1_pipeline gpu-info
```

---

## 🚀 How to Run

### Step 1: Check Dataset Availability

```bash
python -m src.pipeline.stage1_pipeline check
```

This will verify all CERT r4.2 files are accessible and show file sizes.

### Step 2: Run Stage 1 — Data Loading & Normalization

```bash
# Option A: Fast mode (skip http.csv, ~5 minutes)
python -m src.pipeline.stage1_pipeline run

# Option B: Include http.csv (~30 minutes, requires ~20 GB RAM)
python -m src.pipeline.stage1_pipeline run --load-http

# Option C: Sample mode (~2 minutes, for testing)
python -m src.pipeline.stage1_pipeline run --sample

# Option D: Custom output directory
python -m src.pipeline.stage1_pipeline run --output ./my_data
```

### Step 3: Stage 2 — Feature Engineering (coming soon)

```bash
python -m src.pipeline.stage2_pipeline run
```

### Step 4: Stage 3 — Model Training (coming soon)

```bash
# Train on GPU
python -m src.models.train run --gpu

# Train on CPU (slower)
python -m src.models.train run --cpu
```

### Step 5: Launch Dashboard

```bash
cd dashboard
streamlit run app.py
```

---

## 📁 Project Structure

```
insider-threat-detection/
├── src/
│   ├── cert_dataset/           # Data loading & schemas
│   │   ├── __init__.py
│   │   ├── schemas.py          # Schema definitions
│   │   ├── loaders.py          # CSV loaders
│   │   ├── normalizer.py       # Normalization
│   │   └── privacy.py          # Pseudonymization & audit
│   ├── models/                  # ML models (coming soon)
│   │   ├── gnn_detector.py
│   │   ├── transformer_detector.py
│   │   └── ensemble_scorer.py
│   ├── pipeline/               # Pipeline scripts
│   │   └── stage1_pipeline.py  # ← RUN THIS FIRST
│   ├── scoring/                # Risk scoring
│   └── evaluation/             # Evaluation metrics
├── configs/
│   └── config.yaml             # Configuration
├── data/
│   └── normalized/             # Output from Stage 1
│       ├── logon.parquet
│       ├── device.parquet
│       ├── file.parquet
│       ├── email.parquet
│       ├── http.parquet
│       ├── psychometric.parquet
│       ├── ldap.parquet
│       └── ground_truth.parquet
├── dashboard/                  # Streamlit dashboard
├── tests/
├── notebooks/
├── .env.example
├── requirements.txt
└── README.md
```

---

## 💾 Hardware Requirements & Optimization

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 16 GB | 32 GB |
| **GPU VRAM** | 4 GB | 8 GB (RTX 4060 ✅) |
| **Disk** | 30 GB | 50 GB |
| **CPU** | 4 cores | 8+ cores |

### Memory Optimization Guide

**If you have 16 GB RAM:**
```bash
# Use fast mode (skip http.csv)
python -m src.pipeline.stage1_pipeline run
```

**If you have 32 GB RAM:**
```bash
# Can load http.csv
python -m src.pipeline.stage1_pipeline run --load-http
```

**If you have 64 GB+ RAM:**
```bash
# Full pipeline with all data
python -m src.pipeline.stage1_pipeline run --load-http
# Then process http.csv separately
python -m src.pipeline.stage2_pipeline run --include-http
```

---

## 🔧 Configuration

All settings are in `configs/config.yaml`:

```yaml
dataset:
  root: "C:/Darsh/NCPI/r4.2/r4.2"
  ldap_dir: "C:/Darsh/NCPI/r4.2/r4.2/LDAP"
  answers_dir: "C:/Darsh/NCPI/r4.2/answers"

preprocessing:
  chunk_size: 500_000        # Rows per chunk (reduce if OOM)
  use_gpu: 1                 # 1 = use GPU, 0 = CPU only
  gpu_device: 0               # GPU device ID

output:
  parquet_compression: "zstd"  # zstd (fast) or snappy
```

---

## 📊 Dataset: CERT r4.2

| File | Size | Rows (approx) | Description |
|------|------|----------------|-------------|
| `logon.csv` | 58 MB | 975,000 | User logon/logoff events |
| `device.csv` | 29 MB | 484,000 | USB device connections |
| `file.csv` | 193 MB | 966,000 | File copy operations |
| `email.csv` | 1.36 GB | 7.5M | Email send/receive |
| `http.csv` | 14.5 GB | 58M | Web browsing (optional) |
| `psychometric.csv` | 45 KB | 1,000 | Big Five personality traits |
| `LDAP/` | ~1.5 MB | 18 files | Monthly org snapshots |
| **Total** | ~16.6 GB | ~8M+ events | |

---

## 🔐 Privacy

All user/PC identifiers are pseudonymized using SHA-256 with an organization-specific salt. Original IDs cannot be recovered from processed data.

Audit logs are written to `logs/audit.log` for compliance.

---

## 📈 What's Coming

| Stage | Description | Status |
|-------|-------------|--------|
| **Stage 1** | Data Loading & Normalization | ✅ Ready |
| **Stage 2** | Feature Engineering | 🔨 Coming soon |
| **Stage 3** | Model Training | 🔨 Coming soon |
| **Stage 4** | Dashboard | 🔨 Coming soon |

---

## ⚠️ Troubleshooting

**Out of Memory (OOM)?**
```bash
# Reduce chunk size in config.yaml
preprocessing:
  chunk_size: 250_000  # Halve the chunk size
```

**GPU not detected?**
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Set GPU device
set TORCH_DEVICE=cuda:0
```

**Slow processing?**
```bash
# Use zstd compression (faster than gzip)
output:
  parquet_compression: "zstd"
```

---

## 📄 License

MIT License — See LICENSE file for details.
