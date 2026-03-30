# 🚀 AutoAuth Quick Start Guide

## Running the Reorganized Project

### Prerequisites
- Python 3.9 or higher
- Gemini API key from Google AI Studio

---

## Step-by-Step Setup

### 1. **Install Dependencies**

```bash
pip install -r requirements.txt
```

This installs:
- `streamlit` — Web UI framework
- `google-generativeai` — Gemini API client
- `faiss-cpu` — Vector similarity search
- `pandas` — Data processing
- `pypdf` — PDF parsing
- Other utilities

---

### 2. **Configure Environment**

Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env
```

Edit `.env` and add your Gemini API key:

```
GEMINI_API_KEY=your_actual_api_key_here
```

**Get your API key:** https://aistudio.google.com/app/apikey

---

### 3. **Initialize Vector Store** (First Time Only)

The vector store contains embeddings of payer policy PDFs for fast similarity search.

**Option A: Using the setup script (Recommended)**
```bash
python scripts/setup_vector_store.py
```

**Option B: Direct ingestion**
```bash
python src/utils/policy_ingestion.py
```

This will:
- Read all PDFs from `data/policies/`
- Generate embeddings using Gemini
- Save to `data/vector_store/`

**⏱️ Time:** ~2-5 minutes depending on number of policies

---

### 4. **Run the Application**

You have two UI options:

#### **Option A: New Product UI** (Recommended)
Clean, modern, horizontal 3-step flow

```bash
streamlit run src/ui/streamlit_app.py
```

#### **Option B: Legacy UI**
Full-featured dark theme with comprehensive analytics

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📂 Project Structure Reference

```
FILES/
├── app.py                    # Main app (legacy UI) - RUN THIS
├── requirements.txt          # Dependencies
├── .env                      # Your API keys (create this)
│
├── src/
│   ├── agents/              # AI agents (imported by app.py)
│   │   ├── clinical_reader.py
│   │   ├── policy_engine.py
│   │   └── appeal_generator.py
│   │
│   ├── ui/
│   │   └── streamlit_app.py # New UI - RUN THIS
│   │
│   └── utils/
│       ├── env_loader.py
│       └── policy_ingestion.py
│
├── data/
│   ├── policies/            # Payer policy PDFs (already present)
│   ├── patients.csv         # Sample patient data (already present)
│   └── vector_store/        # Generated embeddings (create via step 3)
│
├── output/                  # Runtime outputs (auto-created)
│   ├── dashboard_metrics.json
│   └── run_history.json
│
└── scripts/
    ├── check_setup.py       # Verify installation
    └── setup_vector_store.py # Initialize embeddings
```

---

## ✅ Verify Installation

Run the setup checker:

```bash
python scripts/check_setup.py
```

This verifies:
- ✓ Python dependencies installed
- ✓ `.env` file exists with API key
- ✓ Vector store initialized
- ✓ Data files present

---

## 🎯 Usage Examples

### Example 1: Run with Sample Scenario

1. Start the app:
   ```bash
   streamlit run app.py
   ```

2. In the UI:
   - Select a payer: **UnitedHealthcare**
   - Choose sample case: **Asthma Exacerbation**
   - Click **🚀 Run Authorization Check**

3. Watch the 3-step pipeline:
   - **Step 1:** Clinical Reader extracts codes
   - **Step 2:** Policy Engine evaluates criteria
   - **Step 3:** Decision + Appeal (if denied)

### Example 2: Custom EHR Note

1. Select **EHR Note** input mode
2. Paste your clinical note:
   ```
   PATIENT: Jane Doe | DOB: 03/15/1968
   CHIEF COMPLAINT: Chronic low back pain
   DX: M54.16, M51.06
   CPT: 63030
   Evidence: MRI shows L4-L5 herniation. Failed 8 weeks PT.
   ```
3. Run the pipeline

### Example 3: CSV Patient Data

1. Select **CSV Patient** input mode
2. Enter patient ID (e.g., `1`, `2`, `702`)
3. Run the pipeline

---

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'src'"

**Solution:** Make sure you're running from the project root directory:
```bash
cd "c:\Users\hirah\OneDrive\Desktop\AutoAuth Agent\FILES"
streamlit run app.py
```

### Issue: "Vector store not found"

**Solution:** Initialize the vector store:
```bash
python scripts/setup_vector_store.py
```

### Issue: "Invalid API key"

**Solution:** Check your `.env` file:
```bash
# Make sure it contains:
GEMINI_API_KEY=AIza...your_actual_key
```

### Issue: "No policies found"

**Solution:** Verify PDFs exist:
```bash
ls data/policies/
# Should show: aetna_asthma_policy.pdf, uhc_diabetes_type2_policy.pdf, etc.
```

---

## 📊 Understanding the Output

### Decision Types

- **✅ APPROVED** — Case meets all payer criteria, ready to submit
- **❌ DENIED** — Missing required criteria, appeal letter generated
- **⚑ PENDING** — Manual review needed, additional documentation required
- **⚠️ ERROR** — Processing failed, route to manual review

### Metrics Dashboard (Sidebar)

- **Total Runs** — Number of cases processed
- **Approval Rate** — % of approved cases
- **Days Saved** — Estimated time saved vs. manual processing
- **Payer Behavior** — Approval rates by payer

---

## 🎨 UI Comparison

### New Product UI (`src/ui/streamlit_app.py`)
✨ **Best for:** Clean, modern experience
- Horizontal 3-step flow
- Card-based design
- Insights over raw codes
- AI confidence indicators

### Legacy UI (`app.py`)
📊 **Best for:** Comprehensive analytics
- Dark clinical theme
- Detailed metrics
- Payer behavior charts
- Full feature set

---

## 🔄 Next Steps

1. ✅ Install dependencies
2. ✅ Configure `.env`
3. ✅ Initialize vector store
4. ✅ Run the app
5. 🎯 Process your first case
6. 📊 Review analytics
7. 🚀 Integrate with your workflow

---

## 📞 Need Help?

- Check `README.md` for detailed documentation
- Review `docs/ARCHITECTURE.md` for system design
- Run `python scripts/check_setup.py` to diagnose issues

---

**Ready to go? Run this command:**

```bash
streamlit run app.py
```

🎉 **Your AutoAuth system is ready!**
