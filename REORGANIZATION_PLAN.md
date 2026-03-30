# AutoAuth Project Reorganization Plan

## New Professional Structure

```
autoauth/
├── README.md                          # Project overview and setup
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment template
├── .gitignore                         # Git ignore rules
│
├── src/                               # Source code
│   ├── __init__.py
│   ├── agents/                        # AI agents
│   │   ├── __init__.py
│   │   ├── clinical_reader.py         # Clinical data extraction
│   │   ├── policy_engine.py           # Policy decision logic
│   │   └── appeal_generator.py        # Appeal letter generation
│   │
│   ├── ui/                            # User interfaces
│   │   ├── __init__.py
│   │   └── streamlit_app.py           # Main Streamlit UI
│   │
│   └── utils/                         # Utilities
│       ├── __init__.py
│       ├── env_loader.py              # Environment config
│       └── policy_ingestion.py        # Vector store setup
│
├── data/                              # Data files
│   ├── policies/                      # Payer policy PDFs
│   │   ├── aetna/
│   │   ├── cigna/
│   │   └── uhc/
│   │
│   ├── patients.csv                   # Sample patient data
│   └── vector_store/                  # Embeddings (generated)
│
├── output/                            # Runtime outputs
│   ├── bundles/                       # Clinical bundles
│   ├── results/                       # PA decisions
│   ├── dashboard_metrics.json
│   └── run_history.json
│
├── scripts/                           # Utility scripts
│   ├── setup_vector_store.py         # Initialize embeddings
│   └── check_setup.py                # Verify installation
│
└── docs/                              # Documentation
    └── ARCHITECTURE.md                # System design

## Files to Keep
- Core agents: clinical_reader_agent.py, policy_agent.py, appeal_agent.py
- UI: streamlit_app_v3.py (new version)
- Data: patients.csv, policies/*.pdf, vector_store/
- Config: .env, requirements.txt, .gitignore
- Utils: load_gemini_env.py, ingest_policies_used.py, check_setup.py

## Files to Remove/Archive
- app.py (duplicate/old version)
- streamlit_app_fixed copy.py (duplicate)
- pipeline.py (if not used)
- template.html (if not used)
- .zip (archive file)
- __pycache__/ (generated)
- output/*.json (old test runs - keep only metrics/history)
```
