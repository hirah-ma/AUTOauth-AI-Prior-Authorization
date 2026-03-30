"""
Setup script to initialize the vector store from policy PDFs
Run this once before using the application for the first time
"""

import sys
from pathlib import Path

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.policy_ingestion import main

if __name__ == "__main__":
    print("=" * 60)
    print("AutoAuth Vector Store Setup")
    print("=" * 60)
    print("\nThis will:")
    print("  1. Read all PDFs from data/policies/")
    print("  2. Generate embeddings using Gemini API")
    print("  3. Store in data/vector_store/")
    print("\nNote: This requires a valid GEMINI_API_KEY in .env")
    print("=" * 60)
    
    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Setup cancelled.")
        sys.exit(0)
    
    print("\nStarting vector store initialization...\n")
    main()
    print("\n" + "=" * 60)
    print("Vector store setup complete!")
    print("=" * 60)
