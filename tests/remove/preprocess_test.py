import pandas as pd
import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from src.preprocessing.preprocess import ScamReportPreprocessor, VictimChatbotPreprocessor
from config.settings import get_settings
import json
from pathlib import Path

def check_preprocess_output():
    """Check the output of both preprocessors using sample data."""
    settings = get_settings()
    output_dir = Path("tests/output")
    output_dir.mkdir(exist_ok=True)
    
    # Process scam report data for embedding
    try:
        scam_output_file = output_dir / "scam_preprocess_sample.csv"
        scam_preprocessor = ScamReportPreprocessor()  # VectorStore omitted for demo
        processed_scam_df = scam_preprocessor.preprocess(
            input_file=settings.data.scam_report_csv,
            output_file=scam_output_file
        )
        
        # Print first 2 records (embedding field omitted for simplicity)
        scam_output = processed_scam_df.drop(columns=["embedding"]).head(2).to_dict(orient="records")
        print("Scam Report Preprocessor Output (first 2 records):")
        print(json.dumps(scam_output, indent=2, default=str))
    except Exception as e:
        print(f"Error processing scam reports: {str(e)}")
    
    # Process combined scam report data for chatbot
    try:
        chatbot_output_file = output_dir / "chatbot_preprocess_sample.json"
        chatbot_preprocessor = VictimChatbotPreprocessor()
        processed_chatbot_data = chatbot_preprocessor.preprocess(
            input_file=settings.data.combined_scam_report_csv,
            output_file=chatbot_output_file
        )
        
        # Print first 2 records
        print("\nChatbot Preprocessor Output (first 2 records):")
        print(json.dumps(processed_chatbot_data[:2], indent=2))
    except Exception as e:
        print(f"Error processing chatbot data: {str(e)}")

if __name__ == "__main__":
    check_preprocess_output()