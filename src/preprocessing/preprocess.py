import sys
import os
from datetime import datetime, date
import pandas as pd
from typing import Dict, List, Optional, Union
import json
from pathlib import Path
import random
from abc import abstractmethod

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.database.vector_operations import VectorStore
from src.preprocessing.generator.victim_profile.personas import PersonaGenerator
from src.preprocessing.generator.scam_report.scam_details import ScamDetailsGenerator
from config.settings import get_settings
from config.logging_config import setup_logger


random.seed(42)

class Preprocessor:
    """Base class for data preprocessing with configurable data loading. 
    Data files are shared across the two processors so there may be specific checks (e.g. scam headers) 
    that apply to both subclasses as well as functions that are used by both."""
    
    def __init__(self, vector_store: Optional[VectorStore] = None, input_file: str = None, output_file: Optional[str] = None):
        """Initialize with optional vector store and settings."""
        
        self.settings = get_settings()
        self.vector_store = vector_store
        self.logger = setup_logger("Preprocessor", self.settings.log.subdirectories["preprocessing"])
        # self.templates = self._load_templates()
        self.input_file = input_file or self._get_default_input_file()
        self.output_file = output_file or self._get_default_output_file()
    
    def _get_default_input_file(self) -> str:
        """Return the default input file path for the preprocessor."""
        raise NotImplementedError("Subclasses must define default input file")

    def _get_default_output_file(self) -> str:
        """Return the default output file path for the preprocessor."""
        raise NotImplementedError("Subclasses must define default output file")
    
    # def _load_templates(self) -> Dict:
    #     """Load scam templates from JSON configuration file."""
        
    #     template_file = self.settings.data.scam_templates_json
    #     try:
    #         with open(template_file, 'r') as f:
    #             templates = json.load(f)
    #         self.logger.info(f"Loaded templates from {template_file}")
    #         return templates["response_templates"]
    #     except FileNotFoundError as e:
    #         self.logger.error(f"Template file not found: {template_file}")
    #         raise FileNotFoundError(f"Template file not found: {template_file}") from e
    #     except Exception as e:
    #         self.logger.error(f"Error loading templates from {template_file}: {str(e)}")
    #         raise
    
    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """Load data from a CSV file with header validation."""
        
        file_path = file_path or self.input_file
        if not file_path:
            raise ValueError("No input file path provided for data loading")
        
        try:
            df = pd.read_csv(file_path, dtype={'scam_contact_no': str}).fillna("NA") #scam contact number has to be manually defined as strings due to how pandas reads numerical strings in CSV files
            self.logger.info(f"Loaded {len(df)} records from {file_path}")
            

            return df
        except FileNotFoundError as e:
            self.logger.error(f"CSV file not found: {file_path}")
            raise FileNotFoundError(f"CSV file not found: {file_path}") from e
        except Exception as e:
            self.logger.error(f"Error loading CSV {file_path}: {str(e)}")
            raise
        
    def save_data(self, data: Union[pd.DataFrame, List], output_file: Optional[str] = None) -> None:
        """Save processed data to the specified output file based on file extension."""
        output_file = output_file or self.output_file
        if not output_file:
            self.logger.info("No output file specified, skipping save")
            return

        try:
            output_dir = Path(output_file).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            file_extension = Path(output_file).suffix.lower()

            if file_extension == '.csv' and isinstance(data, pd.DataFrame):
                data.to_csv(output_file, index=False)
                self.logger.info(f"Saved DataFrame to CSV at {output_file}")
            elif file_extension == '.json' and isinstance(data, list):
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)
                self.logger.info(f"Saved list to JSON at {output_file}")
            else:
                raise ValueError(f"Unsupported file extension '{file_extension}' or data type '{type(data)}' for saving to {output_file}")
        except Exception as e:
            self.logger.error(f"Error saving output to {output_file}: {str(e)}")
            raise
    
    @abstractmethod
    def preprocess(self, input_file: Optional[str] = None, output_file: Optional[str] = None) -> Union[pd.DataFrame, List[Dict]]:
        """Preprocess data and save results.

        Args:
            input_file (Optional[str]): Path to input CSV file. Defaults to self.input_file.
            output_file (Optional[str]): Path to output file. Defaults to self.output_file.

        Returns:
            Union[pd.DataFrame, List[Dict]]: Processed data (DataFrame or list of dictionaries).
        """
        pass

class ScamReportPreprocessor(Preprocessor):
    """Preprocessor for scam report data to prepare for embedding seed scam report data. Generates scam_report_processed for database ingestion."""
    
    def _get_default_input_file(self) -> str:
        """Return the default input file path for scam report preprocessing."""
        return self.settings.data.scam_report_csv

    def _get_default_output_file(self) -> str:
        """Return the default output file path for scam report preprocessing."""
        return self.settings.data.scam_report_csv_processed
    
    #Embedding for scam incident description only
    def generate_embedding_text(self, row: pd.Series) -> str: 
        """Generate text for embedding. Only embeds scam_incident_description to ensure better cosine similarity search."""       
  
        description = row.get("scam_incident_description", "")
        if not description:
            self.logger.warning(f"No incident description found in row {row.name}; using empty string for embedding")
        return description  
    
    def preprocess(self, input_file: Optional[str] = None, output_file: Optional[str] = None) -> pd.DataFrame:
        """Preprocess scam report data for embedding and database ingestion."""
        
        self.logger.info(f"Starting preprocessing with input_file={input_file}, output_file={output_file}")
        df = self.load_data(input_file)
        
        records = []
        for index, row in df.iterrows():
            try:
                incident_date = pd.to_datetime(row["scam_incident_date"]).date()
                report_date = pd.to_datetime(row["scam_report_date"]).date()
                text_to_embed=self.generate_embedding_text(row)
                embedding = self.vector_store.get_embedding(text_to_embed) if self.vector_store else None
                
                # Create record 
                record = {
                    "report_id": row["scam_report_no"],
                    "scam_incident_date": incident_date,
                    "scam_report_date": report_date,
                    "scam_type": row["scam_type"],
                    "scam_approach_platform": row.get("scam_approach_platform", "NA"),
                    "scam_communication_platform": row.get("scam_communication_platform", "NA"),
                    "scam_transaction_type": row.get("scam_transaction_type", "NA"),
                    "scam_beneficiary_platform": row.get("scam_beneficiary_platform", "NA"),
                    "scam_beneficiary_identifier": row.get("scam_beneficiary_identifier", "NA"),
                    "scam_contact_no": str(row.get("scam_contact_no", "NA")),
                    "scam_email": row.get("scam_email", "NA"),
                    "scam_moniker": str(row.get("scam_moniker", "NA")),
                    "scam_url_link": row.get("scam_url_link", "NA"),
                    "scam_amount_lost": float(row["scam_amount_lost"]) if row["scam_amount_lost"] != "NA" else 0.0,
                    "scam_incident_description": row["scam_incident_description"],
                    "embedding": embedding
                }
                records.append(record)
            except Exception as e:
                self.logger.error(f"Error processing row {index}: {str(e)}")
                continue
        
        result_df = pd.DataFrame(records)
        self.logger.info(f"Processed {len(result_df)} records for embedding")
        self.save_data(result_df, output_file)
        return result_df

class VictimProfilePreprocessor(Preprocessor):
    """
    Preprocessor for victim chatbot data to create consolidated victim_details.json that includes victim persona (tech_literacy, language_proficiency and emotional_state) and scam details for evaluation.
    """
    
    def _get_default_input_file(self) -> str:
        """
        Return the default input file path for victim profile preprocessing.
        Since this preprocessor generates data programmatically, no default input file is needed.
        Returning None to indicate no file loading is required. 
        """
        return None

    def _get_default_output_file(self) -> str:
        """Return the default output file path for victim profile preprocessing."""
        return self.settings.data.victim_details_json
    
    def preprocess(self, num_records_per_domain: int = 2, input_file: Optional[str] = None, output_file: Optional[str] = None) -> List[Dict[str, Dict]]:
        scam_types = ['ecommerce', 'phishing', 'government officials impersonation']
        scam_generator = ScamDetailsGenerator()
        
        # Generate scams separately for each domain
        domain_scams = {}
        for domain in scam_types:
            proportions = {domain: 1.0}  # 100% for this domain
            scam_df = scam_generator.generate_scam_dataframe(
                total_records=num_records_per_domain,
                proportions=proportions,
                output_file=None
            )
            domain_scams[domain] = scam_df.to_dict(orient='records')
        
        persona_gen = PersonaGenerator()
        
        # Generate all 27 unique profile combinations
        profile_combinations = persona_gen.generate_all_permutation_targets()
        
        processed_data = []
        profile_id_counter = 1  # Start sequential ID from 1
        
        for domain in scam_types:
            scams = domain_scams[domain]
            # Shuffle scams for random pairing
            random.shuffle(scams)
            
            for i, user_profile in enumerate(profile_combinations):
                # Cycle through scams if somehow fewer than 27 (though should be exact)
                scam_row = scams[i % len(scams)]
                
                scam_details = {
                    "scam_report_no": scam_row["scam_report_no"],
                    "scam_incident_date": scam_row["scam_incident_date"].strftime('%Y-%m-%d') if isinstance(scam_row["scam_incident_date"], datetime) else scam_row["scam_incident_date"],
                    "scam_report_date": scam_row["scam_report_date"].strftime('%Y-%m-%d') if isinstance(scam_row["scam_report_date"], datetime) else scam_row["scam_report_date"],
                    "scam_type": scam_row["scam_type"],
                    "scam_approach_platform": scam_row.get("scam_approach_platform", "NA"),
                    "scam_communication_platform": scam_row.get("scam_communication_platform", "NA"),
                    "scam_transaction_type": scam_row.get("scam_transaction_type", "NA"),
                    "scam_beneficiary_platform": scam_row.get("scam_beneficiary_platform", "NA"),
                    "scam_beneficiary_identifier": scam_row.get("scam_beneficiary_identifier", "NA"),
                    "scam_contact_no": str(scam_row.get("scam_contact_no", "NA")),
                    "scam_email": scam_row.get("scam_email", "NA"),
                    "scam_moniker": str(scam_row.get("scam_moniker", "NA")),
                    "scam_url_link": scam_row.get("scam_url_link", "NA"),
                    "scam_amount_lost": float(scam_row["scam_amount_lost"]),
                    "scam_incident_description": scam_row["scam_incident_description"]
                }
                
                processed_data.append({
                    "profile_id": profile_id_counter,
                    "user_profile": user_profile,  
                    "scam_details": scam_details
                })
                profile_id_counter += 1
        
        self.save_data(processed_data, output_file)
        return processed_data
    
#Test functions for preprocessing
if __name__ == "__main__":

    settings = get_settings()
    
    #Preprocess scam details for embedding
    scam_preprocessor = ScamReportPreprocessor()
    scam_preprocessor.preprocess()
    
    # Preprocess victim details and scam details for victim chatbot
    victim_preprocessor = VictimProfilePreprocessor()
    victim_preprocessor.preprocess()
    