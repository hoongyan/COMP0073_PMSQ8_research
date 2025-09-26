
import sys
import os
import pandas as pd
import yaml
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from config.settings import get_settings
from src.preprocessing.generator.person_details.person_details import VictimProfileGenerator
from src.preprocessing.generator.scam_report.scam_details import ScamDetailsGenerator

class CombinedGenerator:
    """Class to generate and combine victim profiles and scam details with consistent proportions."""
    
    def __init__(self, victim_config_path='config/victim_config.yaml', scam_config_path='config/scam_config.yaml'):
        """Initialize with paths to configuration files."""
        
        # Load configuration files
        try:
            with open(victim_config_path, 'r') as f:
                self.victim_config = yaml.safe_load(f)
            with open(scam_config_path, 'r') as f:
                self.scam_config = yaml.safe_load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Configuration file not found: {e}")
        
        self.victim_generator = VictimProfileGenerator(victim_config_path)
        self.scam_generator = ScamDetailsGenerator(scam_config_path)
        self.settings = get_settings()
        self.default_output_file = self.settings.data.full_scam_scenario_csv

    def generate_combined_reports(self, total_records: int, scam_type_weights: dict = None) -> pd.DataFrame:
        """Generate combined victim and scam reports with identical proportions for each scam type."""
        
        # Use default weights from scam_config if none provided
        if scam_type_weights is None:
            scam_type_weights = self.scam_config['scam_type_weights']
        
        if not abs(sum(scam_type_weights.values()) - 1.0) < 1e-6:
            raise ValueError(f"Scam type weights must sum to 1.0, got {sum(scam_type_weights.values())}")
        for scam_type in scam_type_weights:
            if scam_type not in self.victim_config['demographic_configs']:
                raise ValueError(f"Invalid scam type: {scam_type}. Choose from {list(self.victim_config['demographic_configs'].keys())}")

        scam_types = list(scam_type_weights.keys())
        weights = list(scam_type_weights.values())
        scam_type_counts = {
            scam_type: int(total_records * proportion)
            for scam_type, proportion in scam_type_weights.items()
        }
        total_assigned = sum(scam_type_counts.values())
        if total_assigned < total_records:
            max_proportion_scam = max(scam_type_weights, key=scam_type_weights.get)
            scam_type_counts[max_proportion_scam] += total_records - total_assigned
        print(f"Record counts per scam type: {scam_type_counts}")
        
        # Generate victim profiles
        victim_df = self.victim_generator.generate_profiles_dataframe(total_records, scam_type_weights)
        print(f"Generated {len(victim_df)} victim profiles")
        print("Victim profiles per scam type:")
        print(victim_df['scam_type'].value_counts())
        
        # Generate scam reports
        scam_df = self.scam_generator.generate_scam_dataframe(total_records, scam_type_weights)
        print(f"Generated {len(scam_df)} scam reports")
        print("Scam reports per scam type:")
        print(scam_df['scam_type'].value_counts())
        
        # Combine victims and scams by scam type
        combined_dfs = []
        for scam_type in scam_types:
            
            # Convert scam_type to uppercase to match transformed values
            scam_type_upper = scam_type.replace('_', ' ').upper()
            victim_subset = victim_df[victim_df['scam_type'] == scam_type_upper].copy()
            scam_subset = scam_df[scam_df['scam_type'] == scam_type_upper].copy()
            
            print(f"Processing scam type: {scam_type_upper}")
            print(f"  Victim subset size: {len(victim_subset)}")
            print(f"  Scam subset size: {len(scam_subset)}")
            
            if len(victim_subset) == 0 or len(scam_subset) == 0:
                print(f"Skipping scam type {scam_type_upper} due to empty victim or scam subset")
                continue
            
            # Shuffle both subsets
            victim_subset = victim_subset.sample(frac=1, random_state=42).reset_index(drop=True)
            scam_subset = scam_subset.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Use the minimum number of records
            min_records = min(len(victim_subset), len(scam_subset))
            victim_subset = victim_subset.iloc[:min_records]
            scam_subset = scam_subset.iloc[:min_records]
            
            # Drop duplicate scam_type
            scam_subset = scam_subset.drop(columns=['scam_type'])
            
            # Combine victim and scam data
            combined_subset = pd.concat([victim_subset, scam_subset], axis=1)
            combined_dfs.append(combined_subset)
        
        if not combined_dfs:
            print("No combined records generated. Returning empty DataFrame.")
            return pd.DataFrame(columns=self._get_final_columns())
        
        combined_df = pd.concat(combined_dfs, ignore_index=True)
        
        final_columns = self._get_final_columns()
        
        for col in final_columns:
            if col not in combined_df.columns:
                combined_df[col] = "NA"
        
        combined_df = combined_df[final_columns]
        
        return combined_df
    
    def save_to_csv(self, combined_df: pd.DataFrame, output_file: str = None) -> None:
        """Save the combined DataFrame to a CSV file."""
        
        if output_file is None:
            output_file = self.default_output_file
        if not combined_df.empty and output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            combined_df.to_csv(output_file, index=False)
            print(f"Saved {len(combined_df)} combined profiles to {output_file}")

    def _get_final_columns(self) -> list:
        """Return the list of final columns in the specified order."""
        
        return [
            'vic_first_name', 'vic_last_name', 'vic_nric', 'vic_sex', 'vic_dob', 
            'vic_nationality', 'vic_race', 'vic_occupation', 'vic_contact_no', 
            'vic_email', 'vic_blk', 'vic_street', 'vic_unit_no', 'vic_postal_code',
            'scam_report_no', 'scam_incident_date', 'scam_report_date', 'scam_type', 
            'scam_approach_platform', 'scam_communication_platform', 
            'scam_transaction_type', 'scam_beneficiary_platform', 'scam_beneficiary_identifier', 
            'scam_contact_no', 'scam_email','scam_moniker', 'scam_url_link', 'scam_amount_lost', 'scam_incident_description',
            'scam_subcategory', 'scam_item_involved', 'scam_item_type', 'scam_impersonation_type',
            'scam_first_impersonated_entity', 'scam_first_impersonated_entity_name',
            'scam_second_impersonated_entity', 'scam_second_impersonated_entity_name',
            'scam_phished_details', 'scam_use_of_phished_details', 'scam_pretext_for_phishing'
        ]

if __name__ == "__main__":
    
    #test generation
    random.seed(42)  # Set seed
    generator = CombinedGenerator()
    
    #To be adjusted accordingly
    custom_weights = {
        'ecommerce': 0.3,
        'phishing': 0.4,
        'government officials impersonation': 0.3
    }
    
    # Generate combined reports
    combined_df = generator.generate_combined_reports(
        total_records=200,
        scam_type_weights=custom_weights
    )
    
    # Save combined DataFrame
    generator.save_to_csv(combined_df)
    
    print(f"Total combined records generated: {len(combined_df)}")
    if not combined_df.empty:
        print("Records per scam type:")
        print(combined_df['scam_type'].value_counts())
        print("\nSample records:")
        print(combined_df.head())
    else:
        print("No records generated. Check configuration and scam type weights.")