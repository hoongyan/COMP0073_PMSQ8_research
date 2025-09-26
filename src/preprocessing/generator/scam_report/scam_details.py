
import sys
import os
from datetime import datetime, timedelta
from abc import ABC
import random
import pandas as pd
import numpy as np
from faker import Faker
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from config.settings import get_settings

class ScamGenerator(ABC):
    """
    Abstract base class for generating general scam-related data records, such as scam subcategory, report number, phone number, beneficiary platform details and transaction summary.
    """
    
    def __init__(self, fake: Faker, config: dict):
        """Initialize the scam generator with configuration and Faker instance."""
        
        self.fake = fake
        self.config = config
        self.transaction_configs = config['general_scam_configs']['transaction_configs']
        self.prefixes = config['general_configs']['report_no_prefixes']

    def select_subcategory(self, scam_type: str) -> tuple[str, dict]:
        """Select a scam subcategory based on probability."""
        
        subcategories = self.config['scam_specific_configs'][scam_type]['scam_subcategory']
        subcategory_names = list(subcategories.keys())
        probabilities = [subcategories[name]['probability'] for name in subcategory_names]
        if not abs(sum(probabilities) - 1.0) < 1e-6:
            raise ValueError(f"Probabilities for {scam_type} subcategories do not sum to 1: {probabilities}")
        selected_name = np.random.choice(subcategory_names, p=probabilities)
        return selected_name, subcategories[selected_name]

    def generate_report_no(self, report_date: datetime) -> str:
        """Generate a unique report number based on report date."""
        
        date_str = report_date.strftime('%Y%m%d')
        first_digit = random.choice(['7', '2'])
        last_three = str(random.randint(0, 999)).zfill(3)
        suffix = f'{first_digit}{last_three}'
        return f'{random.choice(self.prefixes)}/{date_str}/{suffix}'

    def generate_phone_number(self) -> str:
        """Generate a Singapore phone number."""
        return f"+65{random.randint(90000000, 99999999)}"

    def generate_beneficiary_details(self, transaction_type: str) -> tuple[str, str]:
        """Generate beneficiary platform and identifier for transactions."""
        if transaction_type.lower() == "bank transfer":
            platform = random.choice(self.transaction_configs['Bank Transfer']['beneficiary_platforms'])
            identifier = ''.join(str(random.randint(0, 9)) for _ in range(8))
            return platform, identifier
        return "NA", "NA"

    def generate_transactions(self, incident_date: datetime, amount_lost: float, transaction_type: str, scam_subcategory: str, beneficiary_platform: str, beneficiary_identifier: str) -> tuple[list[tuple], str]:
        """Generate a list of transaction details and a transaction summary."""
        trans_time = datetime.combine(incident_date, datetime.min.time()) + timedelta(
            hours=random.randint(0, 23), minutes=random.randint(0, 59), seconds=random.randint(0, 59)
        )
        date_time_str = trans_time.strftime('%Y-%m-%d')
        trans_summary = f"A transaction of ${amount_lost:.2f} was made to {beneficiary_platform} account {beneficiary_identifier} on {date_time_str}."
        return [(trans_time, amount_lost, transaction_type, beneficiary_platform, beneficiary_identifier)], trans_summary

class EcommerceScam(ScamGenerator):
    """
    Class for generating a consolidated set of e-commerce scam report data. 
    Additional columns (i.e., item_type and item_involved) involving scam specific details were included for scalability and tracking. They will be removed when added into the database.
    """
    
    def __init__(self, fake: Faker, config: dict):
        """Initialize the e-commerce scam details generator."""
        
        super().__init__(fake, config)
        self.scam_type = 'ecommerce'

    def generate_scam_details(self) -> dict:
        """Generate details for all scam details for an e-commerce scam report."""
        
        subcategory_name, subcategory = self.select_subcategory(self.scam_type)
        general = subcategory['general_details']
        specific = subcategory['specific_details']

        approach_platform = np.random.choice(general['approach_platforms'], p=general['approach_platform_probs'])
        comm_platform = np.random.choice(
            list(general['comm_platform_probs'][approach_platform].keys()),
            p=list(general['comm_platform_probs'][approach_platform].values())
        )
        item_type = np.random.choice(specific['item_types'], p=specific['item_type_probs'])
        item_involved = random.choice(specific['item_involved_mapping'][item_type])
        min_loss, max_loss = specific['amount_loss_ranges'][item_type]
        amount_lost = round(np.random.uniform(min_loss, max_loss), 2)
        transaction_type = np.random.choice(general['transaction_type'], p=general['transaction_type_probs'])
        incident_date = datetime.strptime(
            np.random.choice(np.arange(
                np.datetime64(self.config['general_configs']['incident_date_range'][0]),
                np.datetime64(self.config['general_configs']['incident_date_range'][1])
            ).astype(datetime).astype(str)), '%Y-%m-%d'
        )
        report_date = incident_date + timedelta(days=random.randint(*self.config['general_configs']['report_date_delay_days']))
        report_no = self.generate_report_no(report_date)
        beneficiary_platform, beneficiary_identifier = self.generate_beneficiary_details(transaction_type)
        moniker = self.fake.user_name()

        transactions, trans_summary = self.generate_transactions(
            incident_date, amount_lost, transaction_type, subcategory_name, beneficiary_platform, beneficiary_identifier
        )

        details = {
            'scam_report_no': report_no,
            'scam_incident_date': incident_date,
            'scam_report_date': report_date,
            'scam_type': self.scam_type,
            'scam_approach_platform': approach_platform,
            'scam_communication_platform': comm_platform,
            'scam_transaction_type': transaction_type,
            'scam_beneficiary_platform': beneficiary_platform,
            'scam_beneficiary_identifier': beneficiary_identifier,
            'scam_contact_no': "NA",
            'scam_email': "NA",
            'scam_moniker': moniker,
            'scam_url_link': "NA",
            'scam_amount_lost': amount_lost,
            'scam_incident_description': "",
            'scam_subcategory': subcategory_name,
            'scam_item_involved': item_involved,
            'scam_item_type': item_type,
            'scam_impersonation_type': "NA",
            'scam_first_impersonated_entity': "NA",
            'scam_first_impersonated_entity_name': "NA",
            'scam_second_impersonated_entity': "NA",
            'scam_second_impersonated_entity_name': "NA",
            'scam_phished_details': "NA",
            'scam_use_of_phished_details': "NA",
            'scam_pretext_for_phishing': "NA",
            'transactions': transactions
        }
        details['scam_incident_description'] = self.generate_description(details, subcategory, trans_summary)
        return details

    def generate_description(self, details: dict, subcategory: dict, trans_summary: str) -> str:
        """Generate a detailed incident description for an e-commerce scam."""
        
        item_involved = details['scam_item_involved']
        item_type = details['scam_item_type']
        article = "" if item_type == 'FOOD_RELATED' else (" an" if item_involved.lower().startswith(('a', 'e', 'i', 'o', 'u')) else " a")

        return subcategory['description'].format(
            article=article,
            item_involved=item_involved,
            approach_platform=details['scam_approach_platform'],
            scam_moniker=details['scam_moniker'],
            communication_platform=details['scam_communication_platform'],
            transaction_summary=trans_summary
        )

class GovernmentOfficialsImpersonationScam(ScamGenerator):
    """
    Class for generating scam details for a government official impersonation scam report.
    Additional columns (i.e., scam_first_impersonated_entity_name, scam_first_impersonated_entity) involving scam specific details were included for scalability and tracking. They will be removed when added into the database.
    """
    
    def __init__(self, fake: Faker, config: dict):
        """Initialize the government impersonation scam generator."""
        
        super().__init__(fake, config)
        self.scam_type = 'government officials impersonation'

    def generate_scam_details(self) -> dict:
        """Generate details for a government impersonation scam."""
        
        subcategory_name, subcategory = self.select_subcategory(self.scam_type)
        general = subcategory['general_details']
        specific = subcategory['specific_details']

        approach_platform = np.random.choice(general['approach_platforms'], p=general['approach_platform_probs'])
        comm_platform = np.random.choice(general['communication_platforms'], p=general['communication_platform_probs'])
        transaction_type = np.random.choice(general['transaction_type'], p=general['transaction_type_probs'])
        amount_lost = round(np.random.uniform(*general['amount_loss_range']), 2)
        incident_date = datetime.strptime(
            np.random.choice(np.arange(
                np.datetime64(self.config['general_configs']['incident_date_range'][0]),
                np.datetime64(self.config['general_configs']['incident_date_range'][1])
            ).astype(datetime).astype(str)), '%Y-%m-%d'
        )
        report_date = incident_date + timedelta(days=random.randint(*self.config['general_configs']['report_date_delay_days']))
        report_no = self.generate_report_no(report_date)
        scammer_contact_no = self.generate_phone_number()
        beneficiary_platform, beneficiary_identifier = self.generate_beneficiary_details(transaction_type)

        if 'accusations' not in specific or not specific['accusations']:
            raise ValueError(f"No accusations defined for subcategory {subcategory_name}")
        accusation = random.choice(specific['accusations'])

        impersonator_name = random.choice(specific['first_impersonated_entity_names'])
        impersonated_entity = random.choice(specific['first_entity_impersonated'])
        second_impersonator_name = random.choice(specific['second_impersonated_entity_names'])
        second_impersonated_entity = random.choice(specific['second_entity_impersonated'])
        impersonation_type = specific.get('impersonation_type', "NA")

        transactions, trans_summary = self.generate_transactions(
            incident_date, amount_lost, transaction_type, subcategory_name, beneficiary_platform, beneficiary_identifier
        )

        details = {
            'scam_report_no': report_no,
            'scam_incident_date': incident_date,
            'scam_report_date': report_date,
            'scam_type': self.scam_type,
            'scam_approach_platform': approach_platform,
            'scam_communication_platform': comm_platform,
            'scam_transaction_type': transaction_type,
            'scam_beneficiary_platform': beneficiary_platform,
            'scam_beneficiary_identifier': beneficiary_identifier,
            'scam_contact_no': scammer_contact_no,
            'scam_email': "NA",
            'scam_moniker': "NA",
            'scam_url_link': "NA",
            'scam_amount_lost': amount_lost,
            'scam_incident_description': "",
            'scam_subcategory': subcategory_name,
            'scam_impersonation_type': impersonation_type,
            'scam_first_impersonated_entity_name': impersonator_name,
            'scam_first_impersonated_entity': impersonated_entity,
            'scam_second_impersonated_entity': second_impersonated_entity,
            'scam_second_impersonated_entity_name': second_impersonator_name,
            'scam_item_involved': "NA",
            'scam_item_type': "NA",
            'scam_phished_details': "NA",
            'scam_use_of_phished_details': "NA",
            'scam_pretext_for_phishing': "NA",
            'transactions': transactions
        }

        details['scam_incident_description'] = self.generate_description(details, subcategory, accusation, trans_summary)
        return details

    def generate_description(self, details: dict, subcategory: dict, accusation: str, trans_summary: str) -> str:
        """
        Generate a detailed incident description for a government impersonation scam.
        """
        
        approach_platform_prefix = "a call" if details['scam_approach_platform'].lower() == "call" else "a WhatsApp call"
        return subcategory['description'].format(
            approach_platform_prefix=approach_platform_prefix,
            approach_platform=details['scam_approach_platform'],
            scammer_contact_no=details['scam_contact_no'],
            first_impersonated_entity_name=details['scam_first_impersonated_entity_name'],
            first_entity_impersonated=details['scam_first_impersonated_entity'],
            second_impersonated_entity_name=details['scam_second_impersonated_entity_name'],
            second_entity_impersonated=details['scam_second_impersonated_entity'],
            communication_platform=details['scam_communication_platform'],
            accusation=accusation,
            transaction_summary=trans_summary
        )

class PhishingScam(ScamGenerator):
    """
    Class for generating scam details for a phishing scam report.
    Additional columns (i.e., phished details, use of phished details) involving scam specific details were included for scalability and tracking. They will be removed when added into the database.
    """
    
    def __init__(self, fake: Faker, config: dict):
        """Initialize the phishing scam generator."""
        
        super().__init__(fake, config)
        self.scam_type = 'phishing'

    def generate_url_link(self, subcategory_name: str, impersonated_entity: str) -> str:
        """Generate a fake URL for phishing scams."""

        bank = impersonated_entity.lower().replace(' ', '')
        return f"https://secure-{bank}-login.com/verify"


    def generate_scam_details(self) -> dict:
        """Generate details for a phishing scam."""
        
        subcategory_name, subcategory = self.select_subcategory(self.scam_type)
        general = subcategory['general_details']
        specific = subcategory['specific_details']

        approach_platform = np.random.choice(general['approach_platforms'], p=general['approach_platform_probs'])
        comm_platform = np.random.choice(
            general['communication_platforms'], p=general['communication_platform_probs']
        )
        transaction_type = np.random.choice(general['transaction_type'], p=general['transaction_type_probs'])
        amount_lost = round(np.random.uniform(*general['amount_loss_range']), 2)
        incident_date = datetime.strptime(
            np.random.choice(np.arange( 
                np.datetime64(self.config['general_configs']['incident_date_range'][0]),
                np.datetime64(self.config['general_configs']['incident_date_range'][1])
            ).astype(datetime).astype(str)), '%Y-%m-%d'
        )
        report_date = incident_date + timedelta(days=random.randint(*self.config['general_configs']['report_date_delay_days']))
        report_no = self.generate_report_no(report_date)
        beneficiary_platform, beneficiary_identifier = self.generate_beneficiary_details(transaction_type)

        impersonated_entity = random.choice(specific.get('first_impersonated_entity', ['NA']))
        impersonator_name = random.choice(specific.get('first_impersonated_entity_name', ['NA']))
        impersonation_type = specific.get('impersonation_type', "NA")
        moniker = "NA"
        scammer_contact_no = self.generate_phone_number()

        phished_details = np.random.choice(specific['phished_details'], p=specific['phished_details_probs'])
        use_of_phished_details = np.random.choice(specific['use_of_phished_details'], p=specific['use_of_phished_details_probs'])
        pretext_for_phishing = np.random.choice(specific['pretext_for_phishing'], p=specific['pretext_for_phishing_probs']) if 'pretext_for_phishing_probs' in specific else specific['pretext_for_phishing'][0]
        url_link = self.generate_url_link(subcategory_name, impersonated_entity)

        transactions, trans_summary = self.generate_transactions(
            incident_date, amount_lost, transaction_type, subcategory_name, beneficiary_platform, beneficiary_identifier
        )

        details = {
            'scam_report_no': report_no,
            'scam_incident_date': incident_date,
            'scam_report_date': report_date,
            'scam_type': self.scam_type,
            'scam_approach_platform': approach_platform,
            'scam_communication_platform': comm_platform,
            'scam_transaction_type': transaction_type,
            'scam_beneficiary_platform': beneficiary_platform,
            'scam_beneficiary_identifier': beneficiary_identifier,
            'scam_contact_no': scammer_contact_no,
            'scam_email': "NA",
            'scam_moniker': moniker,
            'scam_url_link': url_link,
            'scam_amount_lost': amount_lost,
            'scam_incident_description': "",
            'scam_subcategory': subcategory_name,
            'scam_impersonation_type': impersonation_type,
            'scam_first_impersonated_entity': impersonated_entity,
            'scam_first_impersonated_entity_name': impersonator_name,
            'scam_second_impersonated_entity': "NA",
            'scam_second_impersonated_entity_name': "NA",
            'scam_phished_details': phished_details,
            'scam_use_of_phished_details': use_of_phished_details,
            'scam_pretext_for_phishing': pretext_for_phishing,
            'scam_item_involved': "NA",
            'scam_item_type': "NA",
            'transactions': transactions
        }
        details['scam_incident_description'] = self.generate_description(details, subcategory, trans_summary)
        return details

    def generate_description(self, details: dict, subcategory: dict, trans_summary: str) -> str:
        """Generate a detailed incident description for a phishing scam."""
        
        pretext_map = {
            'unauthorized access to bank account': 'unauthorized access to my bank account',
            'outstanding bills': 'outstanding bills that needed to be paid',
            'payment failed': 'a failed payment attempt'
        }
        phished_details_map = {
            'card credentials': 'my card credentials',
            'banking credentials': 'my banking credentials'
        }
        use_map = {
            'log in from another device': 'log in to my account from another device',
            'increase transfer limit': 'increase my account transfer limit'
        }
        beneficiary_map = {
            'bank account': 'bank account'
        }

        return subcategory['description'].format(
            scammer_contact_no=details['scam_contact_no'],
            first_impersonated_entity_name=details['scam_first_impersonated_entity_name'],
            pretext_for_phishing=pretext_map.get(details['scam_pretext_for_phishing'], details['scam_pretext_for_phishing']),
            first_impersonated_entity=details['scam_first_impersonated_entity'],
            phished_details=phished_details_map.get(details['scam_phished_details'], details['scam_phished_details']),
            use_of_phished_details=use_map.get(details['scam_use_of_phished_details'], details['scam_use_of_phished_details']),
            beneficiary_transactions=beneficiary_map.get(details['scam_beneficiary_platform'], details['scam_beneficiary_platform']),
            approach_platform=details['scam_approach_platform'],
            moniker=details['scam_moniker'],
            url_link=details['scam_url_link'],
            transaction_summary=trans_summary
        )

class ScamDetailsGenerator:
    """Class to generate a consolidated dataframe consisting of different scam types. Proportion of each scam typology are configurable."""
    
    def __init__(self, config_path: str = 'config/scam_config.yaml'):
        """Initialize the scam details generator with configuration file."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.fake = Faker()
        self.settings = get_settings()
        self.default_output_file = self.settings.data.scam_report_csv

    def transform_string_value(self, value) -> str:
        """Capitalize and replace underscores with spaces for scam string values, excluding specific fields."""
        
        if isinstance(value, str) and value != "" and not value.startswith('+65') and not value.replace('.', '').isdigit() and not value.startswith('https://'):
            return value.replace('_', ' ').upper()
        return "NA" if pd.isna(value) else value

    def generate_scam_dataframe(self, total_records: int, proportions: dict, output_file: str = None) -> pd.DataFrame:
        """Generate a specified number of scam reports with given scam type proportions. Returns a Pandas dataframe that is converted into a csv file."""
        
        if not abs(sum(proportions.values()) - 1.0) < 1e-6:
            raise ValueError(f"Scam type proportions must sum to 1.0, got {sum(proportions.values())}")

        scam_type_counts = {
            scam_type: int(total_records * proportion)
            for scam_type, proportion in proportions.items()
        }
        total_assigned = sum(scam_type_counts.values())
        if total_assigned < total_records:
            max_proportion_scam = max(proportions, key=proportions.get)
            scam_type_counts[max_proportion_scam] += total_records - total_assigned
            
        reports = []
        
        for scam_type, count in scam_type_counts.items():
            if count == 0:
                continue
            generator = {
                'ecommerce': EcommerceScam,
                'government officials impersonation': GovernmentOfficialsImpersonationScam,
                'phishing': PhishingScam
            }.get(scam_type)
            
            if not generator:
                continue
            
            for _ in range(count):
                reports.append(generator(self.fake, self.config).generate_scam_details())
        
        for report in reports:
            trans_lines = [
                f"{trans_time.strftime('%Y-%m-%d')},{amount:.2f},{trans_type.replace('_', ' ').upper()},{platform.replace('_', ' ').upper()},{identifier}"
                for trans_time, amount, trans_type, platform, identifier in report.get('transactions', [])
            ]
            report['transactions'] = ";".join(trans_lines)
        
        df = pd.DataFrame(reports)
        
        required_columns = [
            'scam_report_no', 'scam_incident_date', 'scam_report_date', 'scam_type',
            'scam_approach_platform', 'scam_communication_platform', 'scam_transaction_type',
            'scam_beneficiary_platform', 'scam_beneficiary_identifier', 'scam_contact_no',
            'scam_email', 'scam_moniker', 'scam_url_link', 'scam_amount_lost',
            'scam_incident_description', 'scam_subcategory', 'scam_item_involved',
            'scam_item_type', 'scam_impersonation_type', 'scam_first_impersonated_entity',
            'scam_first_impersonated_entity_name', 'scam_second_impersonated_entity',
            'scam_second_impersonated_entity_name', 'scam_phished_details',
            'scam_use_of_phished_details', 'scam_pretext_for_phishing'
        ]
        
        case_sensitive_columns = [
            'scam_incident_description', 'scam_beneficiary_identifier', 'scam_contact_no',
            'scam_email', 'scam_moniker', 'scam_url_link', 'transactions'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = "NA"
        
        for col in required_columns:
            if col not in case_sensitive_columns:
                df[col] = df[col].apply(self.transform_string_value)
        
        df = df[required_columns]
        
        return df
    
    def save_to_csv(self, df: pd.DataFrame, output_file: str = None) -> None:
        """Save the scam DataFrame to a CSV file."""
        if output_file is None:
            output_file = self.default_output_file
        if not df.empty and output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            df.to_csv(output_file, index=False)
            print(f"Saved {len(df)} scam reports to {output_file}")

#Test code
if __name__ == "__main__":
    
    #Set seed
    random.seed(42)
    
    #Define number of scam reports
    total_records = 400
    proportions = {
        'ecommerce': 0.3,
        'phishing': 0.4,
        'government officials impersonation': 0.3
    }
    
    #Generate scam details for reports
    scam_generator = ScamDetailsGenerator()
    scam_df = scam_generator.generate_scam_dataframe(
        total_records,
        proportions,
        output_file=None
    )
    scam_generator.save_to_csv(scam_df)
    print(f"Total records generated: {len(scam_df)}")
    print("Records per scam type:")
    print(scam_df['scam_type'].value_counts())
    print("\nSample records:")
    print(scam_df.head())