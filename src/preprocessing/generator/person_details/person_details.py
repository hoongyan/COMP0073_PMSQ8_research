import sys
import os
import random
import pandas as pd
import yaml
from faker import Faker
from faker.providers import BaseProvider
from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from config.settings import get_settings

class CustomSingaporeProvider(BaseProvider):
    def __init__(self, generator, config):
        super().__init__(generator)
        self.config = config

    def _weighted_choice(self, items):
        """Select an item based on weighted probabilities."""
        choices = [item for item in items]
        weights = [item['probability'] for item in items]
        return random.choices(choices, weights=weights, k=1)[0]

    def nationality(self):
        """Generate nationality based on weights."""
        weights = self.config['demographic_config']['nationality_weights']
        return random.choices(
            list(weights.keys()),
            weights=list(weights.values()),
            k=1
        )[0]

    def race(self, nationality):
        """Generate race based on nationality and weights."""
        race_weights = self.config['demographic_config']['race_weights'][nationality]
        return self._weighted_choice(race_weights)['race']

    def sex(self):
        """Generate sex based on weights."""
        weights = self.config['demographic_config']['sex_weights']
        return random.choices(
            list(weights.keys()),
            weights=list(weights.values()),
            k=1
        )[0]

    def first_name(self, race, sex):
        """Generate first name based on race and sex."""
        names = self.config['race_name_lists'].get(race, self.config['race_name_lists']['Other'])
        name_list = names['male_first_names'] if sex == 'Male' else names['female_first_names']
        # Generate double first names for Chinese race
        if race == 'Chinese':
            first_name1 = random.choice(name_list)
            first_name2 = random.choice(name_list)
            return f"{first_name1} {first_name2}"
        return random.choice(name_list)

    def last_name(self, race):
        """Generate last name based on race."""
        names = self.config['race_name_lists'].get(race, self.config['race_name_lists']['Other'])
        return random.choice(names['last_names'])

    def dob(self):
        """Generate date of birth based on age ranges."""
        age_range = self._weighted_choice(self.config['demographic_config']['age_ranges'])
        min_age, max_age = age_range['min_age'], age_range['max_age']
        current_year = datetime.now().year
        birth_year = random.randint(current_year - max_age, current_year - min_age)
        birth_date = datetime(birth_year, 1, 1) + timedelta(days=random.randint(0, 364))
        return birth_date.strftime('%Y-%m-%d')

    def contact_no(self):
        """Generate Singapore phone number."""
        return f"{self.config['contact_no_prefix']}{random.randint(90000000, 99999999)}"

    def blk(self):
        """Generate random 2 or 3-digit block number."""
        return str(random.randint(10, 999))

    def street(self):
        """Generate street name."""
        return random.choice(self.config['address_components']['streets'])

    def unit_no(self):
        """Generate random unit number in format #FF-UU, floor < 40."""
        floor = random.randint(1, 39)
        unit = random.randint(1, 99)
        return f"{floor:02d}-{unit:02d}"

    def postcode(self):
        """Generate random 6-digit postal code."""
        return str(random.randint(100000, 999999))

    def occupation(self):
        """Generate occupation based on weights."""
        return self._weighted_choice(self.config['demographic_config']['occupations'])['occupation']

    def email(self, first_name, last_name):
        """Generate email address based on first and last name with variations."""
        # Clean names for email (remove spaces, lowercase)
        clean_first = first_name.replace(' ', '').lower()
        clean_last = last_name.replace(' ', '').lower()
        # For firstinitial.lastname, use initials of both first names for Chinese
        if ' ' in first_name:
            initials = ''.join(name[0] for name in first_name.split()).lower()
            first_initial = f"{initials}"
        else:
            first_initial = clean_first[0]
        # Randomly choose email format
        formats = [
            f"{clean_first}.{clean_last}",  # firstname.lastname
            f"{first_initial}.{clean_last}",  # firstinitial.lastname
            f"{clean_first}_{clean_last}"  # firstname_lastname
        ]
        username = random.choice(formats)
        domain = random.choice(self.config['email_domains'])
        return f"{username}@{domain}"

    def nric(self, nationality, dob):
        """Generate synthetic NRIC number (e.g., S5812345A) based on nationality and DOB."""
        birth_year = int(dob[:4])
        year_digits = str(birth_year)[-2:]  # Last two digits of birth year
        if nationality == 'Singaporean':
            prefix = 'S' if birth_year < 2000 else 'T'
        else:
            prefix = 'G'
        digits = ''.join(str(random.randint(0, 9)) for _ in range(5))  # 5 random digits
        suffix = chr(random.randint(65, 90))  # Random letter A-Z
        return f"{prefix}{year_digits}{digits}{suffix}"

    def generate_person(self):
        """Generate a single person profile."""
        nat = self.nationality()
        race = self.race(nat)
        sex = self.sex()
        first_name = self.first_name(race, sex)
        last_name = self.last_name(race)
        dob = self.dob()
        return {
            'first_name': first_name,
            'last_name': last_name,
            'sex': sex,
            'dob': dob,
            'contact_no': self.contact_no(),
            'blk': self.blk(),
            'street': self.street(),
            'unit_no': self.unit_no(),
            'postcode': self.postcode(),
            'occupation': self.occupation(),
            'nationality': nat,
            'race': race,
            'email': self.email(first_name, last_name),
            'nric': self.nric(nat, dob)
        }

class PersonProfileGenerator:
    """A class to generate person profiles using Faker."""
    
    def __init__(self, config_path='config/person_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.fake = Faker()
        self.fake.add_provider(CustomSingaporeProvider(self.fake, config=self.config))
        self.settings = get_settings()
        self.default_output_file = self.settings.data.person_details_csv  

    def transform_string(self, value) -> str:
        """Capitalize string values or return NA for empty fields."""
        if isinstance(value, str) and value != "":
            return value.upper()
        return "NA" if pd.isna(value) else value
    
    def generate_profiles_dataframe(self, total_records: int, output_file: str = None) -> pd.DataFrame:
        """Generate person profiles and return as a Pandas DataFrame with transformed string values."""
        profiles = []
        for _ in range(total_records):
            profile = self.fake.generate_person()
            profiles.append(profile)

        df = pd.DataFrame(profiles)
        columns = [
            'first_name', 'last_name', 'nric', 'sex',
            'dob', 'nationality', 'race', 'occupation', 'contact_no',
            'email', 'blk', 'street', 'unit_no', 'postcode'
        ]
        df = df[columns]

        case_sensitive_columns = ['dob', 'contact_no', 'email']
    
        for col in columns:
            if col not in case_sensitive_columns:
                df[col] = df[col].apply(self.transform_string)

        return df

    def save_to_csv(self, df: pd.DataFrame, output_file: str = None) -> None:
        """Save the person DataFrame to a CSV file."""
        if output_file is None:
            output_file = self.default_output_file
        if not df.empty and output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            df.to_csv(output_file, index=False)
            print(f"Saved {len(df)} person profiles to {output_file}")
            
if __name__ == "__main__":
    
    random.seed(42)  # Set seed
        
    # Define number of profiles
    total_records = 400
    
    # Generate person profiles
    person_generator = PersonProfileGenerator()
    person_df = person_generator.generate_profiles_dataframe(
        total_records,
        output_file=None
    )
    person_generator.save_to_csv(person_df)
    print(f"Total records generated: {len(person_df)}")
    print("\nSample records:")
    print(person_df.head())