import os
import pandas as pd
import time
import json

# AI
import ast
from openai import OpenAI
from dotenv import load_dotenv

# Natural Language Processing
import re
import nltk
from rapidfuzz import fuzz
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')

# remove empty and duplicated data
def refine(df):
    cols_to_keep = ['id', 'publishedAt', 'title', 'location', 'salary', 'applyUrl', 'jobUrl',  'contractType', 'sector', 'workType', 'description', 'benefits', 'companyId', 'companyName', 'companyUrl', 'experienceLevel', 'posterFullName', 'posterProfileUrl']
    # cols_to_keep = ['id', 'publishedAt', 'title', 'location', 'salary', 'applyUrl', 'jobUrl',  'contractType', 'sector', 'workType', 'description', 'workplace', 'benefits', 'companyId', 'companyName', 'companyUrl', 'experienceLevel', 'posterFullName', 'posterProfileUrl']
    df1 = df[cols_to_keep].copy()
    print(f'Processing...length of data {len(df1)}')
    df1 = df1.dropna(subset=['title', 'companyName', 'applyUrl'])
    print(f'droped empty rows, remaining {len(df1)}')
    df1 = df1.drop_duplicates(subset=['applyUrl'])
    df_sorted = df1.sort_values(by='publishedAt', ascending=False)
    df1 = df_sorted.drop_duplicates(subset=['title', 'companyName', 'location'], keep='first').reset_index(drop=True)
    print(f'droped duplicates, remaining {len(df1)}')
    return df1

# remove all job with non-English title
def remove_non_english_jobs(df):
    # Return True if the text contains non-English characters, False if all English
    def contains_non_english(text):
        return bool(re.search(r'[^\x00-\x7F]+', text))  # Matches any non-ASCII characters
    # this function remove non-printable characters and replace non-ASC II characters to ASC II
    def replace_non_ascii(s):
        # remove non-printable characters
        s = ''.join(ch for ch in s if ch.isprintable())
        # Replace some special characters to space
        s = s.replace("–", " ").replace("&", "and").replace('/', ' ')
        return s
    
    df1 = df.copy()
    print(df1.columns)
    df1['title'] = df1['title'].apply(replace_non_ascii)
    return df1[~df1['title'].apply(contains_non_english)].reset_index(drop=True)

# this function standardize the location names to city, state
def standard_location(df):
    us_state_to_abbreviation = {
        "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
        "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
        "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
        "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
        "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI",
        "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
        "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
        "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND",
        "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
        "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
        "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
        "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI",
        "Wyoming": "WY"
    }

    def is_city_state_format(location):
        return bool(re.match(r'^.+,\s*[A-Z]{2}$', location, re.IGNORECASE))

    def is_state_us_format(location):
        return bool(re.match(r'^[A-Za-z\s]+,\s*United States$', location, re.IGNORECASE))

    def is_metro_or_greater_format(location):
        return bool(re.search(r'Metro|^Greater\s+', location, re.IGNORECASE))

    def is_DC(location):
        pattern = r'.*\b(DC|District of Columbia)\b.*'
        return bool(re.search(pattern, location, re.IGNORECASE))
    
    def process_row(row):
        # Check if the location is already in city, state format
        if is_city_state_format(row['location']):
            row['location_st'] = row['location']
            row['state'] = row['location'].split(', ')[1]
            return row

        # Check if the location is in "state, United States" format
        if is_state_us_format(row['location']):
            state_name = row['location'].split(",")[0].strip()
            row['state'] = us_state_to_abbreviation.get(state_name, None)
            row['location_st'] = None
            return row

        # Check if the location contains "Metro" or starts with "Greater"
        if is_metro_or_greater_format(row['location']):
            row['location_st'] = metro_map.get(row['location'], None)
            row['state'] = row['location_st'].split(', ')[1] if row['location_st'] else None
            return row

        # Check if the location is related to DC
        if is_DC(row['location']):
            row['location_st'] = 'Washington DC'
            row['state'] = 'Washington DC'
            return row

        # Default case: leave the row unchanged
        return row
    
    # Get metro-related locations
    metro = df[df['location'].apply(is_metro_or_greater_format)]
    location_prompt = 'output a python dictionary to map the locations string to a string formatted as "city, state(2 letter)": '
    metro_map = transform(metro['location'], location_prompt, 'location')
    
    # Apply the function row-wise to the DataFrame
    df = df.apply(process_row, axis=1)
    return df

load_dotenv('keys.env')
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

def transform(text, prompt, id):
    # Prepare the API prompt
    prompt = f'{prompt}{text}'
    retry = 0

    while retry <= 10:
        print(f"Start API call at ID: {id}, Retry: {retry}")
        try:
            # Call the API
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                timeout=60
            )
            content = str(response.choices[0].message.content)

            # Replace 'null' with Python None
            content = content.replace("null", "None")

            # Extract JSON-like content between `{` and `}`
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1:
                content = content[start:end+1]
                print(f"Retrieved dictionary at ID: {id}")
                print(content)
                # Safely parse JSON or Python-like dictionary
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    print(f"Invalid JSON content at ID: {id}, attempting eval...")
                    return ast.literal_eval(content)

            else:
                print(f"No valid JSON-like content found in response at ID: {id}")

        except Exception as e:
            print(f"Error during API call or processing at ID: {id}: {e}")
        
        retry += 1
        print(f"Retrying {retry}... Waiting before next attempt.")
        time.sleep(1.5)

    print(f"Failed to retrieve dictionary after maximum retries for ID: {id}.")
    return None

import re
import pandas as pd

def process_title(row):
    """
    Processes a job title by:
      1. Removing periods.
      2. Removing any text after a hyphen.
      3. Extracting keywords (jr, junior, sr, senior, lead, principal)
         into a separate column.
      4. Removing any text within parentheses.
      5. Removing the extracted keywords from the title.
      6. Replacing abbreviations with their full forms.
      7. Ignoring case during processing but returning final results in title case.
    
    Returns:
        cleaned_title: The cleaned job title (title-cased).
        extracted_info: The extracted keywords (title-cased) as a comma-separated string, or None.
    """
    title = row['job_title']
    if not isinstance(title, str):
        row['job_level'] = None
        return row

    # Abbreviations mapping (keys and values are in proper title format)
    abbreviations = {
        'CEO': 'Chief Executive Officer',
        'CFO': 'Chief Financial Officer',
        'COO': 'Chief Operating Officer',
        'CRO': 'Chief Revenue Officer',
        'CTO': 'Chief Technology Officer',
        'VP': 'Vice President',
        'Jr': 'Junior',
        'Sr': 'Senior'
    }

    # --- Step 1: Remove periods.
    title = title.replace('.', '')

    # --- Step 2: Remove any text after a hyphen.
    if '-' in title:
        title = title.split('-')[0].strip()

    # --- Store a copy for keyword extraction (before removing parentheses)
    title_for_extraction = title

    # --- Step 3: Extract keywords (ignore case)
    pattern_extract = r'\b(?:jr|junior|sr|senior|lead|principal)\b'
    matches = re.findall(pattern_extract, title_for_extraction, flags=re.IGNORECASE)

    # --- Step 4: Remove any text within parentheses (including the parentheses)
    title = re.sub(r'\([^)]*\)', '', title).strip()

    # --- Step 5: Remove the extracted keywords from the title.
    cleaned_title = re.sub(pattern_extract, '', title, flags=re.IGNORECASE).strip()
    cleaned_title = re.sub(r'\s+', ' ', cleaned_title)  # normalize extra spaces

    # --- Step 6: Replace abbreviations in the cleaned_title.
    for abbr, full in abbreviations.items():
        # Replace whole word matches using case-insensitive flag.
        pattern = r'\b' + re.escape(abbr) + r'\b'
        cleaned_title = re.sub(pattern, full, cleaned_title, flags=re.IGNORECASE)

    # --- Process the extracted keywords: replace abbreviations as needed.
    def replace_keyword(keyword, abbrev_dict):
        # Find matching abbreviation ignoring case, and return its full form if available.
        for key, value in abbrev_dict.items():
            if keyword.lower() == key.lower():
                return value
        # If not found, just return the keyword in title case.
        return keyword.title()

    extracted_info_list = [replace_keyword(m, abbreviations) for m in matches]
    extracted_info = ', '.join(extracted_info_list) if extracted_info_list else None

    # --- Step 7: Capitalize (title-case) the final outputs.
    cleaned_title = cleaned_title.title()
    row['job_title_clean'] = cleaned_title
    if extracted_info:
        # Each component is already replaced with the full form (which is expected to be properly capitalized).
        row['job_title_level'] = extracted_info.title()

    return row


def simplify_titles_finance(row):
    simple_titles = ['assistant','advisor','analyst','accountant','associate','bookkeeper','banker','consultant','counselor','controller','chair','chief of staff','chief executive officer','chief financial officer','chief operating officer','chief revenue officer','chief technology officer','director','head','intern','manager','officer','professor','predsident','specialist','teller','trader','underwriter','vice president']
    seniority = ['junior','senior','lead','principal']
    abbreviations = {
        'CEO': 'Chief Executive Officer',
        'CFO': 'Chief Financial Officer',
        'COO': 'Chief Operating Officer',
        'CRO': 'Chief Revenue Officer',
        'CTO': 'Chief Technology Officer',
        'VP': 'Vice President',
        'Jr': 'Junior',
        'Sr': 'Senior'
    }

    titles_pattern = r'\b(?:' + r'|'.join(r' '.join(part for part in title.split()) for title in simple_titles) + r')\b'
    seniority_pattern = r'\b(' + '|'.join(map(re.escape, seniority)) + r')\b'
    abbrev_pattern = r'\b(' + '|'.join(map(re.escape, abbreviations.keys())) + r')\b'

    title_match = re.search(titles_pattern, row['title'], re.IGNORECASE)
    seniority_match = re.search(seniority_pattern, row['title'], re.IGNORECASE)
    abbrev_match = re.search(abbrev_pattern, row['title'], re.IGNORECASE)
    # function_match = 

    if abbrev_match:
        abbrev = abbrev_match.group(1).upper()  # Match abbreviation and normalize to uppercase
        row['title_simple'] = abbreviations[abbrev]
        if seniority_match:
            matched_seniority = seniority_match.group(1).title()
            row['title_simple'] = f"{row['title_simple']}, {matched_seniority}".strip()
    elif title_match:
        matched_title = title_match.group(0).title()  # Matched title from the titles list
        row['title_simple'] = matched_title.strip()
        if seniority_match:    
            matched_seniority = seniority_match.group(1).title()
            row['title_simple'] = f"{row['title_simple']}, {matched_seniority}".strip()
    else:
        row['title_simple'] = None
    # if function_match:
    #     matched_function = function_match.group(1).capitalize()
    #     row['title_simple'] = row['title_simple'] + f', {matched_function}'

    return row

def simplify_titles_data(row):
    simple_titles = ['Intern', 'Data Analyst', 'Data Scientist', 'Business Intelligence']
    for title in simple_titles:
        if title.lower() in row['title'].lower():
            row['title_simple'] = title
            return row
    row['title_simple'] = None
    return row

def parse_description(df):
    prompt = """Extract information from the following text. Output the result strictly in the  format of a Python dictionary with the following keys:

    1. "min_years_of_experience" (integer): The minimum years of experience required. If not mentioned, the value should be None.
    2. "min_yearly_salary" (integer): The minimum yearly salary. If not mentioned, the value should be None.
    3. "max_yearly_salary" (integer): The maximum yearly salary. If not mentioned, the value should be None.
    4. "degree" (string): The minimum degree required, strictly as abbreviations: BS, MS, or PHD. If no degree is mentioned, the value should be None.
    5. "skills" (list): A list of required or preferred skills. Each skill should be a single keyword or at most two keywords (e.g., "Python", "data analysis"). If no skills are mentioned, the value should be None.

    Use the exact keys as provided. If no information is found for a specific key, set its value to None. The input text is: """

    df = df.reset_index(drop=True)
    df['ai_dict'] = df.apply(
        lambda row: transform(row['description'], prompt, row['id']), axis=1
        )

    normalized_dict = pd.json_normalize(df['ai_dict'])

    # Replace 0 with None in the specified columns
    normalized_dict[['min_yearly_salary', 'max_yearly_salary']] = \
    normalized_dict[['min_yearly_salary', 'max_yearly_salary']].replace(0, None)

    df = pd.concat([df, normalized_dict], axis=1)

    return df

def parse_salary(row):
    # Replace zeros with None
    row = row.replace(0, None)

    # Helper function to extract numeric values from salary strings
    def extract_salary_value(salary_str):
        if isinstance(salary_str, str):
            numbers = re.findall(r'\d{1,3}(?:,\d{3})*(?:\.\d+)?', salary_str)
            return [float(num.replace(',', '')) for num in numbers]
        return []

    # Extract salary values
    salary_values = extract_salary_value(row['salary']) if 'salary' in row else []

    # Process based on salary type
    if salary_values:
        if 'hr' in row['salary'] or 'hour' in row['salary']:
            row['min_yearly_salary'] = min(salary_values) * 2000
            row['max_yearly_salary'] = max(salary_values) * 2000
        elif 'mo' in row['salary'] or 'month' in row['salary']:
            row['min_yearly_salary'] = min(salary_values) * 12
            row['max_yearly_salary'] = max(salary_values) * 12
        elif 'yr' in row['salary'] or 'year' in row['salary']:
            row['min_yearly_salary'] = min(salary_values)
            row['max_yearly_salary'] = max(salary_values)

    # Calculate average salary
    min_salary = row.get('min_yearly_salary')
    max_salary = row.get('max_yearly_salary')

    if pd.notna(min_salary) and pd.notna(max_salary):
        row['avg_salary'] = (min_salary + max_salary) / 2
    elif pd.isna(min_salary) and pd.notna(max_salary):
        row['avg_salary'] = max_salary
    elif pd.notna(min_salary) and pd.isna(max_salary):
        row['avg_salary'] = min_salary
    else:
        row['avg_salary'] = None

    return row
  
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def standard_skills(df):
    df['skills'] = df['skills'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['skills_clean'] = df['skills'].apply(lambda x: clean_skills(x) if isinstance(x, list) else x)
    return df

def clean_skills(skill_list):
    # Words to remove and standard replacements
    words_to_remove = {'skill', 'skills', 'in', 'proficient', 'proficiency', 'suite', 'advanced','ability', 'oral', 'written','effective','us'}
    replacements = {
        r'\bms\b': 'microsoft',
        r'\b(?:ms\s+)?excel\b': 'microsoft excel',
        r'\b(?:ms\s+)?word\b': 'microsoft word',
        r'\b(?:ms\s+)?outlook\b': 'microsoft outlook',
        r'(?<!\bmarket\s)\b(?:ms\s+)?access\b': 'microsoft access',
        r'\b(?:ms\s+)?office\b': 'microsoft office',
        r'\b(?:ms\s+)?power\s?point\b': 'microsoft powerpoint',
        r'\b(?:ms\s+)?power\s?bi\b': 'microsoft powerbi',
        r'\b(?:ms\s+)?power\s?business\s?intelligence\b': 'microsoft powerbi',
        r'.*\bcollaboration\b.*': 'collaboration',
        r'.*\bteamwork\b.*': 'teamwork',
        r'.*\baudit\b.*': 'auditing',
        r'.*\bpayroll\b.*': 'payroll',
        r'.*\bleadership\b.*': 'leadership',
        r'.*\berp\b.*': 'erp',
        
        # r'\bproblemsolving\b': 'problem solving',
    }
    
    def preprocess(skill):
        # Lowercase, remove non-alphanumeric chars, and split into words
        skill = re.sub(r'[^a-z\s]', ' ', skill.lower())
        words = skill.split()

        words = [
        word if word == 'ms' else lemmatizer.lemmatize(word) # lemmatizer may convert "ms" to "m"
        for word in words
        if word not in stop_words and word not in words_to_remove
        ]

        # Normalize using replacements
        skill = ' '.join(words)
        for pattern, replacement in replacements.items():
            skill = re.sub(pattern, replacement, skill, flags=re.IGNORECASE)
        
        # Remove duplicate consecutive words
        skill_clean = ' '.join(w.capitalize() for i, w in enumerate(skill.split()) if i == 0 or w != skill.split()[i-1])
        if len(skill_clean) <= 4:
            skill_clean = skill_clean.upper()
        return skill_clean

    # Apply preprocessing to each skill
    return [preprocess(skill) for skill in skill_list if skill.strip()]

# Function to find the best match and group similar names using fuzzy match
def skills_fuzzymatch(skill_list, threshold=80):
    grouped_skills = []  # List to store the groups of similar skills

    for skill in skill_list:
        # Find if the skill is already grouped
        matched = False
        for group in grouped_skills:
            # Compare with the first skill in each group
            if fuzz.ratio(skill, group[0]) >= threshold:
                group.append(skill)
                matched = True
                break
        
        # If no match found, create a new group
        if not matched:
            grouped_skills.append([skill])

    return grouped_skills

# assign the grouped skills back to a new column in original DataFrame
def assign_group(skill, grouped_skills):
    for group in grouped_skills:
        if skill in group:
            return most_frequent_word(group)  # Return the first skill in the group as the representative
    return skill  # If no group, return the skill itself

def most_frequent_word(text_list):
    # Count the frequency of each word
    word_counts = Counter(text_list)
    # Find the most common word
    if word_counts:
        return word_counts.most_common(1)[0][0]
    return None