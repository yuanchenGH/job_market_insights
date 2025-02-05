import pandas as pd

def refine(df):
    df1 = df.dropna(subset=['description'])
    print(f'droped empty rows, remaining {len(df1)}')
    df_sorted = df1.sort_values(by='posted_date', ascending=False)
    df2 = df_sorted.drop_duplicates(subset=['job_title', 'job_location', 'company_name', 'description'], keep='first').reset_index(drop=True)
    print(f'droped duplicates, remaining {len(df2)}')
    return df2

def reduce_rows(df):
    # list of companies that only collect information, post fake posts, are offering coaching service instead of real jobs, and job board
    # companyName_to_drop = ['SynergisticIT', 'Outlier', 'Credible', 'HireMeFast LLC', 'Phoenix Recruitment', 'Jobs via Dice', 'Underdog.io', 'TekJobs', 'Henry Hire Solutions', 'Lifelancer', 'Global Technical Talent, an Inc. 5000 Company', 'HHS Careers', 'Jerry', 'Talentify.io', 'TELUS Digital AI Data Solutions']
    companyName_to_drop = ['SynergisticIT', 'Outlier']
    # requires US citizenship and security clearance
    cns = ['U.S. Citizenship is required', 'US Citizenship is required', 'security clearance', 'require US Citizenship', 'require U.S. Citizenship']
    title_to_keep = ['Intern', 'Data Analyst', 'Data Scientist', 'Business Intelligence']

    # drop not legit companies
    df1 = df[~df['company_name'].isin(companyName_to_drop)]
    # remove rows contains citizenship and security clearance requirement
    pattern = '|'.join(rf'\b{word}\b' for word in cns)
    df1 = df1[~df1['description'].str.contains(pattern, na=False)] 
    # keep legit job titles
    df1 = df1[df1['job_title'].str.contains('|'.join(title_to_keep), case=False, na=False)]
    # remove duplicates by company name and title, keep the newest post
    # df1 = df1.sort_values(by='posted_date', ascending=False)
    # df1 = df1.drop_duplicates(subset=['companyName', 'title'], keep='first')
    df1 = df1.reset_index(drop=True)
    return df1

def remove_companies(df, path):
    df_companies = pd.read_csv(f'{path}companies_info.csv')
    df1 = df.copy()
    df_merged = pd.merge(df1, df_companies, how='left', on='company_name')
    mask = (df_merged[['Members', 'Posts']] < 2).any(axis=1)
    print('Company dropped: ', df_merged[mask]['company_name'])
    df2 = df_merged[~mask]
    return df2
