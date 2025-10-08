import pandas as pd
import plotly.express as px
import re

# https://en.wikipedia.org/wiki/List_of_wolf_attacks
# https://www.kaggle.com/datasets/danela/global-wolf-attacks?resource=download

wolf_attacks = pd.read_csv("global_wolves.csv")

def count_victims(text):
    text_lower = text.lower()
    word_to_num = { 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 
                    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 
                    'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15 }

    male_count = 0
    female_count = 0
    unknown_count = 0

    # Males
    for synonym in ['male', 'man', 'men', 'boy', 'boys']:
        digit_match = re.search(rf'(\d+)\s+{synonym}s\b', text_lower)  # Must be plural for digit counts
        if digit_match: 
            male_count += int(digit_match.group(1))
            break
        if re.search(rf'\b(a|an|adult)\s+{synonym}\b', text_lower): male_count += 1
        for word, num in word_to_num.items():
            if re.search(rf'\b{word}\s+{synonym}s?', text_lower): male_count += num; break

    # Females
    for synonym in ['female', 'woman', 'women', 'girl', 'girls']:
        digit_match = re.search(rf'(\d+)\s+{synonym}s\b', text_lower)  # Must be plural for digit counts
        if digit_match: 
            female_count += int(digit_match.group(1))
            break
        if re.search(rf'(?:^|[,\s])(a|an)\s+{synonym}\b', text_lower): female_count += 1
        if re.search(rf'\band\s+a\s+{synonym}\b', text_lower): female_count += 1
        if re.search(rf'\badult\s+{synonym}\b', text_lower): female_count += 1
        for word, num in word_to_num.items():
            if re.search(rf'\b{word}\s+{synonym}s?', text_lower): female_count += num; break

    # Edge cases where people are separated by commas
    comma_males = len(re.findall(r',\s*\d*\s*male(?!s)', text_lower))
    comma_females = len(re.findall(r',\s*\d*\s*female(?!s)', text_lower))

    # Edge cases where people are separated by spaces
    space_males = len(re.findall(r'male[A-Z]', text))
    space_females = len(re.findall(r'female[A-Z]', text))

    # Pick the larger of the two counts if we found any
    if male_count == 0: 
        male_count = max(comma_males, space_males)
    if female_count == 0: 
        female_count = max(comma_females, space_females)

    # Edge cases where "and adult males/females" implies at least 2
    if male_count == 0 and re.search(r'and\s+\w+[^,]*,?\s*(?:adult\s+)?males\b', text_lower): male_count = 2
    if female_count == 0 and re.search(r'and\s+\w+[^,]*,?\s*(?:adult\s+)?females\b', text_lower): female_count = 2

    # Edge cases where "his wife" or "her husband" implies at least 1
    if 'his wife' in text_lower: male_count = max(male_count, 1); female_count = max(female_count, 1)
    if 'her husband' in text_lower: male_count = max(male_count, 1); female_count = max(female_count, 1)
    
    # Count unknown victims if and only if no gender info found
    if male_count == 0 and female_count == 0:
        others_match = re.search(r'and\s+(\w+)\s+others?', text_lower) # Special case: "Name and X others" means X+1 total unknown
        if others_match:
            word = others_match.group(1)
            if word.isdigit():
                unknown_count = int(word) + 1
            elif word in word_to_num:
                unknown_count = word_to_num[word] + 1
        elif re.search(r'\b(one|a|an)\s+(resident|person|individual)\b', text_lower):
            unknown_count = 1
        else:
            adults_match = re.search(r'(\w+)\s+adults?\s+and\s+(\w+)\s+children', text_lower) # Check for "X adults and Y children" patterns
            if adults_match:
                word1, word2 = adults_match.groups()
                count1 = int(word1) if word1.isdigit() else word_to_num.get(word1, 0)
                count2 = int(word2) if word2.isdigit() else word_to_num.get(word2, 0)
                unknown_count = count1 + count2
            else:
                for keyword in ['people', 'residents', 'victims', 'persons', 'individuals', 'children']: # Check for "two/three/etc. people/residents/etc."
                    digit_match = re.search(rf'(\d+)\s+{keyword}', text_lower)
                    if digit_match: unknown_count = int(digit_match.group(1)); break
                    for word, num in word_to_num.items():
                        if re.search(rf'\b{word}\s+{keyword}', text_lower): unknown_count = num; break
                    if unknown_count > 0: break
    
    return pd.Series({'Male_Count': male_count, 'Female_Count': female_count, 'Unknown_Count': unknown_count})

def get_month(text):
    # Return just month as plaintext from "Date" column
    # If there is no month, return None
    
    # In cases where there are Dates like "June 1764 – June 1767", return None since they are ranges and not reliable
    if '–' in text or '-' in text or 'and' in text: return None
    month = re.search(r'\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b', text, re.IGNORECASE)
    return month.group(0) if month else None

def apply_functions_to_data_frame(wolf_attacks):
    # Apply data wrangling functions
    wolf_attacks[['Male_Count', 'Female_Count', 'Unknown_Count']] = wolf_attacks['Victims'].apply(count_victims)
    wolf_attacks['Month'] = wolf_attacks['Date'].apply(get_month)


def plot_attacks_by_month(wolf_attacks):
    # Plot all attacks by month
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    wolf_attacks['Month'] = pd.Categorical(wolf_attacks['Month'], categories=month_order, ordered=True)
    month_counts = wolf_attacks['Month'].value_counts().reindex(month_order).fillna(0).reset_index()
    month_counts.columns = ['Month', 'Attack_Count']

    fig = px.line(month_counts, x='Month', y='Attack_Count', title='Number of All Wolf Attacks by Month')
    fig.update_layout(xaxis_title='Month', yaxis_title='Number of Attacks')
    fig.show()

def plot_female_attacks_by_month(wolf_attacks):
    # Plot female attacks by month
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    wolf_attacks['Month'] = pd.Categorical(wolf_attacks['Month'], categories=month_order, ordered=True)
    month_counts = wolf_attacks[wolf_attacks['Female_Count'] > 0]['Month'].value_counts().reindex(month_order).fillna(0).reset_index()
    month_counts.columns = ['Month', 'Attack_Count']

    fig = px.line(month_counts, x='Month', y='Attack_Count', title='Number of Female Wolf Attacks by Month')
    fig.update_layout(xaxis_title='Month', yaxis_title='Number of Attacks')
    fig.show()

def plot_male_attacks_by_month(wolf_attacks):
    # Plot male attacks by month
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    wolf_attacks['Month'] = pd.Categorical(wolf_attacks['Month'], categories=month_order, ordered=True)
    month_counts = wolf_attacks[wolf_attacks['Male_Count'] > 0]['Month'].value_counts().reindex(month_order).fillna(0).reset_index()
    month_counts.columns = ['Month', 'Attack_Count']

    fig = px.line(month_counts, x='Month', y='Attack_Count', title='Number of Male Wolf Attacks by Month')
    fig.update_layout(xaxis_title='Month', yaxis_title='Number of Attacks')
    fig.show()

def plot_males_and_females_by_month(wolf_attacks):
    # Plot male and female attacks by month with bars grouped side by side
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    wolf_attacks['Month'] = pd.Categorical(wolf_attacks['Month'], categories=month_order, ordered=True)
    
    month_counts_female = wolf_attacks[wolf_attacks['Female_Count'] > 0]['Month'].value_counts().reindex(month_order).fillna(0).reset_index()
    month_counts_female.columns = ['Month', 'Attack_Count']
    month_counts_female['Gender'] = 'Female'

    month_counts_male = wolf_attacks[wolf_attacks['Male_Count'] > 0]['Month'].value_counts().reindex(month_order).fillna(0).reset_index()
    month_counts_male.columns = ['Month', 'Attack_Count']
    month_counts_male['Gender'] = 'Male'

    combined_data = pd.concat([month_counts_male, month_counts_female])

    fig = px.bar(combined_data,
                x='Month',
                y='Attack_Count',
                color='Gender',
                title='Wolf Attacks by Month and Gender',
                color_discrete_map={'Male': '#1e78b4', 'Female': '#ff7f0f'},
                barmode='stack' )
    
    fig.update_traces(texttemplate='%{y}', textposition='outside', textfont_color='black') # Black text above bars
    fig.update_layout(xaxis_title='Month', yaxis_title='Number of Attacks', barmode='group')
    fig.show()

# TODO: Another graphic for all genders (including unknown)

apply_functions_to_data_frame(wolf_attacks)
# print(wolf_attacks.iloc[280:290])

# plot_attacks_by_month(wolf_attacks) # Show all (non-gendered) attacks by month
# plot_female_attacks_by_month(wolf_attacks) # Show only female attacks by month
# plot_male_attacks_by_month(wolf_attacks) # Show only male attacks by month
plot_males_and_females_by_month(wolf_attacks) # Show both male and female attacks by month
# TODO: Another graphic for all genders (including unknown)