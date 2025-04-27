import pandas as pd

# Load your original CSV
df = pd.read_csv("athlete_performance_large.csv")

# Remove 'Cycling' and 'Swimming'
df_filtered = df[~df['Sport_Type'].isin(['Cycling', 'Swimming'])]

# New sports
new_sports = ['Tennis', 'Cricket', 'Hockey', 'Boxing', 'Volleyball', 'Baseball', 'Wrestling']
entries_per_sport = 3943 // len(new_sports)
remaining_entries = 3943 % len(new_sports)

# Use original rows as templates
template_rows = df[df['Sport_Type'].isin(['Cycling', 'Swimming'])].sample(n=3943, random_state=42).reset_index(drop=True)

# Assign new sports
new_sport_list = []
for sport in new_sports:
    count = entries_per_sport + (1 if remaining_entries > 0 else 0)
    new_sport_list.extend([sport] * count)
    remaining_entries -= 1

template_rows['Sport_Type'] = new_sport_list

# Combine all
df_updated = pd.concat([df_filtered, template_rows], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

# Save the updated file
df_updated.to_csv("athlete_performance_large.csv", index=False)