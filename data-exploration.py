import os
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from tqdm.auto import tqdm


########################################
# DATA LOADING & INSPECTION
########################################
# Example file loading from CADEC corpus
print("Example: Original annotations")
print(pd.read_csv("./data/CADEC.v2/cadec/original/ARTHROTEC.1.ann", sep='\t', header=None))

print("Example: MedDRA annotations")
print(pd.read_csv("./data/CADEC.v2/cadec/meddra/ARTHROTEC.1.ann", sep='\t', header=None))

print("Example: SCT annotations")
print(pd.read_csv("./data/CADEC.v2/cadec/sct/ARTHROTEC.1.ann", sep='\t', header=None))

########################################
# UNIQUE DRUG NAMES EXTRACTION
########################################
annotation_directory = Path('./data/CADEC.v2/cadec/original')
unique_drugs = set()

for file_path in annotation_directory.glob('*.ann'):
    with open(file_path, 'r') as file:
        for line in file:
            if '\tDrug ' in line:
                drug_name = line.split('\t')[2].strip()
                unique_drugs.add(drug_name)

print('\nUnique drug names:')
for drug in unique_drugs:
    print(drug)
print(f"\nTotal unique drug names: {len(unique_drugs)}\n")


########################################
# REPORTS PER DRUG
########################################
Drugs = ['ARTHROTEC', 'CAMBIA', 'CATAFLAM', 'DICLOFENAC-POTASSIUM', 'DICLOFENAC-SODIUM',
         'FLECTOR', 'LIPITOR', 'PENNSAID', 'SOLARAZE', 'VOLTAREN', 'VOLTAREN-XR', 'ZIPSOR']

reports_per_drug = {}
print('Number of reports per drug:')
for drug in Drugs:
    drug_files = annotation_directory.glob(drug + '*.ann')
    reports_per_drug[drug] = sum(1 for _ in drug_files)
    print(f"{drug}: {reports_per_drug[drug]}")

########################################
# TAG ANALYSIS
########################################
Tags = {}
Empty_tags = {}

print('\nReports without annotations per drug:')
for drug in Drugs:
    print(f"{drug}:")
    lst = []
    count_empty = 0
    for file in Path('./data/CADEC.v2/cadec/original').glob(drug+'*.ann'):
        try:
            data = pd.read_csv(file, sep='\t', header=None)
            lst.extend(data.iloc[:,1].tolist())
        except:
            st = os.stat(file)
            print(f"  Empty file (size={st.st_size}): {file}")
            count_empty += 1
    Tags[drug] = lst
    Empty_tags[drug] = count_empty
    print()

# Example: read a report without tags
with open('./data/CADEC.v2/cadec/text/CAMBIA.1.txt') as f:
    print("Example report without tags:\n", f.read())

print(f"\nTotal number of reports without annotations: {sum(Empty_tags.values())}\n")

########################################
# ENTITY TYPE ANALYSIS
########################################
def extract_first_part(tag):
    return re.findall(r'(\S+)\s', tag)[0]

def process_tags(drug, tags):
    unique_first_parts = sorted(set(extract_first_part(tag) for tag in tags))
    return f"{drug}: {unique_first_parts}"

print("\nTypes of named entities per drug:")
for drug in Drugs:
    print(process_tags(drug, Tags[drug]))


########################################
# PLOTTING
########################################
# Ensure figures folder exists
os.makedirs("figures", exist_ok=True)

# Plot annotations
fig1_name = os.path.join(os.path.abspath("./figures"), "no-annotation-percentage.png")
plt.figure(figsize=(10,5))
plt.title('Percentage of reports without annotations per drug')
plt.xlabel('Drug')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.ylim(0, 100)
plt.bar(Empty_tags.keys(),
        [100 * Empty_tags[drug] / reports_per_drug[drug] for drug in Drugs])
plt.savefig(fig1_name)

# Compute tag distribution
def count_tags(raw_tags):
    entities = ['ADR', 'Disease', 'Drug', 'Finding', 'Symptom']
    return {entity: sum(1 for ele in raw_tags if ele.startswith(entity)) for entity in entities}

df = {drug: list(count_tags(Tags[drug]).values()) for drug in Drugs}
df = pd.DataFrame(df, index=['ADR', 'Disease', 'Drug', 'Finding', 'Symptom']).T
df = df.div(df.sum(axis=1), axis=0)  # normalize by rows
df['Name'] = Drugs

# Plot tag distribution
fig2_name = os.path.join(os.path.abspath("./figures"), "tag-dist.png")
df.plot(x='Name', kind='barh', stacked=True, figsize=(10,6), fontsize=11)
plt.title('Distribution of tags over all documents', fontsize=18)
plt.xlabel('Proportion', fontsize=16)
plt.ylabel('Mentioned drug document', fontsize=16)
plt.legend(title='Named entity', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
plt.savefig(fig2_name)
