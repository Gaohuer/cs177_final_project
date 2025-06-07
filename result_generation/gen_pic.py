# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load data
# df = pd.read_csv("./test_result/ablation_result(1).csv")

# # Create combination labels
# conditions = []
# all_components = ['GCN', 'scGPT', 'GenePT', 'ESM']

# for _, row in df.iterrows():
#     present = [comp for comp in all_components if row[comp]]
#     missing = [comp for comp in all_components if not row[comp]]

#     if len(present) == 4:
#         condition = 'Model'
#     elif len(present) == 3 and len(missing) == 1:
#         condition = f"w/o {missing[0]}"
#     elif present:
#         condition = ' + '.join(present)
#     else:
#         condition = 'w/o all'

#     conditions.append(condition)

# df['Condition'] = conditions

# # Melt for plotting
# df_melted = df.melt(id_vars=['Condition'], 
#                     value_vars=['AUC', 'AUPR', 'F1'],
#                     var_name='Metric', 
#                     value_name='Score')

# # Custom order for hue
# condition_order = [
#     'Model',
#     'w/o GCN',
#     'w/o scGPT',
#     'w/o GenePT',
#     'w/o ESM',
#     'GCN',
#     'scGPT',
#     'GenePT',
#     'ESM',
# ]

# # Plot
# plt.figure(figsize=(15, 6))
# ax = sns.barplot(data=df_melted, x='Metric', y='Score', hue='Condition',
#                  hue_order=condition_order,
#                  palette=sns.color_palette("husl", len(condition_order)))

# plt.title('Performance Metrics by Different Embedding Combinations')
# plt.ylabel('Score')
# plt.ylim(0.5, 0.90)
# plt.xlabel('Metric')
# plt.legend(title='Embedding Combination', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()


# plt.savefig('pic.png')
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("./test_result/ablation_result(1).csv")

# Create combination labels
conditions = []
all_components = ['GCN', 'scGPT', 'GenePT', 'ESM']

for _, row in df.iterrows():
    present = [comp for comp in all_components if row[comp]]
    missing = [comp for comp in all_components if not row[comp]]

    if len(present) == 4:
        condition = 'Model'
    elif len(present) == 3 and len(missing) == 1:
        condition = f"w/o {missing[0]}"
    elif present:
        condition = ' + '.join(present)
    else:
        condition = 'w/o all'

    conditions.append(condition)

df['Condition'] = conditions

# Melt for plotting
df_melted = df.melt(id_vars=['Condition'], 
                    value_vars=['AUC', 'AUPR', 'F1'],
                    var_name='Metric', 
                    value_name='Score')

# Custom order for hue
condition_order = [
    'Model',
    'w/o GCN',
    'w/o scGPT',
    'w/o GenePT',
    'w/o ESM',
    'GCN',
    'scGPT',
    'GenePT',
    'ESM',
]

# Plot
plt.figure(figsize=(15, 6))
ax = sns.barplot(data=df_melted, x='Metric', y='Score', hue='Condition',
                 hue_order=condition_order,
                 palette=sns.color_palette("husl", len(condition_order)))

# Add labels on top of each bar (show Condition names instead of values)
for i, container in enumerate(ax.containers):
    labels = [condition_order[i]] * len(container)  # Repeat the condition name for each bar in the group
    ax.bar_label(container, labels=labels, padding=3, fontsize=8,rotation=60)

plt.title('Performance Metrics by Different Embedding Combinations')
plt.ylabel('Score')
plt.ylim(0.5, 0.90)
plt.xlabel('Metric')
plt.legend(title='Embedding Combination', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig('./pics/pic.png')
plt.show()