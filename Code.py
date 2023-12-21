import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, ttest_ind

# Read the CSV file into a pandas DataFrame
file_path = 'multi_env_data.csv'  # Replace with the actual path to your CSV file
df = pd.read_csv(file_path)
# Display the first few rows of the DataFrame
print("Sample of the DataFrame:")
print(df.head())
# Descriptive statistics
descriptive_stats = df.describe()
print("\nDescriptive Statistics:")
print(descriptive_stats)
descriptive_stats.to_csv('descriptive_statistics.csv')
# Boxplot for each environment
sns.boxplot(x='Environment', y='Phenotypic_Value', data=df)
plt.title('Boxplot of Phenotypic Values by Environment')
plt.savefig('boxplot.png')
plt.close()
# ANOVA test
print("\nOne-way ANOVA test:")
anova_result = f_oneway(df['Phenotypic_Value'][df['Environment'] == 0],
                         df['Phenotypic_Value'][df['Environment'] == 1],
                         df['Phenotypic_Value'][df['Environment'] == 2])
print(anova_result)
anova_result_df = pd.DataFrame({'F-statistic': [anova_result.statistic],
                                'p-value': [anova_result.pvalue]})
anova_result_df.to_csv('anova_result.csv')
# Pairwise t-tests
pairwise_t_tests_results = []
print("\nPairwise t-tests:")
for i in range(3):
    for j in range(i+1, 3):
        env1 = df[df['Environment'] == i]['Phenotypic_Value']
        env2 = df[df['Environment'] == j]['Phenotypic_Value']
        t_stat, p_value = ttest_ind(env1, env2)
        pairwise_t_tests_results.append({'Comparison': f'Environment {i} vs Environment {j}',
                                         't-statistic': t_stat,
                                         'p-value': p_value})

pairwise_t_tests_df = pd.DataFrame(pairwise_t_tests_results)
pairwise_t_tests_df.to_csv('pairwise_t_tests_results.csv')
# Correlation matrix
correlation_matrix = df.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)
correlation_matrix.to_csv('correlation_matrix.csv')
# Scatter plot matrix
sns.pairplot(df, hue='Environment')
plt.title('Scatter Plot Matrix by Environment')
plt.savefig('scatter_plot_matrix.png')
plt.show()
# Heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.show()
# Violin plot to compare distributions
sns.violinplot(x='Environment', y='Phenotypic_Value', data=df, inner='quartile')
plt.title('Violin Plot of Phenotypic Values by Environment')
plt.savefig('violin_plot.png')
plt.show()
# Bar plot of mean phenotypic values by environment
mean_values = df.groupby('Environment')['Phenotypic_Value'].mean()
mean_values.plot(kind='bar', color='skyblue')
plt.title('Mean Phenotypic Values by Environment')
plt.xlabel('Environment')
plt.ylabel('Mean Phenotypic Value')
plt.xticks(rotation=360)
plt.savefig('mean_values_bar_plot.png')
plt.show()
