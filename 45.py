import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set plot style
sns.set(style='whitegrid')

# Load data
df = pd.read_csv('py air data.csv')

# 1️⃣ Data Cleaning & Preprocessing
# Replace "NA" with NaN and convert relevant columns to float
df.replace("NA", np.nan, inplace=True)
df[['pollutant_min', 'pollutant_max', 'pollutant_avg']] = df[['pollutant_min', 'pollutant_max', 'pollutant_avg']].astype(float)
df['station'] = df['station'].str.strip().str.title()
df['last_update'] = pd.to_datetime(df['last_update'], errors='coerce')
df.dropna(subset=['pollutant_avg'], inplace=True)

# 2️⃣ Basic EDA: Pollution by City
pollution_by_city = df.groupby('city')['pollutant_avg'].mean().sort_values(ascending=False)
top10_cities = pollution_by_city.head(10)
important_pollutants = ['PM2.5', 'PM10', 'CO', 'NO2', 'OZONE']

# 3️⃣ Visualizations

# 3.1. Barplot of Top 10 Polluted Cities
plt.figure(figsize=(12, 6))
sns.barplot(x=top10_cities.index, y=top10_cities.values, palette='Spectral')
plt.title('Top 10 Polluted Cities (Avg Pollutant Level)', fontsize=14)
plt.ylabel('Average Pollutant Level')
plt.xlabel('City')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3.2. Violin & Stripplot: Pollutant Distribution
plt.figure(figsize=(12, 6))
sns.violinplot(x='pollutant_id', y='pollutant_avg', data=df, inner=None, palette='Set2')
sns.stripplot(x='pollutant_id', y='pollutant_avg', data=df, color='black', alpha=0.3, jitter=0.2)
plt.title('Distribution of Pollutant Levels by Type')
plt.ylabel('Avg Pollutant Value')
plt.xlabel('Pollutant Type')
plt.tight_layout()
plt.show()

# 3.3. Heatmap of Pollution by City vs Pollutant
filtered_df = df[df['city'].isin(top10_cities.index) & df['pollutant_id'].isin(important_pollutants)]
pivot_table = filtered_df.pivot_table(values='pollutant_avg', index='city', columns='pollutant_id', aggfunc='mean')
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, cmap='coolwarm', annot=True, fmt='.1f', linewidths=0.5)
plt.title("Heatmap: Avg Pollution by City vs Pollutant Type (Top 10 Cities)")
plt.tight_layout()
plt.show()

# 3.4. Pie Chart: Pollutant Type Distribution
pollutant_counts = df['pollutant_id'].value_counts()
plt.figure(figsize=(6, 6))
pollutant_counts.plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title('Pollutant Type Distribution')
plt.ylabel('')
plt.tight_layout()
plt.show()

# 3.5. Boxplot: City Pollution Spread
plt.figure(figsize=(14, 6))
sns.boxplot(data=df[df['city'].isin(top10_cities.index)], x='city', y='pollutant_avg', palette='Set3')
plt.title('Pollution Level Distribution Across Top 10 Cities')
plt.xticks(rotation=45)
plt.ylabel('Pollutant Avg')
plt.tight_layout()
plt.show()

# 3.6. Air Quality Classification
def classify_quality(value):
    if pd.isna(value):
        return 'Unknown'
    elif value <= 50:
        return 'Good'
    elif value <= 100:
        return 'Moderate'
    else:
        return 'Poor'

df['quality'] = df['pollutant_avg'].apply(classify_quality)

# 3.7. Countplot of Air Quality
plt.figure(figsize=(6, 4))
sns.countplot(data=df[df['quality'] != 'Unknown'], x='quality', order=['Good', 'Moderate', 'Poor'], palette='coolwarm')
plt.title('Air Quality Classification ')
plt.ylabel('Number of Records')
plt.tight_layout()
plt.show()

# 3.8. Stacked Bar: Quality per City
city_quality = df[df['quality'] != 'Unknown'].groupby(['city', 'quality']).size().unstack().fillna(0)
city_quality = city_quality.loc[city_quality.sum(axis=1).sort_values(ascending=False).index[:10]]
city_quality.plot(kind='bar', stacked=True, colormap='coolwarm', figsize=(12, 6))
plt.title('Air Quality Classification per City (Top 10)')
plt.ylabel('Number of Records')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3.9. Pairplot of Pollutants
pivot_pair = df[df['city'].isin(top10_cities.index) & df['pollutant_id'].isin(important_pollutants)]
pivot_pair = pivot_pair.pivot_table(index=['city', 'last_update'], columns='pollutant_id', values='pollutant_avg').dropna()

if not pivot_pair.empty:
    sns.pairplot(pivot_pair, corner=True, diag_kind='kde')
    plt.suptitle("Pairwise Comparison of Pollutants in Top Cities", y=1.02)
    plt.show()

# 4️⃣ Grouped Pollutant Average
grouped_pollutants = df.groupby('pollutant_id')['pollutant_avg'].mean().sort_values(ascending=False)

# 5️⃣ T-test: CO vs NO2
co = df[df['pollutant_id'] == 'CO']['pollutant_avg'].dropna()
no2 = df[df['pollutant_id'] == 'NO2']['pollutant_avg'].dropna()
t_stat, p_value = stats.ttest_ind(co, no2, equal_var=False)

# 6️⃣ Correlation: NO2 vs PM2.5
pivot_corr = df.pivot_table(index=['city', 'last_update'], columns='pollutant_id', values='pollutant_avg')
corr = None
if 'NO2' in pivot_corr.columns and 'PM2.5' in pivot_corr.columns:
    corr = pivot_corr['NO2'].corr(pivot_corr['PM2.5'])

# 7️⃣ Manual Linear Regression
df_reg = df.dropna(subset=['pollutant_min', 'pollutant_max', 'pollutant_avg'])
X = df_reg[['pollutant_min', 'pollutant_max']].values
y = df_reg['pollutant_avg'].values
X_with_intercept = np.column_stack((np.ones(X.shape[0]), X))
beta = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y

# Plot predicted vs actual
y_pred = X_with_intercept @ beta
plt.figure(figsize=(8, 5))
plt.scatter(y, y_pred, alpha=0.6, color='purple')
plt.xlabel("Actual Pollutant Average")
plt.ylabel("Predicted Pollutant Average")
plt.title("Manual Linear Regression: Prediction vs Actual")
plt.grid(True)
plt.tight_layout()
plt.show()

# ========================
# ✅ Terminal Output Section
# ========================

# 8️⃣ Average Pollutant Levels
print("\n" + "="*50)
print("📊 Average Pollutant Levels by Type")
print("="*50)
print(f"Pollutant Avg Levels:")
for pollutant, avg in grouped_pollutants.items():
    print(f"{pollutant}: {avg:.2f}")

# 9️⃣ T-test: CO vs NO2
print("\n" + "="*50)
print("🔬 T-Test Between CO and NO2")
print("="*50)
print(f"T-statistic     : {t_stat:.2f}")
print(f"P-value         : {p_value:.4f}")
if p_value < 0.05:
    print("🟢 Result        : Statistically Significant (p < 0.05)")
else:
    print("🔴 Result        : Not Statistically Significant (p ≥ 0.05)")

# 🔗 Correlation: NO2 vs PM2.5
print("\n" + "="*50)
print("🔗 Correlation Between NO2 and PM2.5")
print("="*50)
if corr is not None:
    print(f"Correlation Coefficient (r): {corr:.2f}")
    if abs(corr) >= 0.7:
        strength = "Strong"
    elif abs(corr) >= 0.4:
        strength = "Moderate"
    else:
        strength = "Weak"
    print(f"🧠 Strength: {strength} Correlation")
else:
    print("❌ Data insufficient for correlation calculation.")

# 📈 Manual Linear Regression Coefficients
print("\n" + "="*50)
print("📈 Manual Linear Regression: Predicting avg from min & max")
print("="*50)
print(f"Intercept (b0): {beta[0]:.2f}")
print(f"Min Coeff (b1): {beta[1]:.2f}")
print(f"Max Coeff (b2): {beta[2]:.2f}")
