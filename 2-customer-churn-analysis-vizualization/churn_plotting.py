# ==========================================================
# CUSTOMER CHURN ANALYSIS & VISUALIZATION
# ==========================================================
# Tujuan:
# - Membaca dataset churn pelanggan dari file CSV
# - Menghitung churn rate berdasarkan tipe pembayaran
# - Membuat visualisasi churn rate
# ==========================================================

# --- Import Library ---
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Styling visual
sns.set(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams['figure.figsize'] = (9, 5)

# ----------------------------------------------------------
# 1. Membaca dataset
# ----------------------------------------------------------
df_phone = pd.read_csv('customer_churn_data.csv')

print("📊 Sample Data:")
print(df_phone.head(), "\n")

print("Jumlah data:", len(df_phone))
print("Distribusi churn value:")
print(df_phone['churn_value'].value_counts(), "\n")

# ----------------------------------------------------------
# 2. Menghitung churn rate per tipe pembayaran
# ----------------------------------------------------------
churn_rate = df_phone.groupby('payment')['churn_value'].mean().reset_index()
churn_rate.rename(columns={'churn_value': 'churn_rate'}, inplace=True)
print("📈 Rata-rata churn per tipe pembayaran:")
print(churn_rate, "\n")

# ----------------------------------------------------------
# 3. Visualisasi churn rate per tipe pembayaran
# ----------------------------------------------------------
sns.barplot(
    data=churn_rate,
    x='payment',
    y='churn_rate',
    hue='payment',
    dodge=False,
    legend=False
)

plt.title('Churn Rate by Payment Type (Phone Customers)', fontsize=14, weight='bold')
plt.xlabel('Payment Type')
plt.ylabel('Average Churn Rate')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()