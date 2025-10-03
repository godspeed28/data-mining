import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt

# =========================
# 1. Load data
# =========================
datasets = pd.read_excel('musik_dataset.xlsx')

# Hapus kolom kosong/unnamed
datasets = datasets.drop(columns=[col for col in datasets.columns if 'Unnamed' in col])

# Hapus spasi di nama kolom
datasets.columns = datasets.columns.str.strip()

# =========================
# 2. Preprocessing Usia Pendengar
# =========================
datasets['Usia Pendengar'] = pd.to_numeric(datasets['Usia Pendengar'], errors='coerce')
datasets['Usia Pendengar'] = datasets['Usia Pendengar'].fillna(datasets['Usia Pendengar'].mean())

# =========================
# 3. Preprocessing Genre (One-Hot Encoding)
# =========================
encoder = OneHotEncoder(sparse_output=False)
genre_encoded = encoder.fit_transform(datasets[['Genre']])
genre_df = pd.DataFrame(genre_encoded, columns=encoder.get_feature_names_out(['Genre']))
datasets = pd.concat([datasets.drop('Genre', axis=1), genre_df], axis=1)

# =========================
# 4. Preprocessing Popularitas (Target)
# =========================
label_encoder_target = LabelEncoder()
datasets['Popularitas'] = label_encoder_target.fit_transform(datasets['Popularitas'])

# =========================
# 5. Visualisasi Popularitas
# =========================
value_counts = datasets['Popularitas'].value_counts()
ax = value_counts.plot(kind='bar', color=['skyblue', 'orange', 'green'])
plt.title('Distribusi Popularitas Lagu')
plt.xlabel('Kategori')
plt.ylabel('Jumlah')
for i, count in enumerate(value_counts):
    ax.text(i, count, str(count), ha='center', va='bottom')
plt.show()

# =========================
# 6. Simpan hasil
# =========================
datasets.to_csv('musik-dataset-jadi.csv', index=False)

datasets.info()
