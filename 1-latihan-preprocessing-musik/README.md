## 🎶 Latihan Preprocessing Dataset Musik

Repository ini berisi eksperimen **preprocessing data** menggunakan Python dengan dataset bertema musik.  
Dataset berisi 100 baris data dummy tentang lagu, usia pendengar, genre, durasi, dan popularitas.  
Project ini dibuat untuk latihan **pandas, scikit-learn, dan matplotlib**.

---

## ⚙️ Instalasi

#### 1. Clone Repo
```bash
git clone https://github.com/godspeed28/data-mining.git
```
#### 2. Buat virtual environment
```bash
python -m venv env
source env/bin/activate   # Linux / MacOS
env\Scripts\activate      # Windows
source env\Scripts\activate # Git Bash
```
#### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## 🛠️ Langkah-Langkah

#### 1. Generate Dataset Musik
```bash
python buat.py
```
Akan menghasilkan file:
- musik_dataset.csv
- musik_dataset.xlsx (jika ada openpyxl terinstall)
#### 2. Preprocessing Dataset
```bash
python preprocessing.py
```
Proses preprocessing yang dilakukan:
- Membersihkan kolom Usia Pendengar → konversi ke numerik & isi NaN dengan - rata-rata.
- Melakukan One-Hot Encoding pada kolom Genre.
- Melakukan Label Encoding pada kolom Popularitas.
- Membuat visualisasi distribusi popularitas dengan matplotlib.
- Menyimpan hasil ke musik-dataset-jadi.csv.

