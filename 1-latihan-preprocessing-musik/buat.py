import pandas as pd
import random

# Data dummy
judul_lagu = [
    "Shine On You Crazy Diamond", "Stairway to Heaven", "Bohemian Rhapsody", 
    "Hotel California", "Comfortably Numb", "Blackbird", "Smells Like Teen Spirit", 
    "Nothing Else Matters", "November Rain", "Wish You Were Here"
]

genres = ["Progressive Rock", "Hard Rock", "Classic Rock", "Alternative Rock", "Folk Rock"]
popularitas = ["Rendah", "Sedang", "Tinggi"]

# Generate 100 baris data acak
data = []
for i in range(100):
    lagu = random.choice(judul_lagu)
    usia = random.randint(15, 40)  # usia pendengar
    genre = random.choice(genres)
    durasi = round(random.uniform(2.5, 15.0), 2)  # durasi lagu
    pop = random.choice(popularitas)
    data.append([lagu, usia, genre, durasi, pop])

df = pd.DataFrame(data, columns=["Judul Lagu", "Usia Pendengar", "Genre", "Durasi (menit)", "Popularitas"])

# Simpan ke Excel & CSV
df.to_excel("musik_dataset.xlsx", index=False)
df.to_csv("musik_dataset.csv", index=False)

print("Dataset musik dengan 100 baris berhasil dibuat!")
