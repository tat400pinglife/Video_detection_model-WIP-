from datasets import load_dataset

ds = load_dataset("jian-0/GenVidBench")
print(ds)

# hugging face is a master baiter, downloaded a parquet with links to mp4 files on disk????
# aint no way i have that much space on my cpu