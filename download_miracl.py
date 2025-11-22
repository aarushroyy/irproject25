from datasets import load_dataset

# Load query+relevance dataset
miracl_hi = load_dataset("miracl/miracl", "hi")

# Load Hindi document corpus
miracl_corpus = load_dataset("miracl/miracl-corpus", "hi")

print(miracl_hi)
print(miracl_corpus)


# from datasets import load_dataset

# def main():
#     # 1) Topics + qrels (queries and relevance info)
#     miracl_hi = load_dataset("miracl/miracl", "hi")
#     print("MIRACL Hindi topics splits:", miracl_hi)

#     # Save to disk so we can reuse without redownloading
#     miracl_hi.save_to_disk("data/miracl_hi_topics_qrels")

#     # 2) Document corpus (Hindi Wikipedia passages)
#     miracl_corpus_hi = load_dataset("miracl/miracl-corpus", "hi")
#     print("MIRACL Hindi corpus splits:", miracl_corpus_hi)

#     miracl_corpus_hi.save_to_disk("data/miracl_hi_corpus")

# if __name__ == "__main__":
#     main()
