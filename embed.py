from gensim.models import Word2Vec
import os

data_paths = "data/trace.txt"
save_path = "embeddings/word2vec.embed"
vector_size = 50
window = 10
epoch = 100

def load_data(path):
    page_ids = []
    with open(path, 'r') as f:
        for line in f.readlines():
            page_id = line.split(",")[0]
            page_ids.append(page_id)
    return page_ids

def train_embedding():
    training_data = load_data(data_paths)
    embed = Word2Vec(sentences=[training_data],
                     vector_size=vector_size,
                     window=window,
                     min_count=1,
                     epochs=epoch,
                     workers=os.cpu_count())
    embed.save(save_path)

train_embedding()
model = Word2Vec.load(save_path)
word_vectors = model.wv
print(word_vectors['876217581568'].shape)
print(word_vectors.key_to_index['876217581568'])