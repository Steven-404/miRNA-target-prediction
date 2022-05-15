import logging
from pathlib import Path

import gensim
import numpy as np
import torch
from gensim.models import word2vec

THIS_FILE = Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)
sentences_file = THIS_DIR.joinpath("Data/mRNA_veb.txt")
model_file = THIS_DIR.joinpath("Data/data/pre_train_data/m_vec/m2vec_50_n.model")

sentences = word2vec.LineSentence(str(sentences_file))

model = word2vec.Word2Vec(sentences, vector_size=20,epochs=10,window=5,sg=1)
model.wv.save_word2vec_format(str(model_file))
