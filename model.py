import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

class PositionEmbedding(layers.Layer):
    def __init__(self, max_seq_len, emb_dim):
        super(PositionEmbedding, self).__init__()
        self.max_seq_len = max_seq_len
        self.emb_dim = emb_dim
        self.pos_emb = layers.Embedding(self.max_seq_len, self.emb_dim)
    
    def call(self, x):
        seq_len = tf.shape(x)[1]
        positions = tf.range(seq_len)
        positions = tf.expand_dims(positions, axis=0)
        pos_emb = self.pos_emb(positions)
        
        return x + pos_emb

class FeedForwardLayer(layers.Layer):
    def __init__(self, emb_dim, ffn_dim, dropout):
        super(FeedForwardLayer, self).__init__()
        
        self.seq = keras.Sequential([
            layers.Dense(ffn_dim, activation='relu'),
            layers.Dense(emb_dim),
            layers.Dropout(dropout)
        ])
        self.norm = layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, x):
        return self.norm(x + self.seq(x))

class EncoderLayer(layers.Layer):
    def __init__(
        self,
        emb_dim,
        num_heads,
        ffn_dim,
        dropout=0.1,
        regularizer=None
    ):
        super(EncoderLayer, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        
        self.regularizer = regularizer
        
        self.attn = layers.MultiHeadAttention(self.num_heads, self.emb_dim, kernel_regularizer=self.regularizer)
        self.ffn_layer = FeedForwardLayer(self.emb_dim, self.ffn_dim, dropout)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
    
    def call(self, x):
        attn_output, attn_score = self.attn(query=x, value=x, return_attention_scores=True, use_causal_mask=False)
        self.attn_score = attn_score
        x = self.layernorm1(x + self.dropout1(attn_output))
        
        x = self.ffn_layer(x)
        return x

class Transformer(keras.Model):
    def __init__(self, vocab_size, num_classes, max_seq_len, emb_dim, ffn_dim, num_heads, dropout, num_encoders, regularizer=None):
        super(Transformer, self).__init__()
        
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.emb_dim = emb_dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_encoders = num_encoders
        self.num_classes = num_classes
        
        # We do not need to load the embeddings here as it will be loaded when the 
        # weights of the model is loaded
        self.token_emb = layers.Embedding(self.vocab_size, self.emb_dim)
        self.pos_emb = PositionEmbedding(self.max_seq_len, self.emb_dim)
        
        self.encoder = [
            EncoderLayer(self.emb_dim, self.num_heads, self.ffn_dim, self.dropout, regularizer)
            for _ in range(self.num_encoders)
        ]
        
        self.gap = layers.GlobalAveragePooling1D()
        self.dropout = layers.Dropout(self.dropout)
        self.final_op = layers.Dense(num_classes, activation='softmax')
    
    def call(self, x):
        x = self.token_emb(x)
        x = self.pos_emb(x)
        
        for layer in self.encoder:
            x = layer(x)
        
        x = self.gap(x)
        x = self.dropout(x)
        x = self.final_op(x)
        return x

def load_model(weight_path, max_seq_len, vocab_size):
	num_classes = 3
	emb_dim = 100
	ffn_dim = 128
	num_heads = 4
	dropout = 0.2
	num_encoders = 2
	model = Transformer(vocab_size, num_classes, max_seq_len, 
			emb_dim, ffn_dim, num_heads, dropout, num_encoders)

	model.load_weights(weight_path)

	return model

def load_tokenizer(tokenizer_path, max_seq_len, vocab_size):
	vectorizer = layers.TextVectorization(
			max_tokens = vocab_size,
			output_sequence_length = max_seq_len
		)

	vectorizer.load_assets(tokenizer_path)
	return vectorizer