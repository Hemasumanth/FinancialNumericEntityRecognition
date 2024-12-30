import tensorflow as tf
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, MultiHeadAttention, LayerNormalization, Dropout, Dense
from tensorflow.keras import Model

class CustomNERModelV5(Model):
    def __init__(self, num_tokens, num_tags, d_model, num_heads, dff, lstm_units, rate=0.1):
        super(CustomNERModelV5, self).__init__()

        # Input layer
        self.embedding = Embedding(num_tokens, d_model)

        # Bidirectional LSTM layer
        self.bilstm = Bidirectional(LSTM(units=lstm_units, return_sequences=True))

        # Multi-Head Self-Attention
        self.multi_head_attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)

        # Layer Normalization and Dropout
        self.layer_norm1 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)

        # Additional Dense layer
        self.dense = Dense(dff, activation='relu')

        # Layer Normalization and Dropout
        self.layer_norm2 = LayerNormalization(epsilon=1e-6)
        self.dropout2 = Dropout(rate)

        # Final Dense layer
        self.final_dense = Dense(num_tags + 1, activation='sigmoid')

    def call(self, x, training=False):
        embed = self.embedding(x)

        # Bidirectional LSTM
        lstm_output = self.bilstm(embed)

        # Multi-Head Self-Attention
        attn_output = self.multi_head_attention(lstm_output, lstm_output)

        # Layer Normalization and Dropout
        attn_output = self.layer_norm1(lstm_output + attn_output)
        attn_output = self.dropout1(attn_output, training=training)

        # Additional Dense layer
        dense_output = self.dense(attn_output)

        # Layer Normalization and Dropout
        dense_output = self.layer_norm2(dense_output)
        dense_output = self.dropout2(dense_output, training=training)

        # Final Dense layer
        final_output = self.final_dense(dense_output)

        return final_output
