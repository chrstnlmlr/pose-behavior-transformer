import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
    GlobalAveragePooling1D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, d_model):
        super().__init__()
        self.position_embeddings = self.add_weight(
            shape=(sequence_length, d_model),
            initializer="random_normal",
            trainable=True,
            name="position_embeddings",
        )

    def call(self, inputs):
        return inputs + self.position_embeddings


def transformer_encoder_block(x, d_model, num_heads, ff_dim, dropout_rate):
    attention_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout_rate,
    )(x, x)
    x = LayerNormalization(epsilon=1e-6)(x + attention_output)

    ffn_output = Dense(ff_dim, activation="relu")(x)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    ffn_output = Dense(d_model)(ffn_output)

    x = LayerNormalization(epsilon=1e-6)(x + ffn_output)
    return x


def build_transformer_model(
    input_shape,
    d_model=128,
    num_heads=4,
    ff_dim=256,
    num_layers=2,
    dropout_rate=0.1,
    learning_rate=0.0003,
    output_units=2,
):
    sequence_length, _ = input_shape

    inputs = Input(shape=input_shape)
    x = Dense(d_model)(inputs)
    x = Dropout(dropout_rate)(x)
    x = PositionalEmbedding(sequence_length, d_model)(x)

    for _ in range(num_layers):
        x = transformer_encoder_block(
            x=x,
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
        )

    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(output_units, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)

    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )
    return model