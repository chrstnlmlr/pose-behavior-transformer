from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam


def build_lstm_model(
    input_shape,
    num_layers=3,
    units_layer_1=256,
    units_layer_2=256,
    units_layer_3=256,
    dropout_rate=0.1,
    learning_rate=0.0003,
    output_units=2,
):
    model = Sequential()
    model.add(Input(shape=input_shape))

    units_per_layer = [units_layer_1, units_layer_2, units_layer_3]

    for i in range(num_layers):
        model.add(
            LSTM(
                units=units_per_layer[i],
                return_sequences=(i < num_layers - 1),
            )
        )
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=output_units, activation="sigmoid"))

    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )
    return model