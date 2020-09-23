from generations import *
import tensorflow as tf


def bit_err(y_true, y_pred):
    err = 1 - tf.reduce_mean(
        tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.sign(
                        y_pred - 0.5),
                    tf.cast(
                        tf.sign(
                            y_true - 0.5),
                        tf.float32)),tf.float32),
            1))
    return err

"""#comment the 2 lines below if u dont have cuda-enabled gpu
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

input_bits = tf.keras.Input(shape=(payloadBits_per_OFDM * 2,))
temp = tf.keras.layers.BatchNormalization()(input_bits)
temp = tf.keras.layers.Dense(n_hidden_1, activation='relu')(input_bits)
temp = tf.keras.layers.BatchNormalization()(temp)
temp = tf.keras.layers.Dense(n_hidden_2, activation='relu')(temp)
temp = tf.keras.layers.BatchNormalization()(temp)
temp = tf.keras.layers.Dense(n_hidden_3, activation='relu')(temp)
temp = tf.keras.layers.BatchNormalization()(temp)
out_put = tf.keras.layers.Dense(n_output, activation='sigmoid')(temp)
model = tf.keras.Model(input_bits, out_put)
model.compile(optimizer='adam', loss='mse', metrics=[bit_err])
model.summary()
checkpoint = tf.keras.callbacks.ModelCheckpoint('./temp_trained_25.h5', monitor='val_bit_err',
                                       verbose=0, save_best_only=True, mode='min', save_weights_only=True)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto',
    baseline=None, restore_best_weights=False)
model.fit(
    training_gen(64,25),
    steps_per_epoch=50,
    epochs=10000,
    validation_data=validation_gen(1000, 25),
    validation_steps=1,
    callbacks=[checkpoint,early_stop],
    verbose=2)

model.load_weights('./temp_trained_25.h5')
BER = []
for SNR in range(5, 30, 5):
    y = model.evaluate(
        validation_gen(1024, SNR),
        steps=1
    )
    BER.append(y[1])
    print(y)
print(BER)
BER_matlab = np.array(BER)
import scipy.io as sio
sio.savemat('BER.mat', {'BER':BER_matlab})"""