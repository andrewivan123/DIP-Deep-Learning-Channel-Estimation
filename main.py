from generations import *
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class NeuralNet(nn.Module):
    """
    Neural Network Class Definition
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(payloadBits_per_OFDM*2,n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1,n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2,n_hidden_3)
        self.fc4 = nn.Linear(n_hidden_3,n_output)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

net = NeuralNet()
net.to(device) 
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

n_of_epochs = 100

for epoch in range(n_of_epochs):
    running_loss = 0
    for inputs,labels in training_gen(64,25):
        optimizer.zero_grad()

        inputs = torch.from_numpy(inputs).float()
        labels = torch.from_numpy(labels).float()
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = net(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        print(loss.item())
    if epoch % 5 == 1:    # print every 2000 mini-batches
        print('[%d] loss: %.3f' %
                (epoch + 1, running_loss / 2000))
        running_loss = 0.0



"""def bit_err(y_true, y_pred):
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

#comment the 2 lines below if u dont have cuda-enabled gpu
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