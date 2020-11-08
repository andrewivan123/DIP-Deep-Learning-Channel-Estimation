from generations import *
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import os

PATH = "running_model.pt"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)




class NeuralNet(nn.Module):
    """
    Neural Network Class Definition
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(payloadBits_per_OFDM * 2, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_hidden_3)
        self.fc4 = nn.Linear(n_hidden_3, n_output)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x




n_of_batches = 10000
batch_size = 64
SNRdb = 10 #5 15 20 25

def bit_err(y_true, y_pred):
    err = 1 - torch.mean(
        torch.mean(
                torch.eq(torch.sign(y_pred - 0.5),torch.sign(y_true - 0.5)).type(torch.float),1))
    return err

#check if saved checkpoint exists
"""if os.path.exists(PATH):
    checkpoint = torch.load(PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['loss']"""
#save the BER for the last batch
final_BER = []

for snr in [5,10,15,20,25]:
    net = NeuralNet()
    net.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    # Draw Graph
    x_values = []
    y_values = []
    BER_values = []
    for batch in range(n_of_batches):
        running_loss = 0
        running_BER = 0
        for inputs, labels in training_gen(batch_size, SNRdb):
            optimizer.zero_grad()

            inputs = torch.from_numpy(inputs).float()
            labels = torch.from_numpy(labels).float()
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_BER += bit_err(labels,outputs)
            # print statistics
            running_loss += loss.item()
        if batch % 100 == 0:  # print every 2000 mini-batches
            print('[%d] loss: %.3f' %
                (batch + 1, running_loss / batch_size))
            print('[%d] BER: %.3f' %
                (batch + 1, running_BER / batch_size))

            # Save
            """torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss/batch_size,
                }, PATH)"""

            x_values.append(batch + 1)
            y_values.append(round(loss.item(), 3))
            BER_values.append(bit_err(labels,outputs))
    #save final value
    final_BER.append(BER_values[-1])

    plt.clf()
    plt.title(f'BER Plot (SNR = {snr}dB)')
    plt.xlabel('Batches')
    plt.ylabel('BER')
    plt.plot(x_values, BER_values)
    plt.savefig(f'BER_{snr}dB.png')
print(final_BER)

"""
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