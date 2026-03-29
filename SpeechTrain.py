import os
import pickle
import numpy as np
import librosa
import soundfile

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import MaxPooling2D, Dense, Flatten, Conv2D
from tensorflow.keras.models import Sequential, model_from_json

path = 'SpeechEmotionDataset'

# Audio files known to be problematic — skip them
error = [
    'Actor_01/03-01-02-01-01-02-01.wav',
    'Actor_05/03-01-02-01-02-02-05.wav',
    'Actor_20/03-01-03-01-02-01-20.wav',
    'Actor_20/03-01-06-01-01-02-20.wav',
]

X_train = []
Y_train = []


def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    """Extract MFCC, chroma, and mel-spectrogram features from an audio file."""
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        stft = np.abs(librosa.stft(X)) if chroma else None
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma and stft is not None:
            chroma_feat = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma_feat))
        if mel:
            # 'y=' keyword argument is required in modern librosa
            mel_feat = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel_feat))
    return result


# --- Feature Extraction ---
for root, dirs, directory in os.walk(path):
    for fname in directory:
        name = os.path.basename(root)
        rel_path = name + "/" + fname
        if rel_path not in error:
            full_path = os.path.join(root, fname)
            try:
                mfcc = extract_feature(full_path, mfcc=True, chroma=True, mel=True)
                X_train.append(mfcc)
                arr = fname.split("-")
                label = int(arr[2])
                Y_train.append(label)
                print(f"{name}  {full_path}  shape={mfcc.shape}  label={label}")
            except Exception as e:
                print(f"Skipping {full_path}: {e}")

X_train = np.asarray(X_train).astype('float32') / 255
Y_train = np.asarray(Y_train)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1, 1))
print("X_train shape:", X_train.shape)

# Shuffle
indices = np.arange(X_train.shape[0])
np.random.shuffle(indices)
X_train = X_train[indices]
Y_train = Y_train[indices]
Y_train = to_categorical(Y_train)

# Save preprocessed data (uncomment to regenerate)
# np.save('model/speechX.txt', X_train)
# np.save('model/speechY.txt', Y_train)

# Load from saved files instead (comment out if regenerating above)
X_train = np.load('model/speechX.txt.npy')
Y_train = np.load('model/speechY.txt.npy')
print("Loaded X_train:", X_train.shape)
print("Loaded Y_train:", Y_train.shape)

# --- Model ---
if os.path.exists('model/speechmodel.json'):
    with open('model/speechmodel.json', "r") as json_file:
        loaded_model_json = json_file.read()
    classifier = model_from_json(loaded_model_json)
    classifier.load_weights("model/speech_weights.weights.h5")
    # _make_predict_function() is deprecated and removed in TF2 — not needed
    print(classifier.summary())

    with open('model/speechhistory.pckl', 'rb') as f:
        data = pickle.load(f)
    acc = data['accuracy']
    print("Training Model Accuracy = {:.2f}%".format(acc[-1] * 100))  # Use last epoch safely

else:
    classifier = Sequential([
        # Input shape: (feature_length, 1, 1) — e.g. (180, 1, 1) for 40 MFCC + 12 chroma + 128 mel
        Conv2D(32, (1, 1), input_shape=(180, 1, 1), activation='relu'),
        MaxPooling2D(pool_size=(1, 1)),
        Conv2D(32, (1, 1), activation='relu'),
        MaxPooling2D(pool_size=(1, 1)),
        Flatten(),
        Dense(units=256, activation='relu'),          # 'output_dim' was removed; use 'units'
        Dense(units=Y_train.shape[1], activation='softmax'),
    ])

    print(classifier.summary())
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    hist = classifier.fit(X_train, Y_train, batch_size=16, epochs=100, shuffle=True, verbose=2)

    # Save model and history
    classifier.save_weights('model/speech_weights.weights.h5')
    model_json = classifier.to_json()
    with open("model/speechmodel.json", "w") as json_file:
        json_file.write(model_json)
    with open('model/speechhistory.pckl', 'wb') as f:
        pickle.dump(hist.history, f)

    with open('model/speechhistory.pckl', 'rb') as f:
        data = pickle.load(f)
    acc = data['accuracy']
    print("Training Model Accuracy = {:.2f}%".format(acc[-1] * 100))
