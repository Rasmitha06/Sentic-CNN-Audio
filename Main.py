from tkinter import messagebox, simpledialog, filedialog
import tkinter
from tkinter import *
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

import soundfile
import librosa

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import MaxPooling2D, Dense, Dropout, Activation, Flatten, Conv2D
from tensorflow.keras.models import Sequential, model_from_json

main = tkinter.Tk()
main.title("EMOTION DETECTION USING SPEECH RECOGNITION")
main.geometry("1300x1200")

filename = None
X, Y = None, None
speech_X, speech_Y = None, None
speech_classifier = None

speech_emotion = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']


def upload():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END, filename + " loaded\n")


def processDataset():
    global X, Y, speech_X, speech_Y
    text.delete('1.0', END)
    try:
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
        speech_X = np.load('model/speechX.txt.npy')
        speech_Y = np.load('model/speechY.txt.npy')
        text.insert(END, "Total number of images found in dataset: " + str(len(X)) + "\n")
        text.insert(END, "Total number of speech emotion audio files: " + str(speech_X.shape[0]) + "\n")
        text.insert(END, "Total speech emotions: " + str(speech_emotion) + "\n")
    except FileNotFoundError as e:
        text.insert(END, "Error loading dataset files: " + str(e) + "\n")


def trainSpeechCNN():
    global speech_classifier, speech_X, speech_Y
    text.delete('1.0', END)

    if speech_X is None or speech_Y is None:
        text.insert(END, "Please preprocess the dataset first.\n")
        return

    if os.path.exists('model/speechmodel.json'):
        with open('model/speechmodel.json', "r") as json_file:
            loaded_model_json = json_file.read()
        speech_classifier = model_from_json(loaded_model_json)
        speech_classifier.load_weights("model/speech_weights.weights.h5")
        text.insert(END, "Loaded existing model from disk.\n")
    else:
        speech_classifier = Sequential([
            Conv2D(32, (1, 1), input_shape=(speech_X.shape[1], speech_X.shape[2], speech_X.shape[3]), activation='relu'),
            MaxPooling2D(pool_size=(1, 1)),
            Conv2D(32, (1, 1), activation='relu'),
            MaxPooling2D(pool_size=(1, 1)),
            Flatten(),
            Dense(units=256, activation='relu'),
            Dense(units=speech_Y.shape[1], activation='softmax')
        ])
        speech_classifier.summary()
        speech_classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        hist = speech_classifier.fit(speech_X, speech_Y, batch_size=16, epochs=100, shuffle=True, verbose=2)

        speech_classifier.save_weights('model/speech_weights.weights.h5')
        model_json = speech_classifier.to_json()
        with open("model/speechmodel.json", "w") as json_file:
            json_file.write(model_json)
        with open('model/speechhistory.pckl', 'wb') as f:
            pickle.dump(hist.history, f)

    with open('model/speechhistory.pckl', 'rb') as f:
        data = pickle.load(f)

    acc = data['accuracy']
    accuracy = acc[-1] * 100
    text.insert(END, "CNN Speech Emotion Training Model Accuracy = {:.2f}%\n\n".format(accuracy))


def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma_feat = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma_feat))
        if mel:
            mel_feat = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel_feat))
    return result


def predictSpeechExpression():
    global speech_classifier
    if speech_classifier is None:
        text.insert(END, "Please train the model first.\n")
        return

    file_path = filedialog.askopenfilename(initialdir="testSpeech")
    if not file_path:
        return
    fname = os.path.basename(file_path)
    test = []
    mfcc = extract_feature(file_path, mfcc=True, chroma=True, mel=True)
    test.append(mfcc)
    test = np.asarray(test).astype('float32') / 255
    test = test.reshape((test.shape[0], test.shape[1], 1, 1))

    predict = speech_classifier.predict(test)
    predict_index = np.argmax(predict)
    emotion_index = max(0, min(predict_index, len(speech_emotion) - 1))
    emotion = speech_emotion[emotion_index]

    text.delete('1.0', END)
    text.insert(END, "Uploaded speech file: " + fname + "\nEmotion Recognized: " + emotion + "\n")


def graph():
    if not os.path.exists('model/speechhistory.pckl'):
        text.insert(END, "No training history found. Train the model first.\n")
        return

    with open('model/speechhistory.pckl', 'rb') as f:
        cnn_data = pickle.load(f)

    speech_accuracy = cnn_data['accuracy']
    speech_loss = cnn_data['loss']

    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.plot(speech_accuracy, 'o-', color='blue', label='Speech Emotion Accuracy')
    plt.plot(speech_loss, 'o-', color='red', label='Speech Emotion Loss')
    plt.legend(loc='upper left')
    plt.title('CNN Speech Emotion Accuracy & Loss Graph')
    plt.tight_layout()
    plt.show()


def exit_app():
    main.destroy()


# --- UI Layout ---
font = ('times', 13, 'bold')
font1 = ('times', 12, 'bold')

title = Label(main, text='EMOTION DETECTION USING SPEECH RECOGNITION')
title.config(bg='LightGoldenrod1', fg='medium orchid', font=font, height=3, width=120)
title.place(x=0, y=5)

text = Text(main, height=20, width=100)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=480, y=100)
text.config(font=font1)

Button(main, text="Upload Speech Emotion Dataset",      command=upload,                 font=font1).place(x=50, y=100)
Button(main, text="Preprocess Dataset",                 command=processDataset,         font=font1).place(x=50, y=150)
Button(main, text="Train Speech Emotion CNN Algorithm", command=trainSpeechCNN,         font=font1).place(x=50, y=200)
Button(main, text="Accuracy Comparison Graph",          command=graph,                  font=font1).place(x=50, y=250)
Button(main, text="Predict Speech Emotion",             command=predictSpeechExpression, font=font1).place(x=50, y=300)
Button(main, text="Exit",                               command=exit_app,               font=font1).place(x=50, y=350)

main.config(bg='OliveDrab2')
main.mainloop()
