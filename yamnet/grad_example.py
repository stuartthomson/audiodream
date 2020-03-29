import tensorflow as tf
import numpy as np

from scipy.io import wavfile

from yamnet import yamnet_frames_model, class_names
import params


def basic():
    x: tf.Tensor = tf.constant(3.0)
    with tf.GradientTape() as g:
        g.watch(x)
        y = x * x
    grad_y = g.gradient(y, x)
    print(grad_y)

def layers():
    model = tf.keras.layers.Dense(8)
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')
    input_audio = tf.constant([[1, 2, 3]], dtype=tf.float32)

    with tf.GradientTape() as g:
        g.watch(input_audio)
        human_speech_probability = output_layer(model(input_audio))
    gradient_of_speechiness = g.gradient(human_speech_probability, input_audio)
    print(f"human_speech_probability: {human_speech_probability}")
    print(f"gradient_of_speechiness: {gradient_of_speechiness}")
    learning_rate = 0.1
    input_audio_with_step = input_audio + (learning_rate * gradient_of_speechiness)
    print(f"input_audio_with_step: {input_audio_with_step}")
    print(f"new speechiness: {output_layer(model(input_audio_with_step))}")


def yamnet_grad_test():
    waveform=np.reshape(np.sin(2 * np.pi * 440 * np.linspace(0, 3, num=int(3 *16000))),
            [1, -1])

    print(waveform[0])
    wavfile.write('sine.wav', 16000, waveform[0])
    model = yamnet_frames_model(params)
    model.load_weights('yamnet.h5')
    classes = class_names('yamnet_class_map.csv')

    with tf.GradientTape() as grad_tape:
        audio_tensor = tf.convert_to_tensor(np.reshape(waveform, [1, -1]))
        print(f'Audio Tensor is: {type(audio_tensor)}')
        grad_tape.watch(audio_tensor)
        # scores, spectrograms = model.predict(audio_tensor, steps=1)
        scores, spectrograms = model(audio_tensor)
        print(f'Scores is: {type(scores)}')

        target_scores = scores.numpy()
        assert target_scores.shape == scores.shape
        target_scores[:, 0] = 1
        target_scores = tf.convert_to_tensor(target_scores)

        loss = tf.keras.losses.MSE(target_scores, scores)

    gradient_tensor = grad_tape.gradient(loss, audio_tensor)
    print(scores[0])
    print(classes[np.argsort(scores[0])[-3:]])
    print(gradient_tensor.shape)
    print(audio_tensor.shape)

    output_tensor = audio_tensor + 1000*gradient_tensor
    wavfile.write('speechy.wav', 16000, output_tensor[0].numpy())
    wavfile.write('grad.wav', 16000, 1000*gradient_tensor[0].numpy())

yamnet_grad_test()