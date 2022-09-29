# Import library
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from glob import glob

import librosa as lr
import librosa.display
import IPython.display as ipd

from itertools import cycle

# Style
sns.set_theme(style="white", palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])


# List all the wav files in the folder
audio_files = glob('datasets/audio_data/set_b/*.wav' )



# Play audio file
ipd.Audio(audio_files[5])



# Read in the first audio file, create the time array
audio, sfreq_or_sample_rate = lr.load(audio_files[0])
time = np.arange(0, len(audio)) /  sfreq_or_sample_rate # time gives a array of time gape of each heartbeat

print(time) 



# Plot audio over time
fig, ax = plt.subplots()
ax.plot(time , audio)
ax.set(xlabel='Time (s)', ylabel='Sound Amplitude')
plt.show()


pd.Series(audio).plot(figsize=(10, 4),
                  lw=1,
                  title='Raw Audio Example',
                 color=color_pal[0])
plt.show()


# Trimming leading/lagging silence
y_trimmed, _ = librosa.effects.trim(audio, top_db=20)
pd.Series(y_trimmed).plot(figsize=(10, 4),
                  lw=1,
                  title='Raw Audio Trimmed Example',
                 color=color_pal[1])
plt.show()


pd.Series(audio[30000:31000]).plot(figsize=(10, 5),
                  lw=1,
                  title='Raw Audio Zoomed In Example',
                 color=color_pal[2])
plt.show()


D = librosa.stft(audio)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max) # converting aplitude to decible


# Plot the transformed audio data
fig, ax = plt.subplots(figsize=(10, 5))
img = librosa.display.specshow(S_db,
                              x_axis='time',
                              y_axis='log',
                              ax=ax)
ax.set_title('Spectogram Example', fontsize=20)
fig.colorbar(img, ax=ax, format=f'get_ipython().run_line_magic("0.2f')", "")
plt.show()


S = librosa.feature.melspectrogram(y=audio,
                                   sr=sfreq_or_sample_rate,
                                   n_mels=128 * 2,)
S_db_mel = librosa.amplitude_to_db(S, ref=np.max)


fig, ax = plt.subplots(figsize=(10, 5))
# Plot the mel spectogram
img = librosa.display.specshow(S_db_mel,
                              x_axis='time',
                              y_axis='log',
                              ax=ax)
ax.set_title('Mel Spectogram Example', fontsize=20)
fig.colorbar(img, ax=ax, format=f'get_ipython().run_line_magic("0.2f')", "")
plt.show()






