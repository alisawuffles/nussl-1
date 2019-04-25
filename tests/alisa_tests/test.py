import os
from nussl import AudioSignal, DeepClustering, jupyter_utils, efz_utils
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas as pd
import musdb
from nussl.evaluation import BSSEvalV4
from scipy import signal


def main():
    matplotlib.use('Agg')
    # create_music_pairs()
    # create_urban_pairs()
    # create_voice_pairs()

    music_measures = calculate_measures(data='music', model='music')
    urban_measures = calculate_measures(data='urban', model='music')
    # voice_measures = calculate_measures(data='voice', model='music')
    measures = {'music': music_measures, 'urban': urban_measures}#, 'vocal': voice_measures}
    create_plots(measures)


def create_plots(scores_dict):
    # create boxplot of conf measures
    if len(scores_dict.keys()) >= 2:
        data = []
        for key in scores_dict:
            alpha = [row[0] for row in scores_dict[key]]
            data.append(alpha)
        fig, ax = plt.subplots()
        ax.boxplot(data)
        ax.set_xticklabels(scores_dict.keys())
        plt.savefig('figures/boxplot')

    for key in scores_dict:
        plt.figure()
        plt.scatter([row[0] for row in scores_dict[key]], [row[1] for row in scores_dict[key]])
        plt.title('Confidence measure versus SDR for ' + key + ' mixtures')
        plt.xlabel('Confidence measure')
        plt.ylabel('SDR')
        plt.savefig('figures/' + key + '_scatter')


def calculate_measures(data='music', model='music'):
    if data == 'music':
        path = 'mixtures/musdb'
    elif data == 'voice':
        path = 'mixtures/voice'
    elif data == 'urban':
        path = 'mixtures/UrbanSound8K'

    if model == 'music':
        model_path = efz_utils.download_trained_model('vocals_44k_coherent.pth')
    elif model == 'vocals':
        model_path = efz_utils.download_trained_model('speech_wsj8k.pth')

    scores = []
    titles = set([file[:-7] for file in os.listdir(path + '/paired_sources')])

    for title in titles:
        s0 = AudioSignal(path + '/paired_sources/' + title + '_s0.wav')
        s1 = AudioSignal(path + '/paired_sources/' + title + '_s1.wav')
        sources = [s0, s1]
        mixture = sources[0] + sources[1]

        separator = DeepClustering(
            mixture,
            model_path=model_path,
            num_sources=2,
            percentile=0,
            clustering_type='kmeans',
        )

        separator.run()
        estimates = separator.make_audio_signals()

        # save_estimates(estimates, 'estimates/musdb/' + title)

        sdr = get_sdr(sources, estimates)
        alpha = get_alpha(separator.confidence.flatten())
        scores.append((alpha, sdr))

    return scores


def save_estimates(estimates, path):
    for i, e in enumerate(estimates):
        e.write_audio_to_file(path + '_' + f'est{i}.wav')


def mixture_plots(separator, mixture_title):
    # plots for a particular mixture separation

    # embedding plots and spectrograms
    plt.figure(figsize=(30, 15))
    separator.plot()
    plt.tight_layout()
    plt.savefig('figures/' + mixture_title + '_fig')

    # plot histogram of confidence measure
    plt.figure(figsize=(20,20))
    plt.hist(separator.confidence.flatten(), facecolor='gray')
    plt.ylim((0, 400000))
    plt.title('Confidence measure')
    plt.savefig('figures/' + mixture_title + '_hst')


def get_sdr(sources, estimates):
    # higher of mean SDR score for each estimate
    orderings = list(itertools.permutations(estimates))     # all possible ordering of estimates (there are 2)

    scores = []     # contains SDR for two possible orderings
    for _estimates in orderings:
        evaluator = BSSEvalV4(sources, list(_estimates), compute_permutation=False, win=2.0, hop=1.5)
        _scores = evaluator.evaluate()
        scores.append(_scores)

    best_score_index = np.argmax([np.nanmean(x['raw_values']['SDR']) for x in scores])
    best_score = scores[best_score_index]
    sdr = [np.round(np.nanmean(best_score['raw_values']['SDR'][0]), 3),
           np.round(np.nanmean(best_score['raw_values']['SDR'][1]), 3)]

    return sdr[0]


def get_alpha(alphas, weighted=False, mixture=None):
    if weighted is False:
        return np.nanmean(alphas)
    else:
        magnitude_spectrogram = np.abs(signal.stft(mixture.audio_data))
        weights = magnitude_weights(magnitude_spectrogram).flatten()
        return np.average(alphas, weights=weights)


def magnitude_weights(magnitude_spectrogram):
    weights = magnitude_spectrogram / (np.sum(magnitude_spectrogram) + 1e-6)
    weights *= (magnitude_spectrogram.shape[0] * magnitude_spectrogram.shape[1])
    return np.log10(weights + 1.0)


def create_voice_pairs():
    path = 'sources/voice/'
    titles = [title for title in os.listdir(path)]

    for pair in itertools.combinations(titles, 2):
        # convert each source to mono
        source0 = AudioSignal(path + pair[0], sample_rate=44100, offset=20, duration=10).to_mono(overwrite=True)
        source1 = AudioSignal(path + pair[1], sample_rate=44100, offset=20, duration=10).to_mono(overwrite=True)
        # write to file and update path
        source0.write_audio_to_file('mixtures/voice/paired_sources/' + pair[0] + '_' + pair[1] + '_s0.wav')
        source0.path_to_input_file = 'mixtures/voice/paired_sources/' + pair[0] + '_' + pair[1] + '_s0.wav'
        source1.write_audio_to_file('mixtures/voice/paired_sources/' + pair[0] + '_' + pair[1] + '_s1.wav')
        source1.path_to_input_file = 'mixtures/voice/paired_sources/' + pair[0] + '_' + pair[1] + '_s1.wav'


def create_urban_pairs():
    file_path = 'sources/UrbanSound8K/'
    df = pd.read_csv(file_path + 'metadata/UrbanSound8K.csv', header=0)
    fold1 = df.loc[df['fold'] == 1]             # keep only rows for fold 1
    files = list(fold1['slice_file_name'])      # list of file names

    mixture_ct = 50         # number of mixtures to create
    idx = 0

    # for every pair of file names
    for pair in itertools.combinations(files, 2):
        # if source type is the same, skip this pairing
        if list(pair[0])[6] == list(pair[1])[6]:
            continue

        # convert each source to mono
        source0 = AudioSignal(file_path + 'audio/fold1/' + pair[0], sample_rate=44100).to_mono(overwrite=True)
        source1 = AudioSignal(file_path + 'audio/fold1/' + pair[1], sample_rate=44100).to_mono(overwrite=True)
        # change to same length
        source0 = AudioSignal(audio_data_array=source0[:np.min((len(source0), len(source1)))])
        source1 = AudioSignal(audio_data_array=source1[:np.min((len(source0), len(source1)))])
        # update path to input file
        source0.path_to_input_file = 'sources/UrbanSound8K/audio/fold1/' + pair[0]
        source1.path_to_input_file = 'sources/UrbanSound8K/audio/fold1/' + pair[1]
        # write to file
        source0.write_audio_to_file('mixtures/UrbanSound8K/paired_sources/' + str(idx) + '_s0.wav')
        source1.write_audio_to_file('mixtures/UrbanSound8K/paired_sources/' + str(idx) + '_s1.wav')

        idx += 1
        if idx > mixture_ct:
            break


def create_music_pairs():
    mus = musdb.DB(root_dir='mixtures/musdb')
    tracks = mus.load_mus_tracks()

    mixture_ct = 10
    idx = 0

    for track in tracks:
        print(track)
        duration = range(20*track.rate, 30*track.rate)
        s0 = AudioSignal(audio_data_array=track.targets['vocals'].audio[duration])
        s1 = AudioSignal(audio_data_array=track.targets['accompaniment'].audio[duration]) \
             + AudioSignal(audio_data_array=track.targets['drums'].audio[duration]) \
             + AudioSignal(audio_data_array=track.targets['bass'].audio[duration]) \
             + AudioSignal(audio_data_array=track.targets['other'].audio[duration])
        s0.write_audio_to_file('mixtures/musdb/paired_sources/' + track.name + '_s0.wav')
        s1.write_audio_to_file('mixtures/musdb/paired_sources/' + track.name + '_s1.wav')
        s0.path_to_input_file = 'mixtures/musdb/paired_sources/' + track.name + '_s0.wav'
        s1.path_to_input_file = 'mixtures/musdb/paired_sources/' + track.name + '_s1.wav'

        idx += 1
        if idx > mixture_ct:
            break


def create_my_mixtures():
    vocal_instrumental_mixtures = []

    file_path = 'sources/VoiceInstrumental/'
    for v_file in os.listdir(file_path + 'voice'):
        for i_file in os.listdir(file_path + 'instrumental'):
            if v_file[0] == '.' or i_file[0] == '.':
                continue
            v = AudioSignal(file_path + 'voice/' + v_file, offset=20, duration=10, sample_rate=44100)
            i = AudioSignal(file_path + 'instrumental/' + i_file, offset=20, duration=10, sample_rate=44100)
            vocal = AudioSignal(audio_data_array=v.audio_data[:np.min((len(v), len(i)))])
            instrumental = AudioSignal(audio_data_array=i.audio_data[:np.min((len(v), len(i)))])
            vocal.path_to_input_file = 'sources/VoiceInstrumental/voice/' + v_file
            instrumental.path_to_input_file = 'sources/VoiceInstrumental/voice/' + i_file
            vocal_instrumental_mixtures.append([vocal, instrumental])

    return vocal_instrumental_mixtures


if __name__ == '__main__':
    main()
