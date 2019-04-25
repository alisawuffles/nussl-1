import os
from nussl import AudioSignal, DeepClustering, jupyter_utils, efz_utils
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from nussl.evaluation import BSSEvalV4
import json
import numpy as np
import itertools
import pandas as pd
import musdb


def main():
    music_measures = get_music_measures()
    # matplotlib.use('Agg')
    # create_urban_mixtures()
    # measures_urban = get_urban_measures()
    create_plots(music_measures)


def create_plots(measures):
    # create boxplot of conf measures
    # alpha = [row[0] for row in measures]
    # plt.boxplot(alpha)
    # ax.set_xticklabels(('music', 'urban'))
    # plt.savefig('figures/boxplot')

    plt.scatter([row[0] for row in measures], [row[1] for row in measures])
    plt.title('Confidence measure versus SDR for urban sound mixtures')
    plt.xlabel('Confidence measure')
    plt.ylabel('SDR')
    plt.savefig('figures/scatter')


def get_music_measures(model='music'):
    mus = musdb.DB(root_dir='mixtures/musdb18')
    tracks = mus.load_mus_tracks()

    measures = []
    if model == 'music':
        model_path = efz_utils.download_trained_model('vocals_44k_coherent.pth')
    elif model == 'vocals':
        model_path = efz_utils.download_trained_model('speech_wsj8k.pth')

    mixture_ct = 10
    idx = 0

    for track in tracks:
        print(track)
        s0 = AudioSignal(audio_data_array=track.targets['vocals'].audio)
        s1 = AudioSignal(audio_data_array=track.targets['accompaniment'].audio) \
             + AudioSignal(audio_data_array=track.targets['drums'].audio) \
             + AudioSignal(audio_data_array=track.targets['bass'].audio) \
             + AudioSignal(audio_data_array=track.targets['other'].audio)
        s0.write_audio_to_file('mixtures/musdb/paired_sources/' + track + '_s0.wav')
        s1.write_audio_to_file('mixtures/musdb/paired_sources' + track + '_s1.wav')

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
        save_estimates(estimates, 'estimates/musdb/' + str(idx))

        sdr = get_sdr(sources, estimates)
        alpha = get_alpha(separator.confidence.flatten())

        idx += 1
        if idx > mixture_ct:
            break



def get_urban_measures(model='music'):
    # returns an n x 2 array where rows are mixtures, and columns are alpha and SDR
    path = 'mixtures/UrbanSound8K'

    measures = []
    if model == 'music':
        model_path = efz_utils.download_trained_model('vocals_44k_coherent.pth')
    elif model == 'vocals':
        model_path = efz_utils.download_trained_model('speech_wsj8k.pth')

    for i in range(int(len(os.listdir(path + '/paired_sources'))/2)):
        s0 = AudioSignal(path + '/paired_sources/' + str(i) + '_s0.wav')
        s1 = AudioSignal(path + '/paired_sources/' + str(i) + '_s1.wav')
        sources = [s0, s1]

        mixture = sources[0] + sources[1]
        mixture.write_audio_to_file(path + '/' + str(i) + '_mixture.wav')

        separator = DeepClustering(
            mixture,
            model_path=model_path,
            num_sources=2,
            percentile=0,
            clustering_type='kmeans',
        )

        separator.run()
        estimates = separator.make_audio_signals()
        save_estimates(estimates, 'estimates/UrbanSound8K/' + str(i))

        sdr = get_sdr(sources, estimates)
        alpha = get_alpha(separator.confidence.flatten())
        measures.append((alpha, sdr))

    return measures


def create_urban_mixtures():
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


def save_estimates(estimates, path):
    for i, e in enumerate(estimates):
        e.write_audio_to_file(path + '_' + f'est{i}.wav')


def get_sdr(sources, estimates, type=1):
    # type 1: higher of mean SDR score for each estimate

    orderings = list(itertools.permutations(estimates))     # all possible ordering of estimates (there are 2)

    scores = []         # contains SDR for two possible orderings
    for _estimates in orderings:
        evaluator = BSSEvalV4(sources, list(_estimates), compute_permutation=False, win=2.0, hop=1.5)
        _scores = evaluator.evaluate()
        scores.append(_scores)

    best_score_index = np.argmax([np.nanmean(x['raw_values']['SDR']) for x in scores])
    best_score = scores[best_score_index]
    sdr = [np.round(np.nanmean(best_score['raw_values']['SDR'][0]), 3),
           np.round(np.nanmean(best_score['raw_values']['SDR'][1]), 3)]

    if type == 1:
        return sdr[0]


def get_alpha(alphas, type='mean'):
    if type == 'mean':
        return np.nanmean(alphas)


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


if __name__ == '__main__':
    main()
