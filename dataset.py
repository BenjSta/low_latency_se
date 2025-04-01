import numpy as np
import webrtcvad
import soundfile
import scipy.signal
from torch.utils import data
import torch
import glob
import os
import tqdm

def spectral_augment(x):
    def random_coeff():
        return np.random.uniform(-3 / 8, 3 / 8)

    b = [1, random_coeff(), random_coeff()]
    a = [1, random_coeff(), random_coeff()]
    return [scipy.signal.lfilter(b, a, y, axis=0) for y in x]


def load_paths(dns4_root_datasets_fullband, vctk_txt_root):
    VOCALSET_PATH = os.path.join(
        dns4_root_datasets_fullband, "clean_fullband/VocalSet_48kHz_mono"
    )

    VCTK_VAL_PATHS = [
        os.path.join(
            dns4_root_datasets_fullband,
            "clean_fullband/vctk_wav48_silence_trimmed/" + spk,
        )
        for spk in ["p225", "p226", "p227", "p228"]
    ]

    VCTK_TEST_PATHS = [
        os.path.join(
            dns4_root_datasets_fullband,
            "clean_fullband/vctk_wav48_silence_trimmed/" + spk,
        )
        for spk in ["p229", "p230", "p232", "p237"]
    ]

    training_filelist = glob.glob(
        os.path.join(dns4_root_datasets_fullband, "clean_fullband/**/*.wav"),
        recursive=True,
    )
    training_filelist = [
        p
        for p in training_filelist
        if not VOCALSET_PATH in p
        or any(v in p for v in VCTK_VAL_PATHS)
        or any(v in p for v in VCTK_TEST_PATHS)
    ]

    np.random.seed(1)
    validation_filelist = []
    for p in VCTK_VAL_PATHS:
        validation_filelist += np.random.choice(
            sorted(glob.glob(os.path.join(p, "*.wav"), recursive=True)),
            60,
            replace=False,
        ).tolist()
    np.random.seed()

    val_texts = []
    for v in validation_filelist:
        (dir, file) = os.path.split(v)
        (_, speakerdir) = os.path.split(dir)
        textfile = os.path.join(vctk_txt_root, speakerdir, file[:-9] + ".txt")
        with open(textfile, "r") as f:
            text = f.read()
            val_texts.append(text[:-1])

    np.random.seed(1)
    test_filelist = []
    for p in VCTK_TEST_PATHS:
        test_filelist +=  sorted(glob.glob(os.path.join(p, "*.wav")))

    np.random.seed()

    test_texts = []
    for v in test_filelist:
        (dir, file) = os.path.split(v)
        (_, speakerdir) = os.path.split(dir)
        textfile = os.path.join(vctk_txt_root, speakerdir, file[:-9] + ".txt")
        with open(textfile, "r") as f:
            text = f.read()
            test_texts.append(text[:-1])

    pathlist_noise = glob.glob(
        os.path.join(dns4_root_datasets_fullband, "noise_fullband/*.wav"),
        recursive=True,
    )
    pathlist_noise = sorted(pathlist_noise)
    noise_filepath_list_new = []
    durations = []
    total_noise_available = 0
    for f in tqdm.tqdm(pathlist_noise):
        info = soundfile.info(f)
        dur = info.frames / info.samplerate
        total_noise_available += dur
        if dur > 6.0:
            noise_filepath_list_new.append(f)
            durations.append(dur)
    print("Total available amount of noise: %g hours" % (total_noise_available / 3600))

    noise_filepath_list = noise_filepath_list_new
    np.random.seed(1)
    noise_filepath_list = np.random.choice(
        noise_filepath_list, len(noise_filepath_list), replace=False
    )
    np.random.seed()
    noise_filepath_list_train = noise_filepath_list[:-4400]
    noise_filepath_list_val = noise_filepath_list[-4400:-4000]
    noise_filepath_list_test = noise_filepath_list[-4000:]

    pathlist_rir1 = glob.glob(
        os.path.join(
            dns4_root_datasets_fullband,
            "impulse_responses/SLR26/simulated_rirs_48k/smallroom/**/*.wav",
        ),
        recursive=True,
    )
    pathlist_rir2 = glob.glob(
        os.path.join(
            dns4_root_datasets_fullband,
            "impulse_responses/SLR26/simulated_rirs_48k/mediumroom/**/*.wav",
        ),
        recursive=True,
    )
    pathlist_rir = sorted(pathlist_rir1) + sorted(pathlist_rir2)
    np.random.seed(1)
    pathlist_rir = np.random.choice(pathlist_rir, len(pathlist_rir), replace=False)
    np.random.seed()
    pathlist_rir_train = pathlist_rir[:-300]
    pathlist_rir_val = pathlist_rir[-300:-150]
    pathlist_rir_test = pathlist_rir[:-150:]

    return {
        "clean": [training_filelist, validation_filelist, test_filelist],
        "txt": [None, val_texts, test_texts],
        "noise": [
            noise_filepath_list_train,
            noise_filepath_list_val,
            noise_filepath_list_test,
        ],
        "rir": [pathlist_rir_train, pathlist_rir_val, pathlist_rir_test],
    }


def compute_active_speech_rms(x, fs):
    assert fs == 16000
    VAD_BLOCKSIZE = 480
    MINIMUM_PERCENTAGE_ACTIVE_SPEECH = 0.1

    x16bit = (x * 32768).astype("int16")
    index = 0
    is_speech = []

    vad = webrtcvad.Vad(3)
    while index < x16bit.shape[0] - VAD_BLOCKSIZE:
        framebytes = x16bit[index : index + VAD_BLOCKSIZE].tobytes()
        is_speech += [vad.is_speech(framebytes, 16000)] * VAD_BLOCKSIZE
        index += VAD_BLOCKSIZE
        # print(index)

    is_speech = np.array(is_speech).astype("float32")
    is_speech = np.pad(is_speech, (0, x.shape[0] - is_speech.shape[0]))

    if np.mean(is_speech) > MINIMUM_PERCENTAGE_ACTIVE_SPEECH:
        speech_rms = np.sqrt(np.sum(x**2 * is_speech) / (np.sum(is_speech)))
    else:
        speech_rms = np.nan

    return speech_rms, is_speech


def generate_noisy_speech_sample(
    speech_filepath,
    noise_filepath,
    rir_filepath,
    duration,
    snr_mu_and_sigma,
    speech_rms_mu_and_sigma,
    apply_noise,
    rir_percentage,
    fs,
):
    if duration == None:
        s, fs_speech = soundfile.read(speech_filepath, dtype="float32")
        duration = s.shape[0] / fs_speech
        target_len = int(np.floor(duration * fs))
    else:
        target_len = int(np.floor(fs * duration))
        speech_info = soundfile.info(speech_filepath)
        speech_len = speech_info.frames

        target_len_speech = int(np.floor(speech_info.samplerate * duration))
        speech_start_index = np.random.randint(
            np.maximum(speech_len - target_len_speech, 0) + 1
        )

        s, fs_speech = soundfile.read(
            speech_filepath,
            frames=target_len_speech,
            start=speech_start_index,
            dtype="float32",
        )

    s = scipy.signal.resample_poly(s, fs, fs_speech)
    s = s[:target_len]

    b, a = scipy.signal.butter(2, 50, "high", fs=fs)
    s = scipy.signal.lfilter(b, a, s)

    if s.shape[0] < target_len:
        offset = np.random.randint(target_len - s.shape[0])
        s = np.pad(s, ((offset, target_len - s.shape[0] - offset),))

    if np.random.uniform() < rir_percentage:
        rir, rir_fs = soundfile.read(rir_filepath, dtype="float32", always_2d=True)
        rir = rir[:, 0]
        rir = scipy.signal.resample_poly(rir, fs, rir_fs, axis=0)

        t0_ind = np.argmax(np.abs(rir), axis=0)
        t_ind = np.arange(rir.shape[0])
        t0 = t0_ind / fs
        t = t_ind / fs
        w_rir = np.ones((rir.shape[0],))
        w_rir[t > t0] = 10 ** (3 * (-(t[t > t0] - t0) / 0.15))
        rir_mod = w_rir * rir
        s_target = scipy.signal.fftconvolve(s, rir_mod)[: s.shape[0]]

        s = scipy.signal.fftconvolve(s, rir)[: s.shape[0]]
    else:
        s_target = s.copy()

    s_target_norm_factor = 1 / max(np.max(np.abs(s_target)), 0.1)
    s_target_norm = s_target * s_target_norm_factor
    s_target_rms = (
        compute_active_speech_rms(
            scipy.signal.resample_poly(s_target_norm, 16000, fs), 16000
        )[0]
        / s_target_norm_factor
    )

    s_target_rms = max(s_target_rms, 0.01)

    enough_active_speech = not np.isnan(s_target_rms)
    if not enough_active_speech:
        s_target_rms = (10 ** (-18 / 20)) / s_target_norm_factor

    # truncated normal disttribution
    while True:
        target_speech_rms = np.random.normal(
            speech_rms_mu_and_sigma[0], speech_rms_mu_and_sigma[1]
        )
        if (
            target_speech_rms
            > speech_rms_mu_and_sigma[0] - 2 * speech_rms_mu_and_sigma[1]
            and target_speech_rms
            < speech_rms_mu_and_sigma[0] + 2 * speech_rms_mu_and_sigma[1]
        ):
            break

    factor = 10 ** ((target_speech_rms - 20 * np.log10(s_target_rms)) / 20)

    s_target *= factor
    s *= factor

    if apply_noise:
        noise_info = soundfile.info(noise_filepath)
        noise_len = noise_info.frames
        target_len_noise = int(np.floor(noise_info.samplerate * duration))

        noise_start_index = np.random.randint(
            np.clip(noise_len - target_len_noise + 1, 1, np.inf)
        )

        n, fs_noise = soundfile.read(
            noise_filepath,
            frames=target_len_noise,
            start=noise_start_index,
            dtype="float32",
        )
        n = scipy.signal.resample_poly(n, fs, fs_noise)
        n = n[:target_len]

        if n.shape[0] < target_len:
            n = np.concatenate([n, n])

        n = n[:target_len]

        # truncated normal distribution
        while True:
            target_snr = np.random.normal(snr_mu_and_sigma[0], snr_mu_and_sigma[1])
            if (
                target_snr > snr_mu_and_sigma[0] - 2 * snr_mu_and_sigma[1]
                and target_snr < snr_mu_and_sigma[0] + 2 * snr_mu_and_sigma[1]
            ):
                break

        noise_rms = np.maximum(np.sqrt(np.mean(n**2)), 1e-10)
        snr = target_speech_rms - 20 * np.log10(noise_rms)
        factor = 10 ** ((snr - target_snr) / 20)
        n *= factor
        m = s + n
    else:
        m = s

    return (
        torch.clip(torch.from_numpy(s_target.astype("float32")), -1, 1),
        torch.clip(torch.from_numpy(m.astype("float32")), -1, 1),
        10 ** (target_speech_rms / 20),
    )


class NoisySpeechDataset(data.Dataset):
    def __init__(
        self,
        speech_filepath_list,
        noise_filepath_list,
        rir_filepath_list,
        duration,
        snr_mu_and_sigma,
        speech_rms_mu_and_sigma,
        apply_noise,
        rir_percentage,
        fs,
        weighted_sampling=False,
    ):
        self.speech_filepath_list = speech_filepath_list
        self.noise_filepath_list = noise_filepath_list
        self.rir_filepath_list = rir_filepath_list
        self.duration = duration
        self.snr_mu_and_sigma = snr_mu_and_sigma
        self.speech_rms_mu_and_sigma = speech_rms_mu_and_sigma
        self.apply_noise = apply_noise
        self.rir_percentage = rir_percentage
        self.fs = fs

        self.length = len(speech_filepath_list)
        if weighted_sampling:
            self.weights = np.array(
                [soundfile.info(f).frames for f in tqdm.tqdm(self.speech_filepath_list)]
            )
            self.weights = self.weights / np.sum(self.weights)
        else:
            self.weights = None

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if self.weights is not None:
            # do not use item from dataloader sampler, but sample randomly
            item = np.random.choice(self.length, p=self.weights)

        speech_filepath = self.speech_filepath_list[item]
        ind = np.random.randint(len(self.noise_filepath_list))
        if self.apply_noise:
            noise_filepath = self.noise_filepath_list[ind]
        else:
            noise_filepath = None

        ind = np.random.randint(len(self.rir_filepath_list))
        rir_filepath = self.rir_filepath_list[ind]

        while True:
            try:
                s, m, rms = generate_noisy_speech_sample(
                    speech_filepath,
                    noise_filepath,
                    rir_filepath,
                    self.duration,
                    self.snr_mu_and_sigma,
                    self.speech_rms_mu_and_sigma,
                    self.apply_noise,
                    self.rir_percentage,
                    self.fs,
                )
                break
            except Exception as e:
                print(
                    "Error generating sample from %s, %s, %s: %s"
                    % (speech_filepath, noise_filepath, rir_filepath, str(e))
                )
                print("Retrying...")

        return s, m, rms
