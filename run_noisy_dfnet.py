import argparse
import df.deepfilternet3
import toml
import platform
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import scipy.signal
import tqdm
import os
from ptflops import get_model_complexity_info
import torch
from metrics import compute_dnsmos, compute_pesq, compute_distillmos, compute_xlsr_sqa_mos, compute_sisdr
from dataset import NoisySpeechDataset, load_paths
from models import SpeechEnhancementModel
import glob
import soundfile
from df import enhance, init_df

pathconfig = toml.load("directories.toml")

def main():
    parser = argparse.ArgumentParser(description="Train a speech enhancement model.")
    parser.add_argument(
        "--config", "-c", type=str, required=True, help="Path to the config file."
    )
    parser.add_argument(
        "--cuda_visible_devices",
        "-d",
        type=str,
        default="-1",
        help="CUDA visible devices.",
    )
    parser.add_argument(
        "--resume",
        "-r",
        action="store_true",
        help="Resume training from the latest checkpoint.",
    )

    args = parser.parse_args()

    resume = args.resume

    config = toml.load(args.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pathconfig = toml.load("directories.toml")

    paths = load_paths(pathconfig["DNS4_root"], pathconfig["VCTK_txt_root"])
    clean_train, clean_val, clean_test = paths["clean"]
    _, txt_val, txt_test = paths["txt"]
    noise_train, noise_val, noise_test = paths["noise"]
    rir_train, rir_val, rir_test = paths["rir"]

    np.random.seed(config["data_distribution_seed"])
    val_tensorboard_examples = np.random.choice(len(clean_val), 15, replace=False)
    np.random.seed()


    validation_dataset = NoisySpeechDataset(
        clean_val,
        noise_val,
        rir_val,
        duration=None,
        fs=48000,
        **config["data_distribution"],
    )

    test_dataset = NoisySpeechDataset(
        clean_test,
        noise_test,
        rir_test,
        duration=None,
        fs=48000,
        **config["data_distribution"],
    )

    val_dataloader = DataLoader(
        validation_dataset,
        1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    test_dataloader = DataLoader(
        test_dataset,
        1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    dfnet_model, df_state, _ = init_df()  # Load default model

    validate_dfnet_noisy(
        dfnet_model,
        df_state,
        val_dataloader,
        config,
        device,
        val_tensorboard_examples,
        txt_val,
    )

    test_dfnet_noisy(
        dfnet_model, df_state, test_dataloader, config, device, txt_test
                )

def validate_dfnet_noisy(
    dfnet_model,
    df_state,
    val_dataloader,
    config,
    device,
    val_tensorboard_examples,
    txt_val,
):
    sw_noisy = SummaryWriter(
        os.path.join(
            pathconfig["chkpt_logs_path"],
            "logs", "noisy"))
    sw_dfnet = SummaryWriter(
        os.path.join(
            pathconfig["chkpt_logs_path"],
            "logs", "dfnet"))
    
    np.random.seed(config["data_distribution_seed"])

    y_all, denoised_all, noisy_all = [], [], []

    for batch in tqdm.tqdm(val_dataloader):
        with torch.no_grad():
            y, m, rms = batch
            y16k, m16k = resample_batch(y, m, 48000, 16000)

            noisy_all.append(m16k.numpy()[0, :])
            y_g_hat = enhance(dfnet_model, df_state, m)
            
            denoised_all.append(
                scipy.signal.resample_poly(
                    y_g_hat[0, :].cpu().numpy(), 16000, 48000))
                
            y_all.append(y16k[0, :].numpy())

    lengths_all = np.array([y.shape[0] for y in y_all])
    
    # DFNet metrics
    pesq_all = compute_pesq(y_all, denoised_all, 16000)
    sisdr_all = compute_sisdr(y_all, denoised_all)
    ovr_all, sig_all, bak_all = compute_dnsmos(denoised_all, 16000)
    distillmos_all = compute_distillmos(denoised_all, 16000, device=device)
    xlsr_mos_all = compute_xlsr_sqa_mos(denoised_all, 16000, device=device)
    mean_pesq = np.mean(lengths_all * np.array(pesq_all)) / np.mean(lengths_all)
    mean_ovr = np.mean(lengths_all * np.array(ovr_all)) / np.mean(lengths_all)
    mean_sig = np.mean(lengths_all * np.array(sig_all)) / np.mean(lengths_all)
    mean_bak = np.mean(lengths_all * np.array(bak_all)) / np.mean(lengths_all)
    mean_distillmos = np.mean(lengths_all * np.array(distillmos_all)) / np.mean(lengths_all)
    mean_xlsr_mos = np.mean(lengths_all * np.array(xlsr_mos_all)) / np.mean(lengths_all)
    mean_sisdr = np.mean(lengths_all * np.array(sisdr_all)) / np.mean(lengths_all)

    sw_dfnet.add_scalar("PESQ", mean_pesq, 0)
    sw_dfnet.add_scalar("SI-SDR", mean_sisdr, 0)
    sw_dfnet.add_scalar("DNSMOS-OVR", mean_ovr, 0)
    sw_dfnet.add_scalar("DNSMOS-SIG", mean_sig, 0)
    sw_dfnet.add_scalar("DNSMOS-BAK", mean_bak, 0)
    sw_dfnet.add_scalar("DistillMOS", mean_distillmos, 0)
    sw_dfnet.add_scalar("XLS-R-MOS", mean_xlsr_mos, 0)

    for i in val_tensorboard_examples:
        sw_dfnet.add_audio(
            "%d" % i,
            torch.from_numpy(denoised_all[i]),
            global_step=0,
            sample_rate=16000,
        )

    sw_dfnet.close()

    # Noisy metrics
    noisy_pesq_all = compute_pesq(y_all, noisy_all, 16000)
    noisy_sisdr_all = compute_sisdr(y_all, noisy_all)
    noisy_ovr_all, noisy_sig_all, noisy_bak_all = compute_dnsmos(noisy_all, 16000)
    noisy_distillmos_all = compute_distillmos(noisy_all, 16000, device=device)
    noisy_xlsr_mos_all = compute_xlsr_sqa_mos(noisy_all, 16000, device=device)
    mean_noisy_pesq = np.mean(lengths_all * np.array(noisy_pesq_all)) / np.mean(lengths_all)
    mean_noisy_ovr = np.mean(lengths_all * np.array(noisy_ovr_all)) / np.mean(lengths_all)
    mean_noisy_sig = np.mean(lengths_all * np.array(noisy_sig_all)) / np.mean(lengths_all)
    mean_noisy_bak = np.mean(lengths_all * np.array(noisy_bak_all)) / np.mean(lengths_all)
    mean_noisy_distillmos = np.mean(lengths_all * np.array(noisy_distillmos_all)) / np.mean(lengths_all)
    mean_noisy_xlsr_mos = np.mean(lengths_all * np.array(noisy_xlsr_mos_all)) / np.mean(lengths_all)
    mean_noisy_sisdr = np.mean(lengths_all * np.array(noisy_sisdr_all)) / np.mean(lengths_all)

    sw_noisy.add_scalar("PESQ", mean_noisy_pesq, 0)
    sw_noisy.add_scalar("SI-SDR", mean_noisy_sisdr, 0)
    sw_noisy.add_scalar("DNSMOS-OVR", mean_noisy_ovr, 0)
    sw_noisy.add_scalar("DNSMOS-SIG", mean_noisy_sig, 0)
    sw_noisy.add_scalar("DNSMOS-BAK", mean_noisy_bak, 0)
    sw_noisy.add_scalar("DistillMOS", mean_noisy_distillmos, 0)
    sw_noisy.add_scalar("XLS-R-MOS", mean_noisy_xlsr_mos, 0)

    for i in val_tensorboard_examples:
        sw_noisy.add_audio(
            "%d" % i,
            torch.from_numpy(noisy_all[i]),
            global_step=0,
            sample_rate=16000,
        )
    sw_noisy.close()

    np.random.seed()


def resample_batch(y, m, fs_in, fs_out):
    y = torch.from_numpy(scipy.signal.resample_poly(y.cpu().numpy(), fs_out, fs_in, axis=1))
    m = torch.from_numpy(scipy.signal.resample_poly(m.cpu().numpy(), fs_out, fs_in, axis=1))
    return y, m


def test_dfnet_noisy(
    dfnet_model, df_state, test_dataloader, config, device, txt_test
):
    ## Synthetic data testing
    np.random.seed(config["data_distribution_seed"])
    print("starting to test")

    s_all, denoised_all, lengths_all, noisy_all = [], [], [], []
    os.makedirs(os.path.join(pathconfig["chkpt_logs_path"],
            "logs", "noisy", "synthetic_test"), exist_ok=True)
    os.makedirs(os.path.join(pathconfig["chkpt_logs_path"],
            "logs", "dfnet", "synthetic_test"), exist_ok=True)

    for batch in tqdm.tqdm(test_dataloader):
        with torch.no_grad():
            s, m, rms = batch
            s16k, m16k = resample_batch(s, m, 48000, 16000)

            noisy_all.append(m16k.numpy()[0, :])
            y = enhance(dfnet_model, df_state, m)

            denoised_all.append(
                scipy.signal.resample_poly(
                    y[0, :].cpu().numpy(), 16000, 48000))


            s_all.append(s16k[0, :].numpy())
            lengths_all.append(s16k.shape[1])

    lengths_all = np.array(lengths_all)
    

    # DFNet metrics
    distillmos_all = compute_distillmos(denoised_all, 16000, device=device)
    xlsrmos_all = compute_xlsr_sqa_mos(denoised_all, 16000, device=device)
    pesq_all = compute_pesq(s_all, denoised_all, 16000)
    sisdr_all = compute_sisdr(s_all, denoised_all)
    ovr_all, sig_all, bak_all = compute_dnsmos(denoised_all, 16000)

    mean_pesq = np.mean(lengths_all * np.array(pesq_all)) / np.mean(lengths_all)
    mean_sisdr = np.mean(lengths_all * np.array(sisdr_all)) / np.mean(lengths_all)
    mean_ovr = np.mean(lengths_all * np.array(ovr_all)) / np.mean(lengths_all)
    mean_sig = np.mean(lengths_all * np.array(sig_all)) / np.mean(lengths_all)
    mean_bak = np.mean(lengths_all * np.array(bak_all)) / np.mean(lengths_all)
    mean_distillmos = np.mean(lengths_all * np.array(distillmos_all)) / np.mean(lengths_all)
    mean_xlsr_sqa_mos = np.mean(lengths_all * np.array(xlsrmos_all)) / np.mean(lengths_all)

    with open(os.path.join(pathconfig["chkpt_logs_path"],
            "logs", "dfnet", "synthetic_test", "mean_metrics.txt"), "w") as f:
        f.write(
            "PESQ, SI-SDR, DNSMOS-OVR, DNSMOS-SIG, DNSMOS-BAK, DistillMOS, XLS-R-MOS\n"
            + "%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f"
            % (mean_pesq, mean_sisdr, mean_ovr, mean_sig, mean_bak, mean_distillmos, mean_xlsr_sqa_mos)
        )

    metrics = np.stack((pesq_all, sisdr_all, ovr_all, sig_all, bak_all, distillmos_all, xlsrmos_all)).T
    header = "Index, PESQ, SI-SDR, DNSMOS-OVR, DNSMOS-SIG, DNSMOS-BAK, DistillMOS, XLS-R-MOS"
    data = np.column_stack((np.arange(0, metrics.shape[0]), metrics))
    np.savetxt(
        os.path.join(pathconfig["chkpt_logs_path"],
            "logs", "dfnet", "synthetic_test", "metrics.csv"),
        data,
        fmt="%d, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f",
        delimiter=",",
        header=header,
    )

    # Noisy metrics
    noisy_pesq_all = compute_pesq(s_all, noisy_all, 16000)
    noisy_sisdr_all = compute_sisdr(s_all, noisy_all)
    noisy_ovr_all, noisy_sig_all, noisy_bak_all = compute_dnsmos(noisy_all, 16000)
    noisy_distillmos_all = compute_distillmos(noisy_all, 16000, device=device)
    noisy_xlsr_mos_all = compute_xlsr_sqa_mos(noisy_all, 16000, device=device)
    mean_noisy_pesq = np.mean(lengths_all * np.array(noisy_pesq_all)) / np.mean(lengths_all)
    mean_noisy_sisdr = np.mean(lengths_all * np.array(noisy_sisdr_all)) / np.mean(lengths_all)
    mean_noisy_ovr = np.mean(lengths_all * np.array(noisy_ovr_all)) / np.mean(lengths_all)
    mean_noisy_sig = np.mean(lengths_all * np.array(noisy_sig_all)) / np.mean(lengths_all)
    mean_noisy_bak = np.mean(lengths_all * np.array(noisy_bak_all)) / np.mean(lengths_all)
    mean_noisy_distillmos = np.mean(lengths_all * np.array(noisy_distillmos_all)) / np.mean(lengths_all)
    mean_noisy_xlsr_mos = np.mean(lengths_all * np.array(noisy_xlsr_mos_all)) / np.mean(lengths_all)

    with open(os.path.join(pathconfig["chkpt_logs_path"],
            "logs", "noisy", "synthetic_test", "mean_metrics.txt"), "w") as f:
        f.write(
            "PESQ, SI-SDR, DNSMOS-OVR, DNSMOS-SIG, DNSMOS-BAK, DistillMOS, XLS-R-MOS\n"
            + "%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f"
            % (mean_noisy_pesq, mean_noisy_sisdr, mean_noisy_ovr, mean_noisy_sig, mean_noisy_bak,
               mean_noisy_distillmos, mean_noisy_xlsr_mos)
        )
    metrics = np.stack((noisy_pesq_all, noisy_sisdr_all, noisy_ovr_all, noisy_sig_all, noisy_bak_all,
                        noisy_distillmos_all, noisy_xlsr_mos_all)).T
    header = "Index, PESQ, SI-SDR, DNSMOS-OVR, DNSMOS-SIG, DNSMOS-BAK, DistillMOS, XLS-R-MOS"
    data = np.column_stack((np.arange(0, metrics.shape[0]), metrics))
    np.savetxt(
        os.path.join(pathconfig["chkpt_logs_path"],
            "logs", "noisy", "synthetic_test", "metrics.csv"),
        data,
        fmt="%d, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f",
        delimiter=",",
        header=header,
    )

    ## Blind test
    dns4_test_root = os.path.join(pathconfig["DNS4_root"], "dev_testset")
    blind_test_files = glob.glob(os.path.join(dns4_test_root, "**/*.wav"), recursive=True)

    y_all, noisy_all, lengths_all = [], [], []
    for file in tqdm.tqdm(blind_test_files):
        with torch.no_grad():
            m, fs = soundfile.read(file, always_2d=True)
            m = m[:, 0]  # Take only one channel
            m48k = scipy.signal.resample_poly(m, 48000, fs)
            m16k = scipy.signal.resample_poly(m, 16000, fs)

            m48k = torch.from_numpy(m48k).float()

            y = enhance(dfnet_model, df_state, m48k.unsqueeze(0))
            

            y_all.append(
                scipy.signal.resample_poly(
                    y[0, :].cpu().numpy(), 16000, 48000))
            noisy_all.append(m16k)
            
            lengths_all.append(y.shape[1])

    lengths_all = np.array(lengths_all)

    os.makedirs(os.path.join(pathconfig["chkpt_logs_path"],
            "logs", "dfnet", "blind_test"), exist_ok=True)
    os.makedirs(os.path.join(pathconfig["chkpt_logs_path"],
            "logs", "noisy", "blind_test"), exist_ok=True)

    # DFNet metrics
    ovr_all, sig_all, bak_all = compute_dnsmos(y_all, 16000)

    mean_ovr = np.mean(lengths_all * np.array(ovr_all)) / np.mean(lengths_all)
    mean_sig = np.mean(lengths_all * np.array(sig_all)) / np.mean(lengths_all)
    mean_bak = np.mean(lengths_all * np.array(bak_all)) / np.mean(lengths_all)    

    distillmos_all = compute_distillmos(y_all, 16000, device=device)
    xlsrmos_all = compute_xlsr_sqa_mos(y_all, 16000, device=device)

    mean_distillmos = np.mean(lengths_all * np.array(distillmos_all)) / np.mean(lengths_all)
    mean_xlsr_sqa_mos = np.mean(lengths_all * np.array(xlsrmos_all)) / np.mean(lengths_all)

    with open(os.path.join(pathconfig["chkpt_logs_path"],
            "logs", "dfnet", "blind_test", "mean_metrics.txt"), "w") as f:
        f.write(
            "DNSMOS-OVR, DNSMOS-SIG, DNSMOS-BAK, DistillMOS, XLS-R-MOS\n"
            + "%.3f, %.3f, %.3f, %.3f, %.3f" % (mean_ovr, mean_sig, mean_bak, mean_distillmos, mean_xlsr_sqa_mos)
        )

    metrics = np.stack((ovr_all, sig_all, bak_all, distillmos_all, xlsrmos_all)).T
    header = "filepath, DNSMOS-OVR, DNSMOS-SIG, DNSMOS-BAK, DistillMOS, XLS-R-MOS"
    data = np.column_stack((np.array(blind_test_files, dtype=object), metrics))
    np.savetxt(
        os.path.join(pathconfig["chkpt_logs_path"],
            "logs", "dfnet", "blind_test", "metrics.csv"),
        data,
        fmt="%s, %.3f, %.3f, %.3f, %.3f, %.3f",
        delimiter=",",
        header=header,
    )

    # Noisy metrics
    noisy_ovr_all, noisy_sig_all, noisy_bak_all = compute_dnsmos(noisy_all, 16000)
    noisy_distillmos_all = compute_distillmos(noisy_all, 16000, device=device)
    noisy_xlsr_mos_all = compute_xlsr_sqa_mos(noisy_all, 16000, device=device)
    mean_noisy_ovr = np.mean(lengths_all * np.array(noisy_ovr_all)) / np.mean(lengths_all)
    mean_noisy_sig = np.mean(lengths_all * np.array(noisy_sig_all)) / np.mean(lengths_all)
    mean_noisy_bak = np.mean(lengths_all * np.array(noisy_bak_all)) / np.mean(lengths_all)
    mean_noisy_distillmos = np.mean(lengths_all * np.array(noisy_distillmos_all)) / np.mean(lengths_all)
    mean_noisy_xlsr_mos = np.mean(lengths_all * np.array(noisy_xlsr_mos_all)) / np.mean(lengths_all)

    with open(os.path.join(pathconfig["chkpt_logs_path"],
            "logs", "noisy", "blind_test", "mean_metrics.txt"), "w") as f:
        f.write(
            "DNSMOS-OVR, DNSMOS-SIG, DNSMOS-BAK, DistillMOS, XLS-R-MOS\n"
            + "%.3f, %.3f, %.3f, %.3f, %.3f"
            % (mean_noisy_ovr, mean_noisy_sig,
               mean_noisy_bak, mean_noisy_distillmos, mean_noisy_xlsr_mos)
        )
    metrics = np.stack((noisy_ovr_all, noisy_sig_all, noisy_bak_all,
                        noisy_distillmos_all, noisy_xlsr_mos_all)).T
    header = "filepath, DNSMOS-OVR, DNSMOS-SIG, DNSMOS-BAK, DistillMOS, XLS-R-MOS"
    data = np.column_stack((np.array(blind_test_files, dtype=object), metrics))
    np.savetxt(
        os.path.join(pathconfig["chkpt_logs_path"],
            "logs", "noisy", "blind_test", "metrics.csv"),
        data,
        fmt="%s, %.3f, %.3f, %.3f, %.3f, %.3f",
        delimiter=",",
        header=header,
    )



if __name__ == "__main__":
    main()
