import argparse
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
from losses import combined_loss, MultiResSpecLoss, ComplexCompressedSpecMSE
import glob
import soundfile


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
    parser.add_argument(
        "--validate_no_training",
        "-ov",
        action="store_true",
        help="Validate the model only.",
    )
    parser.add_argument(
        "--test_no_training",
        "-ot",
        action="store_true",
        help="Test the model only.",
    )

    args = parser.parse_args()

    config = toml.load(args.config)
    config['train_name'] = os.path.basename(args.config).split('.')[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pathconfig = toml.load("directories.toml")

    denoise_net = SpeechEnhancementModel(**config["model"])
    macs, params = get_model_complexity_info(
        denoise_net, (20 * 16000,), as_strings=False, print_per_layer_stat=True
    )

    if config["model"]["method"] == "complex_filter": 
        macs = (macs / 20 + #NN cost
            config["fs"] / config["model"]["hopsize"] * (
                4 * (config["model"]["winlen"] // 2 + 1) * denoise_net.num_filter_frames + #complex multiply per bin 
                2 * (config["model"]["winlen"] // 2 + 1) * config["model"]["winlen"] + # forward transform
                2 * (config["model"]["winlen"] // 2 + 1) * config["model"]["hopsize"] * 2) +  # inverse transform)
            config["fs"] / config["model"]["hopsize"] / denoise_net.downsample_factor * 6 * (config["model"]["winlen"] // 2 + 1)) # mapping to complex plane

    elif config["model"]["method"] == "time_domain_filtering":
        macs = (macs / 20 + #NN cost
            config["fs"] / config["model"]["hopsize"] * (
                2 * (config["model"]["winlen"] // 2 + 1) * config["model"]["winlen"]) + # forward transform
            2 * config["fs"] * denoise_net.filtlen * 0.5) # two parallel time-domain convolutions,
        #with 0.5 for possible optimization by non-uniformly partitioned convolutions
    
    print("GMacs/s", macs / 10**9)
    print("M Params", params / 10**6)
    denoise_net.to(device)

    resume = args.resume
    validate_only = args.validate_no_training
    test_only = args.test_no_training

    paths = load_paths(pathconfig["DNS4_root"], pathconfig["VCTK_txt_root"])
    clean_train, clean_val, clean_test = paths["clean"]
    _, txt_val, txt_test = paths["txt"]
    noise_train, noise_val, noise_test = paths["noise"]
    rir_train, rir_val, rir_test = paths["rir"]

    np.random.seed(config["data_distribution_seed"])
    val_tensorboard_examples = np.random.choice(len(clean_val), 15, replace=False)
    np.random.seed()

    trainset = NoisySpeechDataset(
        clean_train,
        noise_train,
        rir_train,
        duration=config["train_seq_duration"],
        fs=config["fs"],
        **config["data_distribution"],
    )

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

    worker_init_fn = None
    if platform.system() == "Linux":
        worker_init_fn = lambda ind=None: np.random.seed(
            int.from_bytes(os.urandom(4), byteorder="little")
        )

    train_dataloader = DataLoader(
        trainset,
        config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
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

    

    optim = torch.optim.AdamW(
        denoise_net.parameters(),
        config["lr_start"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optim,
        config["lr_update_interval"],
        gamma=(config["lr_stop"] / config["lr_start"]) ** (1 / (config["max_steps"]/config["lr_update_interval"])),
    )

    chkpt_dir = os.path.join(
        pathconfig["chkpt_logs_path"], "checkpoints", config["train_name"]
    )
    os.makedirs(chkpt_dir, exist_ok=True)

    if resume:
        try:
            chkpt = torch.load(os.path.join(chkpt_dir, "latest"), weights_only=False)
            optim.load_state_dict(chkpt["optim"])
            denoise_net.load_state_dict(chkpt["denoiser"])
            scheduler.load_state_dict(chkpt["scheduler"])
            best_metric, steps = chkpt["best_metric"], chkpt["steps"]
        except FileNotFoundError:
            print("Checkpoint file not found. Starting from scratch.")
            best_metric, steps = -np.inf, 0
    else:
        best_metric, steps = -np.inf, 0

    log_dir = os.path.join(pathconfig["chkpt_logs_path"], "logs", config["train_name"])
    os.makedirs(log_dir, exist_ok=True)
    sw = SummaryWriter(log_dir)

    if config["loss"]["loss_type"] == 'combined':
        loss_fn = combined_loss(
            config["loss"]["winlen"],
            config["fs"],
            config["loss"]["f_fade_out_complex_low"],
            config["loss"]["f_fade_out_complex_high"],
            config["loss"]["complex_weight"],
            n_ffts=config["loss"]["mres_fft_sizes"],
            gamma=config["loss"]["mres_gamma"],
            mrspec_lambda=config["loss"]["mres_factor"],
            ccmse_lambda=config["loss"]["spec_factor"]
        ).to(device)
        #loss_fn = torch.jit.script(loss_fn, example_inputs=(torch.randn(1, 16000).to(device), torch.randn(1, 16000).to(device)))
    
    elif config["loss"]["loss_type"] == 'multires':
        loss_fn = MultiResSpecLoss(
            config["loss"]["mres_fft_sizes"],
            gamma=config["loss"]["mres_gamma"],
            factor=config["loss"]["mres_factor"],
            f_complex=config["loss"]["spec_factor"]
        ).to(device)
        #loss_fn = torch.jit.script(loss_fn, example_inputs=(torch.randn(1, 16000).to(device), torch.randn(1, 16000).to(device)))
    elif config["loss"]["loss_type"] == 'ccmse':
        loss_fn = ComplexCompressedSpecMSE(
            config["loss"]["winlen"],
            config["fs"],
            config["loss"]["f_fade_out_complex_low"],
            config["loss"]["f_fade_out_complex_high"],
            config["loss"]["complex_weight"]
        ).to(device)
        #loss_fn = torch.jit.script(loss_fn, example_inputs=(torch.randn(1, 16000).to(device), torch.randn(1, 16000).to(device)))

    if validate_only:
        validate(
            denoise_net,
            val_dataloader,
            config,
            device,
            sw,
            steps,
            val_tensorboard_examples,
            txt_val,
            loss_fn
        )
        exit()
        
    if test_only:
        test_model(
        denoise_net, test_dataloader, config, device, log_dir, pathconfig, txt_test
                )
        exit()

    denoise_net.train()
    train_loss_sum = 0
    pbar = tqdm.tqdm(np.arange(config['max_steps']), initial=steps)
    
        
    while steps < config["max_steps"]:
        for batch in train_dataloader:
            y, m, rms = batch
            y, m, rms = y.to(device), m.to(device), rms.to(device)

            optim.zero_grad()
            y_denoised = denoise_net(m)
            #if config["loss"]["loss_type"] in ['combined', 'multires', 'ccmse']:
            loss = loss_fn(y / rms[:, None], y_denoised / rms[:, None])
            loss.backward()
            optim.step()
            train_loss_sum += loss.detach().cpu().numpy()

            steps += 1
            pbar.update(1)
            scheduler.step()

            if steps % 10 == 0:
                pbar.set_description(f"Steps : {steps}, Loss : {loss.detach().cpu().numpy()}, LR : {scheduler.get_last_lr()[0]}")

            if steps % config["val_interval"] == 0:
                torch.save(
                    {
                        "optim": optim.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "denoiser": denoise_net.state_dict(),
                        "steps": steps,
                        "best_metric": best_metric,
                    },
                    os.path.join(chkpt_dir, "latest"),
                )

                denoise_net.eval()
                metric = validate(
                    denoise_net,
                    val_dataloader,
                    config,
                    device,
                    sw,
                    steps,
                    val_tensorboard_examples,
                    txt_val,
                    loss_fn,
                )
                denoise_net.train()

                if metric > best_metric:
                    best_metric = metric
                    torch.save(
                        {
                            "optim": optim.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "denoiser": denoise_net.state_dict(),
                            "steps": steps,
                            "best_metric": best_metric,
                        },
                        os.path.join(chkpt_dir, "best"),
                    )

                train_loss = train_loss_sum / config["val_interval"]
                train_loss_sum = 0
                if config["loss"]["loss_type"] == 'combined':
                    sw.add_scalar("Train Loss (COMBO)", train_loss, steps)
                else:
                    sw.add_scalar("Train Loss", train_loss, steps)
                    
                sw.add_scalar("Learning Rate", scheduler.get_last_lr()[0], steps)

            if steps >= config["max_steps"]:
                break

    # Testing phase
    test_model(
        denoise_net, test_dataloader, config, device, log_dir, pathconfig, txt_test
    )


def validate(
    denoise_net,
    val_dataloader,
    config,
    device,
    sw,
    step,
    val_tensorboard_examples,
    txt_val,
    loss_fn
):
    np.random.seed(config["data_distribution_seed"])

    y_all, denoised_all, loss_all, noisy_all = [], [], [], []

    for batch in tqdm.tqdm(val_dataloader):
        with torch.no_grad():
            y, m, rms = batch
            y, m = resample_batch(y, m, 48000, config["fs"])
            y, m, rms = y.to(device), m.to(device), rms.to(device)

            noisy_all.append(m[0, :].cpu().numpy())
            y_g_hat = denoise_net(m)
            if config["loss"]["loss_type"] in ['combined', 'multires', 'ccmse']:
                loss = loss_fn(y / rms[:, None], y_g_hat / rms[:, None])
            else:
                raise ValueError("Invalid loss type")
            loss_all.append(loss.cpu().numpy())
            denoised_all.append(y_g_hat[0, :].cpu().numpy())
            y_all.append(y[0, :].cpu().numpy())

    lengths_all = np.array([y.shape[0] for y in y_all])
    pesq_all = compute_pesq(y_all, denoised_all, config["fs"])
    sisdr_all = compute_sisdr(y_all, denoised_all)
    ovr_all, sig_all, bak_all = compute_dnsmos(denoised_all, config["fs"])
    distillmos_all = compute_distillmos(denoised_all, config["fs"], device=device)
    xlsr_mos_all = compute_xlsr_sqa_mos(denoised_all, config["fs"], device=device)
    mean_loss = np.mean(lengths_all * np.array(loss_all)) / np.mean(lengths_all)
    mean_pesq = np.mean(lengths_all * np.array(pesq_all)) / np.mean(lengths_all)
    mean_ovr = np.mean(lengths_all * np.array(ovr_all)) / np.mean(lengths_all)
    mean_sig = np.mean(lengths_all * np.array(sig_all)) / np.mean(lengths_all)
    mean_bak = np.mean(lengths_all * np.array(bak_all)) / np.mean(lengths_all)
    mean_distillmos = np.mean(lengths_all * np.array(distillmos_all)) / np.mean(lengths_all)
    mean_xlsr_mos = np.mean(lengths_all * np.array(xlsr_mos_all)) / np.mean(lengths_all)
    mean_sisdr = np.mean(lengths_all * np.array(sisdr_all)) / np.mean(lengths_all)

    if config["loss"]["loss_type"] == 'combined':
        sw.add_scalar("Loss (COMBO)", mean_loss, step)
    else:
        sw.add_scalar("Loss", mean_loss, step)
    sw.add_scalar("PESQ", mean_pesq, step)
    sw.add_scalar("SI-SDR", mean_sisdr, step)
    sw.add_scalar("DNSMOS-OVR", mean_ovr, step)
    sw.add_scalar("DNSMOS-SIG", mean_sig, step)
    sw.add_scalar("DNSMOS-BAK", mean_bak, step)
    sw.add_scalar("DistillMOS", mean_distillmos, step)
    sw.add_scalar("XLS-R-MOS", mean_xlsr_mos, step)

    for i in val_tensorboard_examples:
        sw.add_audio(
            "%d" % i,
            torch.from_numpy(denoised_all[i]),
            global_step=step,
            sample_rate=config["fs"],
        )

    np.random.seed()
    return mean_xlsr_mos + mean_ovr + mean_distillmos


def resample_batch(y, m, fs_in, fs_out):
    y = torch.from_numpy(scipy.signal.resample_poly(y.cpu().numpy(), fs_out, fs_in, axis=1))
    m = torch.from_numpy(scipy.signal.resample_poly(m.cpu().numpy(), fs_out, fs_in, axis=1))
    return y, m


def test_model(
    denoise_net, test_dataloader, config, device, log_dir, pathconfig, txt_test
):
    chkpt_dir = os.path.join(
        pathconfig["chkpt_logs_path"], "checkpoints", config["train_name"]
    )
    chkpt = torch.load(os.path.join(chkpt_dir, "best"), weights_only=False)
    denoise_net.load_state_dict(chkpt["denoiser"])
    denoise_net.eval()
    
    np.random.seed(config["data_distribution_seed"])
    print("starting to test")

    s_all, denoised_all, lengths_all = [], [], []
    os.makedirs(os.path.join(log_dir, "synthetic_test"), exist_ok=True)

    for batch in tqdm.tqdm(test_dataloader):
        with torch.no_grad():
            s, m, rms = batch
            s, m = resample_batch(s, m, 48000, config["fs"])
            s, m, rms = s.to(device), m.to(device), rms.to(device)

            y = denoise_net(m)
            y = y.cpu().numpy()
            s = s.cpu().numpy()
            m = m.cpu().numpy()

            denoised_all.append(y[0, :])
            s_all.append(s[0, :])
            lengths_all.append(s.shape[1])

    lengths_all = np.array(lengths_all)
    distillmos_all = compute_distillmos(denoised_all, config["fs"], device=device)
    xlsrmos_all = compute_xlsr_sqa_mos(denoised_all, config["fs"], device=device)
    pesq_all = compute_pesq(s_all, denoised_all, config["fs"])
    sisdr_all = compute_sisdr(s_all, denoised_all)
    ovr_all, sig_all, bak_all = compute_dnsmos(denoised_all, config["fs"])

    mean_pesq = np.mean(lengths_all * np.array(pesq_all)) / np.mean(lengths_all)
    mean_sisdr = np.mean(lengths_all * np.array(sisdr_all)) / np.mean(lengths_all)
    mean_ovr = np.mean(lengths_all * np.array(ovr_all)) / np.mean(lengths_all)
    mean_sig = np.mean(lengths_all * np.array(sig_all)) / np.mean(lengths_all)
    mean_bak = np.mean(lengths_all * np.array(bak_all)) / np.mean(lengths_all)
    mean_distillmos = np.mean(lengths_all * np.array(distillmos_all)) / np.mean(lengths_all)
    mean_xlsr_sqa_mos = np.mean(lengths_all * np.array(xlsrmos_all)) / np.mean(lengths_all)

    with open(os.path.join(log_dir, "synthetic_test", "mean_metrics.txt"), "w") as f:
        f.write(
            "PESQ, SI-SDR, DNSMOS-OVR, DNSMOS-SIG, DNSMOS-BAK, DistillMOS, XLS-R-MOS\n"
            + "%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f"
            % (mean_pesq, mean_sisdr, mean_ovr, mean_sig, mean_bak, mean_distillmos, mean_xlsr_sqa_mos)
        )

    metrics = np.stack((pesq_all, sisdr_all, ovr_all, sig_all, bak_all, distillmos_all, xlsrmos_all)).T
    header = "Index, PESQ, SI-SDR, DNSMOS-OVR, DNSMOS-SIG, DNSMOS-BAK, DistillMOS, XLS-R-MOS"
    data = np.column_stack((np.arange(0, metrics.shape[0]), metrics))
    np.savetxt(
        os.path.join(log_dir, "synthetic_test", "metrics.csv"),
        data,
        fmt="%d, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f",
        delimiter=",",
        header=header,
    )

    dns4_test_root = os.path.join(pathconfig["DNS4_root"], "dev_testset")
    blind_test_files = glob.glob(os.path.join(dns4_test_root, "**/*.wav"), recursive=True)

    y_all, lengths_all = [], []
    for file in tqdm.tqdm(blind_test_files):
        with torch.no_grad():
            m, fs = soundfile.read(file, always_2d=True)
            m = m[:, 0]  # Take only one channel
            if fs != config["fs"]:
                m = scipy.signal.resample_poly(m, config["fs"], fs)

            m = torch.from_numpy(m).float().to(device)
            m = m[None, :]

            y = denoise_net(m)
            y = y.cpu().numpy()

            y_all.append(y[0, :])
            lengths_all.append(y.shape[1])

    lengths_all = np.array(lengths_all)
    ovr_all, sig_all, bak_all = compute_dnsmos(y_all, config['fs'])

    mean_ovr = np.mean(lengths_all * np.array(ovr_all)) / np.mean(lengths_all)
    mean_sig = np.mean(lengths_all * np.array(sig_all)) / np.mean(lengths_all)
    mean_bak = np.mean(lengths_all * np.array(bak_all)) / np.mean(lengths_all)

    os.makedirs(os.path.join(log_dir, "blind_test"), exist_ok=True)

    distillmos_all = compute_distillmos(y_all, config['fs'], device=device)
    xlsrmos_all = compute_xlsr_sqa_mos(y_all, config['fs'], device=device)

    mean_distillmos = np.mean(lengths_all * np.array(distillmos_all)) / np.mean(lengths_all)
    mean_xlsr_sqa_mos = np.mean(lengths_all * np.array(xlsrmos_all)) / np.mean(lengths_all)

    with open(os.path.join(log_dir, "blind_test", "mean_metrics.txt"), "w") as f:
        f.write(
            "DNSMOS-OVR, DNSMOS-SIG, DNSMOS-BAK, DistillMOS, XLS-R-MOS\n"
            + "%.3f, %.3f, %.3f, %.3f, %.3f" % (mean_ovr, mean_sig, mean_bak, mean_distillmos, mean_xlsr_sqa_mos)
        )

    metrics = np.stack((ovr_all, sig_all, bak_all, distillmos_all, xlsrmos_all)).T
    header = "filepath, DNSMOS-OVR, DNSMOS-SIG, DNSMOS-BAK, DistillMOS, XLS-R-MOS"
    data = np.column_stack((np.array(blind_test_files, dtype=object), metrics))
    np.savetxt(
        os.path.join(log_dir, "blind_test", "metrics.csv"),
        data,
        fmt="%s, %.3f, %.3f, %.3f, %.3f, %.3f",
        delimiter=",",
        header=header,
    )


if __name__ == "__main__":
    main()
