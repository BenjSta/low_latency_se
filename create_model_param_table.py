import os
import toml
import torch
import tqdm
import pandas as pd
from ptflops import get_model_complexity_info
from models import SpeechEnhancementModel


group_column = 'nn_delay'#'nn delay'

def collect_toml_files(root_folder):
    """Recursively collect all TOML files from a directory."""
    toml_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for f in filenames:
            if f.endswith(".toml"):
                toml_files.append(os.path.join(dirpath, f))
    return toml_files


def analyze_model(config_path, cuda_visible_devices="0"):
    """Load config, initialize model, and compute complexity."""
    config = toml.load(config_path)
    config["train_name"] = os.path.basename(config_path).split(".")[0]

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    device = "cuda" if torch.cuda.is_available() else "cpu"

    denoise_net = SpeechEnhancementModel(**config["model"]).to(device)

    macs, params = get_model_complexity_info(
        denoise_net, (20 * 16000,), as_strings=False, print_per_layer_stat=False
    )

    # Adjust MACs based on method
    if config["model"]["method"] == "complex_filter":
        macs = (
            macs / 20
            + config["fs"] / config["model"]["hopsize"]
            * (
                4 * (config["model"]["winlen"] // 2 + 1) * denoise_net.num_filter_frames
                + 2 * (config["model"]["winlen"] // 2 + 1) * config["model"]["winlen"]
                + 2
                * (config["model"]["winlen"] // 2 + 1)
                * config["model"]["hopsize"]
                * 2
            )
            + config["fs"]
            / config["model"]["hopsize"]
            / denoise_net.downsample_factor
            * 6
            * (config["model"]["winlen"] // 2 + 1)
        )

    elif config["model"]["method"] == "time_domain_filtering":
        macs = (
            macs / 20
            + config["fs"] / config["model"]["hopsize"]
            * (2 * (config["model"]["winlen"] // 2 + 1) * config["model"]["winlen"])
            + 2 * config["fs"] * denoise_net.filtlen * 0.5
        )

    # get latest epoch
    pathconfig = toml.load("directories.toml")
    chkpt_dir = os.path.join(
        pathconfig["chkpt_logs_path"], "checkpoints", config["train_name"]
    )
    chkpt_path = os.path.join(chkpt_dir, "latest")
    if os.path.exists(chkpt_path):
        chkpt = torch.load(chkpt_path, weights_only=False)
        optim = torch.optim.AdamW(
            denoise_net.parameters(),
            config["lr_start"],
        )
        optim.load_state_dict(chkpt["optim"])
        step = optim.state[optim.param_groups[0]["params"][-1]]["step"].item()
    else:
        step = 0
        

    if 'cmask' in config_path:
        nn_delay = config['model']['algorithmic_delay_nn']
        hop_size = config['model']['hopsize']
        ds_fact = config['model']['downsample_factor']
        inf_interval = ds_fact*hop_size
        total_delay = max((nn_delay+inf_interval, 3*hop_size))
        total_delay =  round(total_delay/16,3)
        nn_delay = round(nn_delay/16,3)
        hop_size = round(hop_size/16,3)
        inf_interval = round(inf_interval/16,3)
        filt_delay = None
        
    elif 'td' in config_path:
        nn_delay = config['model']['algorithmic_delay_nn']
        hop_size = config['model']['hopsize']
        filt_delay = config['model']['algorithmic_delay_filtering']
        total_delay = max((nn_delay, 2*hop_size))
        total_delay = round(total_delay/16,3)
        nn_delay = round(nn_delay/16,3)
        filt_delay = round(filt_delay/16,3)
        hop_size = round(hop_size/16,3)
        inf_interval = hop_size
    else:
        raise FileNotFoundError       
    
    if config["model"]["method"] == "complex_filter":
        method_str='CMask'
    else:
        method_str='TD'
    return {
        "config_name": os.path.basename(config_path).replace('_','\_'),
        "nn_delay": nn_delay,
        "filter_delay": filt_delay,
        "total_delay": total_delay,
        "inf_interval": inf_interval,
        "hopsize": hop_size,
        "channels": config["model"]["crn_config"].get("num_channels_encoder"),
        "method": method_str,
        "macs": macs,
        "params": params,
        "train_step": step,
    }


def export_latex_table(results, output_path="results.tex"):
    """Export results to a LaTeX table, sorted/grouped."""
    df = pd.DataFrame(results)
    df = df.sort_values(
        by=["nn_delay", "total_delay", "method", "inf_interval"],
        ascending=[False, False, True, False]
    ).reset_index(drop=True)

    header = (
        "\\begin{tabular}{lcccccc|ccc}\n"
        "Config & NN Delay & Filter Delay & Total Delay & Inf Interval & Hopsize & Channels & GMACs & M Params & M Train Step \\\\ \\ \hline\n"
    )

    rows = []
    current_group = None
    for _, r in df.iterrows():
        row = (
            f"{r['config_name']} & {r['nn_delay']} & {r['filter_delay']} & {r['total_delay']} & {r['inf_interval']} & "
            f"{r['hopsize']} & {r['channels']} & {round(r['macs']/1e9,3):,} & {round(r['params']/1e6, 4):,} & {round(r['train_step']/1e6,3)} \\\\"
        )
        group_key = (r[group_column])
        if group_key != current_group:
            if current_group is not None:
                rows.append(r'\hline')
            current_group = group_key
        rows.append(row)

    footer = "\\end{tabular}"

    with open(output_path, "w") as f:
        f.write(header + "\n".join(rows) + "\n" + footer)


def main():
    root_folder = "./configs_exp"  # change to your folder
    results = []
    for toml_file in tqdm.tqdm(collect_toml_files(root_folder)):
        try:
            result = analyze_model(toml_file)
            results.append(result)
        except Exception as e:
            print(f"Error processing {toml_file}: {e}")

    export_latex_table(results, "model_param_table.tex")


if __name__ == "__main__":
    main()