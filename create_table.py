import os
import toml
import pandas as pd
import soundfile as sf
import numpy as np
from dataset import load_paths
pathconfig = toml.load('directories.toml')


paths = load_paths(pathconfig["DNS4_root"], pathconfig["VCTK_txt_root"])
clean_train, clean_val, clean_test = paths["clean"]
_, txt_val, txt_test = paths["txt"]
noise_train, noise_val, noise_test = paths["noise"]
rir_train, rir_val, rir_test = paths["rir"]

# Define configs as tuples for compactness
configs = [
    # (config_name, algorithmic_delay, inference_interval, method)
    ("dfnet", 40, 10, 'DF (DFNet)'),
    ("exp_cmask_40ms_multiframe_lookahead", 40, 10, 'DF'),
    ("exp_cmask_10ms_ds2", 10, 10, 'CMask'),
    ("exp_cmask_5ms_ds2", 5, 5, 'CMask'),
    ("exp_td_5ms_interval10", 5, 10, 'TD'),
    ("exp_cmask_2_5ms_ds4", 2.5, 5, 'CMask'),
    ("exp_cmask_20ms", 20, 10, 'CMask'),
    ("exp_td_2_5ms_interval5", 2.5, 5, 'TD'),
    ("exp_td_2_5ms_interval10", 2.5, 10, 'TD'),
    ("exp_td_-5ms_interval5", -5, 5, 'TD'),
    ("exp_td_20ms", 20, 10, 'TD'),
    ("exp_cmask_10ms", 10, 5, 'CMask'),
    ("exp_td_10ms_interval5", 10, 5, 'TD'),
    ("exp_cmask_5ms_ds4", 5, 10, 'CMask'),
    ("exp_td_10ms_interval10", 10, 10, 'TD'),
    ("exp_td_5ms_interval5", 5, 5, 'TD'),
    ("exp_td_-5ms_interval5_big", -5, 5, 'TD'),
    ("exp_cmask_2_5ms_ds8", 2.5, 10, 'CMask'),
]

# Convert to DataFrame
df = pd.DataFrame(configs, columns=["config_name", "algorithmic_delay", "inference_interval", "method"])

def get_channels(config_name):
    try:
        config = toml.load(os.path.join('configs_exp', config_name + '.toml'))
    except FileNotFoundError:
        print(f"Config file for {config_name} not found.")
        return None
    return config['model']['crn_config']['num_channels_encoder'][1:]

df['channels'] = df['config_name'].apply(get_channels)

df = df.sort_values(by=["algorithmic_delay", "method", "inference_interval"], ascending=[False, True, False]).reset_index(drop=True)
print(df)

def get_metrics(config_name):
    dirs = toml.load('directories.toml')
    p = os.path.join(dirs['chkpt_logs_path'], 'logs', config_name)

    synthetic_path = os.path.join(p, 'synthetic_test', 'metrics.csv')
    blind_path = os.path.join(p, 'blind_test', 'metrics.csv')

    try:
        # load in such a way that it doesnt fail if the filename contains a comma
        pandas_synthetic = pd.read_csv(synthetic_path)

        lines_new = []

        # need a hack for the blind dataset
        with open(blind_path, 'r', encoding='utf-8') as f:
            
            lines = f.readlines()
            
            lines_new.append(lines[0])  # keep the header
            for line in lines[1:]:
                line = '"' + line
                line = line.replace('.wav', '.wav"')
                lines_new.append(line)

        # create string from the lines
        lines_new = ''.join(lines_new)

        # read the modified lines into a pandas Dataframe
        from io import StringIO
        pandas_blind = pd.read_csv(StringIO(lines_new), quoting=1)

    except FileNotFoundError:
        return {'synthetic_pesq': np.nan,
            'synthetic_si_sdr': np.nan,
            'synthetic_dnsmos_ovr': np.nan,
            #'synthetic_dnsmos_sig': synthetic_avg['DNSMOS-SIG'],
            #'synthetic_dnsmos_bak': synthetic_avg['DNSMOS-BAK'],
            'synthetic_distillmos': np.nan,
            'synthetic_xls_r_mos':np.nan,
            'blind_dnsmos_ovr': np.nan,
            #'blind_dnsmos_sig': np.nan,
            #'blind_dnsmos_bak': np.nan,
            'blind_distillmos': np.nan,
            'blind_xls_r_mos': np.nan,
            'combined_nonintrusive': np.nan,
           }
            

    # add a column with the file duration by inspecting the filepath using soundfile
    def get_duration(file_path):
        try:
            sf.info(file_path)
            return sf.info(file_path).duration
        except RuntimeError:
            return None
        
    
    pandas_synthetic['filepath'] = clean_test
    pandas_synthetic['duration'] = pandas_synthetic['filepath'].apply(get_duration)
    pandas_blind['duration'] = pandas_blind['# filepath'].apply(get_duration)

    # calculate the total duration in each dataset
    total_duration_synthetic = pandas_synthetic['duration'].sum()
    total_duration_blind = pandas_blind['duration'].sum()
    print(f"Total duration synthetic: {total_duration_synthetic}, blind: {total_duration_blind}")

    # calculate the average metrics weighted by duration
    def weighted_average(df, metric):
        return (df[' ' + metric] * df['duration']).sum() / df['duration'].sum()

    metrics_synthetic = ['PESQ', 'SI-SDR', 'DNSMOS-OVR', 'DNSMOS-SIG', 'DNSMOS-BAK', 'DistillMOS', 'XLS-R-MOS']
    metrics_blind = ['DNSMOS-OVR', 'DNSMOS-SIG', 'DNSMOS-BAK', 'DistillMOS', 'XLS-R-MOS']

    # cpmpute the average metrics for synthetic and blind datasets
    synthetic_avg = {metric: weighted_average(pandas_synthetic, metric) for metric in metrics_synthetic}
    blind_avg = {metric: weighted_average(pandas_blind, metric) for metric in metrics_blind}

    # for 'DNSMOS-OVR', 'DistillMOS', 'XLS-R-MOS' compute weighted average across both datasets
    combined_avg = {
        'DNSMOS-OVR': (synthetic_avg['DNSMOS-OVR'] + blind_avg['DNSMOS-OVR']) / 2,
        'DistillMOS': (synthetic_avg['DistillMOS'] + blind_avg['DistillMOS']) / 2,
        'XLS-R-MOS': (synthetic_avg['XLS-R-MOS']  + blind_avg['XLS-R-MOS']) / 2}

    # and compute the average of these three metrics
    combined_avg = (combined_avg['DNSMOS-OVR'] + combined_avg['DistillMOS'] + combined_avg['XLS-R-MOS']) / 3

    return {'synthetic_pesq': synthetic_avg['PESQ'],
            'synthetic_si_sdr': synthetic_avg['SI-SDR'],
            'synthetic_dnsmos_ovr': synthetic_avg['DNSMOS-OVR'],
            #'synthetic_dnsmos_sig': synthetic_avg['DNSMOS-SIG'],
            #'synthetic_dnsmos_bak': synthetic_avg['DNSMOS-BAK'],
            'synthetic_distillmos': synthetic_avg['DistillMOS'],
            'synthetic_xls_r_mos': synthetic_avg['XLS-R-MOS'],
            'blind_dnsmos_ovr': blind_avg['DNSMOS-OVR'],
            #'blind_dnsmos_sig': blind_avg['DNSMOS-SIG'],
            #'blind_dnsmos_bak': blind_avg['DNSMOS-BAK'],
            'blind_distillmos': blind_avg['DistillMOS'],
            'blind_xls_r_mos': blind_avg['XLS-R-MOS'],
            'combined_nonintrusive': combined_avg,
           }

# Apply the function to each config_name and join the results to the dataframe
metrics = df['config_name'].apply(get_metrics)
metrics_df = pd.DataFrame(metrics.tolist())
# Join the metrics DataFrame with the main DataFrame
df = df.join(metrics_df)

df.to_csv('table.csv', index=False)

import pandas as pd
import numpy as np

# Remove underscores in column names
df.columns = df.columns.str.replace('_', ' ')

# Remove config name column
df = df.drop(columns=['config name'])

# Sort by Delay (descending), then Inference Interval (descending)
df = df.sort_values(by=['algorithmic delay', 'method', 'inference interval'], ascending=[False, True, False])

# Define metrics
synthetic_metrics = [
    'synthetic pesq', 'synthetic si sdr', 'synthetic dnsmos ovr',
    'synthetic distillmos', 'synthetic xls r mos'
]
blind_metrics = ['blind dnsmos ovr', 'blind distillmos', 'blind xls r mos']
combined_metric = 'combined nonintrusive'

# Highlight best per subgroup for each metric
def highlight_group(sub_df):
    metrics = synthetic_metrics + blind_metrics
    sub_df = sub_df.copy()

    # highlight highest value per column
    for col in metrics:
        if col in sub_df.columns:
            best = sub_df[col].max()
            sub_df[col] = sub_df[col].apply(
                lambda x: f'\\cellcolor{{green!20}}{x:.2f}' if np.isclose(x, best) else f'{x:.2f}'
                if pd.notna(x) else '')
    
    # Double highlight best combined
    if combined_metric in sub_df.columns:
        best_combined = sub_df[combined_metric].max()
        sub_df[combined_metric] = sub_df[combined_metric].apply(
            lambda x: f'\\cellcolor{{green!50}}{x:.2f}' if np.isclose(x, best_combined) else f'{x:.2f}'
            if pd.notna(x) else '')
    
    return sub_df

# Group by delay/interval/method
grouped_rows = []
for (delay,), group in df.groupby(['algorithmic delay'], sort=False):
    # Subgroup by method (CMask vs TD)
    cmask = group[group['method'].str.contains('CMask', na=False)]
    td = group[group['method'].str.contains('TD', na=False)]
    if not cmask.empty:
        grouped_rows.append(highlight_group(cmask))
    if not td.empty:
        grouped_rows.append(highlight_group(td))
    # Insert dashed line if both present
    if not cmask.empty and not td.empty:
        dash = pd.DataFrame([['\\hdashline'] + [''] * (df.shape[1]-1)], columns=df.columns)
        grouped_rows.append(dash)

# Rebuild full table
final_df = pd.concat(grouped_rows)

# Column headers
header_main = [
    r'\textbf{Delay}', r'\textbf{Interval}', r'\textbf{Method}', r'\textbf{Channels}',
    r'\multicolumn{5}{c}{\textbf{Synthetic Test}}',
    r'\multicolumn{3}{c}{\textbf{Blind Test}}',
    r'\textbf{Combined}'
]
header_sub = ['', '', '', ''] + ['PESQ', 'SI-SDR', 'DNSMOS', 'DistillMOS', 'XLS-R'] + \
             ['DNSMOS', 'DistillMOS', 'XLS-R'] + ['']

# Build LaTeX table
latex_lines = []
latex_lines.append(r'\begin{tabular}{llllrrrrrrrrrr}')
latex_lines.append(r'\toprule')
latex_lines.append(' & '.join(header_main) + r' \\')
latex_lines.append(' & '.join(header_sub) + r' \\')
latex_lines.append(r'\midrule')

current_group = None
for idx, row in final_df.iterrows():
    if str(row[0]).startswith('\\hdashline'):
        latex_lines.append(r'\hdashline')
        continue

    group_key = (row['algorithmic delay'])
    if group_key != current_group:
        if current_group is not None:
            latex_lines.append(r'\midrule')
        current_group = group_key

    latex_row = ' & '.join(str(x) for x in row.values) + r' \\'
    latex_lines.append(latex_row)

latex_lines.append(r'\bottomrule')
latex_lines.append(r'\end{tabular}')

# Output LaTeX code
latex_code = '\n'.join(latex_lines)


with open('table.tex', 'w') as f:
    f.write(latex_code)
