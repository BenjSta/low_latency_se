import scipy.signal
import numpy as np
from werpy import wer as compute_wer, normalize as norm_text
import xls_r_sqa.e2e_model
from third_party.dnsmos_local import ComputeScore
from pesq import pesq
import tqdm
from pymcd.mcd import Calculate_MCD
import tempfile
import soundfile
import torch
import whisper
import gc
import distillmos, xls_r_sqa
from xls_r_sqa.config import XLSR_2B_TRANSFORMER_32DEEP_CONFIG


def compute_mcd(list_of_refs, list_of_signals, fs):
    mcd_toolbox = Calculate_MCD(MCD_mode="plain")
    mcdvals = []
    pbar = tqdm.tqdm(zip(list_of_refs, list_of_signals))
    pbar.set_description("Computing MCD...")
    for r, s in pbar:
        with tempfile.NamedTemporaryFile(
            suffix=".wav"
        ) as reffile, tempfile.NamedTemporaryFile(suffix=".wav") as sigfile:
            refname = reffile.name
            signame = sigfile.name

            soundfile.write(refname, r, fs)
            soundfile.write(sigfile, s, fs)

            mcdvals.append(mcd_toolbox.calculate_mcd(refname, signame))

    return np.array(mcdvals)

def compute_distillmos(list_of_signals, fs, device):
    distillmos_model = distillmos.ConvTransformerSQAModel()
    distillmos_model.to(device)
    distillmos_model.eval()

    distillmos_scores = []
    pbar = tqdm.tqdm(list_of_signals)
    pbar.set_description("Computing DistillMOS...")
    with torch.no_grad():
        for s in pbar:
            distillmos_scores.append(
                distillmos_model(
                    torch.from_numpy(
                        scipy.signal.resample_poly(s / np.max(np.abs(s)), 16000, fs)
                    ).to(device)[None, :]
                ).detach().cpu().numpy().squeeze().item()
              
            )

    # Free up memory
    del distillmos_model
    gc.collect()
    torch.cuda.empty_cache()
    
    return np.array(distillmos_scores)


def compute_xlsr_sqa_mos(list_of_signals, fs, device):
    xlsr_sqa_model = xls_r_sqa.e2e_model.E2EModel(config=XLSR_2B_TRANSFORMER_32DEEP_CONFIG, xlsr_layers=10)
    xlsr_sqa_model.to(device)
    xlsr_sqa_model.eval()

    xlsr_sqa_scores = []
    pbar = tqdm.tqdm(list_of_signals)
    with torch.no_grad():
        for s in pbar:
            xlsr_sqa_scores.append(
                xlsr_sqa_model(
                    torch.from_numpy(
                        scipy.signal.resample_poly(s / np.max(np.abs(s)), 16000, fs)
                    ).to(device)[None, :]
                ).detach().cpu().numpy().squeeze().item()
            )

    # Free up memory
    del xlsr_sqa_model
    gc.collect()
    torch.cuda.empty_cache()

    return np.array(xlsr_sqa_scores)


def compute_pesq(list_of_refs, list_of_signals, fs):
    pesqvals = []
    pbar = tqdm.tqdm(zip(list_of_refs, list_of_signals))
    pbar.set_description("Computing PESQ...")
    for r, s in pbar:
        r = scipy.signal.resample_poly(r, 16000, fs)
        s = scipy.signal.resample_poly(s, 16000, fs)
        try:
            pesqval = pesq(16000, r, s, "wb")
        except:
            pesqval = np.nan

        pesqvals.append(pesqval)

    return np.array(pesqvals)


def compute_sisdr(list_of_refs, list_of_signals):
    sisdrvals = []
    pbar = tqdm.tqdm(zip(list_of_refs, list_of_signals))
    pbar.set_description("Computing SISDR...")
    for r, s in pbar:
        shortest = min(r.shape[0], s.shape[0])
        r = r[:shortest]
        s = s[:shortest]
        proj_gain = np.dot(r, s) / np.dot(r, r)
        r_proj = proj_gain * r
        distortion = s - r_proj
        sisdrvals.append(10 * np.log10(np.dot(r_proj, r_proj) / np.dot(distortion, distortion)))
    
    return np.array(sisdrvals)
            

def compute_dnsmos(list_of_signals, fs):
    dnsmos_obj = ComputeScore(
        "third_party/DNSMOS/sig_bak_ovr.onnx", "third_party/DNSMOS/model_v8.onnx"
    )

    dnsmos_sig = []
    dnsmos_bak = []
    dnsmos_ovrl = []
    lengths = []

    pbar = tqdm.tqdm(list_of_signals)
    pbar.set_description("Computing DNSMOS...")

    for s in pbar:
        d = dnsmos_obj(scipy.signal.resample_poly(s / np.max(np.abs(s)), 16000, fs))
        dnsmos_ovrl.append(d["OVRL"])
        dnsmos_bak.append(d["BAK"])
        dnsmos_sig.append(d["SIG"])
        lengths.append(s.shape[0])

    return (np.array(dnsmos_ovrl), np.array(dnsmos_sig), np.array(dnsmos_bak))


def compute_mean_wacc(list_of_signals, list_of_texts, fs, device):
    list_of_transcripts = []

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    asr_model = whisper.load_model("medium.en", device=device)
    # asr_model_faster = WhisperModel("medium.en", device = device)
    pbar = tqdm.tqdm(list_of_signals)
    pbar.set_description("Computing Wacc...")

    for s in pbar:
        list_of_transcripts.append(
            asr_model.transcribe(
                scipy.signal.resample_poly(s / np.max(np.abs(s)), 16000, fs)
            )["text"]
        )
        # text_segments_faster = asr_model_faster.transcribe(scipy.signal.resample_poly(s / np.max(np.abs(s)), 16000, fs))[0]
        # text_faster = ""
        # for segment in text_segments_faster:
        #    text_faster = text_faster+segment.text
        # list_of_transcripts.append(text_faster)
    norm_list_of_transcripts = [
        " " if i == "" else i for i in norm_text(list_of_transcripts)
    ]

    # Free up memory
    del asr_model
    gc.collect()
    torch.cuda.empty_cache()

    return 1 - compute_wer(norm_text(list_of_texts), norm_list_of_transcripts)
