# -*- coding: utf-8 -*-
import math

import numpy as np
import torch
from torch import nn
from models_onnx import TextEncoder
from models_onnx import Generator
from models_onnx import PosteriorEncoder
from models_onnx import ResidualCouplingBlock
from models_onnx import StochasticDurationPredictor
from models_onnx import commons
from text import symbols
import onnxruntime as ort
from torch.nn import functional as F
from commons import convert_pad_shape

def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = np.arange(max_length, dtype=length.dtype)
    return np.expand_dims(x, 0) < np.expand_dims(length, 1)

def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """

    b, _, t_y, t_x = mask.shape
    cum_duration = np.cumsum(duration, -1)

    cum_duration_flat = cum_duration.reshape(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y)
    path = path.reshape(b, t_x, t_y)
    path = path ^ np.pad(path, ((0, 0), (1, 0), (0, 0)))[:, :-1]
    path = np.expand_dims(path, 1).transpose(0, 1, 3, 2)
    return path

class SynthesizerTrn2(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        n_vocab,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        n_speakers=256,
        gin_channels=256,
        use_sdp=True,
        **kwargs
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.use_sdp = use_sdp

        self.enc_p = TextEncoder(
            n_vocab,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )

        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels,
            hidden_channels,
            5,
            1,
            4,
            gin_channels=gin_channels,
        )

        self.dp = StochasticDurationPredictor(
            hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels
        )

        if n_speakers >= 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

    def export_onnx(
        self,
        path,
        max_len=None,
    ):
        noise_scale = 0.667
        length_scale = 1
        noise_scale_w = 0.8
        x = torch.randint(low=0, high=len(symbols), size=(1, 10), dtype=torch.long)
        x_lengths = torch.IntTensor([x.size(1)]).long()
        sid = torch.IntTensor([0]).long()
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
            torch.onnx.export(
                self.emb_g,
                (sid),
                f"{path}/{path}_emb.onnx",
                input_names=["sid"],
                output_names=["g"],
                verbose=True,
            )

        torch.onnx.export(
            self.enc_p,
            (x),
            f"{path}/{path}_enc_p.onnx",
            input_names=[
                "x",
            ],
            output_names=["xout", "m_p", "logs_p", "x_mask"],
            dynamic_axes={
                "x": [0, 1],
                "xout": [0, 2],
                "m_p": [0, 2],
                "logs_p": [0, 2],
                "x_mask": [0, 2],
            },
            verbose=True,
            opset_version=16,
        )

        x, m_p, logs_p, x_mask = self.enc_p(x)

        zin = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale_w

        torch.onnx.export(
            self.dp,
            (x, x_mask, zin, g),
            f"{path}/{path}_dp.onnx",
            input_names=["x", "x_mask", "zin", "g"],
            output_names=["logw"],
            dynamic_axes={"x": [0, 2], "x_mask": [0, 2], "zin": [0, 2], "logw": [0, 2]},
            verbose=True,
        )

        logw = self.dp(x,
                       x_mask,
                       zin,
                       g=g)

        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
            x_mask.dtype
        )
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        torch.onnx.export(
            self.flow,
            (z_p, y_mask, g),
            f"{path}/{path}_flow.onnx",
            input_names=["z_p", "y_mask", "g"],
            output_names=["z"],
            dynamic_axes={"z_p": [0, 2], "y_mask": [0, 2], "z": [0, 2]},
            verbose=True,
        )

        z = self.flow(z_p, y_mask, g=g)
        dec_in = (z * y_mask)[:, :, :max_len]

        torch.onnx.export(
            self.dec,
            (dec_in, g),
            f"{path}/{path}_dec.onnx",
            input_names=["dec_in", "g"],
            output_names=["o"],
            dynamic_axes={"dec_in": [0, 2], "o": [0, 2]},
            verbose=True,
        )
        o = self.dec(dec_in, g=g)
        return o

class OnnxInferenceSession:
    def __init__(self, path, Providers=["CPUExecutionProvider"]):
        self.enc = ort.InferenceSession(path["enc"], providers=Providers)
        self.emb_g = ort.InferenceSession(path["emb_g"], providers=Providers)
        self.dp = ort.InferenceSession(path["dp"], providers=Providers)
        self.flow = ort.InferenceSession(path["flow"], providers=Providers)
        self.dec = ort.InferenceSession(path["dec"], providers=Providers)

    def __call__(
        self,
        seq,
        sid,
        seed=114514,
        noise_scale=0.6,
        noise_scale_w=0.668,
        length_scale=1.2,
    ):
        np.random.seed(seed)
        x = (
            np.expand_dims(np.array(
                [0, 27, 0, 22, 0, 49, 0, 50, 0, 21, 0, 15, 0, 33, 0, 49, 0, 50,
                 0, 2, 0]
            ), 0)
        )
        x = (
            np.expand_dims(np.array(
                [1,2,3,4,5]
            ), 0)
        )

        x_lengths = np.array([x.shape[1]])
        sid = np.array([sid])

        g = self.emb_g.run(None, {"sid": sid.astype(np.int64)})[0]

        g = np.expand_dims(g, -1)

        x, m_p, logs_p, x_mask = self.enc.run(None, {
            "x": x.astype(np.int64),
        })

        print("onnx_enc_x: ", x)

        zin = np.random.randn(x.shape[0], 2, x.shape[2]).astype(np.float32) * noise_scale_w

        print("onnx_enc_zin: ", zin)
        logw = self.dp.run(None, {
            "x": x,
            "x_mask": x_mask,
            "zin": zin,
            "g": g,
        })[0]

        w = np.exp(logw) * x_mask * length_scale
        w_ceil = np.ceil(w)
        y_lengths = np.clip(np.sum(w_ceil, (1, 2)),  a_min=1.0, a_max=100000).astype(
            np.int64
        )
        y_mask = np.expand_dims(sequence_mask(y_lengths, None), 1)
        attn_mask = np.expand_dims(x_mask, 2) * np.expand_dims(y_mask, -1)
        attn = generate_path(w_ceil, attn_mask)

        m_p = np.matmul(attn.squeeze(1), m_p.transpose(0, 2, 1)).transpose(
            0, 2, 1
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = np.matmul(attn.squeeze(1), logs_p.transpose(0, 2, 1)).transpose(
            0, 2, 1
        )  # [b, t', t], [b, t, d] -> [b, d, t']


        z_p_rand = np.random.randn(m_p.shape[0], m_p.shape[1], m_p.shape[2])

        z_p_rand = np.ones_like(m_p) * 0.5

        z_p = (
            m_p
            + z_p_rand
            * np.exp(logs_p)
            * noise_scale
        )

        z = self.flow.run(None,  {
                "z_p": z_p.astype(np.float32),
                "y_mask": y_mask.astype(np.float32),
                "g": g})[0]
        z_in = (z * y_mask)[:, :, :]

        o = self.dec.run(None, {"dec_in": z , "g":g})
        return o

if __name__ == "__main__":
    Session = OnnxInferenceSession(
        {
            "enc": "/Users/yrzhu/Vits/vits-uma-genshin-honkai/onnx/vits_uma_genshin_honkai/vits_uma_genshin_honkai_enc_p.onnx",
            "emb_g": "/Users/yrzhu/Vits/vits-uma-genshin-honkai/onnx/vits_uma_genshin_honkai/vits_uma_genshin_honkai_emb.onnx",
            "dp": "/Users/yrzhu/Vits/vits-uma-genshin-honkai/onnx/vits_uma_genshin_honkai/vits_uma_genshin_honkai_dp.onnx",
            "flow": "/Users/yrzhu/Vits/vits-uma-genshin-honkai/onnx/vits_uma_genshin_honkai/vits_uma_genshin_honkai_flow.onnx",
            "dec": "/Users/yrzhu/Vits/vits-uma-genshin-honkai/onnx/vits_uma_genshin_honkai/vits_uma_genshin_honkai_dec.onnx",
        },
        Providers=["CPUExecutionProvider"],
    )

    audio = Session(seq=None, sid=228)[0][0, 0]
    print("onnx audio: ", audio)
    import soundfile as sf

    # audio 为 numpy.float32 数组，范围通常在 [-1.0, 1.0]
    sf.write("output_onnx.wav", audio, samplerate=22050)