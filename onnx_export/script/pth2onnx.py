import os

from torch import no_grad
import utils
from models import SynthesizerTrn

import torch
from text.symbols import symbols
import onnx
import onnxruntime

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad \
        else tensor.detach().numpy()

class VitsExtractor(object):

    def convert_model(self, json_path: str,
                      model_path: str,
                      ):
        import onnx_infer
        hps = utils.get_hparams_from_file(json_path)

        # Get symbols and initialize synthesizer model
        trn2 = onnx_infer.SynthesizerTrn2(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers = hps.data.n_speakers,
            ** hps.model
        )
        trn2.eval()
        _ = utils.load_checkpoint(model_path, trn2, None)
        trn2.cpu()
        path = "vits_uma_genshin_honkai"
        os.makedirs(path, exist_ok=True)
        trn2.export_onnx(path = path)
        return

    def warp_pth(self, model_config_path: str, model_path: str = None):
        self.convert_model(json_path=str(model_config_path), model_path=str(model_path))


def vits():
    hps = utils.get_hparams_from_file("/Users/yrzhu/Vits/VitsServer/model/config.json")
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model)
    _ = utils.load_checkpoint("/Users/yrzhu/Vits/VitsServer/model/G_953000.pth", net_g, None)
    net_g.eval()
    net_g.cpu()
    with no_grad():
        x_tst = torch.LongTensor([0, 27, 0, 22, 0, 49, 0, 50, 0, 21, 0, 15, 0, 33, 0, 49, 0, 50,
                                0, 2, 0])
        x_tst = x_tst.unsqueeze(0).to("cpu")
        x_tst_lengths = torch.LongTensor([x_tst.size(1)]).to("cpu")
        speaker_id = torch.LongTensor([228]).to("cpu")
        audio = net_g.infer(x_tst, x_tst_lengths, sid=speaker_id, noise_scale=0.6, noise_scale_w=0.668,
                               length_scale=1.2)[0][0, 0].data.cpu().float().numpy()
        print("audio: ", audio)
        import soundfile as sf

        # audio 为 numpy.float32 数组，范围通常在 [-1.0, 1.0]
        sf.write("output.wav", audio, samplerate=hps.data.sampling_rate)

def save_per_module():
    hps = utils.get_hparams_from_file("/Users/yrzhu/Vits/VitsServer/model/config.json")
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model)
    _ = utils.load_checkpoint("/Users/yrzhu/Vits/VitsServer/model/G_953000.pth", net_g, None)
    net_g.eval()
    net_g.save_per_module()


if __name__ == "__main__":

# extract onnx
    model = VitsExtractor().warp_pth(
        model_config_path="/Users/yrzhu/Vits/vits-uma-genshin-honkai/model/config.json",
        model_path="/Users/yrzhu/Vits/vits-uma-genshin-honkai/model/G_953000.pth",
    )

    print("done")
    # original vits model infer
#     vits()

