import argparse

import torch
from torchvision import transforms

from utils.util import mag_warping

# from torchaudio import transforms


class option:

    def __init__(self):
        self.parser = None

    def initialize(self):
        parser = argparse.ArgumentParser()
        # module parameters:
        parser.add_argument("--input_size", default=6, type=int)
        parser.add_argument("--hidden_size", default=256, type=int)
        parser.add_argument("--num_layers", default=1, type=int)
        parser.add_argument("--batch_size", default=256, type=int)
        parser.add_argument("--batch_first", default=True, type=bool)
        parser.add_argument("--attention_flag", action='store_false')
        parser.add_argument("--width", default=50, type=int)
        parser.add_argument("--step", default=15, type=int)
        parser.add_argument("--bidirectional", default=False, type=bool)
        parser.add_argument("--num_heads", default=16, type=int)
        parser.add_argument("--total_length", default=400, type=int)
        parser.add_argument("--cnn_kernel_size", default=3, type=int)
        parser.add_argument("--output_size", default=5, type=int)
        parser.add_argument("--dropout", default=0.0, type=float)
        parser.add_argument("--seed", default=73054772, type=int)

        # training parameters:
        parser.add_argument("--lr", default=.001, type=float)
        parser.add_argument("--epochs", default=50, type=int)
        parser.add_argument("--decoder_mode", default='siamese', type=str)
        parser.add_argument('--test', action='store_true')
        parser.add_argument("--num_reps", default=1, type=int)
        parser.add_argument("--pad_method", default='front', type=str)
        # parser.add_argument("--no_front", dest='front_pad', action='store_false')

        parser.add_argument("--filename", default='', type=str)
        parser.add_argument('--exercise', default='e1', type=str)
        parser.add_argument('--metrics', default='rom', type=str)
        parser.add_argument('--baseline', default='', type=str)
        parser.add_argument("--input", default='', type=str)

        parser.add_argument('--cnn_hidden_size', default=0, type=int)
        parser.add_argument('--rnn_hidden_size', default=0, type=int)

        # data augmentation
        parser.add_argument("--interp", action='store_true')
        # parser.add_argument("--exercise_aug", action='store_true')
        parser.add_argument("--mag_warp", action='store_true')
        parser.add_argument('--random_input', default=-1, type=int)

        # for Sessions/Subjects
        parser.add_argument('--input_filename', default='ROM_E1.npy', type=str)
        parser.add_argument(
            "--init_directory",
            default='segment_sessions_one_repetition_data_E1'
        )
        parser.add_argument("--load", default=True, action='store_false')
        parser.add_argument('--load_directory', default='./bin', type=str)
        parser.add_argument(
            '--has_no_timestamp', dest='has_timestamp', action='store_false'
        )
        # parser.add_argument('--transform', dest='transform', action='store_true')
        # parser.set_defaults(load=False)
        parser.set_defaults(transform=False)
        parser.set_defaults(has_timestamp=True)
        # parser.set_defaults(front_pad=True)

        # Data preprocess:
        parser.add_argument("--pair_up", action='store_false')  # default true!
        parser.add_argument(
            "--standardize", action='store_false'
        )  # default true!
        parser.add_argument(
            "--pad_value", default='0', type=str
        )  # [-inf, inf] + [random]

        # device:
        parser.add_argument("--device", default='cuda', type=str)

        parser.add_argument("--debug", action='store_true')
        parser.set_defaults(debug=False)

        # siamese:
        parser.add_argument("--siamese", default=False, action="store_true")
        parser.add_argument("--loocv", default=False, action="store_true")
        parser.add_argument(
            "--explainable_model", default=False, action='store_true'
        )
        # XAI:
        parser.add_argument("--channel", default=False, action="store_true")

        parser.add_argument("--XAI_channel", default=None, type=list)

        self.parser = parser.parse_args()

        # if self.parser.siamese:
        #     self.parser.pair_up = True
        # else:
        #     self.parser.pair_up = False
        #
        # self.hidden_size_update()
        #
        # self.total_length_update()
        # self.total_length_checker()
        # self.device_update()
        # self.input_filename_update()
    def process(self):
        if self.parser.siamese:
            self.parser.pair_up = True
        else:
            self.parser.pair_up = False

        self.hidden_size_update()
        self.total_length_update()
        self.total_length_checker()
        self.device_update()
        self.input_filename_update()
        self.transforms_update()
        assert self.parser is not None, "Please initialize first before process argparse"
        return self.parser

    def transforms_update(self):
        transform = []
        if self.parser.interp:
            assert False, "NOT IMPLEMENTED!"
        if self.parser.mag_warp:
            transform.append(mag_warping())
        if len(transform) == 0:
            self.parser.transform = None
        else:
            self.parser.transform = transforms.Compose(transform)
        return

    def input_filename_update(self):
        # parser.add_argument('--input_filename', default='ROM_E1.npy', type=str)
        # if not self.parser.load:
        self.parser.init_directory = 'segment_sessions_one_repetition_data'
        if not self.parser.load:
            self.parser.init_directory = 'segment_sessions_one_repetition_data'
        if self.parser.input_filename in ['SPAR.npy']:
            print('IN')
            return

        if self.parser.metrics.lower() in ['rom']:
            self.parser.input_filename = 'ROM'
        elif self.parser.metrics.lower() in [
            'stb', 'res', 'resistance', 'stability'
        ]:
            self.parser.input_filename = 'STB'
        self.parser.input_filename += '_'
        if self.parser.exercise.lower() in ['sa', 'e1']:
            self.parser.input_filename += 'E1'
            self.parser.init_directory += '_E1'
        elif self.parser.exercise.lower() in ['er', 'e2']:
            self.parser.input_filename += 'E2'
            self.parser.init_directory += '_E2'
        elif self.parser.exercise.lower() in ['ff', 'e3']:
            self.parser.input_filename += 'E3'
            self.parser.init_directory += '_E3'
        elif self.parser.exercise.lower() in ['ir', 'e4']:
            self.parser.input_filename += 'E4'
            self.parser.init_directory += '_E4'
        self.parser.input_filename += '.npy'

    def total_length_update(self):
        self.parser.total_length = self.parser.num_reps * 400 if self.parser.num_reps > 1 and self.parser.total_length == 400 else self.parser.total_length
        return

    def hidden_size_update(self):
        if self.parser.cnn_hidden_size == 0:
            self.parser.cnn_hidden_size = self.parser.hidden_size
        if self.parser.rnn_hidden_size == 0:
            self.parser.rnn_hidden_size = self.parser.hidden_size
        return

    def total_length_checker(self):
        assert self.parser.total_length >= 650 if self.parser.num_reps >= 2 else True
        assert self.parser.total_length >= 950 if self.parser.num_reps >= 3 else True
        assert self.parser.total_length >= 1250 if self.parser.num_reps >= 4 else True
        assert self.parser.total_length >= 1550 if self.parser.num_reps >= 5 else True, \
            "checking the total length for repetition that is 5 within our current data collection"
        return

    def device_update(self):
        if self.parser.device in ['cuda', 'gpu', 'mps']:
            if torch.cuda.is_available():
                self.parser.device = torch.device('cuda')
            else:
                self.parser.device = torch.device('cpu')
                print("DEVICE IN GPU IS NOT AVAILABLE, SET TO CPU")
                # try m1 mps
                try:
                    print("Trying MPS..")
                    if torch.backends.mps.is_available():
                        self.parser.device = torch.device('mps')
                    else:
                        self.parser.device = torch.device('cpu')
                        if not torch.backends.mps.is_built():
                            print(
                                "MPS not available because the current PyTorch install was not "
                                "built with MPS enabled."
                            )
                        else:
                            print(
                                "MPS not available because the current MacOS version is not 12.3+ "
                                "and/or you do not have an MPS-enabled device on this machine."
                            )
                except AttributeError:
                    print(
                        "MPS module is not installed, please update pytorch, or MPS is not available in your device"
                    )
                else:
                    print('MPS in Mac M1 is not available')
                    self.parser.device = torch.device('cpu')
        else:
            self.parser.device = torch.device('cpu')
        return

    def __str__(self):
        assert self.parser is not None, "Please initialize first before print argparse"
        return str(self.parser)
