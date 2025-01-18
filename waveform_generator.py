# Tiago Fernandes, 2024
# 09/02/2024 fixes:
# - fixed the random generator;
# - fixed the Virgo notch filter frequencies;
# - started using rng for all random number generation;
# 01/03/2024 changes:
# - calculate the SNR of the signal;


import argparse
import json
import os
import warnings
from typing import Tuple

import numpy as np
from gwpy.timeseries import TimeSeries
from joblib import Parallel, delayed
from pycbc.detector import Detector
from pycbc.waveform import get_td_waveform
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

warnings.filterwarnings("ignore")


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types
    https://github.com/hmallen/numpyencoder
    """

    def default(self, obj):
        int_types = (
            np.int_,
            np.intc,
            np.intp,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        )
        if isinstance(obj, int_types):
            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)


class Generator:
    def __init__(
        self,
        work_dir: str = "outputs",
        noise_h1: str = "O3a_Noise_H1.hdf5",
        noise_l1: str = "O3a_Noise_L1.hdf5",
        noise_v1: str = "O3a_Noise_V1.hdf5",
        initial_seed: int = 0,
        # sample_rate: int = 2048,
    ):
        # self.sample_rate = sample_rate
        # assign variables
        self.work_dir = work_dir
        self.config_dir = os.path.join(self.work_dir, "config")
        self.data_dir = os.path.join(self.work_dir, "sig")
        self.bg_dir = os.path.join(self.work_dir, "bg")
        self.initial_seed = initial_seed

        # Create necessary folders if they don't exist.
        for folder in (self.work_dir, self.config_dir, self.data_dir, self.bg_dir):
            os.makedirs(folder, exist_ok=True)

        # check if the noise files exist
        for noise_file in (noise_h1, noise_l1, noise_v1):
            assert os.path.isfile(noise_file), f"File {noise_file} does not exist"

        # load the noise time-series
        self.noise_h1, self.noise_l1, self.noise_v1 = self.load_noise_TS(
            noise_h1, noise_l1, noise_v1
        )

        # get the sample rate
        self.sample_rate = self.noise_h1.sample_rate.value

    @property
    def configuration(self) -> dict:
        rng = self.rng
        sample_rate = self.sample_rate
        mass_min, mass_max = (1, 500)

        detector_ref = rng.choice(["H1", "L1", "V1"])
        mass1 = np.round(rng.uniform(mass_min, mass_max), 1)
        mass2 = np.round(rng.uniform(mass1, mass_max), 1)
        distance = rng.integers(10, 1000)
        # distance = rng.integers(int(500 * (1 - 0.35)), int(500 * (1 + 0.35)))
        declination = rng.uniform(-np.pi, np.pi)
        polarization = rng.uniform(0, 2 * np.pi)
        right_ascension = rng.uniform(0, 2 * np.pi)
        inclination = rng.random() * 2 * np.pi

        config = {
            "sample_rate": sample_rate,
            "delta_t": 1.0 / sample_rate,
            "mass1": mass1,
            "mass2": mass2,
            "distance": distance,
            "inclination": inclination,
            "declination": declination,
            "polarization": polarization,
            "right_ascension": right_ascension,
            "detector_ref": detector_ref,
            "approximant": "NRSur7dq4",
        }
        return config

    @staticmethod
    def load_noise_TS(path_to_noise1: str, path_to_noise2: str, path_to_noise3: str):
        assert (
            os.path.isfile(path_to_noise1)
            & os.path.isfile(path_to_noise2)
            & os.path.isfile(path_to_noise3)
        ), "One of the files does not exist"
        noise1 = TimeSeries.read(path_to_noise1)
        noise2 = TimeSeries.read(path_to_noise2)
        noise3 = TimeSeries.read(path_to_noise3)
        return noise1, noise2, noise3

    @staticmethod
    def _save_sample(path_to_save, filename, data_1, ts_1, ts_2, ts_3):
        savepath = os.path.join(path_to_save, filename)
        np.savez_compressed(savepath, qgraph=data_1, ts_h1=ts_1, ts_l1=ts_2, ts_v1=ts_3)

    @staticmethod
    def _write_json(data: dict, file: str, mode="w") -> None:
        with open(file, mode) as f:
            json.dump(data, f, indent=4, cls=NumpyEncoder)

    @staticmethod
    def get_asd_hlv(ts_h, ts_l, ts_v, length=2048) -> Tuple[list]:
        fft_length1 = int(max(2, np.ceil(length * ts_h.dt.decompose().value)))
        fft_length2 = int(max(2, np.ceil(length * ts_l.dt.decompose().value)))
        fft_length3 = int(max(2, np.ceil(length * ts_v.dt.decompose().value)))

        asd_h1 = ts_h.asd(overlap=0, fftlength=fft_length1, window="hann", method="welch")
        asd_h1 = asd_h1.interpolate(1.0 / ts_h.duration.decompose().value)

        asd_l1 = ts_l.asd(overlap=0, fftlength=fft_length2, window="hann", method="welch")
        asd_l1 = asd_l1.interpolate(1.0 / ts_l.duration.decompose().value)

        asd_v1 = ts_v.asd(overlap=0, fftlength=fft_length3, window="hann", method="welch")
        asd_v1 = asd_v1.interpolate(1.0 / ts_v.duration.decompose().value)
        return asd_h1, asd_l1, asd_v1

    @staticmethod
    def gen_data_strain(config: dict) -> Tuple[list]:
        hp, hc = get_td_waveform(
            approximant=config["approximant"],
            mass1=config["mass1"],
            mass2=config["mass2"],
            f_lower=20,
            f_final=config["sample_rate"],
            inclination=config["inclination"],
            distance=config["distance"],
            delta_t=config["delta_t"],
        )
        return hp, hc

    @staticmethod
    def whiten_hlv(ts_h, ts_l, ts_v, asd_h, asd_l, asd_v) -> Tuple[list]:
        ts_h = ts_h.whiten(asd=asd_h).bandpass(20, 300).notch(60).notch(120).notch(240)
        ts_l = ts_l.whiten(asd=asd_l).bandpass(20, 300).notch(60).notch(120).notch(240)
        # In the previous code version Virgo had notch filters at 60 Hz and its harmonics.
        ts_v = ts_v.whiten(asd=asd_v).bandpass(20, 300).notch(50).notch(100).notch(200)
        return ts_h, ts_l, ts_v

    @staticmethod
    def get_qtransform(ts_h, ts_l, ts_v, time_window, t_res, f_res) -> Tuple[list]:
        q_transforms = (
            ts.q_transform(
                outseg=time_window,
                tres=t_res,
                norm="median",
                frange=(20, 300),
                fres=f_res,
            ).value
            for ts in (ts_h, ts_l, ts_v)
        )
        return q_transforms

    def _create_samples(self, idx):
        # The rng is initialized for each config, with seed = initial_seed + idx.
        self.rng = np.random.default_rng(self.initial_seed + idx)
        det_names = ("H1", "L1", "V1")
        bgs = {"H1": self.noise_h1, "L1": self.noise_l1, "V1": self.noise_v1}
        dets = {det_name: Detector(det_name) for det_name in det_names}

        # FIXME
        # Resample backgrounds if they don't have the desired sample rate.
        # if int(bgs["H1"].sample_rate.value) != self.sample_rate:
        #     for det_name in det_names:
        #         bgs[det_name] = bgs[det_name].resample(self.sample_rate)

        # FIXME
        # Start background signals' time at 0.
        # for det_name in det_names:
        #     bgs[det_name].times = bgs[det_name].times - bgs[det_name].times[0]

        success = False
        # Generate a waveform, changing the configuration if an error occurs.
        while not success:
            config = self.configuration
            try:
                hp, hc = self.gen_data_strain(config)
                success = True
            except RuntimeError:
                continue
            except Exception as exception:
                print(f"\n\nException: {type(exception).__name__}\n\n")

        # Randomly choose an offset, used to choose the starting time.
        offset = self.rng.integers(len(hp), len(bgs["H1"]) - len(hp), endpoint=True)
        # Align the model and the background.
        t0 = bgs["H1"].times[offset].value
        hp.start_time = hc.start_time = t0
        config["t0"] = t0

        tdelays = {}
        for d in dets.values():
            dt = d.time_delay_from_detector(
                Detector(config["detector_ref"]),
                config["right_ascension"],
                config["declination"],
                t0,
            )
            tdelays[d.name] = dt

        models = dict()  # projected signals, no noise
        for det_name in det_names:
            # Project waves into the respective detector.
            models[det_name] = dets[det_name].project_wave(
                hp,
                hc,
                config["right_ascension"],
                config["declination"],
                config["polarization"],
            )
            # Convert to GWpy format.
            models[det_name] = TimeSeries.from_pycbc(models[det_name])

        # Get the time value of the maximum of the model amplitude.
        t_max = models["H1"].times.value[models[det_name].argmax()]
        t_max = t_max - self.rng.random() * 2*0.1 + 0.1

        sigs = dict()  # projected signals with noise
        for det_name in det_names:
            # Crop around signal maximum.
            bgs[det_name] = bgs[det_name].crop(t_max - 4, t_max + 4)
            # Inject the model into the noise.
            sigs[det_name] = bgs[det_name].inject(models[det_name])
            # Shift signals.
            sigs[det_name].shift(f"{tdelays[det_name]}s")

        # Calculate ASDs of the backgrounds.
        asd_h, asd_l, asd_v = self.get_asd_hlv(
            bgs["H1"], bgs["L1"], bgs["V1"], length=self.sample_rate
        )

        # Whiten the signals.
        sigs["H1"], sigs["L1"], sigs["V1"] = self.whiten_hlv(
            sigs["H1"], sigs["L1"], sigs["V1"], asd_h, asd_l, asd_v
        )

        # Pad the models to have at least sample_rate length
        for det_name in det_names:
            if (diff := self.sample_rate - len(models[det_name])) > 0:
                models[det_name] = models[det_name].pad((int(diff), 0))

        # Crop the models to have less than 2*signal_length
        for det_name in det_names:
            if (diff := len(models[det_name]) - 2 * len(sigs[det_name])) > 0:
                models[det_name] = models[det_name].crop(start=models[det_name].times.value[diff])

        # Whiten the models.
        models["H1"], models["L1"], models["V1"] = self.whiten_hlv(
            models["H1"], models["L1"], models["V1"], asd_h, asd_l, asd_v
        )

        print(
            f"Signals length H1: {len(sigs['H1'])}, L1: {len(sigs['L1'])}, V1: {len(sigs['V1'])}"
        )
        print(f"Signal sample rate: {sigs['H1'].sample_rate.value}")
        print(
            f"Models length H1: {len(models['H1'])}, L1: {len(models['L1'])}, V1: {len(models['V1'])}"
        )
        print(f"Model sample rate: {models['H1'].sample_rate.value}")

        # Calculate SNRs.
        snrs = {}
        for det_name in det_names:
            snrs[det_name] = sigs[det_name].correlate(models[det_name]).max().value

        config["snr"] = np.sqrt(np.sum([snrs[det_name] ** 2 for det_name in det_names]))

        # Create the qgraphs.
        x = y = 275
        time_window = (t_max - 0.28, t_max + 0.28)
        tres = abs(-t_max + 0.28 + t_max + 0.28) / x
        fres = 280 / y

        qg_H1, qg_L1, qg_V1 = self.get_qtransform(
            *sigs.values(),
            time_window,
            tres,
            fres,
        )
        qg_data = np.stack([qg_H1, qg_L1, qg_V1])

        bg_qg_H1, bg_qg_L1, bg_qg_V1 = self.get_qtransform(
            *bgs.values(),
            time_window,
            tres,
            fres,
        )
        bg_qg_data = np.stack([bg_qg_H1, bg_qg_L1, bg_qg_V1])

        # Save data.
        path = self.data_dir
        filename = f"{idx}_sample"
        config_name = self.config_dir + f"/{idx}_config.json"
        self._write_json(config, config_name)
        self._save_sample(
            path, filename, qg_data, sigs["H1"].value, sigs["L1"].value, sigs["V1"].value
        )
        print(f"file created and saved in: {path}/{filename}")

        path = self.bg_dir
        filename = f"{idx}_bg"
        self._save_sample(
            path, filename, bg_qg_data, bgs["H1"].value, bgs["L1"].value, bgs["V1"].value
        )
        print(f"file created and saved in: {path}/{filename}")

    def run_gen(self, n: int, n_jobs: int = 4) -> None:
        print("Generating qgraphs and time-series...")

        print(f"{n_jobs=}")

        if n_jobs > 1:
            Parallel(n_jobs=n_jobs)(
                delayed(self._create_samples)(idx=i) for i in tqdm(range(1, n + 1))
            )
        else:
            for i in tqdm(range(1, n + 1)):
                self._create_samples(idx=i)

def _normalize_qgraph(qgraph):
    qt_h1, qt_l1, qt_v1 = qgraph

    # Normalize each RGB channel.
    qt_h1 = qt_h1 - qt_h1.min() 
    qt_h1 = np.uint8(qt_h1 * 255 / qt_h1.max())
    qt_l1 = qt_l1 - qt_l1.min()
    qt_l1 = np.uint8(qt_l1 * 255 / qt_l1.max())
    qt_v1 = qt_v1 - qt_v1.min()
    qt_v1 = np.uint8(qt_v1 * 255 / qt_v1.max())

    # Stack channels together, and perform the necessary rotations.
    img = np.stack([qt_h1, qt_l1, qt_v1])
    img = Image.fromarray(img.T)
    #img = img.transpose(Image.FLIP_TOP_BOTTOM)

    return img

def generate_img(obj_path: str):
    # read the data
    obj = np.load(obj_path)
    qgraph = obj["qgraph"]
    qgraph = _normalize_qgraph(qgraph)
    #ts_h1 = obj["ts_h1"]
    #ts_l1 = obj["ts_l1"]
    #ts_v1 = obj["ts_v1"]
    # qgraph is (3, H, W)
    # qgraph = qgraph.transpose(1, 2, 0)
    # create the plot
    plt.imshow(qgraph, origin="lower", aspect="auto")
    plt.colorbar()
    plt.show()
    

def create_parser():
    parser = argparse.ArgumentParser(description="Generate GW data.")
    parser.add_argument(
        "-n",
        "--num_files",
        metavar="N",
        type=int,
        nargs="?",
        default=int(1e5),
        help="Number of time-series files to generate.",
    )
    parser.add_argument(
        "-nc",
        "--num_cores",
        metavar="NC",
        type=int,
        nargs="?",
        default=4,
        help="Number of cores.",
    )
    parser.add_argument(
        "-w",
        "--work-dir",
        metavar="PATH",
        type=str,
        nargs="?",
        default="outputs",
        help="Sub-directory for results.",
    )
    # parser.add_argument(
    #     "-sr",
    #     "--sample_rate",
    #     metavar="SR",
    #     type=int,
    #     nargs="?",
    #     default=2048,
    #     help="Sample rate.",
    # )
    parser.add_argument(
        "-nh1",
        "--noise_hanford",
        metavar="NH1",
        type=str,
        nargs="?",
        default="O3a_Noise_H1.hdf5",
        help="path to Hanford noise data.",
    )
    parser.add_argument(
        "-nl1",
        "--noise_livingston",
        metavar="NL1",
        type=str,
        nargs="?",
        default="O3a_Noise_L1.hdf5",
        help="path to Livingston noise data.",
    )
    parser.add_argument(
        "-nv1",
        "--noise-virgo",
        metavar="NV1",
        type=str,
        nargs="?",
        default="O3a_Noise_V1.hdf5",
        help="path to Virgo noise data.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        metavar="S",
        type=int,
        nargs="?",
        default=0,
        help="Seed to use for the random generator.",
    )
    parser.add_argument(
        "-v",
        "--visualize",
        #metavar="PATH",
        #type=str,
        action="store_true",
        help="Visualize the data.",
    )
    return parser


if __name__ == "__main__":
    parser = create_parser() # create parser
    args = parser.parse_args() # parse arguments
    
    # pretty print arguments
    print(f"Arguments: {args}")

    if args.visualize:
        generate_img("outputs/sig/79_sample.npz")

    else:
        gen = Generator(
            work_dir=args.work_dir,
            noise_h1=args.noise_hanford,
            noise_l1=args.noise_livingston,
            noise_v1=args.noise_virgo,
            initial_seed=args.seed,
            # sample_rate=args.sample_rate,
        )
    

        gen.run_gen(n=args.num_files, n_jobs=args.num_cores)
