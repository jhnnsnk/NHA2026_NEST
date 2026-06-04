# %%
import os
import sys

nest_root = "/home/camilojd/AA_Work/Sussex/nest_installations/nest_39"
site_packages = f"{nest_root}/lib/python3.11/site-packages"
bin_dir = f"{nest_root}/bin"

os.environ["PYTHONPATH"] = site_packages + (f":{os.environ['PYTHONPATH']}" if os.environ.get("PYTHONPATH") else "")
os.environ["PATH"] = bin_dir + (f":{os.environ['PATH']}" if os.environ.get("PATH") else "")

if site_packages not in sys.path:
    sys.path.insert(0, site_packages)
    
import nest
import numpy as np
import matplotlib.pyplot as plt

# %%
def encode_input_to_wave(initial_time, end_time, sampling_rate):
    spike_times = np.arange(initial_time, end_time, sampling_rate)
    return spike_times

class DataloaderXOR:
    def __init__(self, n_in, epochs, iterations_steps, resolution, steps_init_input=10, steps_shift_sequences=5):
        self.n_in = n_in
        self.epochs = epochs
        self.iteration_steps = iterations_steps
        self.resolution = resolution

        self.duration_init_input = steps_init_input * resolution
        self.duration_shift_sequences = steps_shift_sequences * resolution

        self.sequences = ["00", "01", "10", "11"]
        self.sequence_weights = [0.25, 0.25, 0.25, 0.25] # Balanced dataset

        self.map_sequence_to_freq = {
            "0": 25,
            "1": 51
        }

        self.data, self.targets, self.pattern_history = self.generate_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, epoch):
        return self.data[epoch], self.targets[epoch]

    def generate_data(self):
        data = []
        targets = []
        pattern_history = []
        for epoch in range(self.epochs):
            pattern_sequence = np.random.choice(
                self.sequences, 
                p=self.sequence_weights)

            first_input = encode_input_to_wave(
                initial_time=(epoch * self.iteration_steps * self.resolution),
                end_time=(epoch+1) * self.iteration_steps * self.resolution,
                sampling_rate=self.map_sequence_to_freq[pattern_sequence[0]]
            ) + self.duration_init_input
            second_input = encode_input_to_wave(
                initial_time=(epoch * self.iteration_steps * self.resolution),
                end_time=(epoch+1) * self.iteration_steps * self.resolution,
                sampling_rate=self.map_sequence_to_freq[pattern_sequence[1]]
            ) + self.duration_init_input + self.duration_shift_sequences
            
            if (pattern_sequence == "01") or (pattern_sequence == "10"):
                target = 1
            else:
                target = 0
                
            data.append([first_input, second_input])
            targets.append(target)
            pattern_history.append({
                "pattern": pattern_sequence,
            })
        return data, targets, pattern_history

# %%
params_setup = {
    "resolution": 1.0,
    "total_num_virtual_procs": 11,  # number of virtual processes, set in case of distributed computing
}

nest.ResetKernel()
nest.set(**params_setup)
nest.set_verbosity("M_FATAL")

# %%
# Training and test parameters
epochs = 100
iteration_steps = 900
group_size = 1

eta_train = 1e-4  # learning rate for training phase 5e-3 * 0.01**2
eta_test = 0.0  # learning rate for test phase
n_iter_validate_every = 10
do_validation = True

test_size = 0.2

# Network parameters
n_in = 2  # number of input neurons
n_rec = 2  # number of recurrent neurons
n_out = 2  # number of readout neurons

steps = {
    "learning_window": 10,
    "offset_gen": 1,  # offset since generator signals start from time step 1
    "delay_in_rec": 1,  # connection delay between input and recurrent neurons
    "extension_sim": 1, # extra time step to close right-open simulation time interval in Simulate()
    "final_update": 3,  # extra time steps to update all synapses at the end of task
    "sequence": iteration_steps
}

steps["delays"] = steps["delay_in_rec"]  # time steps of delays
steps["total_offset"] = steps["offset_gen"] + steps["delays"]  # time steps of total offset

duration = {key: value * params_setup["resolution"] for key, value in steps.items()}

# %%
mm_rec = nest.Create("multimeter", params={
    "interval": params_setup["resolution"], # interval between two recorded time points
    "record_from": ["V_m", "surrogate_gradient", "learning_signal"], # dynamic variables to record
    "start": duration["total_offset"],  # start time of recording
})
mm_out = nest.Create("multimeter", params={
    "interval": params_setup["resolution"],
    "record_from": ["V_m", "readout_signal", "target_signal", "error_signal"],
    "start": duration["total_offset"],
})
sr_in = nest.Create("spike_recorder", params={
    "start": duration["offset_gen"]
})
sr_rec = nest.Create("spike_recorder", params={
    "start": duration["offset_gen"]
})
wr = nest.Create("weight_recorder", params={
    "start": duration["total_offset"]
})

recorders = {
    "mm_rec": mm_rec,
    "mm_out": mm_out,
    "sr_in": sr_in,
    "sr_rec": sr_rec,
    "wr": wr
}

# %%
params_common_syn_eprop = {
    "optimizer": {
        "type": "adam",  # algorithm to optimize the weights
        "batch_size": 1,
        "optimize_each_step": False,  # call optimizer every time step (True) or once per spike (False); both
        # yield same results for gradient descent, False offers speed-up
        "Wmin": -100.0,  # pA, minimal limit of the synaptic weights
        "Wmax": 100.0,  # pA, maximal limit of the synaptic weights
    },
    "weight_recorder": wr,
}
nest.SetDefaults("eprop_synapse", params_common_syn_eprop)

# %%

def calculate_glorot_dist(fan_in, fan_out):
    glorot_scale = 1.0 / max(1.0, (fan_in + fan_out) / 2.0)
    glorot_limit = np.sqrt(3.0 * glorot_scale)
    glorot_distribution = np.random.uniform(low=-glorot_limit, high=glorot_limit, size=(fan_in, fan_out))
    return glorot_distribution

class Network:
    def __init__(self, n_in, n_rec, n_out, delay, recorders):
        """
        Create network
            n_in (int)
            n_rec (int)
            n_out (int)
            recorders (dict): could contain keys sr_in, sr_rec, mm_rec, mm_out
        """
        # Network parameters
        self.n_in = n_in
        self.n_rec = n_rec
        self.n_out = n_out
        self.delay = delay # duration["step"]

        # Recorders
        self.recorders = recorders

        # Setup, creation and connection
        self.setup()
        self.create()
        self.connect()

        self.results_dict = {
            "error": [],
            "loss": [],
            "iteration": [],
            "label": [],
            "y_target": [],
            "y_pred": []
        }

    def setup(self):
        # Setup neuron parameters
        self.params_nrn_out = {
            "C_m": 1.0,  # pF, membrane capacitance - takes effect only if neurons get current input (here not the case)
            "E_L": 0.0,  # mV, leak / resting membrane potential
            "eprop_isi_trace_cutoff": 100,  # cutoff of integration of eprop trace between spikes
            "I_e": 0.0,  # pA, external current input
            "tau_m": 100.0,  # ms, membrane time constant
            "V_m": 0.0,  # mV, initial value of the membrane voltage
        }

        self.params_nrn_rec = {
            "beta": 1.7,  # width scaling of the pseudo-derivative
            "C_m": 1.0,
            "c_reg": 2.0 / duration["sequence"],  # coefficient of firing rate regularization
            "E_L": 0.0,
            "eprop_isi_trace_cutoff": 100,
            "f_target": 500.0,  # spikes/s, target firing rate for firing rate regularization
            "gamma": 0.5,  # height scaling of the pseudo-derivative
            "I_e": 0.0,
            "kappa": 0.99,  # low-pass filter of the eligibility trace
            "kappa_reg": 0.99,  # low-pass filter of the firing rate for regularization
            "surrogate_gradient_function": "piecewise_linear",  # surrogate gradient / pseudo-derivative function
            "t_ref": 0.0,  # ms, duration of refractory period
            "tau_m": 30.0,
            "V_m": 0.0,
            "V_th": 0.1,  # mV, spike threshold membrane voltage
        }
        scale_factor = 1.0 - self.params_nrn_rec["kappa"]  # factor for rescaling due to removal of irregular spike arrival
        self.params_nrn_rec["c_reg"] /= scale_factor**2

        # Setup connection parameters
        self.params_conn_one_to_one = {
            "rule": "one_to_one"
        }
        self.params_conn_all_to_all = {
            "rule": "all_to_all", "allow_autapses": False
        }
        
        # Setup synaptic parameters
        # Static synapses
        self.params_syn_static = {
            "synapse_model": "static_synapse",
            "delay": self.delay,
        }
        # E-Prop synapses
        params_syn_eprop = {
            "synapse_model": "eprop_synapse",
            "delay": self.delay,  # ms,
        }
        # In - Rec
        weights_in_rec = np.array(
            np.random.randn(self.n_in, self.n_rec).T / np.sqrt(self.n_in))
        self.params_syn_in = {
            **params_syn_eprop,
            "weight": weights_in_rec, 
            #"weight": nest.random.normal(0.0, self.n_in), 
        }
        # Rec - Rec
        weights_rec_rec = np.array(
            np.random.randn(self.n_rec, self.n_rec).T / np.sqrt(self.n_rec))
        np.fill_diagonal(weights_rec_rec, 0.0) 
        self.params_syn_rec = {
            **params_syn_eprop,
            "weight": weights_rec_rec, 
        }

        # Rec - Out
        weights_rec_out = np.array(
            calculate_glorot_dist(self.n_rec, self.n_out).T) * scale_factor
        self.params_syn_out = {
            **params_syn_eprop,
            #"weight": nest.random.uniform(-0.01, 0.01), 
            "weight": weights_rec_out, 
        }

        # Out - Rec
        weights_out_rec = np.array(
            np.random.randn(self.n_rec, self.n_out)) / scale_factor
        # Synaptic feedback for the learning signal
        self.params_syn_feedback = {
            "synapse_model": "eprop_learning_signal_connection",
            "delay": self.delay,
            #"weight": nest.random.normal(0, 0.01**2),
            "weight": weights_out_rec,
        }
        # Rate connection for learning window
        self.params_syn_learning_window = {
            "synapse_model": "rate_connection_delayed",
            "delay": self.delay,
            "receptor_type": 1,  # receptor type over which readout neuron receives learning window signal
        }
        # Rate connection for target
        self.params_syn_rate_target = {
            "synapse_model": "rate_connection_delayed",
            "delay": self.delay,
            "receptor_type": 2,  # receptor type over which readout neuron receives target signal
        }
    
    def create(self):
        """
        Network creation
        """
        # Spike input generator
        self.gen_spk_in = nest.Create("spike_generator", self.n_in)

        # Recurrent Network
        self.nrns_in = nest.Create("parrot_neuron", self.n_in)
        self.nrns_rec = nest.Create("eprop_iaf", self.n_rec, self.params_nrn_rec)
        self.nrns_out = nest.Create("eprop_readout", self.n_out, self.params_nrn_out)

        # Output spike generators
        self.gen_rate_target = nest.Create("step_rate_generator", self.n_out)
        self.gen_learning_window = nest.Create("step_rate_generator")
        self.gen_spk_final_update = nest.Create("spike_generator", 1)

    def connect(self):
        self.connect_network()
        self.connect_recorders()
        
    def connect_network(self):
        """
        Connect network following connections on display
        """
        nest.Connect(
            self.gen_spk_in, 
            self.nrns_in, 
            self.params_conn_one_to_one, 
            self.params_syn_static)  # connection 1

        # Should this connections be sparse?? (2,3,4)
        nest.Connect(
            self.nrns_in, 
            self.nrns_rec,
            self.params_conn_all_to_all,
            self.params_syn_in)  # connection 2
        nest.Connect(
            self.nrns_rec, 
            self.nrns_rec,
            self.params_conn_all_to_all,
            self.params_syn_rec)  # connection 3
        nest.Connect(
            self.nrns_rec, 
            self.nrns_out,
            self.params_conn_all_to_all,
            self.params_syn_out)  # connection 4
        
        nest.Connect(
            self.nrns_out, 
            self.nrns_rec, 
            self.params_conn_all_to_all, 
            self.params_syn_feedback)  # connection 5
        
        nest.Connect(
            self.gen_rate_target, 
            self.nrns_out, 
            self.params_conn_one_to_one, 
            self.params_syn_rate_target)  # connection 6
        nest.Connect(
            self.gen_learning_window, 
            self.nrns_out, 
            self.params_conn_all_to_all, 
            self.params_syn_learning_window)  # connection 7

        # Force final update to update all synapses 
        # it include all that have not being transmitted in the last update
        nest.Connect(
            self.gen_spk_final_update, 
            self.nrns_in + self.nrns_rec, 
            "all_to_all", {"weight": 1000.0})

    def connect_recorders(self):
        """
        Connect recorders
        """
        if self.recorders.get("sr_in"):
            nest.Connect(
                self.nrns_in, 
                self.recorders["sr_in"], 
                self.params_conn_all_to_all, 
                self.params_syn_static)
        if self.recorders.get("sr_rec"):
            nest.Connect(
                self.nrns_rec, 
                self.recorders["sr_rec"], 
                self.params_conn_all_to_all, 
                self.params_syn_static)
        if self.recorders.get("mm_rec"):
            nest.Connect(
                self.recorders["mm_rec"], 
                self.nrns_rec, 
                self.params_conn_all_to_all, 
                self.params_syn_static)
        if self.recorders.get("mm_out"):
            nest.Connect(
                self.recorders["mm_out"], 
                self.nrns_out, 
                self.params_conn_all_to_all, 
                self.params_syn_static)

    def evaluate(self, epoch, group_size, steps_sequence, steps_total_offset, steps_learning_window, resolution, phase_label):
        duration_sequence = steps_sequence * resolution
        duration_total_offset = steps_total_offset * resolution
        
        events_mm_out = self.recorders["mm_out"].get("events")
        
        readout_signal = events_mm_out["readout_signal"]
        target_signal = events_mm_out["target_signal"]
        senders = events_mm_out["senders"]
        times = np.around(events_mm_out["times"], 5) # Rounded since decimal errors on cond2

        cond1 = times > (epoch - 1) * group_size * duration_sequence + duration_total_offset
        cond2 = times <= epoch * group_size * duration_sequence + duration_total_offset
        idc = cond1 & cond2

        readout_signal = np.array([readout_signal[idc][senders[idc] == i] for i in set(senders)])
        target_signal = np.array([target_signal[idc][senders[idc] == i] for i in set(senders)])

        readout_signal = readout_signal.reshape((readout_signal.shape[0], 1, group_size, steps_sequence))
        target_signal = target_signal.reshape((target_signal.shape[0], 1, group_size, steps_sequence))

        readout_signal = readout_signal[:, :, :, -steps_learning_window :]
        target_signal = target_signal[:, :, :, -steps_learning_window :]

        loss = 0.5 * np.mean(np.sum((readout_signal - target_signal) ** 2, axis=3), axis=(0, 2))

        y_prediction = np.argmax(np.mean(readout_signal, axis=3), axis=0)
        y_target = np.argmax(np.mean(target_signal, axis=3), axis=0)
        accuracy = np.mean((y_target == y_prediction), axis=1)
        errors = 1.0 - accuracy

        if (epoch % 5 == 0) or phase_label == "validation":
            print(f"Reporting {phase_label} in epoch {epoch}: Loss {loss.item():.4f} | Error {errors.item()}")

        self.results_dict["iteration"].append(epoch)
        self.results_dict["error"].extend(errors)
        self.results_dict["loss"].extend(loss)
        self.results_dict["label"].append(phase_label)
        self.results_dict["y_target"].append(y_target)
        self.results_dict["y_pred"].append(y_prediction)

# %%
net = Network(n_in, n_rec, n_out, params_setup["resolution"], recorders)

# %%
def train_test_split(data_loader, test_size=0.2):
    train_loader = []
    test_loader = []
    for epoch, (data, target) in enumerate(data_loader):
        if epoch < np.rint(len(data_loader) * (1-test_size)):
            train_loader.append([data, target])
        else:
            test_loader.append([data, target])
    return train_loader, test_loader

data_loader = DataloaderXOR(
    n_in=n_in, 
    epochs=epochs, 
    iterations_steps=steps["sequence"], 
    resolution=params_setup["resolution"])

train_loader, test_loader = train_test_split(data_loader, test_size=test_size)

# %%
def run_phase(
        net,
        data,
        target,
        epoch,
        duration,
        steps,
        n_out,
        group_size,
        phase_label,
    ):
    one_hot_target = np.zeros(n_out)
    one_hot_target[target] = 1

    iteration_offset = epoch * group_size * duration["sequence"]
    
    params_gen_spk_in = [
        {"spike_times": data[0] + duration["total_offset"]},
        {"spike_times": data[1] + duration["total_offset"]},
    ]
    params_gen_rate_target = [
        {
            "amplitude_times": np.arange(
                0.0, 
                group_size * duration["sequence"], 
                duration["sequence"])
                + iteration_offset
                + duration["total_offset"],
            "amplitude_values": np.ones(group_size) if one_hot_target[i] else np.zeros(group_size)
        } for i in range(n_out)
    ]
    params_gen_learning_window = {
        "amplitude_times": np.hstack(
            [
                np.array([0.0, duration["sequence"] - duration["learning_window"]])
                + iteration_offset
                + group_element * duration["sequence"]
                + duration["total_offset"]
                for group_element in range(group_size)
            ]
        ),
        "amplitude_values": np.tile([0.0, 1.0], group_size)
    }
    
    nest.SetStatus(net.gen_spk_in, params_gen_spk_in)
    nest.SetStatus(net.gen_rate_target, params_gen_rate_target)
    nest.SetStatus(net.gen_learning_window, params_gen_learning_window)

    nest.Simulate(duration["total_offset"])
    nest.Simulate(duration["extension_sim"])

    if epoch > 0:
        net.evaluate(
            epoch=epoch, 
            group_size=group_size, 
            steps_sequence=steps["sequence"], 
            steps_total_offset=steps["total_offset"], 
            steps_learning_window=steps["learning_window"], 
            resolution=params_setup["resolution"], 
            phase_label=phase_label)

    duration_sim = group_size * duration["sequence"] - duration["total_offset"] - duration["extension_sim"]
    nest.Simulate(duration_sim)

# %%
def select_result_phase(results_dict, label):
    mask = np.array(np.array(net.results_dict["label"])) == label
    it = np.array(net.results_dict["iteration"])[mask]
    loss = np.array(net.results_dict["loss"])[mask]
    error = np.array(net.results_dict["error"])[mask]
    y_target = np.array(net.results_dict["y_target"])[mask]
    y_pred = np.array(net.results_dict["y_pred"])[mask]
    return {
        "it": it, 
        "loss": loss, 
        "error": error, 
        "y_target": y_target, 
        "y_pred": y_pred
    }

# Training
print("Starting train phase")
print("----------------------")
epoch = 0
for data, target in train_loader:
    phase_label = "train"
    # Variable train learning rate?
    if epoch % 20 == 0:
        print(f"Epoch {epoch}/{len(train_loader)}")
    #    eta_train = eta_train / 10
    params_common_syn_eprop["optimizer"]["eta"] = eta_train
    # Validation
    if do_validation and epoch % n_iter_validate_every == 0:
        params_common_syn_eprop["optimizer"]["eta"] = eta_test
        phase_label = "validation"

    nest.SetDefaults("eprop_synapse", params_common_syn_eprop)

    run_phase(
        net=net,
        data=data,
        target=target,
        epoch=epoch,
        duration=duration,
        steps=steps,
        n_out=n_out,
        group_size=group_size,
        phase_label=phase_label
    )
    epoch += 1
print()
print("Training phase finished")
print("----------------------")
train_curve = select_result_phase(net.results_dict, "train")
val_curve = select_result_phase(net.results_dict, "validation")
train_id_min = np.argmin(train_curve['loss'])
val_id_min = np.argmin(val_curve['loss'])
print(f"Total epochs: {epoch}")
print(f"Train last iteration: {train_curve['it'][-1]} | loss: {train_curve['loss'][-1]:.4f}")
print(f"Val last iteration: {val_curve['it'][-1]} | loss: {val_curve['loss'][-1]:.4f}")
print(f"Train best iteration: {train_curve['it'][train_id_min]} | loss: {train_curve['loss'][train_id_min]:.4f}")
print(f"Val best iteration: {val_curve['it'][val_id_min]} | loss: {val_curve['loss'][val_id_min]:.4f}")
print("----------------------")
print()

# %%
# Testing
print("Starting test phase")
print("----------------------")
for data, target in test_loader:
    phase_label = "test"
    params_common_syn_eprop["optimizer"]["eta"] = eta_test
    nest.SetDefaults("eprop_synapse", params_common_syn_eprop)

    run_phase(
        net=net,
        data=data,
        target=target,
        epoch=epoch,
        duration=duration,
        steps=steps,
        n_out=n_out,
        group_size=group_size,
        phase_label=phase_label
    )
    epoch += 1

nest.Simulate(duration["total_offset"])
nest.Simulate(duration["extension_sim"])

net.evaluate(
    epoch=epoch, 
    group_size=group_size, 
    steps_sequence=steps["sequence"], 
    steps_total_offset=steps["total_offset"], 
    steps_learning_window=steps["learning_window"], 
    resolution=params_setup["resolution"], 
    phase_label=phase_label)

# Force spike to update history of last spikes
duration_task = epoch * group_size * duration["sequence"] + duration["total_offset"]

net.gen_spk_final_update.set(
    {"spike_times": [duration_task + duration["extension_sim"] + 1]})

nest.Simulate(duration["final_update"])

# %%
import pandas as pd

# Input
events_sr_in = recorders["sr_in"].get("events")
df_in = pd.DataFrame(events_sr_in)
df_in.set_index(["senders"], inplace=True)

# Recurrent
mm_rec_events = recorders["mm_rec"].get("events")
df_mm_rec = pd.DataFrame(mm_rec_events)
df_mm_rec.set_index("senders", inplace=True)

sr_rec_events = recorders["sr_rec"].get("events")
df_sr_rec = pd.DataFrame(sr_rec_events)
df_sr_rec.set_index("senders", inplace=True)

# Readout
mm_out_events = recorders["mm_out"].get("events")
df_out = pd.DataFrame(mm_out_events)
df_out.set_index("senders", inplace=True)

# Weights
events_wr = recorders["wr"].get("events")

nrns = {
    "in": net.nrns_in.tolist(),
    "rec": net.nrns_rec.tolist(),
    "out": net.nrns_out.tolist(),
}

df_wr = pd.DataFrame(events_wr)
df_wr.set_index(["senders", "targets"], inplace=True)

in_rec_idx = pd.MultiIndex.from_product([nrns["in"], nrns["rec"]])
rec_rec_idx = pd.MultiIndex.from_product([nrns["rec"], nrns["rec"]])
rec_out_idx = pd.MultiIndex.from_product([nrns["rec"], nrns["out"]])

df_in_rec = df_wr.loc[df_wr.index.intersection(in_rec_idx)]
df_rec_rec = df_wr.loc[df_wr.index.intersection(rec_rec_idx)]
df_rec_out = df_wr.loc[df_wr.index.intersection(rec_out_idx)]


def plot_weight_time_course(df, ax, label=""):
    for i, idx in enumerate(df.index.unique()):
        df_filt = df.loc[idx]
        ax.plot(df_filt["times"], df_filt["weights"], "-", 
                label=f"{idx}",
        )

# %%
rows = 12
fig, axs = plt.subplots(rows, 1, figsize=(10, 14), sharex=True)

# Readout Membrane Potential
for i, sender in enumerate(np.unique(df_out.index)):
    mm_out_sender = df_out.loc[sender]
    axs[0].plot(mm_out_sender["times"], mm_out_sender["V_m"], color=f"C{i}", label=sender)
for epoch in range(epochs):
    axs[0].axvline(x=epoch * iteration_steps * params_setup["resolution"], color="k")
axs[0].set_ylabel(r"$v_k$")
axs[0].set_title(r"Readout Membrane Potential $v_k$")
axs[0].legend()
axs[0].grid()


# Plot 1: Recurrent spikes
for sender in np.unique(df_sr_rec.index):
    spk_times = df_sr_rec.loc[sender]
    axs[1].plot(spk_times, np.ones_like(spk_times) * sender, ".")
axs[1].set_ylabel("ID")
axs[1].set_title(r"Recurrent Spike Events $z_j$")
axs[1].grid()

# Plot 2: Recurrent Membrane Potential
for i, sender in enumerate(np.unique(df_mm_rec.index)):
    mm_rec_sender = df_mm_rec.loc[sender]
    axs[2].plot(mm_rec_sender["times"], mm_rec_sender["V_m"], color=f"C{i}", label=sender)
axs[2].set_ylabel(r"$v_j$")
axs[2].set_title(r"Recurrent Membrane Potential $v_j$")
axs[2].legend(loc="upper right")
axs[2].grid()

# 6. Input spike times
for sender in np.unique(df_in.index):
    spk_times = df_in.loc[sender]
    axs[3].plot(spk_times, np.ones_like(spk_times) * sender, ".")
for epoch in range(epochs):
    axs[3].axvline(x=group_size * epoch * iteration_steps * params_setup["resolution"], color="k")
axs[3].set_ylabel("ID")
axs[3].set_title(r"Input Spike Times $z_i$")
axs[3].grid()

# Surrogate gradients
for i, sender in enumerate(np.unique(df_mm_rec.index)):
    mm_rec_sender = df_mm_rec.loc[sender]
    axs[4].plot(mm_rec_sender["times"], mm_rec_sender["surrogate_gradient"], color=f"C{i}")
axs[4].set_ylabel(r"$\psi_j$")
axs[4].set_title(r"Surrogate Gradients $\psi_j$")
axs[4].grid()

# 8. Weight evolution
plot_weight_time_course(df_in_rec, axs[5], label="in_rec")
axs[5].set_ylabel(r"$W_\text{in}$")
axs[5].set_title(r"Weight Evolution $W_\text{in}$")
axs[5].grid()
plot_weight_time_course(df_rec_rec, axs[6], label="rec_rec")
axs[6].set_ylabel(r"$W_\text{rec}$")
axs[6].set_title(r"Weight Evolution $W_\text{rec}$")
axs[6].grid()
plot_weight_time_course(df_rec_out, axs[7], label="rec_out")
axs[7].set_ylabel(r"$W_\text{out}$")
axs[7].set_title(r"Weight Evolution $W_\text{out}$")
axs[7].grid()
#axs[7].legend(loc="upper right")

# 7. Target signal
for i, sender in enumerate(np.unique(df_out.index)):
    mm_out_sender = df_out.loc[sender]
    axs[8].plot(mm_out_sender["times"], mm_out_sender["target_signal"])
#for epoch in range(epochs):
#    axs[8].axvline(x=group_size * epoch * iteration_steps * params_setup["resolution"], linestyle="--", color="k")
axs[8].set_ylabel("Signal")
axs[8].set_title(r"Target $y^*_k$")
#axs[8].legend(loc="upper right")
axs[8].grid()

# Readout signal
for i, sender in enumerate(np.unique(df_out.index)):
    mm_out_sender = df_out.loc[sender]
    axs[9].plot(mm_out_sender["times"], mm_out_sender["readout_signal"], label=sender)
#for epoch in range(epochs):
#    axs[9].axvline(x=group_size * epoch * iteration_steps * params_setup["resolution"], linestyle="--", color="k")
axs[9].set_ylabel("Signal")
axs[9].set_title(r"Readout signal $y_k$")
axs[9].legend(loc="upper right")
axs[9].grid()

# Error signal
for i, sender in enumerate(np.unique(df_out.index)):
    mm_out_sender = df_out.loc[sender]
    axs[10].plot(mm_out_sender["times"], mm_out_sender["error_signal"])
axs[10].set_ylabel("Signal")
axs[10].set_title(r"Error signal $y_k-y^*_k$")
#axs[10].legend(loc="upper right")
axs[10].grid()

for i, sender in enumerate(np.unique(df_mm_rec.index)):
    mm_rec_sender = df_mm_rec.loc[sender]
    axs[11].plot(mm_rec_sender["times"], mm_rec_sender["learning_signal"], color="magenta")
axs[11].set_ylabel("Signal")
axs[11].set_title(r"Learning signal $L_j$")
axs[11].grid()

axs[11].set_xlabel("Time (ms)")

plt.tight_layout()


# %%
# Learning curves
train_curve = select_result_phase(net.results_dict, "train")
val_curve = select_result_phase(net.results_dict, "validation")
test_curve = select_result_phase(net.results_dict, "test") 

fig, ax = plt.subplots(2, 1, figsize=(10,5), sharex=True)
ax[0].plot(train_curve["it"], train_curve["loss"], "*-", label="train")
ax[0].plot(val_curve["it"], val_curve["loss"], "*-", label="val")
ax[0].plot(test_curve["it"], test_curve["loss"], "*-", label="test")
ax[0].grid()
ax[0].set_ylabel(r"$\frac{1}{K,N}\,\mathrm{\sum_{k,n}}\left(\sum_t\left(y_t - y_t^{*}\right)^2\right)$")
ax[0].legend(loc="upper right")
ax[0].set_title("Loss")


ax[1].plot(train_curve["it"], train_curve["error"], "*-", label="train")
ax[1].plot(val_curve["it"], val_curve["error"], "*-", label="val")
ax[1].plot(test_curve["it"], test_curve["error"], "*-", label="test")
ax[1].grid()
ax[1].set_title("Error")
ax[1].set_ylabel(r"$1 - \frac{1}{N}\,\mathrm{\sum_n}\left(\mathbf{1}(y^{*}=\hat{y})\right)$")
ax[1].legend(loc="upper right")
ax[1].set_xlabel("Epoch")

# %%
def compute_confusion_matrix(y_pred, y_target):
    tp, tn, fp, fn = 0, 0, 0, 0
    for y_p, y_t in zip(y_pred, y_target):
        if y_t == 1:
            if y_p == 1:
                tp += 1
            else:
                fn += 1
        elif y_t == 0:
            if y_p == 1:
                fp += 1
            else:
                tn += 1
    return np.array([[tp, fn], [fp, tn]])

y_pred = np.array(test_curve["y_pred"]).squeeze()
y_target = np.array(test_curve["y_target"]).squeeze()
conf_matrix = compute_confusion_matrix(y_pred, y_target)
print("Confusion matrix")
print(conf_matrix)

# Plot confusion matrix (rows: true label [1,0], cols: predicted label [1,0])
fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
im = ax_cm.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax_cm.text(j, i, f"{conf_matrix[i, j]:d}", ha='center', va='center',
                   color='white' if conf_matrix[i, j] > conf_matrix.max() / 2 else 'black')
ax_cm.set_xlabel('Predicted label')
ax_cm.set_ylabel('True label')
ax_cm.set_xticks([0, 1])
ax_cm.set_yticks([0, 1])
ax_cm.set_xticklabels(['1', '0'])
ax_cm.set_yticklabels(['1', '0'])
ax_cm.set_title('Confusion Matrix')
fig_cm.colorbar(im, ax=ax_cm)
plt.tight_layout()

plt.show()
