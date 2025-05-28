import math

class SpikeSequence:
    def __init__(self, spikes):
        self.spikes = spikes

    def get_size(self):
        return len(self.spikes)

    def get_spike(self, index):
        return self.spikes[index]


class WeightMatrix:
    def __init__(self, source_size, target_size, synapse_size):
        self.source_size = source_size
        self.target_size = target_size
        self.synapse_size = synapse_size
        # 3D weight matrix: [source][target][synapse]
        self.weights = [[[0.0 for _ in range(synapse_size)] for _ in range(target_size)] for _ in range(source_size)]

    def get_weight(self, source, target, synapse):
        return self.weights[source][target][synapse]

    def get_source_size(self):
        return self.source_size

    def get_synapse_size(self):
        return self.synapse_size


class SpikeResponseModel:
    def __init__(self, decayT, decayR, thresh):
        self.decayT = decayT
        self.decayR = decayR
        self.thresh = thresh
        self.potential = 0.0

    def neuron_spike(self):
        return self.potential >= self.thresh

    def neuron_potential(self, inputs, output, weights, t_index, sim_time, inhibit_size=None):
        if inhibit_size is not None:
            sp = self.spike_response_potential(inputs, weights, t_index, sim_time, inhibit_size)
        else:
            sp = self.spike_response_potential(inputs, weights, t_index, sim_time)
        rp = self.refractory_period_potential(output, sim_time)
        self.potential = sp + rp

    def spike_response_potential(self, inputs, weights, t_index, sim_time, inhibit_size=None):
        SR_potential = 0.0

        for s_index in range(weights.get_source_size()):
            for g in range(inputs[s_index].get_size()):
                spike_time = inputs[s_index].get_spike(g)

                if sim_time > spike_time:
                    for p_index in range(weights.get_synapse_size()):
                        weight = weights.get_weight(s_index, t_index, p_index)
                        if inhibit_size is not None and s_index < inhibit_size:
                            weight *= -1.0
                        delay = p_index + 1
                        SR_potential += weight * self.spike_response_function(sim_time - spike_time - delay)

        return SR_potential

    def spike_response_function(self, time):
        if time > 0:
            return (time / self.decayT) * math.exp(1 - time / self.decayT)
        return 0.0

    def refractory_period_potential(self, output, sim_time):
        RP_potential = 0.0
        for o in range(output.get_size()):
            spike_time = output.get_spike(o)
            RP_potential += self.refractoriness_function(sim_time - spike_time)
        return RP_potential

    def refractoriness_function(self, time):
        if time > 0:
            return (-2 * self.thresh) * math.exp(-time / self.decayR)
        return 0.0
