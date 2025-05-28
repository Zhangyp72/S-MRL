class RecurrentNetwork:
    def __init__(self, input_size, context_size, hidden_size, output_size, inhibitor_size, synapse_size):
        self.input_size = input_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.inhibitor_size = inhibitor_size
        self.synapse_size = synapse_size
        self.union_size = input_size + context_size

        self.hidden_weights = WeightMatrix(self.union_size, hidden_size, synapse_size)
        self.hidden_weights_i = WeightMatrix(input_size, hidden_size, synapse_size)
        self.hidden_weights_c = WeightMatrix(context_size, hidden_size, synapse_size)
        self.output_weights = WeightMatrix(hidden_size, output_size, synapse_size)

        self.neuron = None
        self.rule = None
        self.error = None

        self.union_spikes = []
        self.hidden_spikes = [SpikeSequence([]) for _ in range(hidden_size)]
        self.output_spikes = [SpikeSequence([]) for _ in range(output_size)]

    def init_network_weight(self, weight_range):
        self.hidden_weights_i.randomise(weight_range)
        self.hidden_weights_c.randomise(weight_range)
        self.hidden_weights.randomise(weight_range)
        self.output_weights.randomise(weight_range)

    def clone_hidden_weight(self):
        for c in range(self.context_size):
            for h in range(self.hidden_size):
                for k in range(self.synapse_size):
                    weight = self.hidden_weights.get_weight(c, h, k)
                    self.hidden_weights_c.set_weight(c, h, k, weight)

    def clone_network_weight(self, other_net):
        self.output_weights.copy_from(other_net.output_weights)
        self.hidden_weights_i.copy_from(other_net.hidden_weights_i)
        self.hidden_weights_c.copy_from(other_net.hidden_weights_c)
        self.hidden_weights.copy_from(other_net.hidden_weights)
        self.neuron = other_net.neuron

    def set_spiking_neuron(self, neuron):
        self.neuron = neuron

    def running(self, sim_duration, time_step):
        time = 0
        while time <= sim_duration:
            for h in range(self.hidden_size):
                self.neuron.neuron_potential(self.union_spikes, self.hidden_spikes[h], self.hidden_weights, h, time)
                if self.neuron.neuron_spike():
                    self.hidden_spikes[h].insert_spike(time)

            for o in range(self.output_size):
                self.neuron.neuron_potential(self.hidden_spikes, self.output_spikes[o], self.output_weights, o, time, self.inhibitor_size)
                if self.neuron.neuron_spike():
                    self.output_spikes[o].insert_spike(time)
            time += time_step

    def set_error_function(self, error_func):
        self.error = error_func

    def spike_error_compute(self):
        return self.error.spike_error()

    def spike_neuron_error_compute(self):
        return self.error.spike_neuron_error()

    def set_learning_rule(self, rule):
        self.rule = rule

    def learning(self, weight_range):
        self.clone_hidden_weight()

        for i in range(self.hidden_size):
            for j in range(self.output_size):
                for k in range(self.synapse_size):
                    delta = (-1.0 if i < self.inhibitor_size else 1.0) * self.rule.adjust_weight_output(i, j, k)
                    new_weight = self.output_weights.clamp_weight(i, j, k, delta, weight_range)
                    self.output_weights.set_weight(i, j, k, new_weight)

        for i in range(self.input_size):
            for h in range(self.hidden_size):
                for k in range(self.synapse_size):
                    delta = self.rule.adjust_weight_hidden(i, h, k)
                    new_weight = self.hidden_weights_i.clamp_weight(i, h, k, delta, weight_range)
                    self.hidden_weights_i.set_weight(i, h, k, new_weight)

        for c in range(self.context_size):
            for h in range(self.hidden_size):
                for k in range(self.synapse_size):
                    delta = self.rule.adjust_weight_hidden(c, h, k)
                    new_weight = self.hidden_weights_c.clamp_weight(c, h, k, delta, weight_range)
                    self.hidden_weights_c.set_weight(c, h, k, new_weight)

        self.hidden_weights.set_weight_from_parts(self.hidden_weights_i, self.hidden_weights_c)

    def learn_failed(self):
        return any(s.get_size() == 0 for s in self.output_spikes)

    def set_inhibitor_size(self, inhibitor_size):
        self.inhibitor_size = inhibitor_size

    def get_inhibitor_size(self):
        return self.inhibitor_size

    def set_synapse_size(self, synapse_size):
        self.synapse_size = synapse_size

    def get_synapse_size(self):
        return self.synapse_size

    def get_hidden_weights(self):
        return self.hidden_weights

    def get_output_weights(self):
        return self.output_weights

    def print_hidden_weights(self):
        self.hidden_weights.print_weights()

    def print_output_weights(self):
        self.output_weights.print_weights()
