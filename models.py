import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F


### Propagation Networks

class RelationEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RelationEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        '''
        Args:
            x: [batch_size, n_relations, input_size]
        Returns:
            [batch_size, n_relations, output_size]
        '''
        B, N, D = x.size()
        x = self.model(x.view(B * N, D))
        return x.view(B, N, self.output_size)


class ParticleEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ParticleEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        '''
        Args:
            x: [batch_size, n_particles, input_size]
        Returns:
            [batch_size, n_particles, output_size]
        '''
        B, N, D = x.size()
        x = self.model(x.view(B * N, D))
        return x.view(B, N, self.output_size)


class Propagator(nn.Module):
    def __init__(self, input_size, output_size, residual=False):
        super(Propagator, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.residual = residual

        self.linear = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, res=None):
        '''
        Args:
            x: [batch_size, n_relations/n_particles, input_size]
        Returns:
            [batch_size, n_relations/n_particles, output_size]
        '''
        B, N, D = x.size()
        if self.residual:
            x = self.linear(x.view(B * N, D))
            x = self.relu(x + res.view(B * N, self.output_size))
        else:
            x = self.relu(self.linear(x.view(B * N, D)))

        return x.view(B, N, self.output_size)


class ParticlePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ParticlePredictor, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.linear_0 = nn.Linear(input_size, hidden_size)
        self.linear_1 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        '''
        Args:
            x: [batch_size, n_particles, input_size]
        Returns:
            [batch_size, n_particles, output_size]
        '''
        B, N, D = x.size()
        x = x.view(B * N, D)
        x = self.linear_1(self.relu(self.linear_0(x)))
        return x.view(B, N, self.output_size)


class PropModule(nn.Module):
    def __init__(self, args, input_dim, output_dim, batch=True, residual=False, use_gpu=False):

        super(PropModule, self).__init__()

        self.args = args
        self.batch = batch

        state_dim = args.state_dim
        attr_dim = args.attr_dim
        relation_dim = args.relation_dim
        action_dim = args.action_dim

        nf_particle = args.nf_particle
        nf_relation = args.nf_relation
        nf_effect = args.nf_effect

        self.nf_effect = args.nf_effect

        self.use_gpu = use_gpu
        self.residual = residual

        # particle encoder
        self.particle_encoder = ParticleEncoder(
            input_dim, nf_particle, nf_effect)

        # relation encoder
        self.relation_encoder = RelationEncoder(
            2 * input_dim + relation_dim, nf_relation, nf_relation)

        # input: (1) particle encode (2) particle effect
        self.particle_propagator = Propagator(
            2 * nf_effect, nf_effect, self.residual)

        # input: (1) relation encode (2) sender effect (3) receiver effect
        self.relation_propagator = Propagator(
            nf_relation + 2 * nf_effect, nf_effect)

        # input: (1) particle effect
        self.particle_predictor = ParticlePredictor(
            nf_effect, nf_effect, output_dim)

    def forward(self, state, Rr, Rs, Ra, pstep, verbose=0):

        if verbose:
            print('state size', state.size(), state.dtype)
            print('Rr size', Rr.size(), Rr.dtype)
            print('Rs size', Rs.size(), Rs.dtype)
            print('Ra size', Ra.size(), Ra.dtype)
            print('pstep', pstep)

        # calculate particle encoding
        particle_effect = Variable(torch.zeros((state.size(0), state.size(1), self.nf_effect)))
        if self.use_gpu:
            particle_effect = particle_effect.cuda()

        # receiver_state, sender_state
        if self.batch:
            Rrp = torch.transpose(Rr, 1, 2)
            Rsp = torch.transpose(Rs, 1, 2)
            state_r = Rrp.bmm(state)
            state_s = Rsp.bmm(state)
        else:
            Rrp = Rr.t()
            Rsp = Rs.t()
            assert state.size(0) == 1
            state_r = Rrp.mm(state[0])[None, :, :]
            state_s = Rsp.mm(state[0])[None, :, :]

        if verbose:
            print('Rrp', Rrp.size())
            print('Rsp', Rsp.size())
            print('state_r', state_r.size())
            print('state_s', state_s.size())

        # particle encode
        particle_encode = self.particle_encoder(state)

        # calculate relation encoding
        relation_encode = self.relation_encoder(torch.cat([state_r, state_s, Ra], 2))

        if verbose:
            print("relation encode:", relation_encode.size())

        for i in range(pstep):
            if verbose:
                print("pstep", i)

            if self.batch:
                effect_r = Rrp.bmm(particle_effect)
                effect_s = Rsp.bmm(particle_effect)
            else:
                assert particle_effect.size(0) == 1
                effect_r = Rrp.mm(particle_effect[0])[None, :, :]
                effect_s = Rsp.mm(particle_effect[0])[None, :, :]

            # calculate relation effect
            relation_effect = self.relation_propagator(
                torch.cat([relation_encode, effect_r, effect_s], 2))

            if verbose:
                print("relation effect:", relation_effect.size())

            # calculate particle effect by aggregating relation effect
            if self.batch:
                effect_agg = Rr.bmm(relation_effect)
            else:
                assert relation_effect.size(0) == 1
                effect_agg = Rr.mm(relation_effect[0])[None, :, :]

            # calculate particle effect
            particle_effect = self.particle_propagator(
                torch.cat([particle_encode, effect_agg], 2),
                res=particle_effect)

            if verbose:
                print("particle effect:", particle_effect.size())

        pred = self.particle_predictor(particle_effect)

        if verbose:
            print("pred:", pred.size())

        return pred


class PropNet(nn.Module):

    def __init__(self, args, residual=False, use_gpu=False):
        super(PropNet, self).__init__()

        self.args = args
        nf_effect = args.nf_effect
        attr_dim = args.attr_dim
        state_dim = args.state_dim
        action_dim = args.action_dim
        position_dim = args.position_dim

        # input: (1) attr (2) state (3) [optional] action
        if args.pn_mode == 'partial':
            batch = False
            input_dim = attr_dim + state_dim
            self.encoder = PropModule(args, input_dim, nf_effect, batch, residual, use_gpu)
            self.decoder = PropModule(args, nf_effect, state_dim, batch, residual, use_gpu)

            input_dim = (nf_effect + action_dim) * args.history_window
            self.roller = ParticlePredictor(input_dim, nf_effect, nf_effect)

        elif args.pn_mode == 'full':
            batch = True
            input_dim = attr_dim + state_dim + action_dim
            self.model = PropModule(args, input_dim, position_dim, batch, residual, use_gpu)

    def encode(self, data, pstep):
        # used only for partially observable case
        args = self.args
        state, Rr, Rs, Ra = data
        return self.encoder(state, Rr, Rs, Ra, pstep, args.verbose_model)

    def decode(self, data, pstep):
        # used only for partially obsevable case
        args = self.args
        state, Rr, Rs, Ra = data
        return self.decoder(state, Rr, Rs, Ra, pstep, args.verbose_model)

    def rollout(self, state, action):
        # used only for partially obsevable case
        if self.args.verbose_model:
            print('latent state', state.size())
            print('action', action.size())
        return self.roller(torch.cat([state, action], 2))

    def to_latent(self, state):
        if self.args.agg_method == 'sum':
            return torch.sum(state, 1, keepdim=True)
        elif self.args.agg_method == 'mean':
            return torch.mean(state, 1, keepdim=True)
        else:
            raise AssertionError("Unsupported aggregation method")

    def forward(self, data, pstep, action=None):
        # used only for fully observable case
        args = self.args
        attr, state, Rr, Rs, Ra = data
        # print(attr.size(), state.size(), Rr.size(), Rs.size(), Ra.size())
        if action is not None:
            state = torch.cat([attr, state, action], 2)
        else:
            state = torch.cat([attr, state], 2)
        return self.model(state, Rr, Rs, Ra, args.pstep, args.verbose_model)


