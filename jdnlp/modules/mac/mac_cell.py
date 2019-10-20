from jdnlp.modules.mac.units import ReadUnit, WriteUnit, ControlUnit

import torch
from torch.autograd import Variable
from torch import nn
from torch.nn.init import kaiming_uniform_, xavier_uniform_, normal
import torch.nn.functional as F


class MACCell(nn.Module):
    def __init__(
        self,
        dim,  # 512
        max_step=12,
        n_memories=3,
        self_attention=False,
        memory_gate=False,
        dropout=0.15,
        save_attns=False,
    ):
        super().__init__()

        self.control = ControlUnit(dim, max_step)
        self.read = ReadUnit(dim, n_memories, save_attns)
        self.write = WriteUnit(dim, self_attention=self_attention, memory_gate=memory_gate)

        self.mem_0 = nn.Parameter(torch.zeros(1, dim))
        self.control_0 = nn.Parameter(torch.zeros(1, dim))

        self.dim = dim
        self.max_step = max_step
        self.dropout = dropout
        self.n_memories = n_memories

        self.save_attns = save_attns
        self.saved_attns = []

    def get_mask(self, x, dropout):
        mask = torch.empty_like(x).bernoulli_(1 - dropout)
        mask = mask / (1 - dropout)

        return mask

    def forward(self, context, question, knowledge):
        """
        In the original implementation, we had:
        context = lstm_out -> [B, S, bidirectional * D] (contextual word embeddings)
        question = h -> [B, L * bidirectional, D] (final hidden layer)
        knowledge = img (KB)
        """
        
        batch_size = question.size(0)

        control = self.control_0.expand(batch_size, self.dim)
        memory = self.mem_0.expand(batch_size, self.dim)

        if self.training:
            control_mask = self.get_mask(control, self.dropout)
            memory_mask = self.get_mask(memory, self.dropout)
            control = control * control_mask
            memory = memory * memory_mask

        controls = [control]
        memories = [memory]

        for i in range(self.max_step):
            control = self.control(i, context, question, control)
            if self.training:
                control = control * control_mask
            controls.append(control)

            read = self.read(memories, knowledge, controls)
            memory = self.write(memories, read, controls)
            if self.training:
                memory = memory * memory_mask
            memories.append(memory)

        return memory
