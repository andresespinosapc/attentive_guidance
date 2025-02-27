import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import Attention, HardGuidance
from machine.models import DecoderRNN as Decoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DecoderRNN(Decoder):
    """
    Subclass of machine.models.DecoderRNN that allows for training with Attentive Guidance.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        sos_id (int): index of the start of sentence symbol
        eos_id (int): index of the end of sentence symbol
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if the encoder is bidirectional (default False)
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        use_attention(bool, optional): flag indication whether to use attention mechanism or not (default: false)
        full_focus(bool, optional): flag indication whether to use full attention mechanism or not (default: false)

    Attributes:
        KEY_ATTN_SCORE (str): key used to indicate attention weights in `ret_dict`
        KEY_LENGTH (str): key used to indicate a list representing lengths of output sequences in `ret_dict`
        KEY_SEQUENCE (str): key used to indicate a list of sequences in `ret_dict`

    Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **inputs** (batch, seq_len, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
        - **encoder_hidden** (num_layers * num_directions, batch_size, hidden_size): tensor containing the features in the
          hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
        - **encoder_outputs** (batch, seq_len, hidden_size): tensor with containing the outputs of the encoder.
          Used for attention mechanism (default is `None`).
        - **function** (torch.nn.Module): A function used to generate symbols from RNN hidden state
          (default is `torch.nn.functional.log_softmax`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0).

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (seq_len, batch, vocab_size): list of tensors with size (batch_size, vocab_size) containing
          the outputs of the decoding function.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs }.
    """

    KEY_ENCODER_HIDDEN = 'encoder_hidden'
    KEY_ENCODER_OUTPUTS = 'encoder_outputs'

    def __init__(self, vocab_size, max_len, hidden_size,
            sos_id, eos_id,
            n_layers=1, rnn_cell='gru', bidirectional=False,
            input_dropout_p=0, dropout_p=0, use_attention=False, attention_method=None, full_focus=False,
            use_k_sparsity=False, initial_k_sparsity=100, k_sparsity_layers=None):

        super(DecoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                input_dropout_p, dropout_p,
                n_layers, rnn_cell)

        if use_k_sparsity and k_sparsity_layers is None:
            raise ValueError('To use k_sparsity you must specify at least one k_sparsity_layer')
        self.use_k_sparsity = use_k_sparsity
        self.k_sparsity = initial_k_sparsity
        self.k_sparsity_layers = k_sparsity_layers

        self.bidirectional_encoder = bidirectional
        input_size = hidden_size

        if use_attention != False and attention_method == None:
                raise ValueError("Method for computing attention should be provided")

        self.attention_method = attention_method
        self.full_focus = full_focus

        # increase input size decoder if attention is applied before decoder rnn
        if use_attention == 'pre-rnn' and not full_focus:
            input_size*=2

        self.rnn = self.rnn_cell(input_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)

        self.output_size = vocab_size
        self.max_length = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id

        self.init_input = None

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        if use_attention:
            self.attention = Attention(self.hidden_size, self.attention_method)
        else:
            self.attention = None

        if use_attention == 'post-rnn':
            self.out = nn.Linear(2*self.hidden_size, self.output_size)
        else:
            self.out = nn.Linear(self.hidden_size, self.output_size)
            if self.full_focus:
                self.ffocus_merge = nn.Linear(2*self.hidden_size, hidden_size)

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None,
                    function=F.log_softmax, teacher_forcing_ratio=0, provided_attention=None):

        # Prepare extra arguments for attention method
        attention_method_kwargs = {}
        if self.attention and isinstance(self.attention.method, HardGuidance):
            attention_method_kwargs['provided_attention'] = provided_attention


        ret_dict = dict()
        ret_dict[DecoderRNN.KEY_ENCODER_HIDDEN] = encoder_hidden
        ret_dict[DecoderRNN.KEY_ENCODER_OUTPUTS] = encoder_outputs
        if self.use_attention:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs,
                                                             function, teacher_forcing_ratio)
        
        if self.use_k_sparsity:
            if 'encoder_hidden' in self.k_sparsity_layers:
                indices = encoder_hidden.abs().topk(self.k_sparsity)[1]
                mask = torch.zeros(encoder_hidden.shape).to(device)
                mask = mask.scatter_(-1, indices, 1)
                encoder_hidden = encoder_hidden * mask
            elif 'encoder_outputs' in self.k_sparsity_layers:
                indices = encoder_outputs.abs().topk(self.k_sparsity)[1]
                mask = torch.zeros(encoder_outputs.shape).to(device)
                mask = mask.scatter_(-1, indices, 1)
                encoder_outputs = encoder_outputs * mask

        decoder_hidden = self._init_state(encoder_hidden)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_output, step_attn):
            decoder_outputs.append(step_output)
            if self.use_attention:
                ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)
            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        # When we use pre-rnn attention we must unroll the decoder. We need to calculate the attention based on
        # the previous hidden state, before we can calculate the next hidden state.
        # We also need to unroll when we don't use teacher forcing. We need perform the decoder steps
        # one-by-one since the output needs to be copied to the input of the next step.
        if self.use_attention == 'pre-rnn' or not use_teacher_forcing:
            unrolling = True
        else:
            unrolling = False

        if unrolling:
            symbols = None
            for di in range(max_length):
                # We always start with the SOS symbol as input. We need to add extra dimension of length 1 for the number of decoder steps (1 in this case)
                # When we use teacher forcing, we always use the target input.
                if di == 0 or use_teacher_forcing:
                    decoder_input = inputs[:, di].unsqueeze(1)
                # If we don't use teacher forcing (and we are beyond the first SOS step), we use the last output as new input
                else:
                    decoder_input = symbols

                # Perform one forward step
                if self.attention and isinstance(self.attention.method, HardGuidance):
                    attention_method_kwargs['step'] = di
                decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs,
                                                                         function=function, **attention_method_kwargs)
                # Remove the unnecessary dimension.
                step_output = decoder_output.squeeze(1)
                # Get the actual symbol
                symbols = decode(di, step_output, step_attn)

        else:
            # Remove last token of the longest output target in the batch. We don't have to run the last decoder step where the teacher forcing input is EOS (or the last output)
            # It still is run for shorter output targets in the batch
            decoder_input = inputs[:, :-1]

            # Forward step without unrolling
            if self.attention and isinstance(self.attention.method, HardGuidance):
                attention_method_kwargs['step'] = -1
            decoder_output, decoder_hidden, attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs, function=function, **attention_method_kwargs)

            for di in range(decoder_output.size(1)):
                step_output = decoder_output[:, di, :]
                if attn is not None:
                    step_attn = attn[:, di, :]
                else:
                    step_attn = None
                decode(di, step_output, step_attn)

        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict

