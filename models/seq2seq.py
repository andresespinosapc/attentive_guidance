from machine.models import Seq2seq as OldSeq2seq


class Seq2seq(OldSeq2seq):
    KEY_MODEL_PARAMETERS = 'model_parameters'

    def forward(self, *args, **kwargs):
        decoder_outputs, decoder_hidden, ret_dict = super().forward(*args, **kwargs)
        ret_dict[self.KEY_MODEL_PARAMETERS] = self.parameters()

        return decoder_outputs, decoder_hidden, ret_dict
