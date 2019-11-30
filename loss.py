from machine.loss import Loss, NLLLoss
import torch

class AttentionLoss(NLLLoss):
    """ Cross entropy loss over attentions

    Args:
        ignore_index (int, optional): index of token to be masked
    """
    _NAME = "Attention Loss"
    _SHORTNAME = "attn_loss"
    _INPUTS = "attention_score"
    _TARGETS = "attention_target"

    def __init__(self, ignore_index=-1):
        super(AttentionLoss, self).__init__(ignore_index=ignore_index, size_average=True)

    def eval_step(self, step_outputs, step_target):
        batch_size = step_target.size(0)
        outputs = torch.log(step_outputs.contiguous().view(batch_size, -1).clamp(min=1e-20))
        self.acc_loss += self.criterion(outputs, step_target)
        self.norm_term += 1


class L1Loss(Loss):
    def __init__(self, input_name='encoding_hidden'):
        self.name = 'L1 %s Loss' % (input_name)
        self.log_name = 'l1_%s_loss' % (input_name)
        self.inputs = input_name
        self.acc_loss = 0
        self.norm_term = 0
        self.criterion = torch.tensor([])

    def get_loss(self):
        if isinstance(self.acc_loss, int):
            return 0

        return self.acc_loss.item() / self.norm_term

    def eval_batch(self, decoder_outputs, other, target_variable):
        outputs = other[self.inputs]
        if self.inputs == 'model_parameters':
            for parameter in outputs:
                self.acc_loss += parameter.abs().sum()
                self.norm_term += 1
        else:
            batch_size = outputs.size(0)
            self.acc_loss += outputs.abs().sum()
            self.norm_term += batch_size
