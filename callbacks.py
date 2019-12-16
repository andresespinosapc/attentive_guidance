from machine.util.callbacks import Callback


class CometLogger(Callback):
    def __init__(self, comet_experiment):
        super().__init__()

        self.experiment = comet_experiment

    def on_epoch_end(self, info=None):
        metrics = {
            'epoch': info['epoch'],
        }
        for split in ['train', 'eval']:
            for metric in info['%s_metrics' % (split)]:
                metrics.update({
                    '%s_%s' % (split, metric.log_name): metric.get_val(),
                })
            for loss in info['%s_losses' % (split)]:
                metrics.update({
                    '%s_%s' % (split, loss.log_name): loss.get_loss(),
                })
        self.experiment.log_metrics(metrics)


class KSParsityDecreaser(Callback):
    def __init__(self, model, comet_experiment,
        factor=0.9, patience=5, metric='seq_acc',
    ):
        super().__init__()

        self.model = model
        self.experiment = comet_experiment
        self.factor = factor
        self.patience = patience
        self.metric = metric
        self.best = 0
        self.num_bad_epochs = 0

    def on_epoch_end(self, info=None):
        # TEMP: Using eval info
        metric = next(filter(lambda x: x.log_name == self.metric, info['eval_metrics']))
        metric_val = metric.get_val()
        if metric_val <= self.best:
            self.num_bad_epochs += 1
        else:
            self.best = metric_val

        if self.num_bad_epochs >= self.patience:
            self.model.k_sparsity *= self.factor
            self.num_bad_epochs = 0

        metrics = {
            'k_sparsity': self.model.k_sparsity,
        }
        self.experiment.log_metrics(metrics)
