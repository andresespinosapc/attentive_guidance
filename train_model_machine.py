from comet_ml import Experiment
from callbacks import CometLogger, KSParsityDecreaser

import os
import argparse
import logging

import random
import numpy as np

import torch
import torchtext
from collections import OrderedDict

from machine.trainer import SupervisedTrainer
from machine.models import EncoderRNN
from machine.loss import NLLLoss
from machine.metrics import WordAccuracy, SequenceAccuracy, FinalTargetAccuracy, SymbolRewritingAccuracy, BLEU
from machine.dataset import SourceField, TargetField
from machine.util.checkpoint import Checkpoint
from machine.dataset.get_standard_iter import get_standard_iter
from machine.tasks import get_task

from trainer import AttentionTrainer
from models import DecoderRNN, Seq2seq

from loss import AttentionLoss, L1Loss
from fields import AttentionField

comet_args = {
    'api_key': '3CY3z4b2eYk08ZWoVOW912Yfl',
    'project_name': 'attentive-guidance',
    'workspace': 'andresespinosapc',
}
if os.environ.get('COMET_DISABLE'):
    comet_args['disabled'] = True
    comet_args['api_key'] = ''
experiment = Experiment(**comet_args)

def log_comet_parameters(opt):
    opt_dict = vars(opt)
    for key in opt_dict.keys():
        experiment.log_parameter(key, opt_dict[key])

TASK_DEFAULT_PARAMS = {
    'task_defaults': {
        'batch_size': 128,
        'k': 3,
        'max_len': 60,
        'patience': 5,
        'epochs': 20,
    },
    'baseline_2018': {
        'full_focus': False,
        'batch_size': 1,
        'embedding_size': 128,
        'hidden_size': 512,
        'rnn_cell': 'gru',
        'attention': 'pre-rnn',
        'attention_method': 'mlp',
        'max_len': 50,
    },
    'Hupkes_2018': {
        'full_focus': True,
        'batch_size': 1,
        'embedding_size': 16,
        'hidden_size': 512,
        'rnn_cell': 'gru',
        'attention': 'pre-rnn',
        'attention_method': 'mlp',
        'max_len': 50,
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

# CONSTANTS
IGNORE_INDEX = -1
LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'


def train_model():
    # Create command line argument parser and validate chosen options
    parser = init_argparser()
    opt = parser.parse_args()
    opt = validate_options(parser, opt)
    log_comet_parameters(opt)

    # Set random seed
    if opt.random_seed:
        random.seed(opt.random_seed)
        np.random.seed(opt.random_seed)
        torch.manual_seed(opt.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(opt.random_seed)

    # Prepare logging and data set
    init_logging(opt)
    src, tgt, train, dev, monitor_data = prepare_iters(opt)

    # Prepare model
    if opt.load_checkpoint is not None:
        seq2seq, input_vocab, output_vocab = load_model_from_checkpoint(
            opt, src, tgt)
    else:
        seq2seq, input_vocab, output_vocab = initialize_model(
            opt, src, tgt, train)

    pad = output_vocab.stoi[tgt.pad_token]
    eos = tgt.eos_id
    sos = tgt.SYM_EOS
    unk = tgt.unk_token

    # Prepare training
    losses, loss_weights, metrics = prepare_losses_and_metrics(
        opt, pad, unk, sos, eos, input_vocab, output_vocab)
    checkpoint_path = os.path.join(
        opt.output_dir, opt.load_checkpoint) if opt.resume_training else None
    # trainer = SupervisedTrainer(expt_dir=opt.output_dir)
    trainer = AttentionTrainer(expt_dir=opt.output_dir)

    # Train
    seq2seq, logs = trainer.train(seq2seq, train,
                                  num_epochs=opt.epochs, dev_data=dev, monitor_data=monitor_data, optimizer=opt.optim,
                                  teacher_forcing_ratio=opt.teacher_forcing_ratio, learning_rate=opt.lr,
                                  resume_training=opt.resume_training, checkpoint_path=checkpoint_path,
                                  losses=losses, metrics=metrics, loss_weights=loss_weights,
                                  checkpoint_every=opt.save_every, print_every=opt.print_every,
                                  custom_callbacks=[
                                      CometLogger(experiment),
                                      KSParsityDecreaser(seq2seq.decoder_module, experiment),
                                    ],
                                  random_seed=opt.random_seed)

    if opt.write_logs:
        output_path = os.path.join(opt.output_dir, opt.write_logs)
        logs.write_to_file(output_path)


def init_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, choices=['lookup', 'symbol_rewriting', 'SCAN'], default='lookup')
    parser.add_argument('--default_params_key', type=str, choices=['task_defaults', 'baseline_2018', 'Hupkes_2018'], default='task_defaults')
    parser.add_argument('--test_name', type=str, default='heldout_tables')
    parser.add_argument('--use_k_sparsity', action='store_true')
    parser.add_argument('--initial_k_sparsity', type=int, default=100)
    parser.add_argument('--k_sparsity_layers', type=str, nargs='*', choices=['encoder_hidden', 'encoder_outputs'])

    # Model arguments
    parser.add_argument('--random_seed', type=int, default=None)
    parser.add_argument('--train', help='Training data')
    parser.add_argument('--dev', help='Development data')
    parser.add_argument('--monitor', nargs='+', default=[],
                        help='Data to monitor during training')
    parser.add_argument('--output_dir', default='../models',
                        help='Path to model directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs', default=6)
    parser.add_argument('--optim', type=str, help='Choose optimizer',
                        choices=['adam', 'adadelta', 'adagrad', 'adamax', 'rmsprop', 'sgd'])
    parser.add_argument('--max_len', type=int,
                        help='Maximum sequence length', default=50)
    parser.add_argument(
        '--rnn_cell', help="Chose type of rnn cell", default='gru')
    parser.add_argument('--bidirectional', action='store_true',
                        help="Flag for bidirectional encoder")
    parser.add_argument('--embedding_size', type=int,
                        help='Embedding size', default=128)
    parser.add_argument('--hidden_size', type=int,
                        help='Hidden layer size', default=128)
    parser.add_argument('--n_layers', type=int,
                        help='Number of RNN layers in both encoder and decoder', default=1)
    parser.add_argument('--src_vocab', type=int,
                        help='source vocabulary size', default=50000)
    parser.add_argument('--tgt_vocab', type=int,
                        help='target vocabulary size', default=50000)
    parser.add_argument('--dropout_p_encoder', type=float,
                        help='Dropout probability for the encoder', default=0.2)
    parser.add_argument('--dropout_p_decoder', type=float,
                        help='Dropout probability for the decoder', default=0.2)
    parser.add_argument('--teacher_forcing_ratio', type=float,
                        help='Teacher forcing ratio', default=0.2)
    parser.add_argument(
        '--attention', choices=['pre-rnn', 'post-rnn'], default=False)
    parser.add_argument('--attention_method',
                        choices=['dot', 'mlp', 'concat', 'general'], default=None)
    parser.add_argument('--metrics', nargs='+', default=['seq_acc'], choices=[
                        'word_acc', 'seq_acc', 'target_acc', 'sym_rwr_acc', 'bleu'], help='Metrics to use')
    parser.add_argument('--full_focus', action='store_true')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size', default=32)
    parser.add_argument('--eval_batch_size', type=int,
                        help='Batch size', default=128)
    parser.add_argument(
        '--lr', type=float, help='Learning rate, recommended settings.\nrecommended settings: adam=0.001 adadelta=1.0 adamax=0.002 rmsprop=0.01 sgd=0.1', default=0.001)
    parser.add_argument('--ignore_output_eos', action='store_true',
                        help='Ignore end of sequence token during training and evaluation')

    # Data management
    parser.add_argument('--load_checkpoint',
                        help='The name of the checkpoint to load, usually an encoded time string')

    parser.add_argument('--save_every', type=int,
                        help='Every how many batches the model should be saved', default=100)
    parser.add_argument('--print_every', type=int,
                        help='Every how many batches to print results', default=100)

    parser.add_argument('--resume-training', action='store_true',
                        help='Indicates if training has to be resumed from the latest checkpoint')

    parser.add_argument('--log-level', default='info', help='Logging level.')
    parser.add_argument(
        '--write-logs', help='Specify file to write logs to after training')
    parser.add_argument('--cuda_device', default=0,
                        type=int, help='set cuda device to use')
    parser.add_argument('--l1_loss_inputs', type=str, nargs='*',
        choices=['encoder_hidden', 'encoder_outputs', 'model_parameters'], default=[])
    parser.add_argument('--use_attention_loss', action='store_true')
    parser.add_argument('--scale_l1_loss', type=float, default=1.)
    parser.add_argument('--scale_attention_loss', type=float, default=1.)
    parser.add_argument('--xent_loss', type=float, default=1.)

    return parser


def validate_options(parser, opt):
    if opt.resume_training and not opt.load_checkpoint:
        parser.error(
            'load_checkpoint argument is required to resume training from checkpoint')

    if not opt.attention and opt.attention_method:
        parser.error(
            "Attention method provided, but attention is not turned on")

    if opt.attention and not opt.attention_method:
        parser.error("Attention turned on, but no attention method provided")

    if torch.cuda.is_available():
        logging.info("Cuda device set to %i" % opt.cuda_device)
        torch.cuda.set_device(opt.cuda_device)

    if opt.attention:
        if not opt.attention_method:
            logging.info("No attention method provided. Using DOT method.")
            opt.attention_method = 'dot'

    return opt


def init_logging(opt):
    logging.basicConfig(format=LOG_FORMAT, level=getattr(
        logging, opt.log_level.upper()))
    logging.info(opt)


def prepare_iters(opt):

    use_output_eos = not opt.ignore_output_eos
    src = SourceField(batch_first=True)
    tgt = TargetField(include_eos=use_output_eos, batch_first=True)
    tabular_data_fields = [('src', src), ('tgt', tgt)]

    if opt.use_attention_loss:
        attn = AttentionField(use_vocab=False, ignore_index=IGNORE_INDEX)
        tabular_data_fields.append(('attn', attn))

    max_len = opt.max_len

    def len_filter(example):
        return len(example.src) <= max_len and len(example.tgt) <= max_len

    task = get_task(opt.task)
    opt.train = task.train_path
    opt.dev = task.valid_path
    opt.monitor = task.test_paths
    # dev_paths = list(filter(lambda x: opt.test_name in x, task.test_paths))
    # if len(dev_paths) <= 0:
    #     raise ValueError('Test data with name %s not found' % (opt.test_name))
    # elif len(dev_paths) == 1:
    #     opt.dev = dev_paths[0]
    # else:
    #     raise ValueError('More than one test data with name %s was found' % (opt.test_name))
    opt.full_focus = TASK_DEFAULT_PARAMS[opt.default_params_key]['full_focus']
    opt.batch_size = TASK_DEFAULT_PARAMS[opt.default_params_key]['batch_size']
    opt.embedding_size = TASK_DEFAULT_PARAMS[opt.default_params_key]['embedding_size']
    opt.hidden_size = TASK_DEFAULT_PARAMS[opt.default_params_key]['hidden_size']
    opt.rnn_cell = TASK_DEFAULT_PARAMS[opt.default_params_key]['rnn_cell']
    opt.attention = TASK_DEFAULT_PARAMS[opt.default_params_key]['attention']
    opt.attention_method = TASK_DEFAULT_PARAMS[opt.default_params_key]['attention_method']
    opt.max_len = TASK_DEFAULT_PARAMS[opt.default_params_key]['max_len']

    # generate training and testing data
    train = get_standard_iter(torchtext.data.TabularDataset(
        path=opt.train, format='tsv',
        fields=tabular_data_fields,
        filter_pred=len_filter
    ), batch_size=opt.batch_size)

    if opt.dev:
        dev = get_standard_iter(torchtext.data.TabularDataset(
            path=opt.dev, format='tsv',
            fields=tabular_data_fields,
            filter_pred=len_filter), batch_size=opt.eval_batch_size)
    else:
        dev = None

    monitor_data = OrderedDict()
    for dataset in opt.monitor:
        m = get_standard_iter(torchtext.data.TabularDataset(
            path=dataset, format='tsv',
            fields=tabular_data_fields,
            filter_pred=len_filter), batch_size=opt.eval_batch_size)
        monitor_data[dataset] = m

    return src, tgt, train, dev, monitor_data


def load_model_from_checkpoint(opt, src, tgt):
    logging.info("loading checkpoint from {}".format(
        os.path.join(opt.output_dir, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.output_dir, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model

    input_vocab = checkpoint.input_vocab
    src.vocab = input_vocab

    output_vocab = checkpoint.output_vocab
    tgt.vocab = output_vocab
    tgt.eos_id = tgt.vocab.stoi[tgt.SYM_EOS]
    tgt.sos_id = tgt.vocab.stoi[tgt.SYM_SOS]

    return seq2seq, input_vocab, output_vocab


def initialize_model(opt, src, tgt, train):
    # build vocabulary
    src.build_vocab(train.dataset, max_size=opt.src_vocab)
    tgt.build_vocab(train.dataset, max_size=opt.tgt_vocab)
    input_vocab = src.vocab
    output_vocab = tgt.vocab

    # Initialize model
    hidden_size = opt.hidden_size
    decoder_hidden_size = hidden_size * 2 if opt.bidirectional else hidden_size
    encoder = EncoderRNN(len(src.vocab), opt.max_len, hidden_size, opt.embedding_size,
                         dropout_p=opt.dropout_p_encoder,
                         n_layers=opt.n_layers,
                         bidirectional=opt.bidirectional,
                         rnn_cell=opt.rnn_cell,
                         variable_lengths=True)
    decoder = DecoderRNN(len(tgt.vocab), opt.max_len, decoder_hidden_size,
                         dropout_p=opt.dropout_p_decoder,
                         n_layers=opt.n_layers,
                         use_attention=opt.attention,
                         attention_method=opt.attention_method,
                         full_focus=opt.full_focus,
                         bidirectional=opt.bidirectional,
                         rnn_cell=opt.rnn_cell,
                         eos_id=tgt.eos_id, sos_id=tgt.sos_id,
                         use_k_sparsity=opt.use_k_sparsity,
                         initial_k_sparsity=opt.initial_k_sparsity,
                         k_sparsity_layers=opt.k_sparsity_layers,)
    seq2seq = Seq2seq(encoder, decoder)

    # This enables using all GPUs available
    if torch.cuda.device_count() > 1:
        logging.info("Using {} GPUs".format(torch.cuda.device_count()))
        seq2seq = torch.nn.DataParallel(seq2seq)

    seq2seq.to(device)

    return seq2seq, input_vocab, output_vocab


def prepare_losses_and_metrics(
        opt, pad, unk, sos, eos, input_vocab, output_vocab):
    use_output_eos = not opt.ignore_output_eos

    # Prepare loss and metrics
    losses = [NLLLoss(ignore_index=pad)]
    # loss_weights = [1.]
    loss_weights = [float(opt.xent_loss)]

    for l1_loss_input in opt.l1_loss_inputs:
        losses.append(L1Loss(input_name=l1_loss_input))
        loss_weights.append(opt.scale_l1_loss)

    if opt.use_attention_loss:
        losses.append(AttentionLoss(ignore_index=IGNORE_INDEX))
        loss_weights.append(opt.scale_attention_loss)

    for loss in losses:
        loss.to(device)

    metrics = []

    if 'word_acc' in opt.metrics:
        metrics.append(WordAccuracy(ignore_index=pad))
    if 'seq_acc' in opt.metrics:
        metrics.append(SequenceAccuracy(ignore_index=pad))
    if 'target_acc' in opt.metrics:
        metrics.append(FinalTargetAccuracy(ignore_index=pad, eos_id=eos))
    if 'sym_rwr_acc' in opt.metrics:
        metrics.append(SymbolRewritingAccuracy(
            input_vocab=input_vocab,
            output_vocab=output_vocab,
            use_output_eos=use_output_eos,
            output_sos_symbol=sos,
            output_pad_symbol=pad,
            output_eos_symbol=eos,
            output_unk_symbol=unk))
    if 'bleu' in opt.metrics:
        metrics.append(BLEU(
            input_vocab=input_vocab,
            output_vocab=output_vocab,
            use_output_eos=use_output_eos,
            output_sos_symbol=sos,
            output_pad_symbol=pad,
            output_eos_symbol=eos,
            output_unk_symbol=unk))

    return losses, loss_weights, metrics


if __name__ == "__main__":
    train_model()
