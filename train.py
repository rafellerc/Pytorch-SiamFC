#!/usr/bin/env python3
import argparse
import logging
from os.path import dirname, abspath, join, isfile

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import training.optim as optimz
from training.summary_utils import SummaryMaker
from training import train_utils
from training.datasets import ImageNetVID, ImageNetVID_val
from training.labels import create_BCELogit_loss_label
import training.models as mdl
import training.losses as losses
import training.metrics as met
from training.train_utils import RunningAverage
from utils.profiling import Timer
from utils.exceptions import IncompleteArgument
import utils.image_utils as imutils

device = torch.device("cuda") if torch.cuda.is_available() \
    else torch.device("cpu")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('-m', '--mode', default='train', choices=['train', 'eval'],
                        help="The mode of execution of the script. Options are "
                        "'train' to train a model, and 'eval' to evaluate a model "
                        "on the ImageNet eval dataset.")
    parser.add_argument('-d', '--data_dir', default='/home/ml2/workspace_rafael/dummy_Imagenet',
                        help="Full path to the directory containing the dataset")
    parser.add_argument('-e', '--exp_name', default='default',
                        help="The name of the experiment folder that contains the "
                             "parameters, checkpoints and logs. Must be in "
                             "training/experiments")
    parser.add_argument('-r', '--restore_file', default=None,
                        help="Optional, name of file to restore from (without its"
                             "extension .pth.tar)")
    parser.add_argument("-t", "--timer", action="store_true", dest="timer",
                        default=False, help="Writes the elapsed time for some "
                                            "sections of code on the log")
    parser.add_argument("-j", "--num_workers", dest="num_workers", type=int,
                        default=4, help="The number of workers for the dataloaders"
                                        " i.e. the number of additional"
                                        " dedicated threads to dataloading.")
    parser.add_argument('-f', '--imutils_flag', default='fast', type=str,
                        choices=imutils.VALID_FLAGS,
                        help="Optional, the flag of the image_utils defining "
                        "the image processing tools.")
    parser.add_argument('-s', '--summary_samples', default=5, type=int,
                        help="Optional, the number of pairs the TensorboardX "
                        "samples during validation to write in the summary. "
                        "For each epoch it saves the ref and the search "
                        "embeddings as well as the final correlation map.")
    args = parser.parse_args()
    return args


def main(args):
    root_dir = dirname(abspath(__file__))
    # Load the parameters from json file
    imagenet_dir = args.data_dir
    exp_dir = join(root_dir, 'training', 'experiments', args.exp_name)
    json_path = join(exp_dir, 'parameters.json')
    assert isfile(json_path), ("No json configuration file found at {}"
                               .format(json_path))
    params = train_utils.Params(json_path)
    # Add the timer option to the parameters
    params.update_with_dict({'timer': args.timer})
    params.update_with_dict({'num_workers': args.num_workers})

    train_utils.set_logger(join(exp_dir, '{}.log'.format(args.mode)))
    logging.info("----Starting train script in mode: {}----".format(args.mode))

    setup_timer = Timer(convert=True)
    setup_timer.reset()
    logging.info("Loading datasets...")

    # Get the correct model
    if params.model == 'BaselineEmbeddingNet':
        model = mdl.SiameseNet(mdl.BaselineEmbeddingNet(), upscale=params.upscale,
                               corr_map_size=33, stride=4)
    elif params.model == 'VGG11EmbeddingNet_5c':
        model = mdl.SiameseNet(mdl.VGG11EmbeddingNet_5c(), upscale=params.upscale,
                               corr_map_size=33, stride=4)
    elif params.model == 'VGG16EmbeddingNet_8c':
        model = mdl.SiameseNet(mdl.VGG16EmbeddingNet_8c(), upscale=params.upscale,
                               corr_map_size=33, stride=4)

    # Freeze all the indicated parameters
    for i, (name, parameter) in enumerate(model.named_parameters()):
        if i in params.parameter_freeze:
            logging.info("Freezing parameter {}".format(name))
            parameter.requires_grad = False

    model = model.to(device)
    # Set the tensorboard summary maker
    summ_maker = SummaryMaker(join(exp_dir, 'tensorboard'),
                              params,
                              model.upscale_factor)

    label_function = create_BCELogit_loss_label
    img_read_fcn = imutils.get_decode_jpeg_fcn(flag=args.imutils_flag)
    img_resize_fcn = imutils.get_resize_fcn(flag=args.imutils_flag)

    logging.info("Validation dataset...")

    metadata_val_file = join(exp_dir, "metadata.val")
    val_set = ImageNetVID_val(imagenet_dir,
                              label_fcn=label_function,
                              pos_thr=params.pos_thr,
                              neg_thr=params.neg_thr,
                              upscale_factor=model.upscale_factor,
                              cxt_margin=params.context_margin,
                              reference_size=params.reference_sz,
                              search_size=params.search_sz,
                              img_read_fcn=img_read_fcn,
                              resize_fcn=img_resize_fcn,
                              metadata_file=metadata_val_file,
                              save_metadata=metadata_val_file,
                              max_frame_sep=params.max_frame_sep)
    val_loader = DataLoader(val_set, batch_size=params.batch_size,
                            shuffle=False, num_workers=params.num_workers,
                            pin_memory=True)
    if params.eval_epoch_size > len(val_loader):
        logging.info('The user set eval_epoch_size ({}) is bigger than the '
                     'size of the eval set ({}). \n Setting '
                     'eval_epoch_size to the eval set size.'
                     .format(params.eval_epoch_size, len(val_loader)))
        params.eval_epoch_size = len(val_loader)

    # Define the model and optimizer

    # fetch loss function and metrics
    loss_fn = losses.BCELogit_Loss
    metrics = met.METRICS
    # Set the optional keyword arguments for the functions that need it
    metrics['center_error']['kwargs']['upscale_factor'] = model.upscale_factor

    try:
        if args.mode == 'train':

            logging.info("Training dataset...")

            metadata_train_file = join(exp_dir, "metadata.train")
            train_set = ImageNetVID(imagenet_dir,
                                    label_fcn=label_function,
                                    pos_thr=params.pos_thr,
                                    neg_thr=params.neg_thr,
                                    upscale_factor=model.upscale_factor,
                                    cxt_margin=params.context_margin,
                                    reference_size=params.reference_sz,
                                    search_size=params.search_sz,
                                    img_read_fcn=img_read_fcn,
                                    resize_fcn=img_resize_fcn,
                                    metadata_file=metadata_train_file,
                                    save_metadata=metadata_train_file,
                                    max_frame_sep=params.max_frame_sep)
            train_loader = DataLoader(train_set, batch_size=params.batch_size,
                                      shuffle=True, num_workers=params.num_workers,
                                      pin_memory=True)

            # Though I'm not a big fan of changing the value of a parameter
            # variable after it has been read, at least I let the user know I'm
            # changing it.
            if params.train_epoch_size > len(train_loader):
                logging.info('The user set train_epoch_size ({}) is bigger than the '
                             'size of the train set ({}). \n Setting '
                             'train_epoch_size to the train set size.'
                             .format(params.train_epoch_size, len(train_loader)))
                params.train_epoch_size = len(train_loader)

            logging.info("Done")
            logging.info("Setup time: {}".format(setup_timer.elapsed))
            parameters = filter(lambda p: p.requires_grad,model.parameters())
            optimizer = optimz.OPTIMIZERS[params.optim](parameters, **params.optim_kwargs)
            # Set the scheduler, that updates the learning rate using a exponential
            # decay. If you don't want lr decay set it to 1.
            logging.info("Using Exponential Learning Rate Decay of {}".format(params.lr_decay))
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, params.lr_decay)

            logging.info("Epoch sizes: {} in train and {} in eval"
                         .format(params.train_epoch_size, params.eval_epoch_size))

            logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
            with Timer(convert=True) as t:
                train_and_evaluate(model, train_loader, val_loader, optimizer,
                                   scheduler, loss_fn, metrics, params, exp_dir,
                                   args, summ_maker=summ_maker)
            if params.timer:
                logging.info("[profiling] Total time to train {} epochs, with {}"
                             " elements on training dataset and {} "
                             "on val dataset: {}"
                             .format(params.num_epochs, len(train_loader),
                                     len(val_loader), t.elapsed))

        elif args.mode == 'eval':
            logging.info("Done")
            with Timer(convert=True) as total:
                logging.info("Starting evaluation")
                # TODO write a decent Exception
                if args.restore_file is None:
                    raise IncompleteArgument("In eval mode you have to specify"
                                             " a model checkpoint to be loaded"
                                             " and evaluated."
                                             " E.g: --restore_file best")
                checkpoint_path = join(exp_dir, args.restore_file + '.pth.tar')
                train_utils.load_checkpoint(checkpoint_path, model)
                # Evaluate
                summ_maker.epoch = 0
                test_metrics = evaluate(model, loss_fn, val_loader, metrics,
                                        params, args, summ_maker=summ_maker)
                save_path = join(exp_dir,
                                 "metrics_test_{}.json".format(args.restore_file))
                train_utils.save_dict_to_json(test_metrics, save_path)
            if params.timer:
                logging.info("[profiling] Total evaluation time: {}"
                             .format(total.elapsed))

    except KeyboardInterrupt:
        logging.info("=== User interrupted execution ===")
        raise
    except Exception as e:
        logging.exception("Fatal error in main loop")
        logging.info("=== Execution Terminated with error ===")
    else:
        logging.info("=== Execution exited normally ===")


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, scheduler,
                       loss_fn, metrics, params, exp_dir, args, summ_maker=None):
    """Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object
            that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that
            fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        scheduler: (torch.optim.lr_scheduler.ExponentialLR) The exponential
            learning rate scheduler.
        loss_fn: a function that takes batch_output and batch_labels and
            computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using
            the output and labels of each batch
        params: (Params) hyperparameters
        exp_dir: (string) directory containing the parameters, weights and
            logs for the current experiment. The full path.
        args: The parser object containing the user informed arguments
        summ_maker: The SummaryMaker object that writes the training information
        to a tensorboard-readable file.
    """
    # reload weights from restore_file if specified
    # TODO load and set best validation error
    if args.restore_file is not None:
        restore_path = join(exp_dir, (args.restore_file + '.pth.tar'))
        logging.info("Restoring parameters from {}".format(restore_path))
        train_utils.load_checkpoint(restore_path, model)

    # best_val_c_error = float("inf")
    best_val_auc = 0
    # Before starting the first epoch do the eval
    logging.info('Pretraining evaluation...')
    # Epoch 0 is the validation epoch before the learning starts.
    summ_maker.epoch = 0
    val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params, args,
                           summ_maker=summ_maker)

    for epoch in range(params.num_epochs):
        # The first epoch after training is 1 not 0
        summ_maker.epoch = epoch + 1
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader, metrics, params,
              summ_maker=summ_maker)

        # Update the Learning rate
        scheduler.step()

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params,
                               args, summ_maker=summ_maker)

        val_auc = val_metrics['AUC']
        is_best = val_auc >= best_val_auc

        # Save weights
        train_utils.save_checkpoint({'epoch': epoch + 1,
                                     'state_dict': model.state_dict(),
                                     'optim_dict': optimizer.state_dict()},
                                    is_best=is_best,
                                    checkpoint=exp_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best auc")
            best_val_auc = val_auc

            # Save best val metrics in a json file in the model directory
            best_json_path = join(exp_dir, "metrics_val_best_weights.json")
            train_utils.save_dict_to_json(val_metrics, best_json_path)
            pass

        # Save latest val metrics in a json file in the model directory
        last_json_path = join(exp_dir, "metrics_val_last_weights.json")
        train_utils.save_dict_to_json(val_metrics, last_json_path)


def train(model, optimizer, loss_fn, dataloader, metrics, params,
          summ_maker=None):
    """Train the model
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and
            computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that
            fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using
            the output and labels of each batch
        params: (Params) hyperparameters
        summ_maker: The SummaryMaker object that writes the training information
        to a tensorboard-readable file.
    """
    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = {metric:RunningAverage() for metric in metrics}
    loss_avg = RunningAverage()
    profiled_values = ['load_data', 'batch']
    profil_summ = {name: RunningAverage() for name in profiled_values}
    timer = Timer()
    # Use tqdm for progress bar
    logging.info("Training on train set")
    with tqdm(total=params.train_epoch_size) as progbar:
        timer.reset()
        for i, sample in enumerate(dataloader):
            ref_img_batch = sample['ref_frame'].to(device)
            search_batch = sample['srch_frame'].to(device)
            labels_batch = sample['label'].to(device)
            # move to GPU if available
            profil_summ['load_data'].update(timer.elapsed)
            timer.reset()

            # compute model output and loss
            output_batch = model(ref_img_batch, search_batch)
            loss = loss_fn(output_batch, labels_batch)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.detach().cpu().numpy()
                labels_batch = labels_batch.detach().cpu().numpy()

                # compute all metrics on this batch
                for (metric_name, metric_dict) in metrics.items():
                    metric_fcn = metric_dict['fcn']
                    kwargs = metric_dict['kwargs']
                    metric_value = metric_fcn(output_batch, labels_batch, **kwargs)
                    summ[metric_name].update(metric_value)

            # update the average loss
            loss_avg.update(loss.item())
            profil_summ['batch'].update(timer.elapsed)
            progbar.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            progbar.update()
            timer.reset()

            if i >= params.train_epoch_size - 1:
                break

    # compute mean of all metrics in summary
    metrics_mean = {metric: values() for (metric, values) in summ.items()}
    metrics_mean['loss'] = loss_avg()
    if summ_maker:
        for (m_name, m_value) in metrics_mean.items():
            summ_maker.add_epochwise_scalar('train', m_name, m_value)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
    if params.timer:
        logging.info("[profiling][train] Mean load_data time: {}".format(profil_summ['load_data']()))
        logging.info("[profiling][train] Mean batch time: {}".format(profil_summ['batch']()))


@torch.no_grad()
def evaluate(model, loss_fn, dataloader, metrics, params, args, summ_maker=None):
    """Evaluate the model
    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and
            computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that
            fetches data
        metrics: (dict) a dictionary of functions that compute a metric using
            the output and labels of each batch
        params: (Params) hyperparameters
        args: The parser object containing the user informed arguments
        summ_maker: The SummaryMaker object that writes the training information
        to a tensorboard-readable file.
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []
    loss_avg = RunningAverage()
    profiled_values = ['load_data', 'batch', 'metrics']
    profil_summ = {name: RunningAverage() for name in profiled_values}
    timer = Timer()
    # compute metrics over the dataset
    logging.info("Validation on val set")
    with tqdm(total=params.eval_epoch_size) as progbar:
        timer.reset()
        # The TensorBoardX summary index, used to keep track of the number of
        # summaries already written.
        tbx_index = 0
        for i, sample in enumerate(dataloader):
            ref_img_batch = sample['ref_frame'].to(device)
            search_batch = sample['srch_frame'].to(device)
            labels_batch = sample['label'].to(device)
            # move to GPU if available
            profil_summ['load_data'].update(timer.elapsed)
            timer.reset()

            # compute model output
            embed_ref = model.get_embedding(ref_img_batch)
            embed_srch = model.get_embedding(search_batch)
            output_batch = model.match_corr(embed_ref, embed_srch)

            loss = loss_fn(output_batch, labels_batch)
            # Make a TensorBoardX summary for the number of pairs informed by
            # user in args.summary_samples. It takes the first n pairs, so it
            # it is guaranteed to save the results for the same pairs in each
            # execution, independently on the batch size.
            if (tbx_index < args.summary_samples) and (summ_maker is not None):
                # The batch_index selects an element of the batch. We get the
                # batch size every time instead of using the user informed batch
                # size to make sure no out of bounds exception raised for
                # the last batch which might contain less elements.
                batch_index = 0
                batch_size = embed_ref.shape[0]
                while (tbx_index < args.summary_samples) and (batch_index < batch_size):
                    # Since the val dataloader does not shuffle, we can use the
                    # tbx_index to get the information about the pairs in the
                    # list_pairs metadata.
                    seq, first_frame, second_frame = dataloader.dataset.list_pairs[tbx_index]
                    seq_name = dataloader.dataset.get_seq_name(seq)
                    index_string = "{}_{}_{}".format(tbx_index,
                                                     seq_name,
                                                     first_frame)

                    summ_maker.add_overlay("Ref_image_{}".format(index_string),
                                           embed_ref[batch_index],
                                           ref_img_batch[batch_index],
                                           cmap='inferno')
                    summ_maker.add_overlay("Search_image_{}".format(index_string),
                                           embed_srch[batch_index],
                                           search_batch[batch_index],
                                           cmap='inferno')
                    summ_maker.add_overlay("Correlation_map_{}-{}".format(index_string,
                                                                          second_frame),
                                           output_batch[batch_index],
                                           search_batch[batch_index],
                                           cmap='inferno',
                                           add_ref=ref_img_batch[batch_index])
                    logging.info("Saving embeddings for summary {}".format(tbx_index))
                    tbx_index += 1
                    batch_index += 1

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.cpu().numpy()
            labels_batch = labels_batch.cpu().numpy()

            profil_summ['batch'].update(timer.elapsed)
            timer.reset()

            # compute all metrics on this batch
            summary_batch = {metric_name: metric_dict['fcn'](output_batch,
                                                             labels_batch,
                                                             **(metric_dict['kwargs']))
                             for metric_name, metric_dict in metrics.items()}
            summary_batch['loss'] = loss.item()
            loss_avg.update(loss.item())
            summ.append(summary_batch)
            profil_summ['metrics'].update(timer.elapsed)
            progbar.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            progbar.update()
            timer.reset()

            if i >= params.eval_epoch_size - 1:
                break

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ])
                    for metric in summ[0]}
    if summ_maker:
        for (m_name, m_value) in metrics_mean.items():
            summ_maker.add_epochwise_scalar('val', m_name, m_value)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    if params.timer:
        logging.info("[profiling][eval] Mean load_data time: {}".format(profil_summ['load_data']()))
        logging.info("[profiling][eval] Mean batch time: {}".format(profil_summ['batch']()))
        logging.info("[profiling][eval] Mean metrics computation time: {}".format(profil_summ['metrics']()))
    return metrics_mean


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
