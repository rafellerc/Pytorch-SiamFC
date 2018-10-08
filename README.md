# SiameseFC PyTorch implementation

## Introduction

This project is the Pytorch implementation of the object Tracker presented in
this paper https://arxiv.org/pdf/1606.09549.pdf, and here
https://www.robots.ox.ac.uk/~luca/siamese-fc.html (project page). The original
version was written in matlab with the MatConvNet framework, available in
https://github.com/bertinetto/siamese-fc (trainining and tracking), but this
python version is adapted from the TensorFlow portability (tracking only),
available in https://github.com/torrvision/siamfc-tf .

## Organization

The project is divided into two major parts: Training and Tracking. The training
part deals with training a Siamese Network in the task of, given a frame sequence
of a video, matching an reference image with its corresponding position in a
subsequent frame.
Once the network has been trained we move to the Tracking part, which incorporates
this network inside a program that tries to follow a target (whose bounding box
is given in the first frame) throughout a whole video, proposing a new bounding
box in each frame. The Tracker includes all the logic needed to deal with how
to propose a search reagion for the network, how to evaluate the network's output,
how to deal with occlusions and scale changes.

###Tracking

The tracking part has a script designed to evaluate the metrics defined
in the OTB paper (Object Tracking Benchmark - Yi Wu, Jongwoo Lim, and Ming-Hsuan Yang,
available in https://www.researchgate.net/publication/273279481_Object_Tracking_Benchmark)
The 3 metrics provided are:
1. Precision: The precision is defined from the center location error, which
    is the distance in pixels between the center of the prediction and the center
    of the ground-truth for each frame. The Precision is then defined as the
    proportion of frames with center location error smaller than a given threshold.
    The choice of threshold is somewhat arbitrary and does not take into account
    the varying image sizes of the sequences, but the value used is 20 pixels,
    the same used in the paper.
2. Precision AUC: The area under the curve of the precision plot. The precision
    plot is a plot of the precision, as defined in 1), with varying thresholds.
    It corresponds to the average precision score for these different thresholds.
3. Mean IOU: The Intersection Over Union is the ratio between the intersection
    of the predicted bounding box and the ground-truth bounding box, and their
    Union. Its max value is 1 and minimum value is 0. The mean IOU is the mean
    of the IOU through all the frames of a sequence.

###Training
The training portion has a main script in the root folder called train.py
which is used to train new models in the ImageNet VID dataset, or simply evaluate
them in its evaluation set (though we have a test set, its ground-truth annotations
are not available). The execution of said script is parametrized by a set of
parameters contained in a .json file. To organize the parameters and outputs of
each execution we define an 'experiment' as the parameters of the execution plus
its outputs and logs. Thus, inside root/training we have a folder called 'experiments'
that should contain all of the folders defining each experiment. It has a default
folder which has the default parameters that can be used by the user to define
his own experiments. Each execution in mode 'train' trains the chosen model with
the whole train set for the given number of epochs (see --Parameters--), and
then validates it in terms of loss and the defined metrics in the eval set. The
values for the metrics is compared to the values for previous epochs and if the
model performs better its weights are saved in the experiment folder as 'best.pth.tar',
and at each epoch the last weights obtained are saved as 'last.pth.tar', and both of these
files can be loaded in further executions of the script by using the argument
'--restore_file best' or '--restore_file last'. Values for the best and last
models are stored in .json files 'metrics_val_best_weights.json' and
'metrics_val_last_weights.json' respectively. The execution also produces a log
file containing the execution information and eventual errors.

## How to Run

### Tracking:

1. Get the validation video sequences available in
https://drive.google.com/file/d/0B7Awq_aAemXQSnhBVW5LNmNvUU0/validationescribed
in https://github.com/torrvision/siamfc-tf and uncompress it.validation

2. Set the path to the root of the uncompressed validation folder in
`Root_Dir/tracking/experiments/<your_experiments>/environment.json` in the field
`root_dataset`. (there is a `<your_experiment>` folder called default, with
the default values of the Parameters, for more info, see [Parameters](#Parameters))
OBS: In case there is a `README` file inside cfnet-validation, remove it. The
videos folder should only have folders, not files.

3. Download the state dictionary of the Network and put it inside of
`Root_Dir/models` and set the value of `net` in
`Root_Dir/tracking/experiments/<your_experiments>/design.json` as the name of
the network (e.g. `baseline-conv5-original`)

4. Edit `Root_Dir/tracking/experiments/<your_experiments>/run.json` to enable
visualization (requires a display) or make_video to make mp4 videos of the
tracked sequences (see [Parameters](#Parameters)).

5. Run the script either by either making it executable with:
    ```bash
    $ chmod -x run_tracker_evaluation.py
    ```
and calling:
    ```bash
    $ ./run_tracker_evaluation.py
    ```
or simply:
    ```bash
    $ python3 run_tracker_evaluation.py
    ```
For user-defined experiments, run it as:
    ```bash
    $ ./run_tracker_evaluation.py --exp <your_experiment_name>
    ```

6. The script evaluates each one of the sequences and prints the corresponding
scores (see [Organization](#Organization)), and speed in terms of frames per second. E.g.:
`5 -- tc_Ball_ce2 -- Precision: 71.48 -- Precisions AUC: 29.89 -- IOU: 51.63 -- Speed: 25.58 --`
After evaluating every sequence it prints the mean results on the dataset.

OBS: The current implementation of the visualization using matplotlib is very
slow, so it slows down a lot the whole execution. I count on reimplementing it
using PyQtGraph to get faster plotting.

### Training
1. Assuming you have a copy of the ImageNet VID dataset, first create your
experiment folder inside root/training/experiments and create a file called
parameters.json with the same contents of the one inside experiments/default.
Change your parameters accordingly (see --Parameters--).

2. Assuming it's your first time executing the script, execute train.py
with the arguments:
    * `--mode train`
    * `--data_dir <path_to_imagenet_root>`
    * `--exp_name <name_of_your_experiment_folder>` (NOT THE FULL PATH)
    * `--timer`     (in case you want to profile the execution times. They will
                    be saved in the train.log file)

3. After your execution is complete you will have new files in your experiment
folder containing the models weights and its performance. A new execution in
train mode in the same experiment folder would overwrite your weight files, but
you can continue your training by executing the train script and further indicating:
    * `--restore_file best`     (Best model thus far in terms of metrics)
        or
    * `--restore_file last`     (Model output of the last epoch)

4. To simply evaluate your model in the eval set, execute in mode eval:
    `--mode eval`
and specify a model to be restored as before. The eval execution generates
a separate eval.log and a separate `metrics_test_best.json` containing the
obtained performance.

#Datasets

The dataset used for the tracker evaluation is a compilation of sequences from
the datasets [TempleColor](http://www.dabi.temple.edu/~hbling/data/TColor-128/TColor-128.html),
[VOT2013, VOT2014, and VOT2016](http://www.votchallenge.net/challenges.html). The Temple
Color and VOT2013 datasets are annotated with upright rectangular bounding boxes
(4 numbers), while VOT2014 and VOT2016 are annotated with rotated bounding boxes
(8 numbers). The order of the annotations seems to be the the following:

* TC: LowerLeft(x , y), Width, Height
* VOT2013: LowerLeft(x, y), Width, Height
* VOT2014: UpperLeft(x, y), LowerLeft(x, y), LowerRight(x, y), UpperRight(x, y)
* VOT2016: LowerLeft(x, y), LowerRight(x, y), UpperRight(x, y), UpperLeft(x, y)

OBS: It is possible that the annotations in VOT204 and 2016 simply represent a
sequence of points that define the contour of the bounding box, in no particular
order (but respecting the adjency of the the points in the rectangle). I didn't
check all the ground-truths to guarantee that all the annotations are in the
particular order described here. If something goes very wrong, you might want to
confirm this.

The dataset used for the training is the 2015 ImageNet VID, which contains
videos of targets where each frame is labeled with a bounding box around the
targets.

#Parameters

Both tracking and training scripts are defined in terms of user-defined parameters,
which define much of their behaviour. The parameters are defined inside .json files
and can be directly modified by the user. In both tracking and training a given
set of parameters defines what we call an `experiment` and thus they are placed
inside folders called experiments inside both training and tracking.
To define new parameters for a new experiment, copy the default experiment folder
with its .json files and name it accordingly, placing it always inside the
training(or tracking)/experiments folder. Here below we give a brief description
of the basic parameters:

##Tracking:

1. design.json:
    * `net`: The name of the file defining the network weights being used. The
        network must be inside RootDir/models.
    * `windowing`: The type of windowing being used for the displacement penalty,
        though currently only `cosine_sum` is supported.
    * `reference_sz`: The final size of the reference patch before being fed to
        the network. Paper defined value = 127
    * `search_sz`: The final size of the search patch before being fed to
        the network. Paper defined value = 255
    * `score_sz`: The size of the final score map produced by the network,
        the default output of the convolution layer is 33 pixels, but the
        network might implement an upscale layer, so the final output must
        be correctly informed.
    * `tot_stride`: The total stride of the network output, that is, the
        displacement in the input corresponding to a displacement of 1
        pixel in the output.
    * `context`: The context margin used to calculate the size of the context
        region around the reference image bounding box.
    * `pad_with_image_mean`: A boolean used to select wheter to use the mean of
        the image or zeros when padding.
2. environment.json:
    * `root_dataset`: The relative path (to the project root) of the VOT dataset.
    * `root_pretrained`: The relative path of the pretrained models folder.
3. evaluation.json:
    * `n_subseq`: The number of subsequences made from each video sequence. This
        parameter allows us to evaluate multiple versions of each sequence,
        each one starting at a different frame of the video (spaced linearly)
        and tracking until the end of the video. This kind of evaluation is
        defined by the OTB authors as TRE (Temporal Robustness Evaluation)
        and aims at giving an equal importance to all the frames in the sequence,
        due to the fact that once the tracker loses the target, all the subsequent
        frames are tracked using an incorrect bounding box prior, and therefore
        the results are pretty much meaningless. To fairly evaluate the later
        frames we simply run the tracker starting from a later frame.
        OBS: If you choose a number of subsequences bigger than the number
        of frames in the sequence some of the subsequences will be reapeated,
        which could introduce a small bias. To avoid that avoid setting n_subseq
        greater than 41 (the lenght of the smallest sequence).
    * `dist_threshold`: The distance threshold used to calculate the Precision
        metric. The value is somewhat arbitrary, but the OTB authors suggest
        using 20 pixels.
    * `stop_on_failure`: A boolen that indicates whether the Tracker should
        stop on failure, that is, when the IOU becomes 0. [NOT IMPLEMENTED YET]
    * `dataset`: The name of the dataset subset to be used in the evaluation,
        default is `validation`.
    * `video`: Indicates which videos should be used for the evaluation. `all`,
        indicates that every video present in the given dataset path should be
        used. If you wish to evaluate on a single video, write the basename
        of the folder corresponding to the wanted sequence.
    * `start_frame`: The start frame for each sequence.
4. hyperparameters.json:
    * `response_up`: The score map upscale factor, used both for its rescaling,
        and for calculating the exact position of the target based on the
        score map.
    * `window_influence`: The influence of the target displacement penalty on
        the score. This penalty penalises positions away from the last target
        position. 0 means no influence and 1 means that score is equal to the
        penalty map.
    * `z_lr`: Template learning rate. This has nothing to do with the training
        learning rate, it controls the rate of change of each update of the
        target template. 0 means there is no update, and only the template of
        the first frame is taken into account, and 1 means that the template
        used is that calculated in the last frame. For every value in between
        the used template is a running average of the templates calculated
        along the sequence.
    * `scale_num`: Number of scales searched. To account for scale change the
        tracker resizes the search image in different scales (half smaller,
        half larger and the original scale) before getting the score map.
        Currently it only supports 3 scales (it doesn't support only one
        scale either).
    * `scale_step`: The proportion each scale reduces or increases the scale in
        each step. For example, a scale_step of 2 doubles or halves the scale
        at each step.
    * `scale_penalty`: The penalty applied to the scaled scores. It penalises a
        change of scale and the penalty applied should be the scale_penalty
        raised to the power of the scale number, that is, if the scale increases
        or reduces the scale by 3 scale_steps its penalty should be raised to
        the 3rd power. Not implemented though, since currently we only use 3
        scales (-1, 0 and 1 steps).
    * `scale_lr`: The scale learning rate. It controls the rate of change of the
        predicted scale of the target. 0 means there is no scale update, and
        1 means that, in each frame, the predicted scale is the scale of the
        peak score map.
    * `scale_min`: Minimum allowed scale change.
    * `scale_max`: Maximum allowed scale change.
5. run.json:
    * `visualization`: Boolean, when set to 1 displays each tracked frame using
        matplotlib. Needs a display to do so. Otherwise set to 0.
    * `make_video`: In the case where no display is available or you want to
        use results later, set make_video to 1 to save the tracked sequences
        to mp4 files.
    * `output_video_folder`: The full path to the folder where you want to save
        the videos made with the option make_video.


##Training:

* `model_version`: A descriptive name for your model
* `batch_size`: The batch size in terms of reference/search region pairs. The
    authors of the paper suggested using a batch of 8 pairs.
* `num_epochs`: Total number of epochs.
* `train_epoch_size`: The number of iterations for each train epoch. If it is
    bigger than the total number of frames in the dataset, the epoch size
    defaults to the whole dataset, warning the user of it.
* `eval_epoch_size`: The number of iterations for each validation epoch. If it
    is bigger than the total number of frames in the dataset, the epoch size
    defaults to the whole dataset, warning the user of it.
* `save_summary_steps`: The number of batches between the metrics evaluation.
    If set to 0 all batches are evaluated in terms of the metrics after the
    loss is calculated. If set to 10, every tenth batch is evaluated.
* `optim`: The optimizer to be used during training. Options include *SGD* for
    stochastic gradient descent, and *Adam* for Adaptative Momentum.
* `optim_kwargs`: The keywords associated with each optimizer's initialization.
    It is itself a dictionary, and should follow the pytorch documentation.
    For example, if optim is `SGD` we could specify it as
    {`lr`: 1e-3, `momentum`:0.9}, for a learning rate of 0.001 and momentum
    of 0.9. Each optimizer has its available keywords, cf.
    https://pytorch.org/docs/stable/optim.html for more info.
* `max_frame_sep`: The maximum frame separation, the maximum distance between
    frames in each pairs chosen by the dataset. Default value is 50.
* `reference_sz`: The examplar region size in pixels.
* `search_sz`: The search region size in pixels.
* `final_sz`: The final size after the pairs are passed throught the model.
* `upscale`: A boolean to indicate if the network should have a bilinear upscale
    layer. OBS: Might slow training a lot.
* `pos_thr`: The positive threshold distance in the label, the threshold of
    the distance to the center that is considered a positive pixel.
* `neg_thr`: The negative threshold distance in the label, the threshold of
    the distance to the center that is considered a negative pixel. Every
    pixel with a distance between the positive and negative thresholds is
    considered neutral, and is not penalised either way.
* `context_margin`: The context margin for the examplar region.

