# SiameseFC PyTorch implementation

## Introduction

This project is the Pytorch implementation of the object tracker presented in 
[Fully-Convolutional Siamese Networks for Object Tracking](https://arxiv.org/pdf/1606.09549.pdf),
also available at [their project page](https://www.robots.ox.ac.uk/~luca/siamese-fc.html).
The original version was written in matlab with the MatConvNet framework, available
[here](https://github.com/bertinetto/siamese-fc) (trainining and tracking), but this
python version is adapted from the TensorFlow portability (tracking only),
available [here](https://github.com/torrvision/siamfc-tf).

## Organization

The project is divided into three major parts: [**Training**](#training), [**Tracking**](#tracking) and a [**Visualization Application**](#visualization-application).

## Training

The main focus of this work, the **Training** part deals with the training of the Siamese Network in order to learn a similarity metric between patches, as
described in the paper.

<center>
    <figure>    
        <img src="images/schema.png" height="50%" width="60%"/>
        <figcaption> The <b>Siamese Network</b> passes both images through
        the embedding network and then does the correlation between the embeddings, as shown below.
        </figcaption>
    </figure>        
</center>

<center>
    <figure>
        <img src="images/correlation_better.gif" height="50%" width="50%">
        <figcaption> The peak of the <b>correlation map</b> is supposed to be 
        located at the center of the map (because the images are both centered in the target).
        </figcaption>
    </figure>
</center>

<center>
    <figure>
        <img src="images/catpair.png" height="60%" width="60%">
        <figcaption>Here we overlay the correlation map with the search image, the peak value indicates the estimated center position of the target.
        </figcaption>
    </figure>
</center>

### Results - Training

The only training metric comparable between different parameters and implementations is the average *center error*  (or *center displacement*). The authors provide this metric in **appendix B** of their paper [Learning feed-forward one-shot learners](https://arxiv.org/pdf/1606.05233.pdf), which is 7.40 pixels for validation and 6.26 pixels for training.

Our Baseline Results are shown below:

<center>
    <figure>
        <img src="images/results_train.png" height="60%" width="100%">
        <figcaption>
        </figcaption>
    </figure>
</center>

We are around 4 pixels behind the authors, which we hypothesize that is mainly due to:

* The lack of a bicubic upscaling layer on the correlation map, which effectively causes our correlation map to have a resolution 4 times lower than the original image (due to the network's stride).
* The lack of class normalization of the loss to deal with the unbalance between negative and positive elements on the correlation map label (way more negative than positive positions). 

On the other hand, **we are way less prone to overfitting**, because we sample the pairs differently on each training epoch, as opposed to the authors, that choose all the pairs beforehand and use the same pairs on each training epoch.

This trained model is made available as *BaselinePretrained.pth.tar*.

### How to Run - Training

1. **Prerequisites:** The project was built using **python 3.6** and tested on Ubuntu 16.04 and 17.04. It was tested on a **GTX 1080 Ti** and a **GTX 950M**. Furthermore it requires [PyTorch 4.1](https://pytorch.org/). The rest of the dependencies can be installed with:  
```
# Tested on scipy 1.1.0
pip install scipy
# Tested on scikit-learn 0.20.0
pip install scikit-learn 
# Tested on tqdm 4.26.0
pip install tqdm
# Tested on tensorboardx 1.4
pip install tensorboardx
# Tested on imageio 2.4.1
pip install imageio
# To run the TensorBoard files later on install TensorFlow. 
pip install tensorflow

# To run the visualization app you need PyQt5 and pyqtgraph
# Tested on pyqt 5.6.0 
pip install pyqt5
# Tested on pyqtgraph 0.10.0
pip install pyqtgraph
```
In case you have Anaconda, install the conda virtual environment with:
```
# Used conda 4.5.11
conda env create -f environment.yaml
conda activate siamfc

# The pyqtgraph is not included and needs to be installed with pip
pip install pyqtgraph

```

(**OPTIONAL:** To accelerate the dataloading refer to [this section](#accelerating-data-loading))

2. Download the ImageNet VID Dataset in http://bvisionweb1.cs.unc.edu/ILSVRC2017/download-videos-1p39.php and extract it on the folder of your choice (*OBS: data reading is done in execution time, so if available extract the dataset in your SSD partition*). You can get rid of the *test* part of the dataset, since it has no Annotations. 

3. For each new training we must create an *experiment folder* (the folder stores the training parameters and the training output):
```
# Go to the experiments folder
cd training/experiments
# Create your experiment folder named <EXP_NAME>
mkdir <EXP_NAME>
# Copy the parameter file from the default experiment folder
cp default/parameters.json <EXP_NAME>/
```
4. Edit the *parameters.json* file with the desired parameters. The description of each parameter can be found [here](#training-parameters).
5. Run the *train.py* script:
```
# <EXP_NAME> is the name of the experiment folder, NOT THE PATH. 
python train.py --data_dir <FULL_PATH_TO_DATASET_ROOT> --exp_name <EXP_NAME>
```
<center>
    <figure>
        <img src="images/quick_train_screen.gif" height="60%" width="100%">
        <figcaption>This gif illustrates the execution of the training script. It uses very few epochs just to give a feel of the execution. A serious training execution could take a whole day.
        </figcaption>
    </figure>
</center>


* Use `--time` in case you want to profile the execution times. They will be saved in the train.log file.

6. The outputs will be:
* `train.log`: The log of the training, most of which is also displayed in the terminal.
* `metadata.train` and `metadata.val`: The metadata of the training and validation datasets, which is written on the start of the program. **Simply copy these files to any new experiment folder to save time on set up (about 15 minutes in my case).**      
* `metrics_val_last_weights.json`: The json containing the metrics of the most recent validation epoch. Human readable.  
* `metrics_val_best_weights.json`: The json containing the metrics of the validation epoch with the best AUC score. Human readable.   
* `best.pth.tar` and `last.pth.tar`: Dictionary containing the state_dictionary among other informations about the model. Can be loaded again later, for
training, validation or inference. Again *last* is the current epoch and *best* is the best one.   
* `tensorboard`: Folder containing the **tensorboard** files summarizing the training. It is separated in a *val* and a *train* folder so that the curves can be plotted in the same plot. To launch it type: 
```
# You need TensorFlow's TensorBoard installed to do so.
tensorboard --logdir <path_to_experiment_folder>/tensorboard 
``` 

<center>
    <figure>
        <img src="images/scalars.png" height="60%" width="60%">
        <figcaption>The three metrics stored are the mean <b>AUC</b> of the the ROC curve of the binary classification error between the label correlation map (defined by the parameters) and the actual correlation map, as well as the <b>Center Error</b>, which is the distance in pixels between the peak position of the correlation map and the actual center. Lastly, we also plot the mean <b>Binary Cross-Entropy Loss</b>, used to optimize the model.
        </figcaption>
    </figure>
</center>

**OBS**: Our definition of the loss is slightly different than the author's, but should be equivalent. Refer to [Loss Definition](#loss-definition).
<center>
    <figure>
        <img src="images/pairs.png" height="60%" width="60%">
        <figcaption>We also store Ref/Search/Correlation Map trios for each epoch, for debugging and exploration reasons. They are collected in each validation epoch, thus the first image corresponds to the validation before th e first training epoch. They allow us to see the evolution of the network's estimation after each training epoch. </figcaption>
    </figure>
</center>

**OBS:** To set the number of trios stored in each epoch, use the `--summary_samples <NUMBER_OF_TRIOS>` flag:
```
python train.py -s 10 -d <FULL_PATH_TO_DATASET_ROOT> -e <EXP_NAME> 
```
The images might take a lot of space though, especially if the number of epochs is large.

### Additional Uses

#### Retraining/Loading Pretrained Weights

You can continue training a network or load pretrained weights by calling the train script with the flag `--restore_file <NAME_OF_MODEL>` where <NAME_OF_MODEL> is the filename **without** the *.pth.tar* extension (e.g. *best*, for *best.pth.tar*). The program then searchs for the file NAME_OF_MODEL.pth.tar inside the experiment folder and loads its state as the initial state, the rest of the training script continues normally.  
```
python train.py -r <NAME_OF_MODEL> -d <FULL_PATH_TO_DATASET_ROOT> -e <EXP_NAME>
```

#### Evaluation Only

Once you finished training and dispose of a *.pth.tar file containing the network's weigths, you can evaluate it on the dataset by using the `--mode eval` combined with `--restore_file <NAME_OF_MODEL>`:
```
python train.py -m eval -r <NAME_OF_MODEL> -d <FULL_PATH_TO_DATASET_ROOT> -e <EXP_NAME>
```
The results of the evaluation are then stored in `metrics_test_best.json`.


### Tracking

The tracking part has a script designed to evaluate the metrics defined
in the OTB paper [Object Tracking Benchmark - Yi Wu, Jongwoo Lim, and Ming-Hsuan Yang](https://www.researchgate.net/publication/273279481_Object_Tracking_Benchmark)
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


### How to Run - Tracking

1. Get the validation video sequences available in
https://drive.google.com/file/d/0B7Awq_aAemXQSnhBVW5LNmNvUU0/validationescribed
in https://github.com/torrvision/siamfc-tf and uncompress it.

2. Set the path to the root of the uncompressed validation folder in
`Root_Dir/tracking/experiments/<your_experiments>/environment.json` in the field
`root_dataset`. (there is a `<your_experiment>` folder called default, with
the default values of the Parameters, for more info, see [Parameters](#parameters))
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


## Datasets

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

## Visualization Application

<center>
    <figure>
        <img src="images/viz_app.gif" height="60%" width="80%">
        <figcaption>
        </figcaption>
    </figure>
</center>
[Coming Soon]

## Parameters

Both tracking and training scripts are defined in terms of user-defined parameters,
which define much of their behaviour. The parameters are defined inside .json files
and can be directly modified by the user. In both tracking and training a given
set of parameters defines what we call an `experiment` and thus they are placed
inside folders called experiments inside both training and tracking.
To define new parameters for a new experiment, copy the default experiment folder
with its .json files and name it accordingly, placing it always inside the
training(or tracking)/experiments folder. Here below we give a brief description
of the basic parameters:

### Training Parameters:

* `model`: The Embedding Network to be used as the branch of the Siamese Network, the models are defined in [models.py](training/models.py). The models available are *BaselineEmbeddingNet*, *VGG11EmbeddingNet_5c*, *VGG16EmbeddingNet_8c*.
* `parameter_freeze`: A list of the layers of the Embedding Net that will be frozen (parameters will not be change). The numeration refers the *nn.Sequential* class that defines the Network in [models.py](training/models.py). E.g. [0, 1, 4, 5] with *BaselineEmbeddingNet* freezes the two first convolutional layers (0 and 4) along with the two first BatchNorm layers (1 and 5).
* `batch_size`: The batch size in terms of reference/search region pairs. The
    authors of the paper suggested using a batch of 8 pairs.
* `num_epochs`: Total number of training epochs. One validation epoch is done before training starts and after each training epoch.
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
* `reference_sz`: The reference region size in pixels. Default is 127.
* `search_sz`: The search region size in pixels. Default is 255.
* `final_sz`: The final size after the pairs are passed throught the model.
* `upscale`: A boolean to indicate if the network should have a bilinear upscale
    layer. OBS: Might slow training a lot.
* `pos_thr`: The positive threshold distance in the label, the threshold of
    the distance to the center that is considered a positive pixel.
* `neg_thr`: The negative threshold distance in the label, the threshold of
    the distance to the center that is considered a negative pixel. Every
    pixel with a distance between the positive and negative thresholds is
    considered neutral, and is not penalised either way.
* `context_margin`: The context margin for the reference region.


### Tracking Parameters:

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


### Accelerating Data Loading

[Coming Soon]

### Loss Definition

[Coming Soon]