# AI-Challenge-RTVC

Five video classification methods implemented in Keras and TensorFlow

Exploring the UCF101 video action dataset

[h/t @joshumaule and @surlyrightclick for the epic artwork.]
Classifying video presents unique challenges for machine learning models. As I’ve covered in my previous posts, video has the added (and interesting) property of temporal features in addition to the spatial features present in 2D images. While this additional information provides us more to work with, it also requires different network architectures and, often, adds larger memory and computational demands.

Today, we’ll take a look at different video action recognition strategies in Keras with the TensorFlow backend. We’ll attempt to learn how to apply five deep learning models to the challenging and well-studied UCF101 dataset.

Want the code? It’s all available on GitHub: Five Video Classification Methods. Pull requests encouraged!
This is part 3 in my series about video classification. If you missed the first two posts (gasp!), see here:

Continuous online video classification with TensorFlow, Inception and a Raspberry Pi

Or, using convolutional neural networks to identify what’s on TV
medium.com	
Continuous video classification with TensorFlow, Inception and Recurrent Nets

Part 2 of a series exploring continuous classification methods.
medium.com	
The video classification methods
We’ll look at each of our five methods in turn to see which one achieves the best top 1 and top 5 accuracy on UCF101. We’ll explore:

Classifying one frame at a time with a ConvNet
Using a time-distributed ConvNet and passing the features to an RNN, in one network
Using a 3D convolutional network
Extracting features from each frame with a ConvNet and passing the sequence to a separate RNN
Extracting features from each frame with a ConvNet and passing the sequence to a separate MLP
Constraints
Each of these methods could be its own blog post (or ten), so we’ll impose a few constraints to help simplify things and also to keep down computational complexity for future applications in real-time systems:

We won’t use any optical flow images. This reduces model complexity, training time, and a whole whackload of hyperparemeters we don’t have to worry about.
Every video will be subsampled down to 40 frames. So a 41-frame video and a 500 frame video will both be reduced to 40 frames, with the 500-frame video essentially being fast-forwarded.
We won’t do much preprocessing. A common preprocessing step for video classification is subtracting the mean, but we’ll keep the frames pretty raw from start to finish.
Every model has to fit into the 12 GiB of memory provided to us in the GPU in the new AWS p2.xlarge instances.
With these constraints, we know we won’t hit the ~94% state-of-the-art accuracy, but we’ll see if we can go in that direction.

Dataset
We’re going to use the popular UCF101 dataset. I find this dataset to have a great balance of classes and training data, as well as a lot of well-documented benchmarks for us to judge ourselves against. And unlike some of the newer video datasets (see YouTube-8M), the amount of data is manageable on modern systems.

UCF summarizes their dataset well:

With 13,320 videos from 101 action categories, UCF101 gives the largest diversity in terms of actions and with the presence of large variations in camera motion, object appearance and pose, object scale, viewpoint, cluttered background, illumination conditions, etc, it is the most challenging data set to date.
Challenge accepted!

Data preparation
The first thing we need to do is get the data in a format we can train on. We accomplish this in three steps:

Split all the videos into train/test folders
Extract jpegs of each frame for each video
Summarize the videos, their class, train/test status and frame count in a CSV we’ll reference throughout our training.
One important note is that the training set has many videos from the same “group”. This means there may be multiple videos of the same person from the same angle in the same setting performing the same action. It’s crucial that we don’t end up with videos from the same group in both the train and test groups, as we’d score unrealistically high on these classes.

UCF provides three train/test split recommendations that we can follow. For the sake of time, we use just split #1 for all of our experiments.

A note about the graphs below
Each graph includes three series:

The CNN-only top 1 accuracy in red, used as a baseline.
The top 5 categorical accuracy in green.
The top 1 categorical accuracy in blue.
I apologize for the lack of legend and the fugliness of the Matplotlib charts!

Method #0: Randomly guess or always choose the most common
This isn’t a real classification method, but if our model can’t beat random or the most used, we’re definitely not on the right track!

Trying to randomly guess the best result gives us ~0.9% accuracy. This makes sense since there are 101 classes, and, well… math.

Always guessing the most common class, “TennisSwing”, yields 1.32%. Also makes sense since TennisSwing labels are ~1.32% of our dataset. Okay, we’re on the right track, and we have something to beat. Let’s build some models.

Method #1: Classify one frame at a time with a CNN
For our first method, we’ll ignore the temporal features of video and attempt to classify each clip by looking at a single frame. We’ll do this by using a CNN, AKA ConvNet. More specifically, we’ll use Inception V3, pre-trained on ImageNet.

We’ll use transfer learning to retrain Inception on our data. This takes two steps.

First, we fine-tune the top dense layers for 10 epochs (at 10,240 images per epoch) in an attempt to retain as much of the previous learning as possible. (Updated Sep 26, ’07: A reader pointed out a bug in the train_cnn.py file that was causing the fine tuning to not update weights. I previously wrote that this step oddly did not yield meaningful results. The below chart is not updated to reflect this.) Fine tuning the top dense layers get us to ~52% top-1 validation accuracy, so it’s a great shortcut!

Next, we retrain the top two inception blocks. Coming up on 70 epochs, we’re looking really good, achieving a top 1 test accuracy of about 65%!


The blue line denotes the top 1 accuracy, while the green line denotes the top 5 accuracy.
It’s worth noting that we’re literally looking at each frame independently, and classifying the entire video based solely on that one frame. We aren’t looking at all the frames and doing any sort of averaging or max-ing.

Let’s spot check a random sample of images from the test set to see how we did:


Predictions: HighJump: 0.37, FloorGymnastics: 0.14, SoccerPenalty: 0.09. Actual: Nunchucks. Result: Fail

Predictions: WallPushups: 0.67, BoxingPunchingBag: 0.09, JugglingBalls: 0.08 . Actual: Wallpushups. Result: Top 1 correct!

Predictions: Drumming: 1.00. Actual: Drumming. Result: Top 1 correct!

Predictions: HandstandWalking: 0.32, Nunchucks: 0.16, JumpRope: 0.11 . Actual: JumpRope. Result: Top 5 correct!
Final test accuracy: ~65% top 1, ~90% top 5

Method #2: Use a time-distributed CNN, passing the features to an RNN, in one network
Now that we have a great baseline with Inception to try to beat, we’ll move on to models that take the temporal features of video into consideration. For our first such net, we’ll use Kera’s awesome TimeDistributed wrapper, which allows us to distribute layers of a CNN across an extra dimension — time. Obviously.

For the ConvNet part of the model, we’ll use a very small VGG16-style network. Ideally we’d use a deeper network for this part, but given that we have to load the whole thing into GPU memory, our options are fairly limited.

For the RNN part of the net, we’ll use a three-layer GRU, each consisting of 128 nodes, and a 0.2 dropout between each layer.

Unlike with method #1, where we got to use the pre-trained ImageNet weights, we’ll have to train the whole model on our data from scratch here. This could mean that we’ll require too much data or a much bigger network to achieve the same level of accuracy as the Inception whopper produced. However, it also means that our CNN weights will be updated on each backprop pass along with the RNN. Let’s see how it does!


In all charts, the red line is the Method #1 benchmark, green is the top 5 categorical accuracy, and blue is the top 1 categorical accuracy.
Yikes. How disappointing. Looks like we may need a more complex CNN to do the heavy lifting.

Final test accuracy: 20% top 1, 41% top 5

Method #3: Use a 3D convolutional network
Okay so training a CNN and an LSTM together from scratch didn’t work out too well for us. How about 3D convolutional networks?

3D ConvNets are an obvious choice for video classification since they inherently apply convolutions (and max poolings) in the 3D space, where the third dimension in our case is time. (Technically speaking it’s 4D, since our 2D images are represented as 3D vectors, but the net result is the same.)

However, they have the same drawback we ran into with method #2: memory! In Learning Spatiotemporal Features with 3D Convolutional Networks, the authors propose a network they call C3D that achieves 52.8% accuracy on UCF101. I was excited to attempt to reproduce these results, but I was stalled out with memory limitations of the 12 GiB GPU in the P2. The C3D simply wouldn’t run, even as I hacked off layer after layer.

As a plan B, I designed a smaller derivative, consisting of just three 3D convolutions, growing in size from 32 to 64 to 128 nodes. How’d we do?


After 28 epochs, we aren’t even close to hitting the benchmark we set with Inception. I did reduce the learning rate from 5e-5 to 1e-6 and trained for another 30 epochs (not graphed), which got us a little better, but still not in the ballpark.

Maybe training from top-to-bottom isn’t the way to go. Let’s go another direction.

Final test accuracy: 28% top 1, 51% top 5

Method #4: Extract features with a CNN, pass the sequence to a separate RNN
Given how well Inception did at classifying our images, why don’t we try to leverage that learning? In this method, we’ll use the Inception network to extract features from our videos, and then pass those to a separate RNN.

This takes a few steps.

First, we run every frame from every video through Inception, saving the output from the final pool layer of the network. So we effectively chop off the top classification part of the network so that we end up with a 2,048-d vector of features that we can pass to our RNN. For more info on this strategy, see my previous blog post on continuous video classification.

Second, we convert those extracted features into sequences of extracted features. If you recall from our constraints, we want to turn each video into a 40-frame sequence. So we stitch the sampled 40 frames together, save that to disk, and now we’re ready to train different RNN models without needing to continuously pass our images through the CNN every time we read the same sample or train a new network architecture.

For the RNN, we use a single, 4096-wide LSTM layer, followed by a 1024 Dense layer, with some dropout in between. This relatively shallow network outperformed all variants where I tried multiple stacked LSTMs. Let’s take a look at the results:


The RNN handily beats out the CNN-only classification method.
Hey that’s pretty good! Our first temporally-aware network that achieves better than CNN-only results. And it does so by a significant margin.

Final test accuracy: 74% top 1, 91% top 5

Method #5: Extract features from each frame with a CNN and pass the sequence to an MLP
Let’s apply the same CNN extraction process as in the previous method, but instead of sending each piece of the sequence to an RNN, we’ll flatten the sequence and pass the new (2,048 x 40) input vector into a fully connected network, AKA a multilayer perceptron (MLP). The hypothesis is that the MLP will be able to infer the temporal features from the sequence organically, without it having to know it’s a sequence at all.

(Plus, “multilayer perceptron” is one of the coolest terms in data science.)

After trying quite a few deep, shallow, wide and narrow networks, we find that the most performant MLP is a simple two-layer net with 512 neurons per layer:


Another method that beats the CNN-only benchmark! But not nearly as impressively as the RNN did. My gut tells me there’s room for parameter tuning on this to do better.

For most of our methods, I’m only showing results for our top performing network. However, with the MLP, something interesting happened when we tried deeper and wider networks: The top 1 accuracy floundered, but the top 5 accuracy went off the charts! Here’s a four-layer, 2,048-wide MLP:


That is basically perfect top 5 classification accuracy. But I digress…

Final test accuracy: 70% top 1, 88% top 5

And the winner is…

…the Inception ConvNet to extract features followed by a single-layer LSTM RNN!

Now it’s time to make a concession: 74% accuracy on UCF101 is nowhere near state-of-the-art, which stands at over 94% last I checked. However, with a single stream of images (no optical flow), minimal preprocessing, limited memory and very little parameter tuning, I think we’ve outlined some great jumping off points for diving deeper into each of these five classification methods.

My primary takeaway from this research is how powerful convolutional networks are at a wide range of tasks. Classifying “out in the wild” moving images with 101 classes while only looking at individual images at 65% accuracy is just astounding. And using it to extract features for other models turns out to be key to achieving satisfactory results with low memory availability.

My second takeaway is that there is a lot of work to be done here. This excites me greatly, and I hope this post helps kick start ideas and motivates others to explore the important world of video classification as well!

Want the code? It’s all available on GitHub: Five Video Classification Methods. Pull requests encouraged!

https://github.com/harvitronix/five-video-classification-methods

## References

https://blog.coast.ai/five-video-classification-methods-implemented-in-keras-and-tensorflow-99cad29cc0b5

https://www.analyticsvidhya.com/blog/2017/08/introduction-to-multi-label-classification/
