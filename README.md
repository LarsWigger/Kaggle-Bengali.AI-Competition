# OCR Kaggle Competition

## Competition Description
The competition can be viewed [here](https://www.kaggle.com/c/bengaliai-cv19).
The task was a multilabel classification problem: For every image, one of 168 Grapheme Roots, 11 Vowel Diacritics and 7 Consonant Diacritics had to be predicted. The training data consisted of 200840 137x236 grayscale images (given as arrays/dataframes of numerical pixel values divided among four .parquet files). The private validation data consisted of a similar, possibly identical, amount of images.

## Result
I scored 92.71% on the private leaderboard, making me 312 out of 2059 or top 16%. You can validate this on [my Kaggle profile](https://www.kaggle.com/larswigger/competitions).

## Experience Level
I had not prior practical experience actually working with machine learning or even deep learning. Everything I learned (which was a lot), I learned along the way. I did, however, have a reasonable understanding of how it worked in theory before trying my hand on it.

## The GPU-Problem
Because I do not own any GPU other than an Intel UHD 620 (which is obviously useless for deep learning), I had to use cloud services. Since I did not want to pay for my experiments, I was somewhat limited in my options. There were two services I used:
* Kaggle Kernels: These give you 30h of GPU-time per week. They can run up to 9h in one go and are nicely integrated into the Kaggle ecosystem. Unfortunately, 30h of GPU-time is very little for a deep learning problem, so I had to look for something else.
* Google Colab: Google Colab is offered by Google for free. It allows the user to use spare GPU capacities for scientific computing. It does, however, have a few drawbacks: 
  + It disconnects if left idle for too long. This forces me to watch over it while it runs and makes overnight experiments impossible. 
  + It is a bit of a pain to set up because every time I open it I get the default VM and need to download the dataset as well as install the needed packages - it does not even come with TensorFlow 2 by default. 
  + I have to take precautions against losing my progress: If it suddenly disconnects (which is possible) I obviously do not want to lose all my progress on the current experiment. My solution to this was to connect it to Google Drive and create a backup in a Google Drive folder after every epoch. 
  + It is unpredictable what GPU - if any - I get. Usually, the GPUs were about half as fast as those on Kaggle. Sometimes, they were almost as fast, sometimes they were very slow and sometimes I did not get a GPU at all. The more I used it, the less likely I was to get a decent (or any) GPU.
  
While both of these are definitely better than no GPU at all, I am effectively unable to get the same amount of GPU-time I could get if I had even a simple RTX 2070 Super which I could run 24/7. 
  
### Structural Implications of the GPU-Problem  
If I had a GPU, I could do all of it in one environment and automate a lot of the work with bash scripts. Because I do not have a GPU, I am limited to the confines of both products. This has several implications that make it very hard to set up a smooth workflow:
+ I have to use Jupyter-Notebooks, not scripts (Google Colab cannot deal with Scripts)
+ There are differences in the way the environments work. In Kaggle, the input is in a certain directory. In Colab, it is not. Additionally, they use different packages. Because of this, I cannot just run a file that ones on one system on the other one.
+ I cannot make my code modular. Working with multiple code files in Kaggle is annoying, but at least somewhat doable. In Colab, it is very annoying. Even if I would do it, I would still have to set it up first (in Colab, I would need to download the helper scripts first and add them to the path before being able to import them). Because it is very hard to make the code modular, every single helper function is declared directly in the notebooks.
+ Version Control is not as straightforward as it would be if I had full control over the environment.

All of these limitations lead to the fact that the code is anything but well-structured. I would have preferred if I could do it differently, but as elaborated above, this was very hard. Additionally, I was experimenting a lot and did not have any prior practical experience, which made it even harder to create structure. Most of my changes consisted of small adjustments to code that had worked before, running that code and evaluating the results. I could easily remember everything I tried out, so this was not really a limitation for me, but it obviously is for other people. I also do not remember the exact results for every experiment. For my purposes, it was enough to know which was better. Partly because of this, I wrote a summary of my experiments. Additionally, I linked to some notebooks so that you can look at particular versions and aspects of the code. I only list the important steps and experiments below.

## Experiments

+ My first step was to do a thorough preprocessing of the data. The result is [this kernel](https://www.kaggle.com/larswigger/bengali-image-preprocessing). I was particulalry interested in reducing the size of the data to overcome the GPU-Problem. Since I did not have any prior experience with this, I used [Iafoss very popular kernel](https://www.kaggle.com/iafoss/image-preprocessing-128x128) as a starting point. I replicated most of the code to get familiar with the tools used (I did not simply copy it). The Kernel cut out anything but the part containing the Grapheme, inverted the image (it was black writing on white background originally), set low-value noise to 0 and normalized the image. Additionally, I made some small changes, which made my version a bit more sophisticated: 
  + I avoided a bug in his Kernel, which is that crop_resize() returns the borders within a cropped array, which he then uses on the uncropped array. As a consequence, they are obviously wrong. This is somewhat offset, but not corrected, by the code that follows in his Kernel.
  + I made sure that the borders I cropped off beforehand where not left out if the cropped array was not reduced in size further. This made sure that there would be no loss of information while still reducing most images in size.
  + I changed the size of the data to 64x64 rather than 128x128 to save even more resources.
  + I saved all of them in a single .npy file to be able to load them all at once later on. I did not save them as a folder of .png images as Iafoss did, because Kaggle takes forever to load 200840 very small files, whereas one big file is ready almost immediately.
  
+ I used the kernel [Bengali Graphemes: Starter EDA+ Multi Output CNN by Kaushal Shah](https://www.kaggle.com/kaushal2896/bengali-graphemes-starter-eda-multi-output-cnn) as a starting point for my TensorFlow experiments. I implemented anything but the MultiOutputDataGenerator(which is obviously more complex), added some Dropout close to the head and let my experiment run. My first (debugged) submission scored 93.68% on the public leaderboard after (I think?) 40 Epochs. 
+ In [version 7](https://www.kaggle.com/larswigger/bengali-training?scriptVersionId=28793209) of my TensorFlow training kernel I copied the MultiOutputDataGenerator mentioned beforehand to my experiment. It was a custom class because Keras did not include anything for multilabel augmentation. With some basic rotation, zooming and shifting, the public leaderboard score jumped to 95.39%.
+ In [version 11](https://www.kaggle.com/larswigger/bengali-training?scriptVersionId=28859297) I tried to train the model only on the Grapheme Root data. I hoped that this would allow the model to focus on the Grapheme Root without distractions. The result was that it performed significantly worse in every single regard. One model for all labels was definitely the way to go.
+ In [version 17](https://www.kaggle.com/larswigger/bengali-training?scriptVersionId=29049731) I managed to try out several new things. It took me several attempts to get this working. In the end however, it just scored a mere 94.81% after 25 epochs. I made these changes:
  + I switched to a new height of 128x128. Because this multiplied my memory need by 4, the VM crashed. Because I could not find out how to properly do it with Keras, I created a python generator that loaded data from several files rather than one and kept only one file in memory at once. To speed it up, I used the @background decorator (not a default package). It was much slower than before, but at least it worked.
  + I used ResNet50 as a base for the model.
  + I copied the GridMask augmentation and applied it to the model

+ I got extremely annoyed by TensorFlow during my previous experiments. It was nice whenever I wanted to do something that was supposed to be done with the API and a royal pain whenever I wanted to do something that was not supposed to be done with the API. Additionally, the forums said that CutMix and MixUp were extremely useful for this competition, but they were simply not available for Keras. Everyone seemed to be using PyTorch. So I decided to switch my complete environment over in the middle of the competition. I used [Abhishek Thakurs video series](https://www.youtube.com/watch?v=8J5Q4mEzRtY&list=PL98nY_tJQXZntH5WUtKB0bghZeKVIJHJc) as a starting point for the competition. After a lot of work, the [new environment](https://www.kaggle.com/larswigger/pytorch-bengali-training?scriptVersionId=29175628) had the following advantages:
  + I used stratified folds for splitting the data rather than a simple shuffle as I had before.
  + I used a PyTorch data loader to load my data. It was significantly faster than the makeshift TensorFlow version I used beforehand. Because I found out about the numpy.memmap() function I had minimal memory usage (such a relieve compared to before) and it was a lot more flexible.
  + I could use the actual competition metric to evaluate my experiments compared to just standard accuracy with Keras.
  + I could easily control any aspect of the training loop. This also had the downside that I had to manually implement all of my metric tracking which TensorFlow gave me by default.
  
+ My first, mostly experimental, submission scored 95.28% (after debugging)
+ I added CutMix and MixUp. The results immediately got better and I tried it with both ResNet50(96.4%) and ResNet101(96.52%, one of the final submission, available [here](https://www.kaggle.com/larswigger/pytorch-bengali-submission?scriptVersionId=29326570)).
+ At this point, I switched my experiments to Google Colab, so it got a bit chaotic. You can see an example of the training structure I used in Colab [here](https://github.com/LarsWigger/Kaggle_Bengali-AI_Competition/blob/master/Notebooks/Minimal_Bengali_Training.ipynb).
+ I read in a post that the author got better results by plain resize compared to centering the Grapheme first as I had done. I tried it out and got clearly better results during the first 30 epochs. I decided to stop using my sophisticated preprocessing during the following experiments.
+ I tried using EfficientNetB3, SEResNet50 and DenseNet121. All of them performed worse than a simple ResNet50 and took longer to train.
+ I put the head directly on top of the model rather than using the Dropout and Dense Layers I used beforehand. The results were clearly better during training compared to using my intermediate layers.
+ Surprisingly, none of these improvements in my experiments led to a notable improvement on the public leaderboard - most were actually worse. So ultimately I just chose a ResNet50 version without my sophisticated preprocessing (scored 96.46% on the public leaderboard) as my second submission (I was allowed to select two for evaluation).
