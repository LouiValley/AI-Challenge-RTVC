# AI-Challenge-RTVC

## 1. What is Multi-Label Classification?

![Multi-Label Classification](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/08/25230246/beautiful_scenery_05_hd_picture_166257.jpg)

### TODO List

0.Windows 10 Envoriment prepare: 

Annaconda Python 3.6.6 + Tensorflow-gpu 1.11 + CUDA 9 + Cudnn 9.0

1.Split all the videos into train/test folders

compare to UCF-101 video files summary:

| Datasets      | Size                         | FPS   | Length | Images |
| ------------- |:----------------------------:| -----:|-------:|-------:|
| UCF-101       | 320X240                      | 25    | 4~10s  | 44     | 
| RTVC2018      | 540X960,480X684,1280X720...  | 30    | 10s    | 50     |

2.Extract jpegs of each frame for each video

3.Summarize the videos, their class, train/test status and frame count in a CSV we’ll reference throughout our training.

4.

5.

### Troubleshoots

1.GPU CUDA_ERROE_OUT_OF_MEMORY

```
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
```

2.Load our data from file. But with blank value

```
with open(os.path.join('data', 'data_file.csv'), 'r') as fin:
  reader = csv.reader(fin)
  data = list(reader)
print(data)  
 ```
 
 to 
 
 ```
 with open(os.path.join('data', 'data_file.csv'), 'r') as fin:
   data = pandas.read_csv(fin, header=None)
 print(data.values.tolist())  
 ```
 
3.Get the parts in Windows10

```
parts = video.split(os.path.sep)
```
to
```
parts = video.split("/")
```

1_move_files.py in Windows10

```
            if not os.path.exists(filename):
                print("Can't find %s to move. Skipping." % (filename))
                continue

            # Move it.
            dest = os.path.join(group, classname, filename)
            print("Moving %s to %s" % (filename, dest))
            os.rename(filename, dest)
```
to

```
            fullfilename = "UCF-101/" + classname + "/"+ filename
            print(fullfilename)
            if not os.path.exists(fullfilename):
                print("Can't find %s to move. Skipping." % (fullfilename))
                continue

            # Move it.
            dest = os.path.join(group, classname, filename)
            print("Moving %s to %s" % (fullfilename, dest))
            os.rename(fullfilename, dest)
```

4.MemoryError

```
Epoch 1/1000
  6/100 [>.............................] - ETA: 2:08 - loss: 1.4358 - acc: 0.6094 - top_k_categorical_accuracy: 0.8542Traceback (most recent call last):
  File "train_cnn.py", line 142, in <module>
    main(weights_file)
  File "train_cnn.py", line 138, in main
    [checkpointer, early_stopper, tensorboard])
  File "train_cnn.py", line 119, in train_model
    callbacks=callbacks)
  File "C:\Users\Administrator\AppData\Local\conda\conda\envs\tensorflow_gpu\lib\site-packages\keras\legacy\interfaces.py", line 91, in wrapper
    return func(*args, **kwargs)
  File "C:\Users\Administrator\AppData\Local\conda\conda\envs\tensorflow_gpu\lib\site-packages\keras\engine\training.py", line 1418, in fit_generator
    initial_epoch=initial_epoch)
  File "C:\Users\Administrator\AppData\Local\conda\conda\envs\tensorflow_gpu\lib\site-packages\keras\engine\training_generator.py", line 180, in fit_generator
    generator_output = next(output_generator)
  File "C:\Users\Administrator\AppData\Local\conda\conda\envs\tensorflow_gpu\lib\site-packages\keras\utils\data_utils.py", line 601, in get
    six.reraise(*sys.exc_info())
  File "C:\Users\Administrator\AppData\Local\conda\conda\envs\tensorflow_gpu\lib\site-packages\six.py", line 693, in reraise
    raise value
  File "C:\Users\Administrator\AppData\Local\conda\conda\envs\tensorflow_gpu\lib\site-packages\keras\utils\data_utils.py", line 595, in get
    inputs = self.queue.get(block=True).get()
  File "C:\Users\Administrator\AppData\Local\conda\conda\envs\tensorflow_gpu\lib\multiprocessing\pool.py", line 644, in get
    raise self._value
  File "C:\Users\Administrator\AppData\Local\conda\conda\envs\tensorflow_gpu\lib\multiprocessing\pool.py", line 119, in worker
    result = (True, func(*args, **kwds))
  File "C:\Users\Administrator\AppData\Local\conda\conda\envs\tensorflow_gpu\lib\site-packages\keras\utils\data_utils.py", line 401, in get_index
    return _SHARED_SEQUENCES[uid][i]
  File "C:\Users\Administrator\AppData\Local\conda\conda\envs\tensorflow_gpu\lib\site-packages\keras_preprocessing\image.py", line 1441, in __getitem__
    return self._get_batches_of_transformed_samples(index_array)
  File "C:\Users\Administrator\AppData\Local\conda\conda\envs\tensorflow_gpu\lib\site-packages\keras_preprocessing\image.py", line 1916, in _get_batches_of_transformed_samples
    dtype=self.dtype)
MemoryError
```

## References

https://blog.coast.ai/five-video-classification-methods-implemented-in-keras-and-tensorflow-99cad29cc0b5

https://www.analyticsvidhya.com/blog/2017/08/introduction-to-multi-label-classification/

https://medium.com/coinmonks/multi-label-classification-blog-tags-prediction-using-nlp-b0b5ee6686fc

https://www.depends-on-the-definition.com/guide-to-multi-label-classification-with-neural-networks/

https://www.kdnuggets.com/2018/09/object-detection-image-classification-yolo.html

https://medium.com/@RiterApp/capsule-networks-as-a-new-approach-to-image-recognition-345d4db0831
