# AI-Challenge-RTVC

## 1. What is Multi-Label Classification?

![Multi-Label Classification](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/08/25230246/beautiful_scenery_05_hd_picture_166257.jpg)

### TODO List

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

## References

https://blog.coast.ai/five-video-classification-methods-implemented-in-keras-and-tensorflow-99cad29cc0b5

https://www.analyticsvidhya.com/blog/2017/08/introduction-to-multi-label-classification/

https://medium.com/coinmonks/multi-label-classification-blog-tags-prediction-using-nlp-b0b5ee6686fc

https://www.depends-on-the-definition.com/guide-to-multi-label-classification-with-neural-networks/

https://www.kdnuggets.com/2018/09/object-detection-image-classification-yolo.html

https://medium.com/@RiterApp/capsule-networks-as-a-new-approach-to-image-recognition-345d4db0831
