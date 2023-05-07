四个文件都是**二进制格式**：分别是训练、测试的图片及标签数据

- 可参考以下网站该来进行处理

  - [网站1](https://xinancsd.github.io/MachineLearning/mnist_parser.html)

  - [网站2](https://github.com/jeffreyforkfolder/tensorflow-learning/blob/master/other/MNIST/MNIST%E6%95%B0%E6%8D%AE%E9%9B%86%E4%BA%8C%E8%BF%9B%E5%88%B6%E6%A0%BC%E5%BC%8F%E8%BD%AC%E6%8D%A2%E4%B8%BA%E5%9B%BE%E7%89%87.md)

以**训练集图像文件**`train-images-idx3-ubyte`为例：

![img](https:////upload-images.jianshu.io/upload_images/2795802-c0c679831719840e.png?imageMogr2/auto-orient/strip|imageView2/2/w/773/format/webp)

图像文件的

- 第1-４个byte（字节，１byte=8bit），即前32bit存的是文件的magic number，对应的十进制大小是2051；
- 第5-8个byte存的是number of images，即图像数量60000；
- 第9-12个byte存的是每张图片行数/高度，即28；
- 第13-16个byte存的是每张图片的列数/宽度，即28。
- 从第17个byte开始，每个byte存储一张图片中的一个像素点的值。

因为`train-images-idx3-ubyte`文件总共包含了60000张图片数据，按照以上的存储方式，我们算一下该文件的大小：

- 一张图片包含28x28=784个像素点，需要784bytes的存储空间；
- 60000张图片则需要784x60000=47040000 bytes的存储空间；
- 此外，文件开始处使用了16个bytes用于存储magic number、图像数量、图像高度和图像宽度，因此，训练集图像文件的大小应该是47040000+16=47040016 bytes。

我们查看解压后的`train-images-idx3-ubyte`文件的属性：

![img](https:////upload-images.jianshu.io/upload_images/2795802-d6c7ee436fb87d75.png?imageMogr2/auto-orient/strip|imageView2/2/w/472/format/webp)

文件实际大小和我们计算的结果一致。

类似地，我们查看**训练集标签文件**`train-labels-idx1-ubyte`的存储格式：

![img](https:////upload-images.jianshu.io/upload_images/2795802-ee2ed520378f764b.png?imageMogr2/auto-orient/strip|imageView2/2/w/526/format/webp)

和图像文件类似：

- 第1-４个byte存的是文件的magic number，对应的十进制大小是2049；
- 第5-8个byte存的是number of items，即label数量60000；
- 从第9个byte开始，每个byte存一个图片的label信息，即数字0-9中的一个。

计算一下**训练集标签文件**`train-labels-idx1-ubyte`的文件大小：

- 1x60000+8=60008 bytes。

与该文件实际的大小一致：



![img](https:////upload-images.jianshu.io/upload_images/2795802-1dcff1a1b877a28c.png?imageMogr2/auto-orient/strip|imageView2/2/w/479/format/webp)

另外两个文件，即**测试集图像文件、测试集标签文件的存储方式和训练图像文件、训练标签文件相似**，只是图像数量由60000变为10000。

#### 1.3 使用python访问MNIST数据集文件内容

知道了MNIST二进制文件的存储方式，下面介绍如何使用python访问文件内容。同样以**训练集图像文件**`train-images-idx3-ubyte`为例：

首先，使用**open()函数**打开文件，并使用**read()方法**将所有的文件数据读入到一个字符串中：



```bash
yan@yanubuntu:~/codes/Deep-Learning-21-Examples/chapter_1/MNIST_data$ python
Python 2.7.12 (default, Nov 12 2018, 14:36:49) 
[GCC 5.4.0 20160609] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> with open('train-images.idx3-ubyte', 'rb') as f:
...  file = f.read()
... 
>>> 
```

file是str类型，其中的每个元素就存储的１个字节的内容。我们现在查看前４个字节，即magic number的内容，看下是否是前面说的2051:



```bash
>>> magic_number=file[:4]
>>> magic_number
'\x00\x00\x08\x03'
>>> magic_number.encode('hex')
'00000803'
>>> int(magic_number.encode('hex'),16)
2051
```

可以看出前4个byte的值确实是2051，但是不能直接输出magic number的内容，需要将其编码，然后才能转成十进制的int类型（有关字节编码的知识暂时没懂，先用着）。
 同样的方式，查看图像数量、图像高度和图像宽度信息：



```bash
>>> num_images = int(file[4:8].encode('hex'),16)
>>> num_images
60000
>>> h_image = int(file[8:12].encode('hex'),16)
>>> h_image
28
>>> w_image = int(file[12:16].encode('hex'),16)
>>> w_image
28
```

现在获取第１张图片的像素信息，然后利用numpy和cv2模块转换其格式，并保存成`.jpg`格式的图片：



```bash
>>> image1 = [int(item.encode('hex'), 16) for item in file[16:16+784]]
>>> len(image1)
784
>>> import numpy as np
>>> import cv2
>>> image1_np = np.array(image1, dtype=np.uint8).reshape(28,28,1)
>>> image1_np.shape
(28, 28, 1)
>>> cv2.imwrite('image1.jpg', image1_np)
True
>>> 
```

保存下来的图片image1.jpg如下图所示：

![img](https:////upload-images.jianshu.io/upload_images/2795802-00206d247837d2b0.png?imageMogr2/auto-orient/strip|imageView2/2/w/207/format/webp)

该图片的标签是５，我们可以验证一下**训练集标签文件**`train-labels-idx1-ubyte`文件的第一个标签是否和图像内容一一对应：



```bash
>>> with open('train-labels.idx1-ubyte', 'rb') as f:
...  label_file = f.read()
... 
>>> label1 = int(label_file[8].encode('hex'), 16)
>>> label1
5
>>> 
```

训练标签文件的第一张图片标签是第9个byte（索引从0开始，所以第9个byte是label_file[8]），结果没问题。



