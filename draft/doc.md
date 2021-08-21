# 0819
embrace the change, do the thing that challenge

# workflow
- augument image by limiting window 
- save pkls every nob, to boost load speed 



# done 
- read discussions

# throw 
- pytorch video data  baseline run 
- 27bra baseline run 


# >>>>>>>>>>>>>>>>>>>>  27bra >>>>>>>>>>>>>>>>>>>>>>

# recom args
center_crop_ratio = 0.9
random_crop_ratio = 0.8
img_size: [192,192] 
model: resnet18


 
# some thoughts to 27bra
- main split
way1, nob > window
way2, dcm > window > pie  # should better 
 
- tumor is a part in the pic, not the whold image can treat like classical image
need some technique that focus on the tumor
- what is a tmuor looks like ?  maybe need some technique to analyze 
search medical info
- edge information 
try diff dim 
- add all pix and then scale them 
pie ok 


# p1
- improve multi thread

# p9
- write self-define softmax in range[0,1] and can scale
- png 3 channel sample, view the tumor range 
- try directly crop little 
- chop edges very high
- unify all the pic direction 
- try 3 slice technique
- try auto detect tumor legacy way 
- wirte a loader that can multithread load, and multithread save , and has progress bar 


# done
-  plot surface of image 
- add all dcms in a cohort to 1 png












# <<<<<<<<<<<<<<<<<<<  27bra  <<<<<<<<<<<<<<<<<<<  

# meme
- transform before cache: improve traning efficiency, lose cache flexibility, 









# ich doc
- introduce
- usage
- args


## introduce
just a tool to handle image classification task


## usage
```
python train.py --epoch 300
```


## args
- args
get args from input
- subset-ratio
danger op, just for quick test 




