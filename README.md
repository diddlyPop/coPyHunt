# coPyHunt
scavenger hunt game supervised by an image classifier

------------------------------------------------------------

# install instructions
known PySimpleGui problems on Mac, working well on Windows w/ python3.7.1. will move to python3.7.6 soon.
```
git clone https://github.com/diddlypop/coPyHunt.git
cd coPyHunt
```

initialize your own virtual environment
keep in mind that the weights file is 230MB+

```
pip install requirements.txt
cd yolo-coco
wget https://pjreddie.com/media/files/yolov3.weights
```

# play instructions

```
python gui.py
```

ctrl+shift+s to screengrab or copy photos to clipboard
scan when you've grabbed your item
