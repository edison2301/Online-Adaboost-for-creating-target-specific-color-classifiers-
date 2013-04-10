### LICENSE

```
Software License Agreement (BSD License)

Copyright (c) 2013-, Filippo Basso and Matteo Munaro
                     {filippo.basso, matteo.munaro}@dei.unipd.it
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

* Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following
disclaimer in the documentation and/or other materials provided
with the distribution.
* Neither the name of the copyright holder(s) nor the names of its
contributors may be used to endorse or promote products derived
from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

### DESCRIPTION
This package contains a library implementing the Online Adaboost algorithm and a demo file showing 
the learning process of an Online Adaboost classifier based on color features.
An example image with 9 colored squares is used.
A color classifier is learned for a selected square (by default it is the central one).
As features, color features extracted from the RGB histogram of the target are used.
The target square is used as positive example, while negative examples are randomly selected from the rest of the image.
After every iteration, the confidence values of every square with respect to the classifier learned for the target
square are shown.
Moreover, the mean color value of the most weighted features is shown in a histogram where the height is proportional
to the feature weight. The written number, instead, represents the volume of the corresponding parallelepiped.

 For Online Adaboost, we used our implementation of the algorithm described in:
 
 [1] H. Grabner and H. Bischof. _On-line boosting and vision._
 	   In Proc. of the IEEE Conference on Computer Vision and Pattern Recognition, pages 260–267, Washington, DC, USA, 2006.

 For the color features, we used the original implementation described in:
 
 [2] F. Basso, M. Munaro, S. Michieletto and E. Menegatti. _Fast and robust multi-people tracking from RGB-D data for a mobile robot._
     In Proceedings of the 12th Intelligent Autonomous Systems (IAS) Conference, Jeju Island (Korea), 2012.

 If you use part of this code, please cite [2].
 
### BUILDING 
This package needs the open source computer vision library [OpenCV](http://opencv.org/).
Once you installed OpenCV, you should go to the package directory and type:

```
cmake .
make
```

Tested with OpenCV 2.3.X and 2.4.X on Ubuntu 12.04 LTS.

### DEMO APPLICATION
After building the code, you have to go to the `bin` folder and type:
```
./demo_adaboost
```


