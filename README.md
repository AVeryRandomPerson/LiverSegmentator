## LiverSegmentator
This is an image segmentation software implementing Linear Support Vector Classification from scikit-learn, using
LBP features from scikit-image.

CT Images are used. No direct support for DICOM yet, but soon.
## Note : Refer to API documentation to customize model. [ In The Folder ]

### Preprocessing
```markdown

# LBP
# Histogram Equilization
# Gamma Correction
# Canny Edge

```

### SVC features

```markdown

# LBP histogram
# Standard Deviation of intensity
# Cost metric of distance from centre
```

### INSTALLATION 

```markdown
Requirements :

1.	Python 3.5 - AMD 64x Bit
	
	How to install ?
	Download and run binary installer : https://www.python.org/downloads/

2.	Python Packages:

	A.	Imutils			[https://pypi.python.org/pypi/imutils]
	B.	Matplotlib		[https://matplotlib.org/]
	C.	Numpy (with mkl)[http://www.numpy.org/]
	D.	OpenCV			[http://opencv.org/]
	E.	Scikit-Learn	[http://scikit-learn.org/stable/install.html]
	F.	Scikit-Image	[http://scikit-image.org/]
	G.	SciPy 			[https://www.scipy.org/]

	How to install?

	Head on to their website and follow the installation instructions. 
	Different operating system platforms will have slightly different install actions.

	*If you are using Windows and are having trouble you may use this alternate method:*

	I.	Head to : http://www.lfd.uci.edu/~gohlke/pythonlibs/
	II.	Use Ctrl+F and find the package of interest. Download it.
	III.Move the downloaded file (.whl) into {Your Python Directory}/Scripts/
	IV.	Open cmd.
	V.	Enter this command:cd '{Your Python Directory}/Scripts/'
	VI. Enter this command:pip install '{downloaded file name}.whl'

```

```markdown
TEN Things you need to know :

1.	System Files/LiverSegmentator contains the source code of the project program.
2.	System Files/Database should be the location of image files for the program to work with.
3.	Final Report contains only the final report of the project. 
4.	The README.md inside System Files folder is a readme file for the GitHub Repository.	

*5.	The source code runs on python 3.5 - 64x Bit AMD system.
6.	The system is not stable on 32x Bit systems but future implementation will seek to establish compatibility.
7.	There are dependencies on the code the following packages would be required:
	[ For an installation guide please refer to Install-Guide.txt]

	A.	Imutils
	B.	Matplotlib
	C.	Numpy (with mkl)
	D.	OpenCV
	E.	Scikit-Learn
	F.	Scikit-Image
	G.	SciPy

8.	There is an API documentation available in the API-Documentation folder. (System Files/API-DOCUMENTATION/{Target})
	[ Do not use word-wrap. Or at least expand your window to full screen to get a better text alignment.]

9.	The source code is constantly being updated / maintained.
	See : GitHub - https://github.com/AVeryRandomPerson/LiverSegmentator
	Note : As of the writing of this readme, github repo is Private. Will be made public eventually.

10.	Results.csv are the summary of all experimental results during the course of the project.


Any enquiries please email : PlatinumHoboGold@gmail.com
```
