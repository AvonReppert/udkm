# udkm
a collection of python modules for data analysis in [udkm group](https://www.uni-potsdam.de/en/udkm)

The documentation including code examples can be found here: https://udkm.readthedocs.io

## modules
This library is subdivided into modules that are adapted to the different experimental setups and 
- ```pxs``` : library for the X-ray diffraction setup using the Plasma X-ray source
- ```moke```: routines for the trMOKE setup in the femto-magnetism lab
- ```kmc3```:  data evaluation using the h5 files that come out of the xpp endstation at kmc3
- ```tools```:  general functions, fits, constants and plot layouts 
- ```sim```: general material database and example for udkm1dsim code
- ```calc```: general collection of more or less frequently used calculation scripts

## installation of the code

### option 1: download the code repository via git:
This allows you to have editable source file that you can add to the python search path. 

1. install git that is available from ```https://git-scm.com/downloads```
2. run git and initialize your name and e-mail adress via:
  ``` 
  git config --global user.name "John Doe"
  git config --global user.email johndoe@example.com
  ```
3. go to folder where you want to store the code repository for example ```C:\Users\Aleks\Documents\Code``` 
4. open a console by typing  ```cmd``` into your explorer line
5. clone the repsoitory via ```git clone https://github.com/AleksUDKM/udkm.git```
6. add the repository to your search path of your python distribution 
   in Spyder I have ```C:\Users\Aleks\Documents\Code\udkm``` in the PYTHONPATH manager  
7. Klick the synchronize button and restart Spyder to be sure that the changes are applied

To see if your installation works try:
```
import udkm.tools.functions as tools
print(tools.teststring)
```
which should yield: "Successfully loaded udkm.tools.functions"


### option 2: via pip using git:
Allows usage of the code without the option of changing it

```pip install git+https://github.com/AleksUDKM/udkm.git ```

## contributing to the repository:

To contribute code to the repository you need a GitHub account. 
Once you have that let me know and I will add you as contributor after a brief intro into our concept. 

1. open a command window in the directory of the repository (for example: ```C:\Users\Aleks\Documents\Code\udkm``` )
2. check that you have no conflicts by typing ``` git status``` 
3. get the most recent version of the repository via ``` git pull ```
4. modify the code on your local machine and test that it works. 
5. once you are satisfied you can add your changes to the repository by  ```git add * ```
6. commit your changes via ```git commit -m  "short description of the commit" ```
7. push your commits into the online repository via ```git push``` 

# usage of the matplotlib style file

In udkm.tools you find the file ```udkm_base.mplstyle``` that we use as a default for plotting with matplotlib.
To add it to the preinstalled matplotlib styles follow these steps:

1. Look for the function path of matplotlib using
```
import matplolitb
matplotlib.matplotlib_fname()
```
2. There you will find a folder called ```stylelib``` to which you copy the file ```udkm_base.mplstyle``` 
3. After restarting your kernel you should be able to use the new plotstyle via:
```
import matplolitb.pyplot as plt
plt.style.use("udkm_base")
``` 
Alternatively you can import the plotstyle file into your runtime using the absolute or relative path
to the style file via ```plt.style.use(path_to_file+"/udkm_base.mplstyle")```, potentially also in the startup routine.

