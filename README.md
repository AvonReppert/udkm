# udkm
a collection of python commands for data analysis in udkm group

## tools
general functions and definitions for data analysis, which includes:
- ```helpers```: general data analysis routines
- ```colors``` : frequently used colors and colormaps
- ```constants```: frequently used constants
- ```testing```: a sandbox file for testing git

## moke
library for the MOKE setup in the femto-magnetism lab, which includes:
- ```functions```: data analysis functions

## pxs
library for the X-ray diffraction setup using the Plasma X-ray source
- ```functions```: data analysis functions

# installation of the code

## option 1: download the code repository via git:
This allows you to have editable source file that you can add to the pyhton search path. 

1. install git that is available from ```https://git-scm.com/downloads```
2. run Git and initialize your name and e-mail adress via 
3. go to folder where you want to store the code repository and type ```cmd```
4. clone the repsoitory via ```git clone https://github.com/AleksUDKM/udkm.git```
5. add it to your search path of your pyhton distribution

To see if your installation works try:
```
import udkm.tools.helpers as helpers
print(helpers.teststring)
```
which should yield: "Successfully loaded udkm.tools.helpers"


## option 2: via pip using git:
Allows usage of the code without the option of changing it

```pip install git+https://github.com/AleksUDKM/udkm.git ```

# contributing to the repository:

To contribute code to the repository you need a GitHub account. 
Once you have that let me know and then I will add you as contributor. 
For testing purposes I recommend modifications of the "udkm/tools/testing" file

1. check that you have no conflicts and the most recent version via ``` git status``` 
2. get the most recent version of the repository via ``` git pull ```
3. modify the code on your local machine and test that it works. 
4. once you are satisfied you can add your changes to the repository by  ```git add * ```
5. commit your changes via ```git commit -m  "short description of the commit" ```
6. push your commits into the online repository via ```git push``` 
