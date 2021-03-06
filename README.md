# pr-assignment-2

https://www.overleaf.com/3677423679tvgqqpxdgrxg

### **How to run**

The image classification pipeline from task 1 and task 2 have been implemented in python. The Gene pipeline has been
implemented in a python notebook. 

**Image Classification pipeline**

 *Tested on MacOS and Linux* 
1. Go to the project root and run: pip install -r requirements.txt
2. Run: cd task_1/big_cats
3. To run 10-fold cross validation run: python3 main.py
4. To inspect the data run: python3 explore_data.py

We think it should work on all operating systems, since we handled the paths etc with the 
os library. However, we were not able to test it on Windows.
Depending on your computer, running 10-fold cross validation can take quite a long while. 
Results are written to the screen and also to the folder data/. 

**Gene Pipeline**

Execute the steps in the notebook

**Task_2**
Run task_2/task_2.py. Make sure creditcard.csv is in the task_2 folder, so the folder has the following files:
creditcard.csv
result_processing.py
task_2.py
