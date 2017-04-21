## Machine Learning exercises


### Source
The exercise problems are taken from [Andrew Ng's Machine Learning MOOC](https://www.coursera.org/learn/machine-learning/home). I look forward to add more problem sets from other sources in future.

---

### Folder layout

1. `datasets` : contains all the sample datasets which are used in the scripts. 
2. `scripts` : contains the solutions in Python.
3. `plots` : folder having some plots of the datasets and their respective  hypothesis.

---

### Steps for running a solution script

1. Switch to the parent directory.<br/>
```bash
cd <path-to-ml-exercises>
```

2. Run a script from this directory. This makes sure that the paths of the dataset-files supplied to the scripts are consistent.<br/>
```python
python scripts/<script-name>.py
```

3. (Optional) If you'd like to have pretty XKCD-style graphs, you can un-comment *plt.xkcd()* statements in the scripts. <br/>
More information about setting XKCD-style font for your system could be found [here](https://gist.github.com/ashishraste/e4ef570fba4fce30f04ee0a99f47ce00).

---

### Sample plots

1. Plot of Logistic regression classifier with regularization and the computed decision boundary (exercise 2, data 2).
![Exercise 2, Data 2](https://github.com/ashishraste/ml-exercises/blob/master/plots/ex2data2_plot.png) 

---
