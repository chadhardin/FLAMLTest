
from flaml import AutoML
from flaml.data import load_openml_dataset
from flaml.data import get_output_from_log
import matplotlib.pyplot as plt
import statsmodels.api as sm
import logging
import pandas as pd
import numpy as np

X_train, X_test, y_train, y_test = load_openml_dataset(dataset_id=1169, data_dir='./')
automl = AutoML()
settings = {
    "time_budget": 30,  # total running time in seconds
    "metric": 'accuracy',  # can be: 'r2', 'rmse', 'mae', 'mse', 'accuracy', 'roc_auc', 'roc_auc_ovr',
                           # 'roc_auc_ovo', 'log_loss', 'mape', 'f1', 'ap', 'ndcg', 'micro_f1', 'macro_f1'
    "task": 'classification',  # task type
    "log_file_name": 'airlines_experiment.log',  # flaml log file
    "seed": 7654321,    # random seed
}
automl.fit(X_train=X_train, y_train=y_train, **settings)

 
time_history, best_valid_loss_history, valid_loss_history, config_history, metric_history = \
    get_output_from_log(filename=settings['log_file_name'], time_budget=240)
from collections import Counter
bestmodels = []
for i in config_history:
    print(i['Current Learner'])
    bestmodels.append(i['Current Learner'])
bestmodels = Counter(bestmodels)
print(bestmodels)
t = "Learning Curve"
xl = "Wall Clock Time (s)"
yl = "Validation Accuracy"
pt = "feature"
"""automl.visualization(type = "feature_importance", level = 1, plotfilename = "Feature_importance")
automl.visualization(type = "validation_accuracy", xlab = xl, ylab = yl, plotfilename = "Validation_accuracy", logfilename = settings['log_file_name'])
automl.visualization(type = "feature_importance", level = 2, problem_type = "classification")
"""

'''
if plottype == "scatter":
            plt.title(title)
            plt.xlabel(xlab)
            plt.ylabel(ylab)
            plt.scatter(time_history, 1 - np.array(valid_loss_history))
            plt.step(time_history, 1 - np.array(best_valid_loss_history), where='post')
            plt.show()
elif plottype == "feature":
            plt.title(title)
            plt.barh(self.feature_names_in_, self.feature_importances_)
            plt.show()
        elif plottype == "Model":
            # pie graph that shows percentage of each model with the two best
            print("best model b")
        elif plottype == "parameters":
            # ANOVA 
            print("dees the best")
'''
