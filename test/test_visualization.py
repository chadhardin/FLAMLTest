
from flaml import AutoML
from flaml.data import load_openml_dataset

X_train, X_test, y_train, y_test = load_openml_dataset(dataset_id=1169, data_dir='./')
automl = AutoML()
settings = {
    "time_budget": 60,  # total running time in seconds
    "metric": 'accuracy',  # can be: 'r2', 'rmse', 'mae', 'mse', 'accuracy', 'roc_auc', 'roc_auc_ovr',
                           # 'roc_auc_ovo', 'log_loss', 'mape', 'f1', 'ap', 'ndcg', 'micro_f1', 'macro_f1'
    "task": 'classification',  # task type
    "log_file_name": 'airlines_experiment.log',  # flaml log file
    "seed": 7654321,    # random seed
}
automl.fit(X_train=X_train, y_train=y_train, **settings)

xl = "Wall Clock Time (s)"
yl = "Validation Accuracy"
automl.visualization(type = "feature_importance", level = 1, plotfilename = "Feature_importance")
automl.visualization(type = "validation_accuracy", xlab = xl, ylab = yl, plotfilename = "Validation_accuracy", settings = settings)
automl.visualization(type = 'best_model', settings = settings)
#automl.visualization(type = "feature_importance", level = 2, problem_type = "classification")