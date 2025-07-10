from sagemaker.sklearn.estimator import SKLearn

estimator = SKLearn(
    entry_point="train.py"
)
