from sagemaker.sklearn.estimator import SKLearn

estimator = SKLearn(
    entry_point="train.py",
    # ml.t3.medium, ml.m4.xlarge
    instance_type="local",
    framework_version="1.2-1",
    hyperparameters={}
)

estimator.fit()
