from sagemaker.sklearn.estimator import SKLearn
from sagemaker.local import LocalSession

sagemaker_session = LocalSession()
sagemaker_session.config = {"local": {"local_code": True}}

estimator = SKLearn(
    entry_point="./app/train.py",
    # ml.t3.medium, ml.m4.xlarge
    instance_type="local",
    framework_version="1.2-1",
    hyperparameters={},
    sagemaker_session=sagemaker_session,
    role="lab-sagemaker-execution-role"
)

estimator.fit({
    "train": "file://data/train",
})
