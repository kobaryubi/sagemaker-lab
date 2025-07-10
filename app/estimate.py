import boto3
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.local import LocalSession

boto_session = boto3.Session(region_name="ap-northeast-1")
sagemaker_session = LocalSession(boto_session=boto_session)

estimator = SKLearn(
    entry_point="train.py",
    # ml.t3.medium, ml.m4.xlarge
    instance_type="local",
    framework_version="1.2-1",
    hyperparameters={},
    sagemaker_session=sagemaker_session,
)

estimator.fit()
