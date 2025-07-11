import itertools
import pandas as pd
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.local import LocalSession

sagemaker_session = LocalSession()
sagemaker_session.config = {"local": {"local_code": True}}

estimator = SKLearn(
    source_dir="./local",
    entry_point="train.py",
    instance_type="local",
    framework_version="1.2-1",
    hyperparameters={},
    sagemaker_session=sagemaker_session,
    role="lab-sagemaker-execution-role",
)

estimator.fit({
    "train": "file://data/train",
})

predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type="local",
)

# Make inference requests
shape = pd.read_csv("data/train/iris.csv", header=None)

a = [50 * i for i in range(3)]
b = [40 + i for i in range(10)]
indices = [i + j for i, j in itertools.product(a, b)]

test_data = shape.iloc[indices[:-1]]
test_X = test_data.iloc[:, 1:]
test_y = test_data.iloc[:, 0]

print(predictor.predict(test_X.values))
print(test_y.values)

predictor.delete_endpoint()
