import argparse
import os
import pandas as pd
from sklearn import tree
import joblib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    args = parser.parse_args()

    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    raw_data = [ pd.read_csv(file, header=None, engine="python") for file in input_files ]
    train_data = pd.concat(raw_data)

    train_y = train_data.iloc[:, 0]
    train_X = train_data.iloc[:, 1:]

    clf = tree.DecisionTreeClassifier(max_leaf_nodes=30)
    clf = clf.fit(train_X, train_y)

    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))


def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf
