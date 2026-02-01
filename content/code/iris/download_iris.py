import os
import numpy as np
from sklearn.datasets import load_iris


def main(data_dir="./data"):
    os.makedirs(data_dir, exist_ok=True)
    iris = load_iris()
    X = iris["data"]  # shape: (150, 4)
    y = iris["target"]  # shape: (150,)

    out_path = os.path.join(data_dir, "iris.npz")
    np.savez_compressed(out_path, X=X, y=y)
    print(f"Saved Iris dataset to {out_path}")


if __name__ == "__main__":
    main()
