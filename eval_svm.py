import svm
import argparse
import numpy as np

def __set_args():
    parser = argparse.ArgumentParser(
            description='This script is to test SVM',
            add_help=True
            )

    parser.add_argument(
            '-f', '--file',
            help='Training file path',
            required=True
            )

    parser.add_argument(
            '-k', '--kernel',
            help='''
            use Kernel Trick
            0 : Polynomial Kernel
            1 : Gaussian Kernel
            2 : Sigmoid Kernel
            ''',
            type=int,
            default=3
            )

    parser.add_argument(
            '-d', '--diameter',
            help="polynomial kernel's parameter",
            type=int,
            default=2
            )

    parser.add_argument(
            '-s', '--sigma',
            help="gaussian kernel's parameter",
            type=int,
            default=10
            )

    parser.add_argument(
            '-a',
            help="sigmoid kernel's parameter1",
            type=int,
            default=0.5
            )

    parser.add_argument(
            '-b',
            help="sigmoid kernel's parameter2",
            type=int,
            default=-1
            )

    return parser.parse_args()

def main():
    x, y = [], []
    args = __set_args()

    # ファイルの読み込み
    f = open(args.file, 'r')
    linears = f.readlines()
    f.close()
    R = len(linears)  # テストデータの数

    for i in range(R):
        sentence = linears[i].split()
        x_data = []
        for l in range(len(sentence)-1):
            x_data.append(float(sentence[l]))
        x.append(x_data)
        y.append(float(sentence[-1]))

    x = np.array(x)  # テストの入力データ (D, R) D: 入力データの次元
    y = np.array(y)  # 教師データの答え (1, R)

if __name__ == "__main__":
    main()
