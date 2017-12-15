import svm
import argparse
import numpy as np


def eval_svm(x, y, kernel, n):
    if len(x) < n:  # xの長さが短いときはNoneを返す
        print("サンプルデータ数が分割数に対して少ないです。")
        return
    sep_x = np.array(np.split(x, n))
    sep_y = np.array(np.split(y, n))
    correct_num = 0
    for i in range(n):
        ind = np.ones(n, dtype=bool)
        ind[i] = False
        train_data = sep_x[ind].reshape(-1, 2)
        train_ans = sep_y[ind].flatten()
        eval_data = sep_x[i]
        eval_ans = sep_y[i]

        alpha = svm.get_alpha(train_data, train_ans, kernel)
        if alpha is None:
            print("aborted!")
            return
        w, theta = svm.get_param(train_data, train_ans, alpha, kernel)
        predict_ans = svm.classify(train_data, train_ans, alpha,
                                   theta, kernel, eval_data)
        correct_num += len(np.where(eval_ans == predict_ans)[0])

    return correct_num / len(x)


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

    parser.add_argument(
            '-n',
            help="What divide data into, Default is 10",
            type=int,
            default=10
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

    # コマンドライン引数からカーネルを設定
    # デフォルトはカーネル無し
    kernel = svm.Kernel().inner_prod  # ただの内積
    if args.kernel == 0:
        kernel = svm.Kernel(d=args.diameter).polynomial
    elif args.kernel == 1:
        kernel = svm.Kernel(sigma=args.sigma).gaussian
    elif args.kernel == 2:
        kernel = svm.Kernel(a=args.a, b=args.b).sigmoid

    percent = eval_svm(x, y, kernel, args.n)
    if percent is not None:
        print("正答率は", percent, "です。")


if __name__ == "__main__":
    main()
