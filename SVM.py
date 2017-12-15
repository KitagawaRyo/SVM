import argparse
import numpy as np
import cvxopt as co
import matplotlib.pyplot as plt


class Kernel:
    """
    カーネル用のクラスをここにまとめる。
    inner_prod()は普通の内積を扱う
    """
    def __init__(self, d=2, sigma=10, a=2, b=-10):
        self.d = d
        self.sigma = sigma
        self.a = a
        self.b = b

    def inner_prod(self, x_k, x_l):
        if x_k.ndim == 3 or x_l.ndim == 3:  # 3次元配列が来たら崩さずに返す
            return np.sum(np.multiply(x_k, x_l), axis=2)
        else:
            return np.sum(np.multiply(x_k, x_l))

    def polynomial(self, x_k, x_l):
        a = 1 + self.inner_prod(x_k, x_l)
        return pow(a, self.d)

    def gaussian(self, x_k, x_l):
        if x_k.ndim == 3 or x_l.ndim == 3:  # 3次元配列が来たら崩さずに返す
            a = - np.sum((x_k - x_l)**2, axis=2) / 2 / (self.sigma**2)
        else:
            a = - np.sum((x_k - x_l)**2) / 2 / (self.sigma**2)
        return np.exp(a)

    def sigmoid(self, x_k, x_l):
        s = (self.a * self.inner_prod(x_k, x_l)) + self.b
        return np.tanh(s)


def __get_P(x, y, kernel):
    P = [[y[i] * y[l] * kernel(x[i], x[l])
          for l in range(0, len(x))] for i in range(0, len(x))]
    P += np.diag([1.0e-9 for i in range(0, len(x))])
    return co.matrix(P)


def get_alpha(x, y, kernel):
    """
    cvxoptを使ってSVMの最適化問題を解く
    http://cvxopt.org/userguide/coneprog.html#quadratic-cone-programs
    """
    try:
        R = len(y)
        P = __get_P(x, y, kernel)
        q = co.matrix(-1., (R, 1))
        G = co.matrix(np.diag([-1. for i in range(0, R)]))
        h = co.matrix(0., (R, 1))
        A = co.matrix(y).trans()
        b = co.matrix([0.])
        sol = co.solvers.qp(P, q, G, h, A, b)
        return sol['x'].trans()
    except ValueError:
        pass


def get_param(x, y, alpha, kernel):
    """
    get_alphaで求めたαから、ωとθを求める。
    """
    alphaY = np.multiply(alpha.trans(), co.matrix(y))
    w = co.matrix(x).trans() * co.matrix(alphaY)

    svNumber = np.argmax(alpha)
    theta = np.sum([alpha[i] * y[i] * kernel(x[i], x[svNumber])
                    for i in range(0, len(x))]) - y[svNumber]
    return w, theta


def get_sv_index(alpha):
    """
    サポートベクターのindexを取得
    """
    average = np.average(alpha) / 10
    return np.where(alpha > average)


def classifier(x, y, alpha, theta, kernel, xAxis, yAxis):
    """
    識別器の実行内容
    x, y, α, θ, カーネルからZ軸の値を出力する。
    """
    alphaY = np.multiply(alpha.trans(), co.matrix(y))

    # サポートベクターのインデックスを取得
    svNumber = get_sv_index(alpha)[1]
    X = np.array([[[i, l] for i in xAxis] for l in yAxis])
    ZAxis = [[0 for i in range(0, len(xAxis))] for l in range(0, len(yAxis))]
    for i in svNumber:
        sv = np.full((len(xAxis), len(yAxis), 2), x[i])
        data = kernel(sv, X) * alphaY[i]
        ZAxis = np.add(ZAxis, data)
    ZAxis -= theta
    return ZAxis


def __set_args():
    parser = argparse.ArgumentParser(
            description='This script is to use SVM',
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

    # matplotlibの宣言
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for i in range(R):
        sentence = linears[i].split()
        x_data = []
        for l in range(len(sentence)-1):
            x_data.append(float(sentence[l]))
        x.append(x_data)
        y.append(float(sentence[-1]))
        color = 'red' if sentence[-1] == '1' else 'blue'
        ax.scatter(sentence[0], sentence[1], c=color)

    x = np.array(x)  # テストの入力データ (D, R) D: 入力データの次元
    y = np.array(y)  # 教師データの答え (1, R)

    # コマンドライン引数からカーネルを設定
    # デフォルトはカーネル無し
    kernel = Kernel().inner_prod  # ただの内積
    if args.kernel == 0:
        kernel = Kernel(d=args.diameter).polynomial
    elif args.kernel == 1:
        kernel = Kernel(sigma=args.sigma).gaussian
    elif args.kernel == 2:
        kernel = Kernel(a=args.a, b=args.b).sigmoid

    # サポートベクターマシンの二次計画問題を解く
    alpha = get_alpha(x, y, kernel)
    if alpha is None:
        print("aborted!")
        return
    w, theta = get_param(x, y, alpha, kernel)

    print("重みw = ", w.trans(), "閾値θ = ", theta)

    # グラフのX軸とY軸を設定
    if len(x[0]) == 2:
        xAxis = np.arange(0, 50.1, 0.1)  # (501, 1) グラフのx軸
        yAxis = np.arange(0, 50.1, 0.1)  # (501, 1) グラフのy軸
        XAxis, YAxis = np.meshgrid(xAxis, yAxis)  # ともに(501, 501)の配列

        # 識別器による計算
        ZAxis = classifier(x, y, alpha, theta, kernel, xAxis, yAxis)
        ax.contour(XAxis, YAxis, ZAxis,
                   colors=['b', 'k', 'r'], levels=[-10, 0, 10])
        plt.show()


if __name__ == "__main__":
    main()
