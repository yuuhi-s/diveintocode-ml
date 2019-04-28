import numpy as np

def train_test_split(X, y, train_size=0.8):
    """
    学習用データを分割する。

    Parameters
    ----------
    X : 次の形のndarray, shape (n_samples, n_features)
      学習データ
    y : 次の形のndarray, shape (n_samples, )
      正解値
    train_size : float (0<train_size<1)
      何割をtrainとするか指定

    Returns
    ----------
    X_train : 次の形のndarray, shape (n_samples, n_features)
      学習データ
    X_test : 次の形のndarray, shape (n_samples, n_features)
      検証データ
    y_train : 次の形のndarray, shape (n_samples, )
      学習データの正解値
    y_test : 次の形のndarray, shape (n_samples, )
      検証データの正解値
    """
    #Xの配列をシャッフルし、行番号を抽出
    permu = np.random.permutation(X[:, 0].size)

    #Xとyを、シャッフルした行番号順に並び替える
    X = X[permu]
    y = y[permu]

    #X_train, X_testをtrain_sizeの割合で縦に分割
    X_train, X_test = np.vsplit(X, [int(X[:, 0].size * train_size)])
    #y_train, y_testをtrain_sizeの割合で分割
    y_train, y_test = np.split(y, [int(y.size * train_size)])

    return X_train, X_test, y_train, y_test
