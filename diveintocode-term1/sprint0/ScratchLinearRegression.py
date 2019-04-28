from sklearn.linear_model import LinearRegression
def ScratchLinearRegression(X_train, X_test, y_train):
    '''
    線形回帰のパイプラインの関数
    Parameters
    ---------------
    X_train : {array-like, sparse matrix}, shape (n_samples, n_features)
        学習用データ
    X_test : array_like or sparse matrix, shape (n_samples, n_features)
        検証用データ
    y_train : array-like, shape (n_samples,)
        学習用ラベル

    Returns
    -----------
    array, shape [n_samples]
        予測されたクラスラベル
    '''
    linear = LinearRegression()
    
    #modelをfit
    linear.fit(X_train, y_train)
    
    return linear.predict(X_test)