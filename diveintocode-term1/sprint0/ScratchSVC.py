from sklearn.svm import SVC
def ScratchSVC(X_train, X_test, y_train):
    '''
    ロジスティック回帰のパイプラインの関数
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
    svm = SVC()
    
    #modelをfit
    svm.fit(X_train, y_train)
    
    return svm.predict(X_test)