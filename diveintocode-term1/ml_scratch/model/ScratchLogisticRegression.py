import numpy as np

class ScratchLogisticRegression():
    '''
    ロジスティック回帰のスクラッチ実装

    Parameters
    ----------
    num_iter : int
        イテレーション数
    eta : float
        学習率(0 < eta <= 1の範囲)
    lam : float
        正則化パラメータ
    bias : bool
        バイアス項を入れない場合はFalse
    verbose : bool
        学習過程を出力する場合はTrue

    Attributes
    ----------
    self.coef_ : 次の形のndarray, shape (n_features,)
        パラメータ
    self.loss : 次の形のndarray, shape (self.iter,)
        学習用データに対する損失の記録
    self.val_loss : 次の形のndarray, shape (self.iter,)
        検証用データに対する損失の記録
    self.y : list
        学習用データのラベルを入れるリスト
    self.y_val : list
        検証用データのラベルを入れるリスト
    '''
    def __init__(self, num_iter=200, eta=0.05, lam=1, bias=True, verbose=False):
        # ハイパーパラメータを属性として記録
        self.iter        = num_iter      #イテレーション数
        self.eta        = eta                #学習率
        self.lam        = lam               #正則化パラメータ
        self.bias       = bias              #バイアス項(True : あり, False : なし)
        self.verbose = verbose      #学習過程(True : 出力する, False : 出力しない)

        # 損失を記録する配列を用意
        self.loss       = np.zeros(self.iter)
        self.val_loss = np.zeros(self.iter)
        
        #パラメータを用意
        np.random.seed(1)
        self.coef_ = 0
        
        #yのラベルを入れるリストを用意
        self.y = []
        self.y_val = []

        
    def fit(self, X, y, X_val=None, y_val=None):
        '''
        ロジスティック回帰を学習する。検証用データが入力された場合はそれに対する損失と精度もイテレーションごとに計算する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            学習用データの特徴量
        y : 次の形のndarray, shape (n_samples, )
            学習用データの正解値
        X_val : 次の形のndarray, shape (n_samples, n_features)
            検証用データの特徴量
        y_val : 次の形のndarray, shape (n_samples, )
            検証用データの正解値     
        '''
        #1次元なら2次元にする
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if y.ndim == 1:
            y = y[:, np.newaxis]
            
        #yのラベルのユニーク値をリストに保管
        self.y = []
        self.y.append(np.unique(y)[0])
        self.y.append(np.unique(y)[1])

        #yをコピーし、0, 1に変換
        y_copy = y.copy()
        y_copy[y_copy == np.unique(y_copy)[0]] = 0
        y_copy[y_copy == np.unique(y_copy)[1]] = 1
    
        #検証用データあり
        if X_val is not None and y_val is not None:
            if X_val.ndim == 1:
                X_val = X_val[:, np.newaxis]
            if y_val.ndim == 1:
                y_val = y_val[:, np.newaxis]
                
            #y_valのラベルのユニーク値をリストに保管
            self.y_val = []
            self.y_val.append(np.unique(y)[0])
            self.y_val.append(np.unique(y)[1])

            #y_valを変換0, 1に変換
            y_val_copy = y_val.copy()
            y_val_copy[y_val_copy == np.unique(y_val_copy)[0]] = 0
            y_val_copy[y_val_copy == np.unique(y_val_copy)[1]] = 1    

        #バイアス項あり
        if self.bias == True:
            #1列目に1の配列を挿入
            X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
            
            #検証用データあり
            if X_val is not None and y_val is not None:
                #1列目に1の配列を挿入
                X_val = np.insert(X_val, 0, np.ones(X_val.shape[0]), axis=1)

        #パラメータをランダムに設定
        self.coef_ = np.random.randn(X.shape[1])[np.newaxis, :]     
        
        '''
        ここから学習
        '''
        for i in range(self.iter):
            # 線形結合
            linear_join = self._logistic_hypothesis(X)

            #シグモイド関数で推定
            y_hat = self._sigmoid(linear_join)
   
            #エラーの計算
            error = y_hat - y_copy
            
            #損失の記録を格納
            cost = np.sum((-y_copy * np.log(y_hat)) - ((1 - y_copy) * np.log(1 - y_hat))) / X.shape[0]
            self.loss[i] += cost
            
            #パラメータの更新
            self.coef_ = self._gradient_descent(X, error)

            #検証用データあり
            if X_val is not None and y_val is not None:
                # 線形結合
                val_linear_join = self._logistic_hypothesis(X_val)

                #シグモイド関数で推定
                y_val_hat = self._sigmoid(val_linear_join)

                #損失の記録を格納
                cost_val = np.sum((-y_val_copy * np.log(y_val_hat)) - 
                                                  ((1 - y_val_copy) * np.log(1 - y_val_hat))) / X_val.shape[0]
                self.val_loss[i] += cost_val
                
            #学習過程を出力する場合
            if self.verbose == True:
                print('学習用データの学習過程' + str(i + 1) + '番目 : ' + str(self.loss[i]))

                #検証用データあり
                if X_val is not None or y_val is not None:
                    print('検証用データの学習過程' + str(i + 1) + '番目 : ' + str(self.val_loss[i]))


    def predict(self, X, threshold=0.5):
        '''
        ロジスティック回帰を使いラベルを推定する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            検証用データの特徴量
        threshold : float
            しきい値

        Returns
        -------
        y_pred : 次の形のndarray, shape (1, n_samples)
            予測したラベル
        '''
        if self.bias == True:
            #Xの1列目に1を挿入
            X = np.insert(X, 0, np.ones(X.shape[0]), axis=1) 
        
        #推定(元のラベルに戻す)
        y_pred = np.where(self._sigmoid(np.dot(X, self.coef_.T)) >= threshold, self.y[1], self.y[0])
            
        return y_pred
    

    def predict_proba(self, X):
        '''
        ロジスティック回帰を使い各クラスに属する確率を推定する

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            検証用データの特徴量

        Returns
        -------
        y_pred_proba : 次の形のndarray, shape (n_samples, n_classes)
             各クラスのサンプルの確率
        '''
        if self.bias == True:
            #Xの1列目に1を挿入
            X = np.insert(X, 0, np.ones(X.shape[0]), axis=1) 

        #推定
        y_pred_proba = np.concatenate([1 - self._sigmoid(np.dot(X, self.coef_.T)), 
                                                                   self._sigmoid(np.dot(X, self.coef_.T))], 1)

        return y_pred_proba
                

    def _logistic_hypothesis(self, X):
        '''
        線型結合を行う

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
          学習データ

        Returns
        -------
        次の形のndarray, shape (n_samples, )
            線形結合した結果
        '''
        return np.dot(X, self.coef_.T)
    
    
    def _sigmoid(self, z):
        '''
        シグモイド関数を計算する関数

        Parameter
        --------------
        z : 次の形のndarray, shape (n_sanples, 1)
            ある範囲の配列
        
        Returns
        -----------
        シグモイド関数
        '''
        return 1 / (1 + np.exp(-z))
    

    def _target_function(self, X, y, y_hat):
        '''
        目的関数を計算する関数

        Parameter
        --------------
        X : 次の形のndarray, shape (n_samples, n_features)
            学習用データの特徴量
        y : 次の形のndarray, shape (n_samples, 1)
            学習用データの正解値
        y_hat : 次の形のndarray, shape (n_samples, 1)
            仮定関数
            
        Returns
        -----------
        j : float
            目的関数
        '''
        j = (np.sum((-y * np.log(y_hat)) - ((1 - y) * np.log(1 - y_hat))) / X.shape[0]) +  ((self.lam / 2 * X.shape[0]) * np.sum(self.coef_**2))
        
        return j


    def _gradient_descent(self, X, error):
        '''
        最急降下法でパラメータを更新する関数

        Parameter
        --------------
        X : 次の形のndarray, shape (n_samples, n_features)
            学習用データ
        error : 次の形のndarray, shape (n_samples, 1)
            推定した値と正解値の差

        Returns
        -----------
        self.coef_ : 次の形のndarray, shape (n_samples, 1)
            更新されたパラメータ   
        '''
        #バイアス項がある場合
        if self.bias == True:
            #特徴量のインデックスが0の場合
            self.coef_[:, 0] = self.coef_[:, 0] - self.eta * (np.dot(error.T, X[:, 0]) / X.shape[0])
            #特徴量のインデックスが1以上の場合
            self.coef_[:, 1:] = self.coef_[:, 1:] - self.eta * ((np.dot(error.T, X[:, 1:]) / X.shape[0]) +  ((self.lam / X.shape[0]) * self.coef_[:, 1:]))
        
        #バイアス項がない場合
        else:
            self.coef_= self.coef_- self.eta * ((np.dot(error.T, X) / X.shape[0]) + ((self.lam / X.shape[0]) * self.coef_))
        
        return self.coef_