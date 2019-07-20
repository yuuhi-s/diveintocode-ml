import numpy as np

class ScratchLinearRegression():
    '''
    線形回帰のスクラッチ実装

    Parameters
    ----------
    num_iter : int
      イテレーション数
    lr : float
      学習率
    bias : bool
      バイアス項を入れない場合はFalse
    verbose : bool
      学習過程を出力する場合はTrue

    Attributes
    ----------
    self.loss : 次の形のndarray, shape (self.iter,)
      学習用データに対する損失の記録
    self.val_loss : 次の形のndarray, shape (self.iter,)
      検証用データに対する損失の記録
    self.coef_ : 次の形のndarray, shape (n_features, 1)
      パラメータ      
    '''
    def __init__(self, num_iter=500, lr=0.002, bias=True, verbose=False):
        # ハイパーパラメータを属性として記録
        self.iter        = num_iter  #イテレーション数
        self.lr           = lr             #学習率
        self.bias       = bias         #バイアス項(True : あり, False : なし)
        self.verbose = verbose   #学習過程(True : 出力する, False : 出力しない)
        
        # 損失を記録する配列を用意
        self.loss        = np.zeros(self.iter) #学習用データを記録する0配列
        self.val_loss  = np.zeros(self.iter) #検証用データを記録する0配列
        
        #パラメータを用意
        np.random.seed(41)
        self.coef_ = 0

        
    def fit(self, X, y, X_val=None, y_val=None):
        '''
        線形回帰を学習する。検証用データが入力された場合はそれに対する損失と精度もイテレーションごとに計算する。

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
            
        #検証用データあり
        if X_val is not None and y_val is not None:
            if X_val.ndim == 1:
                X_val = X_val[:, np.newaxis]
            if y_val.ndim == 1:
                y_val = y_val[:, np.newaxis]            

        
        #バイアス項あり
        if self.bias == True:
            #1列目に1の配列を挿入
            X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
            
            #検証用データあり
            if X_val is not None and y_val is not None:
                #1列目に1の配列を挿入
                X_val = np.insert(X_val, 0, np.ones(X_val.shape[0]), axis=1)
            
        #パラメータを設定
        self.coef_ = np.random.randn(X.shape[1])[np.newaxis, :]  
        
        '''
        ここから学習
        '''
        for i in range(self.iter):
            #仮定関数より推定
            y_hat = self._linear_hypothesis(X)

            #errorの計算
            error = y_hat - y

            #損失の記録を格納
            mse = MSE(y_hat, y)
            self.loss[i] += mse

            #パラメータの更新
            self.coef_ = self._gradient_descent(X, error)
                    
            #検証用データあり
            if X_val is not None and y_val is not None:
                        
                #推定
                y_val_hat = self._linear_hypothesis(X_val)
                        
                #損失の記録を格納
                mse_val = MSE(y_val_hat, y_val)
                self.val_loss[i] += mse_val

            #学習過程を出力する場合
            if self.verbose == True:
                print('学習用データの学習過程' + str(i + 1) + '番目 : ' + str(self.loss[i]))

                #検証用データあり
                if X_val is not None or y_val is not None:
                    print('検証用データの学習過程' + str(i + 1) + '番目 : ' + str(self.val_loss[i]))


    def predict(self, X):
        '''
        線形回帰を使い推定する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            検証用データの特徴量

        Returns
        -------
        y_pred : 次の形のndarray, shape (n_samples, 1)
            線形回帰による推定結果
        '''
        if self.bias == True:
            #Xの1列目に1を挿入
            X = np.insert(X, 0, np.ones(X.shape[0]), axis=1) 

        #推定
        y_pred = np.dot(X, self.coef_.T)

        return y_pred
                

    def _linear_hypothesis(self, X):
        """
        線形の仮定関数を計算する

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            学習データ

        Returns
        -------
        y_hat : 次の形のndarray, shape (n_samples, 1)
            線形の仮定関数による推定結果
        """        
        y_hat = np.dot(X, self.coef_.T)
        
        return y_hat
                
        
    def MSE(y_pred, y):
        """
        平均二乗誤差の計算(2で割っている)

        Parameters
        ----------
        y_pred : 次の形のndarray, shape (n_samples, 1)
            推定した値
        y : 次の形のndarray, shape (n_samples, 1)
            正解値

        Returns
        ----------
        numpy.float
            平均二乗誤差の1/2
        """
        return (1/ len(y)) * ((y_pred - y)**2).sum()
    

    def _gradient_descent(self, X, error):
        """
        最急降下法の計算

        Parameters
        ----------------
        X : 次の形のndarray, shape (n_samples, n_features)
          学習データ
        error : 次の形のndarray, shape (n_samples, 1)
            推定した値と正解値の差

        Retruns
        ----------
        self.coef_ : 次の形のndarray, shape (n_samples, 1)
            パラメータ
        """
        self.coef_ = self.coef_ - (self.lr / X.shape[0]) * (np.dot(error.T, X))
        return self.coef_