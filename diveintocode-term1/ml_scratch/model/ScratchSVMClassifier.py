import numpy as np

class ScratchSVMClassifier:
    '''
    SVMのスクラッチ実装

    Parameters
    ----------
    num_iter : int
        イテレーション数
    eta : float
        学習率(0 < eta <= 1の範囲)
    threshold : float
        しきい値
    bias : bool
        バイアス項を入れない場合はFalse
    kernel : object
        カーネルの式, 多項式の場合は'poly'
    gamma : float
        'poly'のカーネル係数
    degree : int
        'poly'のカーネル係数
    coef0 : float
        'poly'の独立項

    Attributes
    ----------
    self.lam_ : 次の形のndarray, shape (n_features,)
        ラグランジュ乗数
    self.y_label : list
        ラベルのユニーク値
    self.sv_X : 次の形のndarray, shape (n_samples, n_features)
        サポートベクターの特徴量
    self.sv_lam : 次の形のndarray, shape (n_samples, 1)
        サポートベクターのラグランジュ乗数
    self.sv_y : 次の形のndarray, shape (n_samples, 1)
        サポートベクターのラベル
    self.theta : 次の形のndarray, shape (n_samples, n_features)
        分類境界の勾配
        
    '''
    def __init__(self, num_iter=50, eta=0.05, threshold=1e-5, 
                             bias=True, kernel='linear', gamma=5, degree=5, coef0=0):
        # ハイパーパラメータを属性として記録
        self.iter              = num_iter     #イテレーション数
        self.eta              = eta               #学習率
        self.threshold   = threshold     # しきい値
        self.bias             = bias             #バイアス項(True : あり, False : なし)
        self.kernel         = kernel          #カーネルの式(linear :  線形カーネル, poly : 多項式カーネル)
        self.gamma       = gamma       #カーネル係数
        self.degree       = degree        #カーネル係数
        self.coef0         = coef0          #'poly'の独立項

        #　インスタンス変数
        self.lam        = None      #ラグランジュ乗数
        self.y_label  = None      #ラベルのユニーク値
        self.sv_X      = None      #サポートベクターの特徴量
        self.sv_lam  = None      #サポートベクターのラグランジュ乗数
        self.sv_y      = None      #サポートベクターのラベル
        self.theta     = None     #分類境界の勾配
        

    def fit(self, X, y):
        '''
        SVMで学習する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            学習用データの特徴量
        y : 次の形のndarray, shape (n_samples, )
            学習用データの正解値
        '''
        #yのラベルのユニーク値をリストに保管
        self.y_label = []
        self.y_label.append(np.unique(y)[0])
        self.y_label.append(np.unique(y)[1])
        
        #yをコピーし、-1, 1に変換
        y_copy = y.copy()
        y_copy[y_copy == np.unique(y_copy)[0]] = -1
        y_copy[y_copy == np.unique(y_copy)[1]] = 1
        
        #ラグランジュ乗数の初期化
        self.lam = np.ones(X.shape[0])[:, np.newaxis]
        
        #イテレーションの回数実行
        for i in range(self.iter):
            self.lam = self._gradient_descent(X, y_copy)
        
        #しきい値以上のサポートベクターを取り出す
        sv = np.where(self.lam > self.threshold)[0]
            
        #サポートベクターの数を取得
        sv_nam = len(sv)

        #サポートベクターのデータを格納
        self.sv_X = X[sv, :] #特徴量ベクトル
        self.sv_lam = self.lam[sv] #ラグランジュ乗数
        self.sv_y = y_copy[sv] #ラベル

        #thetaの計算
        self.theta = np.zeros(X.shape[1]) #Xの特徴量の数の0配列を用意
        for n in range(sv_nam):
            self.theta += self.sv_lam[n] * self.sv_y[n] * self.sv_X[n] #サポートベクターの数だけ足す

        #theta0の計算
        self.theta0 = np.sum(self.sv_y) #サポートベクターのラベル名の合計
        self.theta0 -= np.sum(np.dot(self.theta, self.sv_X.T)) #thetaとサポートベクターの特徴量の合計を引く
        self.theta0 /= sv_nam  #サポートベクターの数で割る


    def predict(self, X):
        '''
        SVMを使いラベルを推定する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            検証用データの特徴量

        Returns
        -------
        y_pred : 次の形のndarray, shape (1, n_samples)
            予測したラベル
        '''        
        #線形カーネルの時
        if self.kernel == 'linear':
            y_pred = (np.sum((self.sv_lam.T * self.sv_y) 
                               * self._kernel_func(X, self.sv_X), axis=1))
        
        #多項式カーネルの時
        elif self.kernel == 'poly':
            y_pred = (np.sum((self.sv_lam.T * self.sv_y) 
                              * self._poly_kernel_func(X, self.sv_X), axis=1))

        #バイアス項がないとき
        if self.bias == False:
            y_pred = np.dot(self.theta, X.T)
        
        #予測結果が0未満なら-1側, 0より大きいなら1側のラベルを割り振る
        for i in range(X.shape[0]):
            if y_pred[i] < 0:
                y_pred[i] = self.y_label[0]
            else:
                y_pred[i] = self.y_label[1]

        return y_pred
    
    
    def _kernel_func(self, Xi, Xj):
        '''
        線形カーネル計算を行う関数
        Parameters
        --------------
        Xi : 次の形のndarray, shape (n_samples, n_features)
            検証用データの特徴量
        Xj : 次の形のndarray, shape (n_samples, n_features)
            検証用データの特徴量
            
        Returns
        ----------
        線形カーネルの計算結果
        ''' 
        return np.dot(Xi, Xj.T)
    

    def _poly_kernel_func(self, Xi, Xj):
        '''
        多項式カーネル計算を行う関数
        Parameters
        --------------
        Xi : 次の形のndarray, shape (n_samples, n_features)
            検証用データの特徴量
        Xj : 次の形のndarray, shape (n_samples, n_features)
            検証用データの特徴量
        
        Returns
        ----------
        多項式カーネルの計算結果
        ''' 
        return self.gamma * (np.dot(Xi, Xj.T) + self.coef0)**self.degree
            
    
    def _gradient_descent(self, X, y):
        '''
        最急降下法でパラメータを更新する関数

        Parameter
        --------------
        X : 次の形のndarray, shape (n_samples, n_features)
            学習用データの特徴量
        y : 次の形のndarray, shape (n_samples, 1)
            学習用データの正解ラベル

        Returns
        -----------
        lam : 次の形のndarray, shape (n_samples, 1)
            更新されたパラメータ   
        '''
        #カーネル関数の計算
        #線形
        if self.kernel == 'linear':
            k = self._kernel_func(X, X)
        #多項式
        elif self.kernel == 'poly':
            k = self._poly_kernel_func(X, X)
                
        #サンプル数の一時変数
        samples = len(y)

        #ラグランジュ乗数の未定乗数法による再急降下
        for i in range(samples):
            ans = 0  #初期化
            for j in range(samples):
                #1つのインデックスに対してすべてのインデックスをかけ、合計する
                ans += self.lam[j] * y[i] * y[j] * k[i, j]

            #ラグランジュ乗数を更新
            self.lam[i] += self.eta * (1 - ans)
            
            #ラグランジュ乗数が0未満の時、0を代入
            if  self.lam[i] < 0:
                self.lam[i] = 0

        return self.lam