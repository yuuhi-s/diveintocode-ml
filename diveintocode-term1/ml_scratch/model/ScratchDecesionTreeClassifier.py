import numpy as np

class ScratchDecesionTreeClassifier():
    '''
    決定木のスクラッチ実装

    Parameters
    --------------
    max_depth : int
        決定木の深さ

    Attributes
    -------------
    self.node : instance
        Nodeクラスのインスタンス
    '''

    def __init__(self, max_depth=None):
        # ハイパーパラメータ
        self.max_depth = max_depth  # 木の最大の深さ

        # インスタンス変数
        self.node = None  # Nodeクラスのインスタンス
        

    def fit(self, X, y):
        '''
        決定木を学習する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            学習用データの特徴量
        y : 次の形のndarray, shape (n_samples, )
            学習用データの正解値
        '''
        # 1次元なら2次元にする
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if y.ndim == 1:
            y = y[:, np.newaxis]

        # Nodeクラスをインスタンス化
        self.node = Node(self.max_depth)

        # Nodeクラスで学習
        self.node.node_split(X, y, 0)


    def predict(self, X):
        '''
        決定木を使いラベルを推定する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            検証用データの特徴量

        Returns
        -------
        y_pred : 次の形のndarray, shape (n_samples, 1)
            予測したラベル
        '''
        # 1次元なら2次元にする
        if X.ndim == 1:
            X = X[:, np.newaxis]

        # ラベルを格納していくリスト
        y_pred = []

        # データのサンプル数だけ予測
        for i in X:
            y_pred.append(self.node.predict(i))
        
        #numpy配列でReturn
        return np.array(y_pred)


class Node():
    '''
    決定木のノードを分けるクラス

    Parameters
    --------------
    max_depth : int
        決定木の深さ

    Attributes
    -------------
    self.depth : int
        木の深さ
    self.info_gein : float
        最大の情報利得
    self.feature : int
        情報利得が最大の時の特徴量の列番号
    self.threshold : float
        情報利得が最大の時のしきい値
    self.left_node : instance
        左側のノード
    self.right_node : instance
        右側のノード
    self.label : int
        予測ラベル
    '''
    def __init__(self, max_depth=None):
        # ハイパーパラメータ
        self.max_depth = max_depth  # 木の最大の深さ

        # インスタンス変数
        self.depth          = None  # 木の深さ
        self.info_gain    = None  # 最大の情報利得
        self.feature        = None  # 情報利得が最大のときの特徴量の列番号
        self.threshold     = None  # 情報利得が最大のときのしきい値
        self.left_node    = None  # 左ノードのインスタンス
        self.right_node  = None  # 右ノードのインスタンス
        self.label             = None  # 予測ラベル


    def node_split(self, X, y, depth):
        '''
        ノードを分ける関数

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            学習用データの特徴量
        y : 次の形のndarray, shape (n_samples, 1)
            学習用データの正解値
        depth : int
            木の深さ
        '''
        # 木の深さ
        self.depth = depth

        # dict{クラス名: ラベルの数}
        class_count = {i: len(y[y == i]) for i in np.unique(y)}

        # ラベルの数が最大のクラス名
        self.label = max(class_count.items(), key=lambda x: x[1])[0]

        # 木の深さが最大のときReturn
        if self.depth == self.max_depth:
            return

        # 特徴量の数
        num_features = X.shape[1]

        # 情報利得の初期値
        self.info_gain = 0

        # 特徴量の列数実施
        for f in range(num_features):
            # 特徴量の行数実施
            for threshold in X[:, f]:

                # あるしきい値を設定したときのノードと情報利得
                left = y[X[:, f] < threshold]  # 左ノード
                right = y[X[:, f] >= threshold]  # 右ノード
                ig = self._information_gain(y, left, right)  # 情報利得

                # 情報利得が今までより上回った時更新
                if self.info_gain < ig:
                    self.info_gain = ig  # 情報利得
                    self.feature = f  # 特徴量の列番号
                    self.threshold = threshold  # しきい値

        # 情報利得が0のときReturn
        if self.info_gain == 0:
            return

        # 左側のノード
        X_left = X[X[:, self.feature] < self.threshold]  # サンプル
        y_left = y[X[:, self.feature] < self.threshold]  # ラベル
        self.left_node = Node(self.max_depth)  # インスタンス化
        self.left_node.node_split(X_left, y_left, depth + 1)  # 再度分割

        # 右側のノード
        X_right = X[X[:, self.feature] >= self.threshold]  # サンプル
        y_right = y[X[:, self.feature] >= self.threshold]  # ラベル
        self.right_node = Node(self.max_depth)  # インスタンス化
        self.right_node.node_split(X_right, y_right, depth + 1)  # 再度分割


    def predict(self, X):
        '''
        ラベルを推定する

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            検証用データの特徴量

        Returns
        ----------
        self.label : int
            予測したラベル
        self.left_node.predict(X) : method
            左のノードで予測するメソッド
        self.right_node.predict(X) : method
            右のノードで予測するメソッド
        '''
        #特徴量の列番号がない(ノードの時など) or 深さが最大ならラベルをReturn
        if self.feature == None or self.depth == self.max_depth:
            return self.label

        else:
            '''
            ある列のある特徴量がしきい値未満のときは左ノード、
            しきい値以上のときは右ノードを返し、多数決で予測
            '''
            if X[self.feature] < self.threshold:
                return self.left_node.predict(X) #左ノード
            else:
                return self.right_node.predict(X) #右ノード

    
    def _gini_impure(self, y):
        '''
        ジニ不純度を計算する

        Parameters
        ----------
        y : 次の形のndarray, shape (n_samples, 1)
            学習用データのラベル

        Returns
        ----------
        gini : float
            ジニ不純度
        '''
        # yのクラスのユニーク値を取得
        y_unique = np.unique(y)

        # ラベルの総数を取得
        labels = len(y)

        # ラベルの総数が0の時はジニ不純度を0とする
        if labels == 0:
            gini = 0
            return gini

        # 初期値
        gini = 1

        # クラスの数だけ実行
        for i in y_unique:
            val = len(y[y == i]) / labels # あるクラスに属するサンプル数とノードのサンプルの総数を割る
            gini = gini - val**2.0  # 2乗したものを初期値から引く

        return gini
    

    def _information_gain(self, y_p, y_left, y_right):
        '''
        情報利得を計算する

        Parameters
        ----------
        y_p : 次の形のndarray, shape (n_samples, 1)
            親ノードの学習用データのラベル
        y_left : 次の形のndarray, shape (n_samples, 1)
            左側のノードの学習用データのラベル
        y_right : 次の形のndarray, shape (n_samples, 1)
            右側のノードの学習用データのラベル

        Returns
        ----------
        ig : float
            情報利得
        '''
        #ジニ不純度の計算
        p = self._gini_impure(y_p)  #親ノード
        left = self._gini_impure(y_left) #左側のノード
        right = self._gini_impure(y_right) #右側のノード

        # 情報利得の計算
        ig = p - (len(y_left) / len(y_p) * left) - (len(y_right) / len(y_p) * right)

        return ig