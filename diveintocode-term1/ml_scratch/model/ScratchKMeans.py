class ScratchKMeans():
    '''
    K-meansのクラス
        
    Parameters
    --------------
    n_clusters : int
        クラスターの数
    n_init : int
        K-meansアルゴリズムが異なる重心で実行される回数
    max_iter : int
        K-meansアルゴリズムの最大反復回数
    tol : float
        収束する際の誤差
        
    Attributes
    -------------
    center : 次の形のndarray, shape (n_clusters, n_features)
        中心点
    clusters : 次の形のndarray, shape (n_samples, )
        クラスタ
    sse : float
        クラスタ内誤差平方和
    '''
    def __init__(self, n_clusters=4, n_init=10, max_iter=100, tol=1e-4):
        #ハイパーパラメータ
        self.n_clusters = n_clusters #クラスター数
        self.n_init         = n_init         #異なる重心で実行される回数
        self.max_iter    = max_iter   #最大のイテレーション回数
        self.tol               = tol              #収束する際の誤差
        
        #インスタンス変数
        self.center   = None #中心点
        self.clusters = None #クラスタ
        self.sse         = None #SSE
        
        
    def fit(self, X):
        '''
        クラスタリングの学習
        
        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            学習用データの特徴量
        '''
        #中心点を設定
        self.center = self._choice_center(X)
        
        #イテレーションの回数実行
        for iteration in range(self.max_iter):
            
            #中心点との距離を入れる配列
            distance = np.zeros((X.shape[0], self.n_clusters))

            #クラスタの数だけデータとの距離を求める
            for i in range(self.n_clusters):
                euclid = np.linalg.norm(X - self.center[i], axis=1)
                
                #求めた距離を格納
                distance[:, i] += euclid            

            #各データと中心が最小の距離
            min_distance = np.amin(distance, axis=1, keepdims=True)
            
            #各データが割り当てられるクラスタの配列
            _, clusters = np.where(distance==min_distance)
            
            # dict{クラスター名: データの数}
            cluster_count = {i: len(clusters[clusters == i]) for i in np.unique(clusters)}
            
            #クラスターに割り当てられるデータ点が０のとき、そのクラスター名を取得
            dic = [k for k, v in cluster_count.items() if v == 0]            
            
            #距離が一番遠いものを選ぶ
            for i in dic:
                max_distance = max(distance[:, i])
                
                #距離が一番遠い時の行数を取得
                x_idx, _ = np.where(distance==max_distance)
                
                #中心点を一番遠い所にする
                self.center[i, :] == X[x_idx, :]            

            #重心を求める
            new_center = np.array([X[clusters == i].mean(axis=0) for i in range(self.n_clusters)])

            #中心点と重心が指定した誤差以下となればbreak
            if np.all(abs(new_center - self.center) <= self.tol):
                break
                
            #クラスターが同じならbreak
            if np.all(self.clusters == clusters):
                break
            
            #中心点の更新
            self.center = new_center

            #クラスターの更新
            self.clusters = clusters

            
    def fit_predict(self, X):
        '''
        クラスタリングの学習と推定
        
        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            学習用データの特徴量
            
        Returns
        ----------
        self.clusters : 次の形のndarray, shape (n_samples, )
            推定したクラスター
        '''
        #中心点を設定
        self.center = self._choice_center(X)
        
        #イテレーションの回数実行
        for iteration in range(self.max_iter):
            
            #中心点との距離を入れる配列
            distance = np.zeros((X.shape[0], self.n_clusters))

            #クラスタの数だけデータとの距離を求める
            for i in range(self.n_clusters):
                euclid = np.linalg.norm(X - self.center[i], axis=1)
                
                #求めた距離を格納
                distance[:, i] += euclid            

            #各データと中心が最小の距離
            min_distance = np.amin(distance, axis=1, keepdims=True)
            
            #各データが割り当てられるクラスタの配列
            _, clusters = np.where(distance==min_distance)
            
            # dict{クラスター名: データの数}
            cluster_count = {i: len(clusters[clusters == i]) for i in np.unique(clusters)}
            
            #クラスターに割り当てられるデータ点が０のとき、そのクラスター名を取得
            dic = [k for k, v in cluster_count.items() if v == 0]            
            
            #距離が一番遠いものを選ぶ
            for i in dic:
                max_distance = max(distance[:, i])
                
                #距離が一番遠い時の行数を取得
                x_idx, _ = np.where(distance==max_distance)
                
                #中心点を一番遠い所にする
                self.center[i, :] == X[x_idx, :]            

            #重心を求める
            new_center = np.array([X[clusters == i].mean(axis=0) for i in range(self.n_clusters)])

            #中心点と重心が指定した誤差以下となればbreak
            if np.all(abs(new_center - self.center) == self.tol):
                break
                
            #クラスターが同じならbreak
            if np.all(self.clusters == clusters):
                break
            
            #中心点の更新
            self.center = new_center

            #クラスターの更新
            self.clusters = clusters
            
        return self.clusters
    

    def predict(self, X):
        '''
        クラスタリングによる推定
        
        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            学習用データの特徴量
            
        Returns
        ----------
        self.clusters : 次の形のndarray, shape (n_samples, )
            推定したクラスタ
        '''
        #中心点との距離を入れる配列
        distance = np.zeros((X.shape[0], self.n_clusters))
        
        #クラスタの数だけデータとの距離を求める
        for i in range(X.shape[0]):
            euclid = np.linalg.norm(X[i] - self.center, axis=1)
                
            #求めた距離を格納
            distance[i, :] += euclid    
            
        #各データと中心が最小の距離
        min_distance = np.amin(distance, axis=1, keepdims=True)
            
        #各データが割り当てられるクラスタの配列
        _, clusters = np.where(distance==min_distance)
        
        return clusters
                
    
    def _choice_center(self, X):
        '''
        中心点を決める
        
        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            学習用データの特徴量
            
        Returns
        ----------
        self.cenrer : 次の形のndarray, shape (n_clusters, n_features)
            推定した中心点
        '''
        sse_min_list = [] #sseを入れるリスト
        center = [] #中心点を入れるリスト
        
        #イテレーション数実行
        for i in range(self.n_init):

            #データの行番号をランダムに取得(重複なし)
            row = X.shape[0]
            idx = np.random.choice(row, self.n_clusters, replace=False)

            #中心点の初期値
            self.center = X[idx]
            
            #中心点をリストに格納
            center.append(self.center)
            
            #中心点との距離を入れる配列
            distance = np.zeros([X.shape[0], self.center.shape[0]])
            
            #クラスタの数だけデータとの距離を求める
            for i in range(self.n_clusters):
                euclid = np.linalg.norm(X - self.center[i], axis=1)
                
                #求めた距離を格納
                distance[:, i] += euclid

            #各データと中心が最小の距離
            min_distance = np.amin(distance, axis=1, keepdims=True)
            
            #各データが割り当てられるクラスタの配列
            _, clusters = np.where(distance==min_distance)
            
            #sseを入れるリスト(1つの中心点)
            sse_list = []
            
            for k in range(self.n_clusters):
                for n in range(X.shape[0]):
                    
                    #データがクラスタに属していたら1, それ以外は0
                    if clusters[n] == k:
                        r = 1
                    else:
                        r = 0
                        
                    #SSEをリストに格納
                    sse_list.append(r * np.linalg.norm(X[n] - self.center[k])**2)
  
            #1イテレーションのSSEをリストに格納
            sse_min_list.append(np.sum(sse_list))
        
        #全てのイテレーションの中で一番小さいSSEを取得
        self.sse = min(sse_min_list)
        
        #SSEが一番小さい時のindexを取得
        min_index = sse_min_list.index(self.sse)
        
        #中心点を決める
        self.center = center[min_index]
       
        return self.center