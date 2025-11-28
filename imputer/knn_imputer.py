from sklearn.impute import KNNImputer as SKLearnKNNImputer

class KNNImputer:
    """
    KNN填补方法
    """
    
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
    
    def impute(self, raw_data, verbose=True):
        """
        执行KNN填补
        
        Parameters:
        - raw_data: numpy.ndarray, 包含NaN值的原始数据
        - verbose: bool, 是否显示详细信息
        """
        if verbose:
            print(f"(IMPUTATION) KNN\n\tMatrix: {raw_data.shape}\n\tn_neighbors: {self.n_neighbors}")
            
        # 使用重命名的sklearn KNNImputer
        imputer = SKLearnKNNImputer(n_neighbors=self.n_neighbors)
        imputed_data = imputer.fit_transform(raw_data)
        return imputed_data

# 为了向后兼容，保留函数
def knn_impute(raw_data, n_neighbors=5):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    return imputer.impute(raw_data)