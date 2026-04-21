from sklearn.decomposition import PCA

class PCAReducer:
    def __init__(self, n_components=8):
        self.pca = PCA(n_components=n_components)

    def fit(self, X_train):
        self.pca.fit(X_train)
        variance = self.pca.explained_variance_ratio_.sum()
        print(f"Explained variance: {variance:.3f}")
        assert variance >= 0.85, "PCA must preserve at least 85% variance"
        return self

    def transform(self, X):
        return self.pca.transform(X)