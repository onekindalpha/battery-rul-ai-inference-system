import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

class CustomRobustScaler3D:
    """
    Scikit-Learn의 RobustScaler를 3D 시계열 데이터에 적용하기 위한 Wrapper입니다.
    복잡한 커스텀 로직 대신 검증된 라이브러리를 사용합니다.
    """
    def __init__(self, p=None, min_iqr=None, clip_value=10.0):
        # p, min_iqr 등은 기존 코드 호환성을 위해 인자로만 받고 실제로는 무시하거나 기본값 사용
        # 배터리 데이터는 이상치(Outlier)가 많으므로 RobustScaler가 가장 적합합니다.
        self.scaler = RobustScaler()
        self.clip_value = clip_value

    def fit(self, X):
        # X shape: (Batch, Time, Features) -> (Batch * Time, Features)
        # 3차원 데이터를 2차원으로 납작하게 펴줍니다.
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if X.ndim == 3:
            N, T, F = X.shape
            X_flat = X.reshape(N * T, F)
        else:
            X_flat = X
            
        self.scaler.fit(X_flat)
        return self

    def transform(self, X):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 모양 기억하기
        original_shape = X.shape
        
        # 2차원으로 변형
        if X.ndim == 3:
            N, T, F = X.shape
            X_flat = X.reshape(N * T, F)
        else:
            X_flat = X
            
        # 스케일링 적용 (sklearn)
        X_scaled = self.scaler.transform(X_flat)
        
        # 너무 큰 값 자르기 (안정성 확보)
        if self.clip_value is not None:
            X_scaled = np.clip(X_scaled, -self.clip_value, self.clip_value)
            
        # 원래 모양(3차원)으로 복구
        if len(original_shape) == 3:
            return X_scaled.reshape(original_shape)
        else:
            return X_scaled

# RUL Scaler는 기존 로직이 간단하고 확실하므로 그대로 둡니다.
class RULScaler:
    def __init__(self, mode: str = "minmax"):
        self.mode = mode

    def scale(self, y: np.ndarray, max_rul: float) -> np.ndarray:
        max_rul = float(max_rul) if max_rul is not None else 1.0
        if self.mode == "minmax":
            return np.clip(y / (max_rul + 1e-8), 0, 1)
        else:
            # log1p(y) / log1p(max_rul) 방식도 가능하지만 기존 유지
            return np.log1p(np.clip(y, 0, max_rul * 1.5))

    def unscale(self, y: np.ndarray, max_rul: float) -> np.ndarray:
        max_rul = float(max_rul) if max_rul is not None else 1.0
        if self.mode == "minmax":
            return np.maximum(y * (max_rul + 1e-8), 0)
        else:
            return np.maximum(np.expm1(y), 0)

def scale_rul_array(y, m, mx):
    mx = float(mx) if mx is not None else 1.0
    return np.clip(y / (mx + 1e-8), 0, 1) if m == 'minmax' else np.log1p(np.clip(y, 0, mx * 1.5))

def unscale_rul_array(y, m, mx):
    mx = float(mx) if mx is not None else 1.0
    return np.maximum(y * (mx + 1e-8), 0) if m == 'minmax' else np.maximum(np.expm1(y), 0)