from typing import Union

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix

# Custom types
MatrixLike = Union[np.ndarray, pd.DataFrame, spmatrix]
ArrayLike = Union[np.ndarray, pd.Series, list]
