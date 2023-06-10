import pandas as pd
import numpy as np
from preprocessor import Preprocessor


preprocessor = Preprocessor('abcnews-date-text.csv')
preprocessor.preprocessAndSave()

