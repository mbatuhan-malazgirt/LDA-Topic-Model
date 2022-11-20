from ast import Break
from asyncio.windows_events import NULL
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import random


#------------------main------------------
colors = ['blue', 'black', 'green', 'orange', 'purple', 'violet', 'lightpink', 'darkolivegreen', 'rosybrown', 'lime']
k = 2
simpleDataset = [2.25, 15.75, 2.75, 16.35, 2.3, 16.7, 2.1, 16.3, 2.95, 15.85, 2.55, 16.05, 3.4, 16.4, 3.1, 16.85, 2.7, 14.95, 3.3, 15.8, 2.85, 15.55, 3.9, 15.45, 3.85, 15.95, 3.4, 15.15, 1.95, 15.65, 4.65, 16.5, 2.35, 15.65, 4, 17.1, 4.45, 15.65, 3.65, 14.7, 2.9, 14.25, 4.4, 14.6, 5.55, 15.65, 1.9, 14.65, 2.85, 17.6, 4, 16.1, 5, 14.9, 5.05, 16.1, 5.75, 17.35, 4.15, 16.75, 3.6, 16.95, 4.05, 17.5, 4.85, 16.95, 3.6, 17.4, 2.65, 17.2, 2.05, 17.55, 1.9, 16.75, 2.3, 17.15, 1.6, 15.95, 1.2, 16.45, 6.1, 16.65, 5.1, 17.5, 5.25, 16.4, 5.25, 16.75, 5.6, 17, 4.65, 15.35, 4.25, 15.05, 4, 14, 3.05, 13.9, 5.55, 14, 4.2, 13.2, 5.2, 14.65, 4.85, 13.8, 4.55, 14.05, 4.95, 14.25, 3.15, 14.4, 3.6, 13.75, 3.35, 13.25, 2.65, 13.95, 2.3, 14.75, 1.85, 13.65, 6.85, 14.8, 6.85, 15.95, 8.45, 16.8, 7.2, 17.75, 7.2, 16.4, 7.6, 16.85, 6.65, 17.2, 6.25, 15.25, 5.85, 14.9, 6.05, 16.05, 11.9, 9.15, 11.45, 9.9, 13.1, 11.15, 11.8, 11.45, 12.25, 10.7, 12.05, 10.15, 12.65, 10.35, 11.65, 10.8, 12.4, 11, 11.35, 10.6, 10.6, 10.55, 10.75, 9.9, 10.85, 11.15, 12.5, 11.35, 12.4, 11.6, 13.65, 11.8, 13.95, 11.05, 13.45, 10.4, 13.25, 10.25, 13.3, 8.95, 12.55, 8.8, 12.6, 9.8, 13.95, 9.8, 14.25, 8.9, 13.35, 9.55, 14.35, 10.15, 14, 10.75, 15.1, 9.9, 15.35, 8.8, 14.9, 8.4, 14.3, 8, 13.45, 8.1, 12.65, 8.1, 11.85, 8.2, 11.05, 8.35, 10.8, 9, 10.8, 9.45, 15.45, 10.8, 11.85, 9.25, 11.4, 8.5, 10.4, 7, 9.7, 8.3, 10.1, 8.9, 12.15, 7.3, 11.05, 7.75, 10.15, 8.35, 10.05, 7.5, 10, 9.7, 9.7, 9.9, 9.6, 8.95, 14.35, 7.55, 12.7, 7.45, 11.6, 7.35, 12.05, 6.75, 13.3, 7.3, 14, 7.15, 13.65, 6.15, 12.8, 6, 11.5, 6.05, 11.2, 5.55, 11.15, 5.15, 10.8, 5, 10.1, 5.5, 10.1, 6.3, 10.3, 6.75, 11.1, 6.45, 10.85, 5.55, 11.1, 6.75, 9.8, 6.5, 9.3, 6.1, 9, 6.55, 9.25, 7.3, 9.1, 7.85, 9, 8.5, 12.1, 5.75, 13.15, 5.45, 22.8, 15.85, 21.85, 14.5, 22.9, 14.55, 21.15, 15.5, 21.5, 15.85, 14.8, 12.4, 13.05, 13.25, 21, 13.95, 20.3, 14.8, 19.8, 15.55, 20.55, 15.85, 20.8, 16.65, 21.3, 16.75, 22.1, 16.8, 19.3, 16.85, 20.4, 17.65, 20.05, 15.95, 22.5, 17.4, 23.8, 16, 23.95, 15.1, 22.65, 14.55, 22.4, 12.8, 21.1, 13.55, 21.6, 14.65, 20.95, 14.6, 19.6, 14.35, 19.55, 14.25, 21.55, 15.15, 13.1, 8.35, 11.55, 9.65, 11.45, 9.05, 12.05, 8.2, 12.8, 7.15, 11.9, 6, 13, 6.9, 13.85, 7.25, 14.15, 9.25, 12.65, 8.55, 10.55, 7.95, 11.8, 9.65, 12.75, 9.45, 12.45, 8.8, 12.4, 7.9, 12.3, 7.1, 22, 13.5, 22.25, 13.65, 23.95, 13.95, 24, 14.15, 24.15, 16.15, 22.9, 15.95, 21.95, 15.45, 4.65, 14.4, 5.3, 13.25, 5.9, 14.15, 5.5, 16.7, 5.6, 17.8, 3.45, 11.2, 3.7, 12.65, 1.9, 12.2, 4.4, 12.25, 2.65, 11.95, 2.85, 13.1, 7.3, 14.05, 6.55, 13.55, 5.75, 12.7, 21.95, 11.3, 20.9, 12.2, 20.55, 12.7, 20.65, 13.05, 19.55, 13.35, 19.55, 14.25, 18.75, 15.15, 19, 15.75, 19.5, 16.15, 19.9, 16.5, 20.65, 17.35, 21.25, 17.55, 22, 17.3, 22.5, 16.35, 22.5, 15.8, 22.1, 14.55, 21.15, 14.4, 19.8, 14.9, 19.65, 15.25, 20.35, 13.45, 22.25, 13.3, 23.25, 13.3, 22.85, 11.85, 23.7, 14.1, 23.55, 14.85, 22.9, 15.15, 21.95, 15.7, 23.6, 17.9, 24.45, 17.3, 24.75, 15.45, 24.6, 14.1, 24.95, 13.3, 25.35, 13.4, 24.7, 12.85, 24.4, 12.05, 23.4, 12.35, 24.3, 12.05, 23.6, 14, 25.55, 13.6, 26, 13.9, 25.95, 14.9, 25.35, 15.8, 25.25, 17.2, 25.5, 17.45, 26.45, 16.95, 26.25, 15.85, 25.9, 15.1, 26.4, 12.95, 26.35, 12.3, 25.85, 11.55, 25.25, 11.1, 24.1, 10.65, 23.4, 10.5, 23.05, 10.55, 22.15, 10.75, 21.3, 11.1, 19.7, 12.15, 24.85, 11.8, 24.9, 11.7, 24.15, 11.25, 22.7, 11.4, 21.65, 12.65, 23.65, 13.25, 24.85, 13.4, 24.7, 14.55, 23.55, 13.8, 22.8, 13.65, 21.45, 13.75, 21.75, 12.3, 21.95, 11.95, 21.65, 11.9, 23.65, 12.9, 24.6, 11.9, 25.3, 12.5, 24.1, 11.65, 23.45, 11.4, 23.1, 11.6, 24.35, 15.75, 24.6, 16.55, 22.85, 16.85, 23.5, 16.95, 23.9, 17.15, 24.25, 16.15, 26, 16.25, 25.25, 16.55, 25.35, 16.15, 23.8, 15.3, 21.65, 16.6, 20.9, 16.05, 20.4, 15.4, 20.35, 13.25, 21.85, 12.6, 24.5, 14.25, 23.8, 15.1, 22.1, 15.2, 24.1, 14.95, 25.65, 14.6, 26.9, 13.8, 27.3, 12.85, 27.1, 12.2, 27.4, 14.95, 27.5, 15.2, 26.8, 15.35, 6.7, 12.8, 8.05, 15, 7.05, 16.15, 7.25, 14.95, 5.95, 12.8, 5, 12.5, 4.4, 11.55, 3.9, 12.15, 5.05, 13.65, 6.15, 12.75, 5.7, 12.2, 5.45, 12, 5.05, 12.1, 6.35, 13.55, 6.9, 14.85, 6.35, 16.6, 7.65, 17.3, 12.75, 12.1, 12.2, 12.25, 11.6, 12.05, 13.05, 11.5, 14.35, 11.05, 13.75, 9.65, 13.4, 7.95, 12.35, 6.85, 12.15, 4.85, 13.75, 4.85, 14.05, 5.15, 13.35, 5.65, 14.05, 6, 14.3, 6.75, 15.1, 6.7, 16.1, 6.4, 15.65, 4.65, 10.15, 16.05, 8.35, 15.95, 9.45, 17.05, 22.95, 9.95, 26.05, 10.1, 27.95, 11.55, 29.8, 14.1, 28.75, 17.25, 28.7, 16.05, 28.25, 13.75]
hardDataset = [6.55, 4.6, 6.55, 4.65, 6.75, 5.6, 6.95, 6.1, 7.3, 6.85, 7.6, 7.55, 7.75, 7.85, 8.4, 9.15, 8.8, 9.7, 9.3, 10.85, 10, 12, 10.45, 12.7, 11, 13.2, 12.05, 12.55, 12.75, 10.9, 13.15, 9.75, 13.5, 8, 13.75, 6.8, 14.2, 5.05, 14.4, 4.25, 14.25, 4.8, 14.2, 6.3, 13.95, 7.2, 13.85, 8.15, 13.5, 9.3, 13.15, 10.45, 12.5, 12, 11.7, 12.6, 11.2, 12.9, 10.4, 12.15, 7.95, 9.2, 7.5, 8.45, 7.5, 8.15, 7.6, 7.55, 6.9, 5.55, 7.3, 8.1, 8.55, 9.75, 9.05, 10.9, 9.45, 11.4, 9.95, 12.1, 9.15, 9.8, 8.35, 8.25, 8.35, 8.6, 8.1, 8.1, 8.85, 9.75, 9.45, 10.8, 10.25, 11.9, 10.65, 12.1, 14.05, 9.25, 14.25, 8.2, 14.5, 5.7, 14.6, 4.8, 14.45, 5.35, 14.25, 8.05, 13.6, 9.75, 13.15, 11.05, 12.7, 11.6, 12.3, 11.95, 14.6, 4, 14.4, 3.4, 14.4, 3.15, 14.4, 2.35, 14.55, 1.85, 15, 1.45, 16.05, 2.5, 16.25, 3, 16.5, 4.25, 16.75, 5.65, 16.85, 7, 16.9, 8.25, 16.65, 10.55, 16.4, 11.65, 16.3, 13.2, 16.15, 14.25, 16.15, 15.55, 16.2, 16.05, 16.1, 16.4, 16.15, 16.7, 16.2, 15, 16.15, 14, 16.15, 12.6, 16.2, 11.85, 16.3, 10.95, 16.4, 8.5, 16.45, 7.75, 16.75, 6.4, 16.75, 4.75, 16.3, 3.65, 15.65, 2.3, 14.8, 2.05, 14.5, 2.35, 14.95, 2.45, 16.5, 4.15, 17.35, 6.7, 17.2, 8.4, 16.9, 9.15, 16.8, 10.05, 16.45, 8.65, 16.2, 9.7, 16.25, 9.85, 16.3, 9.8, 12.6, 18.4, 13.35, 18.35, 14.5, 18.45, 15.15, 18.4, 16.25, 18.4, 17.15, 18.35, 18.3, 18.35, 19.55, 18.45, 19.6, 18.45, 16.65, 18.35, 15.05, 18.1, 13.25, 18.2, 13.05, 18.25, 13.75, 18.35, 14.3, 18.25, 17.1, 18.4, 17.95, 18.35, 18.7, 18.35, 19.8, 18.5, 18.1, 18.35, 15.9, 18.4, 15.9, 18.4, 15.45, 18.35, 15.25, 18.3, 16.1, 18.3, 17.85, 18.3, 18.55, 18.3, 16.5, 16.55, 16.5, 16.55, 17.25, 15, 17.8, 14.25, 18.95, 12.6, 19.75, 11.5, 20.45, 10.4, 21, 9.75, 21.75, 8.8, 22.5, 7.55, 23.7, 6, 24.15, 5.4, 25.05, 4.05, 25.5, 3.4, 25.75, 3.05, 25.7, 3.2, 25.1, 4.6, 24.55, 5.25, 23.8, 6.3, 23.25, 7.05, 22.75, 7.6, 22.15, 8.4, 20.85, 9.85, 20.3, 10.65, 19.85, 11.15, 19.3, 11.65, 18.6, 12.55, 18.1, 13.25, 17.55, 14.3, 16.9, 15.15, 16.45, 16.15, 16.6, 15.8, 17.75, 14.4, 18.2, 13.8, 18.8, 12.65, 19.35, 11.5, 20.05, 10.35, 20.55, 9.75, 21.35, 8.9, 21.9, 8.6, 22.85, 7.55, 23.95, 6.15, 24.8, 5.25, 25.7, 4, 26.2, 3, 26, 4.05, 26.15, 5.95, 26.25, 8.1, 26.2, 9.3, 26.3, 10.4, 26.5, 12.15, 26.65, 14, 26.6, 15.5, 26.55, 15.95, 26.4, 17.25, 26.4, 17.2, 26.7, 15.4, 26.45, 14.3, 26.3, 13.3, 26.2, 12.15, 26.25, 10.95, 26.25, 9.75, 26.2, 8.3, 26.05, 6.8, 25.8, 5.3, 25.75, 4.3, 26, 3.75, 26, 6.15, 26.35, 7.8, 26.15, 10.25, 26.3, 12.35, 26.35, 13, 26.5, 14.3, 26.7, 16.8, 26.7, 17.3, 6.1, 15.35, 6.9, 15.5, 7.95, 15.5, 8.8, 15.4, 9.7, 15.35, 11.65, 15.35, 13.35, 15.4, 14.45, 15.3, 15.75, 15.2, 15.75, 15.2, 13.8, 15.3, 12.3, 15.2, 11, 15.15, 9.5, 15.1, 7.4, 15.1, 5.8, 15.2, 6.65, 15.15, 8.65, 15.25, 9.6, 15.25, 11.1, 15.2, 11.6, 15.15, 12.55, 15.15, 13.45, 15.2, 14.5, 15.3, 15.15, 15.25, 15.2, 15.3, 12.45, 15.3, 10.65, 15.3, 10.25, 15.3, 9.05, 15.2, 7.9, 15.1, 7.3, 15.25, 6.55, 15.3, 8.2, 15.4, 8.8, 15.3, 11.75, 15.35, 14.05, 15.35, 14.4, 15.35, 15.55, 15.5, 16.2, 15.45, 9.5, 12, 9.1, 10.85, 8.05, 9, 8.6, 10.25, 9, 10.5, 9.1, 10.6, 7.95, 7.45, 7.1, 5.95, 6.4, 4.9, 6.7, 6.5, 7.2, 7.5, 7.45, 7.6, 9.5, 9.95, 13.5, 9.6, 13.7, 8.55, 13.6, 8.75, 13.6, 9.15, 14.15, 7.9, 14.25, 6.55, 14.35, 6, 14.3, 4.85, 14.3, 4.3, 14.35, 2.1, 14.6, 1.85, 15.8, 2.45, 16.7, 3.95, 17.2, 5.6, 17.2, 6.35, 16.95, 7.15, 16.9, 8.3, 16.8, 9.15, 16.65, 9.85, 16.5, 11.2, 16.35, 12.6, 16.2, 13.55, 16.3, 14.9, 16.55, 15.45, 16.7, 15.85, 18.4, 12.65, 18.8, 12.1, 19.2, 11.55, 19.15, 12, 20.1, 10.75, 21.45, 9.1, 22.35, 8.25, 22.65, 7.85, 23.1, 7.25, 24.25, 6.05, 25.15, 4.7, 25.75, 3.7, 25.75, 4.25, 25.6, 5.75, 25.7, 6.15, 25.95, 5.3, 26, 4.75, 25.8, 4.45, 25.85, 7.5, 26.3, 9, 26.35, 9.2, 26.5, 8.4, 26.35, 6.75, 25.85, 7.45, 25.95, 8.05, 25.9, 9.8, 26.05, 10.5, 26.15, 11.35, 26.2, 11.7, 26.45, 12.45, 26.55, 13.55, 26.6, 14.05, 26.6, 15.05, 26.65, 16.45, 27.05, 17.5, 16.35, 15.7, 16.2, 14.1, 16.1, 13.6, 16.1, 13.45, 16.15, 12.45, 16.15, 11.7, 16.2, 11.5, 16.25, 10.35, 16.35, 9.05, 16.55, 8.05, 16.7, 6.95, 16.75, 5.45, 20.5, 2.35, 20.55, 2.45, 21.05, 3.9, 21.25, 4.35, 21.55, 5.15, 22, 5.9, 22.3, 6.15, 21.85, 5.65, 20.9, 3.95, 20.8, 3.4, 20.9, 3.6, 20.8, 3.1, 20.8, 3.25, 21.75, 5.3, 22.15, 6.1, 22.45, 6.45, 22.95, 7.05, 23.5, 7.5, 24.45, 7.95, 25, 8.45, 25.75, 8.8, 26.3, 8.9, 25, 8.35, 24.1, 7.75, 23.7, 7.3, 24.45, 7.95, 26.45, 8.75, 27.4, 8.85, 28.05, 8.7, 29, 8.45, 29.85, 7.8, 30.5, 7.3, 31.25, 6.7, 31.6, 6.2, 32, 5.55, 32.3, 4.7, 32.7, 3.85, 33.3, 3.1, 33.35, 2.95, 32.55, 4.3, 32.2, 5, 31.25, 6.45, 30.65, 7.2, 29.5, 7.45, 28.3, 7.95, 27.1, 8.2, 24.85, 8.2, 24.4, 7.95, 25.4, 8.45, 26.55, 8.4, 27.15, 8.2, 28.75, 7.95, 29.15, 7.75, 30.05, 7.05, 31.15, 6.15, 32.25, 5.1, 32.65, 4.65, 33.1, 3.95, 33.5, 3.3, 33.9, 4.25, 34.55, 5.45, 35.05, 6.9, 35.6, 8.45, 35.9, 9.55, 36.3, 10.7, 36.5, 12.25, 36.5, 13.6, 36.55, 14.5, 36.6, 15.3, 36.55, 16, 36.55, 16.5, 36.55, 16.5, 36.7, 15.15, 36.5, 14.25, 36.4, 13.4, 36.3, 12.95, 36.25, 12.05, 36.15, 10.25, 36.1, 8.9, 35.8, 8.05, 35.35, 6.85, 34.85, 5.55, 34.25, 4.3, 33.85, 3.85, 34.1, 4.65, 35, 6.1, 35.35, 7.45, 36.15, 9.85, 36.3, 10.9, 36.25, 11.35, 28.5, 14.9, 28.55, 14.95, 28.8, 15, 29.15, 15.05, 29.7, 15.1, 29.75, 15.1, 30.5, 15.1, 30.9, 15.05, 31.65, 15.15, 32.2, 15.15, 32.6, 15.15, 33.05, 15.15, 34.05, 15.15, 33.65, 15.1, 33.3, 15.15, 31.35, 10.7, 31.4, 10.7, 31.7, 10.75, 31.95, 10.8, 32.35, 10.8, 32.9, 10.7, 33.5, 10.65, 34, 10.65, 34.4, 10.65, 34.5, 10.65, 30.7, 10.95, 30.7, 10.95, 30.85, 10.8, 30.9, 10.75, 30.3, 10.7, 30.05, 10.75, 21.4, 12.4, 21.95, 12.45, 22.25, 12.45, 22.75, 12.35, 23.15, 12.35, 23.55, 12.4, 23.95, 12.3, 8.95, 6.1, 9.25, 6.1, 9.95, 6.25, 10.8, 6.2, 11.3, 6.3, 11.9, 6.3, 29.2, 1.65, 30, 1.85, 30.35, 1.95, 30.75, 1.95, 31.7, 1.95, 32.4, 1.8, 21.1, 12.5, 21.55, 12.55, 23.1, 12.55, 24.1, 12.5]

#plot dataset
def plotDataset(dataset):
    plt.scatter(dataset[::2], dataset[1::2])
    plt.show()

#my kmeans algorithm
def kMeansAlgorithm(k, dataset):
    #calculate the new centroids
    #repeat until the centroids don't change
    iteration = 0
    objectiveFunctions = []
    iterations = []
    #take k random points from dataset list as centroids
    centroids = []
    allClusters = []
    for i in range(k):
        rNumber = random.randint(0, len(dataset)-2)
        if rNumber % 2 == 1:
            rNumber += 1
        centroids.append(dataset[rNumber])#x
        centroids.append(dataset[rNumber+1])#y
    #----------------------we have k random centroids [x,y...] now---------------------

    #assign each point to the closest centroid
    lastObjectiveFunction = 9999999
    while(True):
        iteration += 1
        for i in range(k):
            allClusters.append([])
        for i in range(0, len(dataset), 2):
            minDistance = euclidianDistance(dataset[i], centroids[0], dataset[i+1], centroids[1])
            minIndex = 0
            for j in range(2, len(centroids), 2):
                distance = euclidianDistance(dataset[i], centroids[j], dataset[i+1], centroids[j+1])
                if distance < minDistance:
                    minDistance = distance
                    minIndex = j
            allClusters[minIndex//2].append(dataset[i])
            allClusters[minIndex//2].append(dataset[i+1])
        
        newCentroids = []
        for i in range(k):
            totalofx = 0
            totalofy = 0

            for j in range(0, len(allClusters[i]), 2):
                totalofx += allClusters[i][j]
                totalofy += allClusters[i][j+1]

            newCentroids.append(totalofx/(len(allClusters[i])/2))
            newCentroids.append(totalofy/(len(allClusters[i])/2))
            #new centroids are calculated
        #here we are gonna calculate the objective function.
        #Objective function is the sum of the distances between each point and its centroid
        #we need to hold the last objective function value to compare with the new one, if the new one is smaller than the last one, we need to continue the algorithm

        #plot the first 3 iterations here using matplotlib but put different colors on clusters.
        if iteration < 4:
            for i in range(k):
                plt.scatter(allClusters[i][::2], allClusters[i][1::2], color=colors[i])

            plt.scatter(splitXandY(newCentroids)[0], splitXandY(newCentroids)[1], color='yellow', marker="X")
            plt.title("Iteration " + str(iteration) + " of my K-Means Algorithm")
            plt.show()

        objectiveFunction = 0

        for i in range(k):
            for j in range(0, len(allClusters[i]), 2):
                objectiveFunction += euclidianDistance(allClusters[i][j], newCentroids[i*2], allClusters[i][j+1], newCentroids[i*2+1])

        objectiveFunctions.append(objectiveFunction)

        if lastObjectiveFunction - objectiveFunction < 0.0000001:
            #plot the last iteration here
            for i in range(k):
                plt.scatter(allClusters[i][::2], allClusters[i][1::2], color=colors[i])
            plt.scatter(splitXandY(newCentroids)[0], splitXandY(newCentroids)[1], color='red', marker="X")
            plt.title("Last Iteration of my K-Means Algorithm")
            plt.show()

            #plot the objective function here
            for i in range(iteration):
                iterations.append(i+1)
            plt.plot(iterations, objectiveFunctions)
            plt.xlabel("Iteration")
            plt.ylabel("Objective Function")
            plt.title("Objective Function vs Iteration")
            plt.show()
            break

        lastObjectiveFunction = objectiveFunction

        centroids = newCentroids
        allClusters = [[]]

    return allClusters

#calculate the euclidian distance between two points
def euclidianDistance(x1, x2, y1, y2):
    #calculate the distance between two points
    #return the distance

    distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)

    return distance

#scikit learn kmeans algorithm
def scikitKMeans(k, dataset):
    #use the scikit-learn kmeans algorithm
    #what should be the return value?
    
    kMeans = KMeans(n_clusters=k)
    kMeans.fit(pairUp(dataset))
    #print(kMeans.labels_)
    return kMeans.cluster_centers_

#pair up the x and y values in dataset
def pairUp(dataset):
    result = []
    for i in range(0, len(dataset), 2):
        result.append([dataset[i], dataset[i+1]])
    return result

#split the x and y values in dataset
def splitXandY(dataset):
    #split the dataset into x and y values
    #return the x and y values
    
    x = []
    y = []
    for i in range(0, len(dataset), 2):
        x.append(dataset[i])
        y.append(dataset[i+1])
    #returns x and y as a list.
    return x, y

#find the optimal k value using silhuette score
def findOptimalK(dataset):
    #elbow method
    #find the optimal k value
    #use silhuette score to find the optimal k value between 2 and 10
    #return the optimal k value and the score

    max = 0
    maxIndex = 0
    for i in range(0, 10):
        kMeans = KMeans(n_clusters=i+2)
        kMeans.fit(pairUp(dataset))
        score = silhouette_score(pairUp(dataset), kMeans.labels_, metric='euclidean')
        if score > max:

            max = score
            maxIndex = i+2
    return maxIndex, max

#this function is to use sci kit and plot the clusters
def sciKitUse(k, dataset):
    #use the scikit kmeans algorithm and plot the result

    #we have labels at hand, labels will tell us which cluster the point belongs to
    #we need to plot the points with different colors according to their labels
    #we need to plot the centroids with different colors
    #we need to plot the objective function vs iteration

    kMeans = KMeans(n_clusters=k)
    pairedData = pairUp(dataset)
    kMeans.fit(pairedData)
    labels = kMeans.labels_
    centroids = kMeans.cluster_centers_

    for i in range(len(pairedData)):
        plt.scatter(pairedData[i][0], pairedData[i][1], color = colors[labels[i]])
    for centroid in centroids:
        plt.scatter(centroid[0], centroid[1], color = 'red', marker = "X")

    plt.title("SciKit KMeans Library Output")
    plt.show()

optimalK, score = findOptimalK(hardDataset)
print("Optimal K: ", optimalK)
print("Score: ", score)
#kMeansAlgorithm(k, hardDataset)
sciKitUse(k, hardDataset)


