import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

array = [[599  , 5,  74,  98,  55,   14,  12,   9, 117,  17],
 [ 16 ,738 , 12 , 65 ,  9 ,  26,   7 ,  6 , 40,  81],
 [ 31  , 0 ,523 ,168 ,136 ,  86  ,33  ,14  , 9 ,  0],
 [ 10  , 1 , 31 ,652 , 90 , 175 , 19 , 15  , 5  , 2],
 [  6  , 0 , 34 ,132 ,717 ,  55  ,16 , 31  , 9 ,  0],
 [  5  , 1 , 17 ,233 , 53 , 661  ,10 , 15 ,  4 ,  1],
 [  2  , 1 , 39 ,157 ,105 ,  48 ,637 ,  3 ,  7 ,  1],
 [  6  , 0 , 14 , 97 ,103 ,  96 ,  5, 637  , 5 ,  1],
 [ 41  , 7 , 28  ,84 , 19,   18  , 6,   4, 783  ,10],
 [ 25  ,28 ,  8,  77 , 29  , 27   ,5 , 19 , 59, 723]]

df_cm = pd.DataFrame(array, range(10),
                  range(10))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 12})# font size
plt.show()