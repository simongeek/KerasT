import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

array = [[772  , 5  ,24,  19,  17,   6,  18,  10,  66,  63],
 [ 14 ,637  , 1 ,  4 ,  3 ,  7 , 19 ,  2,  38, 275],
 [ 81  , 0 ,538,  50  ,88,  86  ,93,  27  ,19 , 18],
 [ 20   ,1  ,52, 468  ,60 ,180 ,143 , 33  ,13,  30],
 [ 19   ,1  ,51 , 59 ,662  ,33  ,91  ,66  ,16  , 2],
 [ 12   ,0  ,34 ,135  ,37 ,664  ,53  ,41  ,11  ,13],
 [  7   ,0  ,23  ,29  ,26  ,13 ,885   ,2  ,10  , 5],
 [ 10   ,0  ,24  ,45  ,48  ,69  ,20 ,756  , 5  ,23],
 [ 74   ,4  , 3  , 9,   4   ,6 ,  8  , 4, 854  ,34],
 [ 18   ,8 ,  5  ,13 ,  9   ,4,  10   ,6 , 34 ,893]]

df_cm = pd.DataFrame(array, range(10),
                  range(10))
#plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 12})# font size
plt.show()