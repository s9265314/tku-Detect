# tku-Detect


校慶密室逃脫 返校
其中之一的解謎道具


---

## 硬體![](https://i.imgur.com/ALHFbnE.jpg)
利用雷射切割機製作樣式之拼圖
並利用python辨識拼圖是否正確後
傳資料給arduino
控制下一道提示從天花板上掉下

---
## 程式

### train
利用resnet50遷移學習
訓練成3類
data/mark - 正確  

data/error - 缺失  

data/unknow - 擺放錯誤  


### py2ino
將訓練好的模型匯入後
辨識拼圖是否正確
並傳訊息給arduino

### ino2py
為arduino上燒錄之程式
可以接收python辨識結果掉落提示
並有按鈕可以回到將提示所回便於場復

