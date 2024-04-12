# ﻿cat_classification
## Environment package installation
[![Python](https://img.shields.io/badge/Python-3.10.11-blue.svg)](https://www.python.org/downloads/release/python-31011/)

請參考以下安裝方式：
```
git clone https://github.com/cookiedream/cat_classification.git
```

安裝資源包請使用下面的command line：

    pip install -r requirements.txt

## 需要調整參數
可以去 `train.yaml` 裡面去調整

## 如何開啟 `tensorboard` 查看訓練圖
### 請在 `cmd` 中路徑指向 `run` 這個資料夾
	cd ./runs
 ### 接下來在 `cmd` 中執行
	tensorboard --logdir==<model_name>
 ### 會看到以下圖片
 ![image](https://github.com/cookieyu2000/cat_classification/assets/105692097/604a79fc-1649-4b3d-b841-698995dad260)
 
請複製反灰連結至瀏覽器中


## 程式架構
    ```
    $tree -L 2
    
    
	├── README.md
	├── Model
	├── data
	├── fig
	│   └── fig.png
	├── README.md
	├── dataloader.py
	├── requirements.tx
	├── train.py
	├── api.py
	└── requirements.txt
    ```

## 使用git上傳請注意
```
## git command line ( if you open the project first time) ##
step 1:
	git checkout Wu / Chen (choose whos branch you want to checkout)
step 2:
	git merge main (merge the main branch to your branch)

## git if you want to push your code ##
step 1:
	git add .
step 2:
	git commit -m "What you want to say"
step 3:
	git push
```
