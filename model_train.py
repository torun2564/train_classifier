
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import timm

classes_ja = ['E5',
              'E6',
              'E7',
              'JR中央線E233系',
              'JR八高線E231系',
              'JR京浜東北線E233系',
              'JR京葉線E233系',
              'JR武蔵野線E231系',
              'JR埼京線E233系',
              'JR湘南新宿ラインE231系',
              'JR湘南新宿ラインE233系',
              'JR総武線E231系',
             'JR山手線E235系',
             'JR横須賀線E217系',
             'JR横須賀線E235系',
              'N700',
             'メトロ10000系',
             'メトロ17000系',
             'メトロ7000系',
             '東武10000系',
             '東武10030系',
             '東武20000系',
             '東武30000系',
             '東武50000系',
             '東武50090系',
             '東武60000系',
             '東武70000系',
             '東武8000系',
             '東武9000系',
             '東急4000型']

classes_en = ['E5',
 'E6',
 'E7',
 'JR_chuoh_E233',
 'JR_hachiko_E231',
 'JR_keihintohoku_E233',
 'JR_keiyo_E233',
 'JR_musashino_E231',
 'JR_saikyo_E233',
 'JR_shonanshinjuku_E231',
 'JR_shonanshinjuku_E233',
 'JR_sobu_E231',
 'JR_yamanote_E235',
 'JR_yokosuka_E217',
 'JR_yokosuka_E235',
 'N700',
 'metoro10000',
 'metoro17000',
 'metoro7000',
 'tobu10000',
 'tobu10030',
 'tobu20000',
 'tobu30000',
 'tobu50000',
 'tobu50090',
 'tobu60000',
 'tobu70000',
 'tobu8000',
 'tobu9000',
 'tokyu4000']



img_size = 256

# 画像認識モデル
timm_model = "mobilenetv3_small_100"
net = timm.create_model(model_name=timm_model, pretrained=True, in_chans=3, num_classes=len(classes_ja))

# 訓練済みパラメータの読み込みと設定
net.load_state_dict(torch.load("model_cnn_"+timm_model+".pth", map_location=torch.device("cpu")))

def predict(img):
    transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
                                   ])
    img = transform(img)
    x = img.reshape(1, 3, img_size, img_size) # 1枚 x 3チャンネル(RGB) x 幅 x 高さ
    
    # 予測
    net.eval()
    y = net(x)
    
    # 結果を返す
    y_pred = F.softmax(torch.squeeze(y))
    sorted_prob, sorted_indices = torch.sort(y_pred, descending=True) # 予測確率を降順にソート
    
    return [(classes_ja[idx.item()], classes_en[idx.item()], prob.item()) for idx, prob in zip(sorted_indices, sorted_prob)]
        
