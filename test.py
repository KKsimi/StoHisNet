import os
import torch
from PIL import Image
from torchvision import transforms
import json
from Model.StoHisNet import StoHisNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor(),

    ])

val_path = '/Data/wei_data/test/'


try:
    json_file = open('./class_wei.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

model = StoHisNet(num_classes=4)
model_weight_path = 'weight/gas_data.pth'
model.load_state_dict(torch.load(model_weight_path, map_location=device))
model.to(device)
model.eval()
a = []
b0 = b1 = b2 = b3 = b4 = b5 = 0
acc = 0
nam = []
i = 0

res = {}
resname = []
label = []
pre = []
conf = []
a = []
d = []

y_true = []
y_pred = []


with torch.no_grad():
    # predict class
    b0 = b1 = b2 = b3 = b4 = b5 = i = j = 0
    na = nam
    a = []
    tt = []
    cuo = []
    kk = 1
    for i in range(0, 4):
        val_pa = val_path + str(i) + '/'
        name_list = os.listdir(val_pa)
        su = len(name_list)
        d = 0
        for name in name_list:
            img = Image.open(val_pa+name).convert('RGB')
            img = data_transform(img)
            img = torch.unsqueeze(img, dim=0)
            outputs = model(img.to(device))
            predict = torch.softmax(outputs, dim=1)
            predict_y = torch.max(predict, dim=1)
            temp2 = predict_y[1].data.cpu().numpy().tolist()  #class
            print(temp2)
            print(kk)
            kk += 1
            y_true.append(i)
            y_pred.append(temp2[0])

            if i == temp2[0]:
                d += 1
            else:
                cuo.append(name)

        a.append(su)
        tt.append(d)

print(a)
print(tt)
print(cuo)
print('==============================')
print(y_true)
print(y_pred)

