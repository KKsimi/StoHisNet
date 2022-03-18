import torch
from torch import nn
from torchvision import transforms, datasets
import json
import torch.optim as optim
from loss import focal_loss
from Model.StoHisNet import StoHisNet

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    net = StoHisNet(num_classes=4)
    # model_weight_path = './weight/resnet50.pth'
    # net.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cuda:0')))
    net.to(device)

    data_transform = {

        "train": transforms.Compose([
                                     transforms.Resize((224,224)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.ColorJitter(contrast=(0.8,1.2),saturation=(0.8, 1.2), hue=(-0.2,0.2)),
                                     transforms.ToTensor(),
                                     ]),
        "val": transforms.Compose([
                                   transforms.Resize((224,224)),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomVerticalFlip(),
                                   transforms.ToTensor(),
                                   ])}

    data_root = '/Data/wei_data/train/'
    val_path = '/Data/wei_data/val/'

    image_path = data_root

    train_dataset = datasets.ImageFolder(root=image_path,
                                         transform=data_transform['train'])

    class_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in class_list.items())
    # wirte dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_wei.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,shuffle=True,num_workers=8
    )
    validate_dataset = datasets.ImageFolder(root=val_path,
                                            transform=data_transform['val'])

    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size,
                                                  shuffle=True, num_workers=8)

    net.to(device)

    #loss_function = nn.CrossEntropyLoss()
    loss_function = focal_loss(alpha=[0.32, 0.24, 0.17, 0.27], gamma=2, num_classes=4)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=4e-8)
    train_loss = []
    valid_loss = []


    best_acc = 0.0
    save_path = 'weight/gas_data.pth'
    for epoch in range(70):
        # train
        net.train()
        running_loss = 0.0

        for step, data in enumerate(train_loader, start=0):
            images, labels = data
            images,labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = net(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            # print train process
            rate = (step + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate * 100), a, b, loss), end="")

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        valid_epoch_loss = []
        with torch.no_grad():
            for val_data in validate_loader:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                loss = loss_function(outputs, val_labels.to(device))
                valid_loss.append(loss.item())
                valid_epoch_loss.append(loss.item())
                acc += (predict_y == val_labels.to(device)).sum().item()

            # valid_epochs_loss.append(np.average(valid_epoch_loss))
            val_accurate = acc * 1.0 / val_num
            # valid_epochs_acc.append(val_accurate)
            print()
            print('acc is :', acc)
            print('best_acc is :', best_acc)
            if best_acc < val_accurate:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)
            print('[epoch %d] train_loss: %.3f test_accuracy: %.3f' % (
                epoch + 1, running_loss / step, val_accurate
            ))

        scheduler.step()

    print('Finish')

if __name__=='__main__':
    main()