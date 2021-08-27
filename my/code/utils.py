import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# check if val loss is decreasing
def is_val_loss_decreasing(epoch, patience, losses):
    if epoch <= patience:
        return True
    if losses[-2] <= losses[-1]:
        return False
    return True

# check loader
def check_loader(loaders):
    X, y = next(iter(loaders))
    print('X[0] shape : ', X[0].shape)
    print('y[0] value : ', y[0])
    print('X length : ', len(X))

# check image
def check_image(image):
    image = image.numpy()
    image = np.transpose(image, (1, 2, 0))
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)

# return transforms
def get_transformer(aug_flag=True):
    transformer = dict()
    transformer['origin'] = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(), 
                                transforms.Normalize((0.4124234616756439, 0.3674212694168091, 0.2578217089176178), 
                                                     (0.3268945515155792, 0.29282665252685547, 0.29053378105163574))
                                ])

    if aug_flag:
        transformer['aug1'] = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                transforms.RandomRotation(5),
                                transforms.RandomAffine(degrees=11, translate=(0.1,0.1), scale=(0.8,0.8)),
                                transforms.ToTensor(),
                                ])

        transformer['aug2'] = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.CenterCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4124234616756439, 0.3674212694168091, 0.2578217089176178), 
                                                    (0.3268945515155792, 0.29282665252685547, 0.29053378105163574))
                                ])

        transformer['aug3'] = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.CenterCrop(224),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.ToTensor(),
                                ])

    return transformer