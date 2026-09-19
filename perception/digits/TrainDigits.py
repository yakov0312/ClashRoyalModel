import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from perception.digits.DigitNet import DigitNet


class RGBTransform:
    def __call__(self, image):
        image = np.array(image)

        image = cv2.resize(
            image,
            (28, 28),
            interpolation=cv2.INTER_AREA,
        )

        image = image.astype(np.float32) / 255.0

        return torch.from_numpy(image).permute(2, 0, 1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = datasets.ImageFolder(
    "dataset",
    transform=RGBTransform(),
)

print(f"Found {len(dataset)} images")
print(f"Classes: {dataset.classes}")

trainSize = int(len(dataset) * 0.8)
validationSize = len(dataset) - trainSize

trainSet, validationSet = random_split(
    dataset,
    [trainSize, validationSize],
)

trainLoader = DataLoader(
    trainSet,
    batch_size=32,
    shuffle=True,
)

validationLoader = DataLoader(
    validationSet,
    batch_size=32,
)

model = DigitNet().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
)

lossFunction = torch.nn.CrossEntropyLoss()

for epoch in range(17):
    model.train()
    totalLoss = 0.0

    for images, labels in trainLoader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        output = model(images)
        loss = lossFunction(output, labels)

        loss.backward()
        optimizer.step()

        totalLoss += loss.item()

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in validationLoader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            predictions = output.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total else 0

    print(
        f"Epoch {epoch + 1}/20 | "
        f"Loss: {totalLoss / len(trainLoader):.4f} | "
        f"Accuracy: {accuracy * 100:.2f}%"
    )

torch.save(model.state_dict(), "digitnet.pt")
print("Saved digitnet.pt")