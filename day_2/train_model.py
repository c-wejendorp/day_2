import click
import torch
from models.model import MyAwesomeModel
import matplotlib.pyplot as plt
from data.make_dataset import load_data
import os


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--epochs", default=30, help="number of epochs to train for")
def train(lr, epochs):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    model = MyAwesomeModel()
    model.train()
    train_set, _ = load_data()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    loss_hist = []

    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_set):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print("Epoch: {}, Loss: {}".format(epoch, loss.item()))
        loss_hist.append(loss.item())

    # save the model in subfolder
    if not os.path.exists("models/trained_models"):
        os.makedirs("models/trained_models")

    torch.save(model, f"models/trained_models/model_{lr}_{epochs}.pt")

    # visualize the loss
    plt.plot(loss_hist)
    plt.xlabel("Epochs")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Loss vs Epochs")
    # save in the visualization folder
    if not os.path.exists("reports/figures"):
        os.makedirs("reports/figures")
    plt.savefig(f"reports/figures/loss_{lr}_{epochs}.png")
    plt.close()


cli.add_command(train)


if __name__ == "__main__":
    cli()
