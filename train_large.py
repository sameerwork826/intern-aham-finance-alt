from model_train import train_model

if __name__ == "__main__":
    auc = train_model("data/applications_large.csv")
    print(f"AUC: {auc:.3f}")


