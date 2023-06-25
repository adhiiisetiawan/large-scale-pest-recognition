from ip102_datamodule import IP102DataModule

dm = IP102DataModule(batch_size=32)
# dm.prepare_data()
dm.setup()
feature, target = next(iter(dm.train_dataloader()))

print(feature.shape)