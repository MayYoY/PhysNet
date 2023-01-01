from dataset import ubfc_rppg, pure, utils
from configs import preprocess, running
# from models import train_test
from efficientphys import train_test, cnn, transformer


"""ops = ubfc_rppg.Preprocess(preprocess.PreprocessUBFC.output_path, preprocess.PreprocessUBFC())
ops.read_process()
ops = pure.Preprocess(preprocess.PreprocessPURE.output_path, preprocess.PreprocessPURE())
ops.read_process()"""

"""
model_path = "/code/physnet/save/PURE_PURE_UBFC_physnet_normalized_Epoch8.pth"
train_test.fixSeed(42)
train_test.train_test("./save", running.TrainConfig, running.TestConfig)
# train_test.train_test("./save", running.TrainConfig, running.TestConfig,
#                       mode="Test", model_path = model_path)
"""

train_test.fixSeed(100)
# train_test.train_test("./save/cnn", running.TrainEfficient, running.TestEfficient)
# train_test.train_test("./save/cnn_nocrop", running.TrainEfficient, running.TestEfficient)
model_path = "/code/physnet/save/cnn/efficient_epoch7.pt"
train_test.train_test("./save/cnn", running.TrainEfficient, running.TestEfficient,
                      mode="Test", model_path=model_path)
