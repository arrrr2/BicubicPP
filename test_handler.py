import unittest
import torch
import torch.nn as nn
from handler import ModelHandler

class TestModelHandler(unittest.TestCase):

    def setUp(self):
        self.config_path = 'config.yaml'
        self.handler = ModelHandler(self.config_path)

    def test_build_models(self):
        self.handler.build_models()
        self.assertIsInstance(self.handler.model, nn.Module)

    def test_set_stage(self):
        self.handler.set_stage('prunning')
        self.assertEqual(self.handler.get_stage(), 'prunning')

    def test_preparation(self):
        self.handler.build_models()
        self.handler.set_stage('pretraining')
        self.handler.preparation()
        self.assertIn(self.handler.device, str(next(self.handler.model.parameters()).device),)

    def test_train(self):
        self.handler.build_models()
        self.handler.set_stage('pretraining')
        self.handler.preparation()
        # train_loader = torch.utils.data.DataLoader(...)
        train_loader = None
        self.handler.train(train_loader)
        self.assertIsNotNone(self.handler.models['pretraining'])

    def test_validation(self):
        self.handler.build_models()
        self.handler.set_stage('pretraining')
        self.handler.preparation()
        # val_loader = torch.utils.data.DataLoader(...)
        val_loader = None
        val_loss = self.handler.validation(val_loader)
        # self.assertIsInstance(val_loss, float)

    # def test_save_load(self):
    #     self.handler.build_models()
    #     self.handler.set_stage('pretraining')
    #     self.handler.preparation()
    #     self.handler.save()
    #     self.handler.load()
    #     self.assertIsNotNone(self.handler.models)

    def test_remove_bias(self):
        self.handler.build_models()
        self.handler.train(None)
        self.handler.set_stage('prunning')
        self.handler.preparation()
        self.handler.remove_bias()
        

if __name__ == '__main__':
    unittest.main()