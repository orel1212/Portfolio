import torch
import torch.nn as nn
from torchvision.models import resnet34

class CRNN(nn.Module):
  def __init__(self,num_classes, max_sequence = 4*6, channels=3, img_h=50, img_w=250, lstm_hidden_size=512, num_lstm_layers=2, dropout_rate = 0.5):
    super(CRNN, self).__init__()

    resnet = resnet34(pretrained=True)
    resnet_trained_layers = list(resnet.children())[:-3]
    self.cnn_seq_1 = nn.Sequential(*resnet_trained_layers)

    self.cnn_seq_2 = nn.Sequential(
      nn.Conv2d(256, 256, kernel_size=(3, 6), stride=1, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True)
    )
    self.channels = channels
    self.img_h = img_h
    self.img_w = img_w
    self.lstm_dim, self.linear1_dim = self._get_dims_by_example()
    self.linear1 = nn.Linear(in_features=self.linear1_dim, out_features=max_sequence)
    self.lstm_hidden_size = lstm_hidden_size
    self.num_lstm_layers = num_lstm_layers
    self.lstm_part = nn.LSTM(input_size=self.lstm_dim, hidden_size=self.lstm_hidden_size,
                             num_layers=self.num_lstm_layers)
    self.mlp = nn.Linear(in_features=self.lstm_hidden_size, out_features=num_classes)
    self.dropout = nn.Dropout(dropout_rate)

  def _get_dims_by_example(self):
    rand_to_dim = torch.randn(1, self.channels, self.img_h, self.img_w)
    out_put_cnn_1 = self.cnn_seq_1(rand_to_dim)
    out_put_cnn_2 = self.cnn_seq_2(out_put_cnn_1)
    b, c, h, w = out_put_cnn_2.size()
    return c * h, w

  def forward(self, input_tensor):
    out_put_cnn_1 = self.cnn_seq_1(input_tensor)
    out_put_cnn_2 = self.cnn_seq_2(out_put_cnn_1)
    b, c, h, w = out_put_cnn_2.size()
    tensor_output = self.linear1(out_put_cnn_2.view(-1, c * h, w))

    tensor_output_permuted = tensor_output.permute(2, 0, 1)
    out_lstm, _ = self.lstm_part(tensor_output_permuted)
    do_lstm = self.dropout(out_lstm)

    logits = self.mlp(do_lstm)  # seq,batch,classes

    return logits