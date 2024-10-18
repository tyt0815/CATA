import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 레이어
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer (출력 레이어)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # 초기 은닉 상태 및 셀 상태를 0으로 설정
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM에 입력 데이터 통과
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # 마지막 타임스텝의 출력을 FC 레이어로 전달
        out = self.fc(out[:, -1, :])
        return out