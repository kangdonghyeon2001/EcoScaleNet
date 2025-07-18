import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, dropout=0.2, residual_connection=None):
        super(ResidualBlock, self).__init__()

        self.residual_connection = residual_connection

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.residual_connection is not None:
            residual = self.residual_connection(x)

        out += residual
        out = F.relu(out)
        out = self.dropout2(out)

        return out
    
class EcoScaleNet(nn.Module):
    def __init__(self, n_classes, kernel_size_list, last_layer_activation='sigmoid'):

        super(EcoScaleNet, self).__init__()
   
        self.conv1 = nn.Conv1d(12, 64, kernel_size=7, padding=3, stride=2, bias=False) # N, 64, 2048
        self.bn1 = nn.BatchNorm1d(64)
        self.max_pool = nn.MaxPool1d(kernel_size = 3, stride=2, padding=1)

        self.kernel_size_list = kernel_size_list

        block = ResidualBlock

        self.layer1 = self._make_layer(block, 64, 64, 2, kernel_size=3, stride=2, dropout=0.2) 
        self.layer2 = self._make_layer(block, 64, 128, 2, kernel_size=3, stride=2, dropout=0.2) 
        self.layer3 = self._make_layer(block, 128, 256, 2, kernel_size=3, stride=2, dropout=0.2)
        self.layer4 = self._make_layer(block, 256, 512, 2, kernel_size=3, stride=2, dropout=0.2)

        self.os1 = EcoScaleBlock(self.kernel_size_list[0], 64, 32, 32)
        self.os2 = EcoScaleBlock(self.kernel_size_list[1], 128, 64, 64)
        self.os3 = EcoScaleBlock(self.kernel_size_list[2], 256, 128, 128)
        self.os4 = EcoScaleBlock(self.kernel_size_list[3], 512, 256, 256)
  
        self.avgpool = nn.AdaptiveAvgPool1d(output_size=(1))
        
        self.fc = nn.Linear(512, n_classes)
        self.last_layer_activation = nn.Sigmoid() if last_layer_activation == 'sigmoid' else nn.Identity() # for multi-label

    def _make_layer(self, block, in_ch, out_ch, blocks, kernel_size=3, stride=1, dropout=0.2):
        residual_connection = nn.Sequential(
                nn.MaxPool1d(kernel_size=stride, stride=stride),
                nn.Conv1d(in_ch, out_ch,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        layers = []
        layers.append(block(in_ch, out_ch, kernel_size, stride, dropout, residual_connection))
        for i in range(1, blocks):
            layers.append(block(out_ch, out_ch, kernel_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.os1(x)
        x = self.layer2(x)
        x = self.os2(x)
        x = self.layer3(x)
        x = self.os3(x)
        x = self.layer4(x)
        x = self.os4(x)

        x = self.avgpool(x)
        x = x.squeeze(2)
        x = self.fc(x)
        x = self.last_layer_activation(x)
        
        return x
    
class OS_stage(nn.Module):
    def __init__(self, kernel_size_list, in_ch, out_ch):
        super(OS_stage, self).__init__()

        self.convs = nn.ModuleList([])
        for kernel_size in kernel_size_list:
            conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding='same')
            self.convs.append(conv)

    def forward(self, x):
        output = []
        for conv in self.convs:
            output.append(conv(x))

        out = torch.cat(output, dim=1)
        return out
    
class EcoScaleBlock(nn.Module):
    
    def __init__(self, kernel_size_list, in_ch, out_ch, last_out_ch, dropout=0.2):
        super(EcoScaleBlock, self).__init__()

        self.conv1_1 = nn.Conv1d(in_ch, in_ch // 2, kernel_size=1,  bias=False)
        self.conv1 = OS_stage(kernel_size_list, in_ch // 2, out_ch)
        self.bn1 = nn.BatchNorm1d(out_ch * len(kernel_size_list))
        self.dropout1 = nn.Dropout(dropout)

        self.conv2_1 = nn.Conv1d((out_ch * len(kernel_size_list)), (out_ch * len(kernel_size_list)) // 2, kernel_size=1,  bias=False)
        self.conv2 = OS_stage(kernel_size_list, out_ch * len(kernel_size_list) // 2, out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch * len(kernel_size_list))
        self.dropout2 = nn.Dropout(dropout)

        self.conv3_1 = nn.Conv1d((out_ch * len(kernel_size_list)), (out_ch * len(kernel_size_list)) // 2, kernel_size=1,  bias=False)
        self.conv3 = OS_stage([1,2],out_ch * len(kernel_size_list) // 2, last_out_ch)
        self.bn3 = nn.BatchNorm1d(last_out_ch * 2)
        self.conv4 = nn.Conv1d(last_out_ch * 2, last_out_ch * 2, kernel_size=1, stride=1, bias=False)
        self.bn4 = nn.BatchNorm1d(last_out_ch * 2)
        self.dropout3 = nn.Dropout(dropout)
    
        self.residual = nn.Sequential(
            nn.Conv1d(in_ch, last_out_ch * 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(last_out_ch*2)
        )

    def forward(self, x):

        residual = self.residual(x)

        x = self.conv1_1(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.conv2_1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.conv3_1(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)

        x += residual

        x = F.relu(x)
        x = self.dropout3(x)

        return x