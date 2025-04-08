import torch
import torch.nn as nn

class Yolo(nn.Module):
    def __init__(self, num_classes=20, 
                 anchors=[(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), 
                                              (9.47112, 4.84053), (11.2364, 10.0071)]):
        super(Yolo, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors  # 5 anchors tổng cộng
        self.num_anchors = len(anchors)  # 5

        # Stage 1: Backbone (giữ nguyên như mã của bạn)
        self.stage1_conv1 = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32),
                                          nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))
        self.stage1_conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64),
                                          nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))
        self.stage1_conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv4 = nn.Sequential(nn.Conv2d(128, 64, 1, 1, 0, bias=False), nn.BatchNorm2d(64),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv5 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128),
                                          nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))
        self.stage1_conv6 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv7 = nn.Sequential(nn.Conv2d(256, 128, 1, 1, 0, bias=False), nn.BatchNorm2d(128),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv8 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256),
                                          nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))
        self.stage1_conv9 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv10 = nn.Sequential(nn.Conv2d(512, 256, 1, 1, 0, bias=False), nn.BatchNorm2d(256),
                                           nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv11 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512),
                                           nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv12 = nn.Sequential(nn.Conv2d(512, 256, 1, 1, 0, bias=False), nn.BatchNorm2d(256),
                                           nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv13 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512),
                                           nn.LeakyReLU(0.1, inplace=True))

        # Stage 2a: Nhánh cho scale 13x13 (vật thể lớn)
        self.stage2_a_maxpl = nn.MaxPool2d(2, 2)
        self.stage2_a_conv1 = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1, bias=False), nn.BatchNorm2d(1024),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv2 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1, 0, bias=False), nn.BatchNorm2d(512),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv3 = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1, bias=False), nn.BatchNorm2d(1024),
                                            nn.LeakyReLU(0.1, inplace=True))

        # Đầu ra dự đoán cho scale 13x13
        self.scale_13x13 = nn.Conv2d(1024, self.num_anchors * (5 + num_classes), 1, 1, 0, bias=False)

        # Stage 2b: Nhánh cho scale 26x26 (vật thể trung bình/nhỏ)
        self.stage2_b_conv1 = nn.Sequential(nn.Conv2d(512, 256, 1, 1, 0, bias=False), nn.BatchNorm2d(256),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.stage2_b_conv2 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512),
                                            nn.LeakyReLU(0.1, inplace=True))

        # Đầu ra dự đoán cho scale 26x26
        self.scale_26x26 = nn.Conv2d(512, self.num_anchors * (5 + num_classes), 1, 1, 0, bias=False)

    def forward(self, input):
        # Stage 1: Backbone
        output = self.stage1_conv1(input)   # 416x416x3 → 208x208x32
        output = self.stage1_conv2(output)  # 208x208x32 → 104x104x64
        output = self.stage1_conv3(output)  # 104x104x64 → 104x104x128
        output = self.stage1_conv4(output)  # 104x104x128 → 104x104x64
        output = self.stage1_conv5(output)  # 104x104x64 → 52x52x128
        output = self.stage1_conv6(output)  # 52x52x128 → 52x52x256
        output = self.stage1_conv7(output)  # 52x52x256 → 52x52x128
        output = self.stage1_conv8(output)  # 52x52x128 → 26x26x256
        output = self.stage1_conv9(output)  # 26x26x256 → 26x26x512
        output = self.stage1_conv10(output) # 26x26x512 → 26x26x256
        output = self.stage1_conv11(output) # 26x26x256 → 26x26x512
        output = self.stage1_conv12(output) # 26x26x512 → 26x26x256
        output = self.stage1_conv13(output) # 26x26x256 → 26x26x512
        residual = output  # 26x26x512

        # Stage 2a: Nhánh cho scale 13x13
        output_13 = self.stage2_a_maxpl(residual)  # 26x26x512 → 13x13x512
        output_13 = self.stage2_a_conv1(output_13) # 13x13x512 → 13x13x1024
        output_13 = self.stage2_a_conv2(output_13) # 13x13x1024 → 13x13x512
        output_13 = self.stage2_a_conv3(output_13) # 13x13x512 → 13x13x1024
        pred_13x13 = self.scale_13x13(output_13)   # 13x13x1024 → 13x13x125 (5 anchors, 20 classes)

        # Stage 2b: Nhánh cho scale 26x26
        output_26 = self.stage2_b_conv1(residual)  # 26x26x512 → 26x26x256
        output_26 = self.stage2_b_conv2(output_26) # 26x26x256 → 26x26x512
        pred_26x26 = self.scale_26x26(output_26)   # 26x26x512 → 26x26x125 (5 anchors, 20 classes)

        return pred_13x13, pred_26x26  # Trả về 2 đầu ra: 13x13x125 và 26x26x125
    
if __name__ == "__main__":
    modeltest = Yolo(80)
    print(type(modeltest))