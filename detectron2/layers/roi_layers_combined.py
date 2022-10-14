import torch.nn as nn
import torch


class ROIFeatureCombine(nn.Module):
    def __init__(self):
        super(ROIFeatureCombine, self).__init__()
        self.transconv_op = nn.ConvTranspose2d(256,256,
                                                kernel_size=2,
                                                stride=2,
                                                padding=0)
        self.conv_op = nn.Conv2d(256, 256, 
                                kernel_size=3,
                                padding=1)

        self.downconv_op = nn.Conv2d(256,256,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1)

        self.up_block = nn.Sequential(
                            self.transconv_op,
                            self.conv_op,
                            nn.BatchNorm2d(256),
                            nn.ReLU()
                        )

        self.keep_block = nn.Sequential(
                            self.conv_op,
                            nn.BatchNorm2d(256),
                            nn.ReLU()
                        )

        self.down_block = nn.Sequential(
                            self.downconv_op,
                            nn.BatchNorm2d(256),
                            nn.ReLU()
                        )

    '''
    def forward(self, list_pooled_mask_features):
        import pdb;pdb.set_trace()
        x1 = list_pooled_mask_features[-1]
        x2 = self.transconv_op(x1)
        x3 = self.up_block(x1)
        x4 = self.transconv_op(torch.add(x2,x3))

        x5 = list_pooled_mask_features[-2]
        x6 = self.up_block(x5)
        x7 = self.transconv_op(torch.add(x4, x6))

        x8 = list_pooled_mask_features[-3]
        # # tmp
        # x8 = self.keep_block(x8)
        # # end tmp
        # return torch.add(torch.add(x4, x6), x8)
        x9 = self.up_block(x8)
        x10 = torch.add(x7, x9)
        
        x11 = list_pooled_mask_features[-4]
        x12 = self.keep_block(x11)
        
        mask_features = torch.add(x10, x12)
        return mask_features
    '''

    def forward(self, list_pooled_mask_features):
        x1 = list_pooled_mask_features[-1]
        x2 = self.up_block(x1)
        x2 = torch.add(list_pooled_mask_features[-2], x2) # skip connect

        x3 = self.up_block(x2)
        x3 = torch.add(list_pooled_mask_features[-3], x3)
        
        x4 = self.up_block(x3)
        x4 = torch.add(list_pooled_mask_features[-4], x4)

        return self.down_block(self.down_block(self.down_block(x4)))
