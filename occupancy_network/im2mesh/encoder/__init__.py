from im2mesh.encoder import (
    pointnet, vnn, evn
)


encoder_dict = {
    'pointnet_resnet': pointnet.ResnetPointnet,
    'vnn_pointnet_resnet': vnn.VNN_ResnetPointnet,
    'evn_pointnet_resnet': evn.EVN_ResnetPointnet
}
