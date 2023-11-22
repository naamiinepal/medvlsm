import numpy as np
import torch
import yaml

# from models import YOLOv1


def xywhc2label(bboxs, S, B, num_classes):
    # bboxs is a xywhc list: [(x,y,w,h,c),(x,y,w,h,c),....]
    label = np.zeros((S, S, 5 * B + num_classes))
    for x, y, w, h, c in bboxs:
        x_grid = int(x // (1.0 / S))
        y_grid = int(y // (1.0 / S))
        # xx = x / (1.0 / S) - x_grid
        # yy = y / (1.0 / S) - y_grid
        xx, yy = x, y
        label[y_grid, x_grid, 0:5] = np.array([xx, yy, w, h, 1])
        label[y_grid, x_grid, 5:10] = np.array([xx, yy, w, h, 1])
        label[y_grid, x_grid, 10 + c] = 1
    return label


def pred2xywhcc(pred, S, B, num_classes, conf_thresh, iou_thresh):
    # pred is a 7*7*(5*B+C) tensor, default S=7 B=2

    # bboxs = torch.zeros((S * S * B, 5 + num_classes))  # 98*15
    # for x in range(S):
    #     for y in range(S):
    #         # bbox1
    #         # bboxs[B * (x * S + y), 0:4] = torch.Tensor(
    #         #     [(pred[x, y, 0] + x) / S, (pred[x, y, 1] + y) / S, pred[x, y, 2], pred[x, y, 3]])
    #         # bboxs[B * (x * S + y), 4] = pred[x, y, 4]
    #         # bboxs[B * (x * S + y), 5:] = pred[x, y, 10:]
    #         #
    #         # # bbox2
    #         # bboxs[B * (x * S + y) + 1, 0:4] = torch.Tensor(
    #         #     [(pred[x, y, 5] + x) / S, (pred[x, y, 6] + y) / S, pred[x, y, S], pred[x, y, 8]])
    #         # bboxs[B * (x * S + y) + 1, 4] = pred[x, y, 9]
    #         # bboxs[B * (x * S + y) + 1, 5:] = pred[x, y, 10:]
    #
    #         # bbox1
    #         bboxs[B * (x * S + y), 0:4] = torch.Tensor([pred[x, y, 0], pred[x, y, 1], pred[x, y, 2], pred[x, y, 3]])
    #         bboxs[B * (x * S + y), 4] = pred[x, y, 4]
    #         bboxs[B * (x * S + y), 5:] = pred[x, y, 10:]
    #
    #         # bbox2
    #         bboxs[B * (x * S + y) + 1, 0:4] = torch.Tensor([pred[x, y, 5], pred[x, y, 6], pred[x, y, 7], pred[x, y, 8]])
    #         bboxs[B * (x * S + y) + 1, 4] = pred[x, y, 9]
    #         bboxs[B * (x * S + y) + 1, 5:] = pred[x, y, 10:]

    bboxs = torch.zeros((S * S, 5 + num_classes))  # 49*25
    for x in range(S):
        for y in range(S):
            # bbox1
            # bboxs[B * (x * S + y), 0:4] = torch.Tensor(
            #     [(pred[x, y, 0] + x) / S, (pred[x, y, 1] + y) / S, pred[x, y, 2], pred[x, y, 3]])
            # bboxs[B * (x * S + y), 4] = pred[x, y, 4]
            # bboxs[B * (x * S + y), 5:] = pred[x, y, 10:]
            #
            # # bbox2
            # bboxs[B * (x * S + y) + 1, 0:4] = torch.Tensor(
            #     [(pred[x, y, 5] + x) / S, (pred[x, y, 6] + y) / S, pred[x, y, S], pred[x, y, 8]])
            # bboxs[B * (x * S + y) + 1, 4] = pred[x, y, 9]
            # bboxs[B * (x * S + y) + 1, 5:] = pred[x, y, 10:]

            conf1, conf2 = pred[x, y, 4], pred[x, y, 9]
            if conf1 > conf2:
                # bbox1
                bboxs[(x * S + y), 0:4] = torch.Tensor(
                    [pred[x, y, 0], pred[x, y, 1], pred[x, y, 2], pred[x, y, 3]]
                )
                bboxs[(x * S + y), 4] = pred[x, y, 4]
                bboxs[(x * S + y), 5:] = pred[x, y, 10:]
            else:
                # bbox2
                bboxs[(x * S + y), 0:4] = torch.Tensor(
                    [pred[x, y, 5], pred[x, y, 6], pred[x, y, 7], pred[x, y, 8]]
                )
                bboxs[(x * S + y), 4] = pred[x, y, 9]
                bboxs[(x * S + y), 5:] = pred[x, y, 10:]

    # apply NMS to all bboxs
    xywhcc = nms(bboxs, num_classes, conf_thresh, iou_thresh)
    return xywhcc


def nms(bboxs, num_classes, conf_thresh=0.1, iou_thresh=0.3):
    # Non-Maximum Suppression, bboxs is a 98*15 tensor
    bbox_prob = bboxs[:, 5:].clone().detach()  # 98*10
    bbox_conf = bboxs[:, 4].clone().detach().unsqueeze(1).expand_as(bbox_prob)  # 98*10
    bbox_cls_spec_conf = bbox_conf * bbox_prob  # 98*10
    bbox_cls_spec_conf[bbox_cls_spec_conf <= conf_thresh] = 0

    # for each class, sort the cls-spec-conf score
    for c in range(num_classes):
        rank = torch.sort(
            bbox_cls_spec_conf[:, c], descending=True
        ).indices  # sort conf
        # for each bbox
        for i in range(bboxs.shape[0]):
            if bbox_cls_spec_conf[rank[i], c] == 0:
                continue
            for j in range(i + 1, bboxs.shape[0]):
                if bbox_cls_spec_conf[rank[j], c] != 0:
                    iou = calculate_iou(bboxs[rank[i], 0:4], bboxs[rank[j], 0:4])
                    if iou > iou_thresh:
                        bbox_cls_spec_conf[rank[j], c] = 0

    # exclude cls-specific confidence score=0
    bboxs = bboxs[torch.max(bbox_cls_spec_conf, dim=1).values > 0]

    bbox_cls_spec_conf = bbox_cls_spec_conf[
        torch.max(bbox_cls_spec_conf, dim=1).values > 0
    ]

    ret = torch.ones((bboxs.size()[0], 6))

    # return null
    if bboxs.size()[0] == 0:
        return torch.tensor([])

    # bbox coord
    ret[:, 0:4] = bboxs[:, 0:4]
    # bbox class-specific confidence scores
    ret[:, 4] = torch.max(bbox_cls_spec_conf, dim=1).values
    # bbox class
    ret[:, 5] = torch.argmax(bboxs[:, 5:], dim=1).int()
    return ret


def calculate_iou(bbox1, bbox2):
    # bbox: x y w h
    bbox1, bbox2 = (
        bbox1.cpu().detach().numpy().tolist(),
        bbox2.cpu().detach().numpy().tolist(),
    )

    area1 = bbox1[2] * bbox1[3]  # bbox1's area
    area2 = bbox2[2] * bbox2[3]  # bbox2's area

    max_left = max(bbox1[0] - bbox1[2] / 2, bbox2[0] - bbox2[2] / 2)
    min_right = min(bbox1[0] + bbox1[2] / 2, bbox2[0] + bbox2[2] / 2)
    max_top = max(bbox1[1] - bbox1[3] / 2, bbox2[1] - bbox2[3] / 2)
    min_bottom = min(bbox1[1] + bbox1[3] / 2, bbox2[1] + bbox2[3] / 2)

    if max_left >= min_right or max_top >= min_bottom:
        return 0
    else:
        # iou = intersect / union
        intersect = (min_right - max_left) * (min_bottom - max_top)
        return intersect / (area1 + area2 - intersect)


def parse_cfg(cfg_path):
    # cfg = {}
    # with open(cfg_path, 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         if line[0] == '#' or line == '\n':
    #             continue
    #         line = line.strip().split(':')
    #         key, value = line[0].strip(), line[1].strip()
    #         cfg[key] = value

    with open(cfg_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)  # dict
    print("Config:", cfg)
    return cfg


# def build_model(weight_path, S, B, num_classes):
#     model = YOLOv1(S, B, num_classes)
#     # model = YOLOv1ResNet(S, B, num_classes)

#     # load pretrained model
#     if weight_path and weight_path != '':
#         model.load_state_dict(torch.load(weight_path))
#     return model

if __name__ == "__main__":
    print(xywhc2label([(15, 15, 30, 30, 4), (10, 10, 20, 30, 5)], 4, 2, 10))
    pass
