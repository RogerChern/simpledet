import mxnet as mx
import mxnext as X
from utils.patch_config import patch_config_as_nothrow
from utils.deprecated import deprecated


class FasterRcnn:
    _rpn_output = None
    _p = None

    def __init__(self, pDetector):
        FasterRcnn._p = pDetector

    @classmethod
    def get_train_symbol(cls, backbone, neck, rpn_head, roi_extractor, bbox_head):
        p = cls._p
        backbone_batch_size = p.backbone_batch_size
        rcnn_batch_size = p.rcnn_batch_size
        rbbox_target_mean = p.rbbox_target.mean
        rbbox_target_std = p.rbbox_target.std

        gt_bbox = X.var("gt_bbox")
        gt_rbbox = X.var("gt_rbbox")
        im_info = X.var("im_info")

        rpn_feat = backbone.get_rpn_feature()
        rcnn_feat = backbone.get_rcnn_feature()
        rpn_feat = neck.get_rpn_feature(rpn_feat)
        rcnn_feat = neck.get_rcnn_feature(rcnn_feat)

        rpn_head.get_anchor()
        rpn_loss = rpn_head.get_loss(rpn_feat, gt_bbox, im_info)
        proposal, bbox_cls, bbox_target, bbox_weight, proposal_assigned_gt_idx = rpn_head.get_sampled_proposal(rpn_feat, gt_bbox, im_info, return_idx=True)

        # HACK: reference bbox_target in the graph or otherwise no memory will be allocated for it.
        # We required kWriteTo for its memory in the ProposalTarget operator.

        """
        # debug bbox_target
        def print_fun(gid, niter, bbox_target):
            print("bbox_target: \n{}".format(bbox_target[:10]))
        bbox_target = X.forward_debug(bbox_target, callback=print_fun)
        """

        proposal = mx.sym.broadcast_add((bbox_target * 0).sum(), proposal)

        # create rbbox target and weight from bbox sampling results
        inds = mx.sym.arange(0, backbone_batch_size, repeat=rcnn_batch_size).reshape(-1)
        proposal_assigned_gt_idx = mx.sym.stack(inds, proposal_assigned_gt_idx.reshape(-1))
        proposal_assigned_gt = mx.sym.gather_nd(gt_rbbox, proposal_assigned_gt_idx).reshape((backbone_batch_size, rcnn_batch_size, -1))

        """
        # debug proposal
        def print_fun(gid, niter, proposal):
            print("proposal: \n{}".format(proposal[:10]))
        proposal = X.forward_debug(proposal, callback=print_fun)
        """

        # debug proposal_assigned_gt
        def print_fun(gid, niter, proposal_assigned_gt):
            # print("proposal_assigned_gt.shape: {}".format(proposal_assigned_gt.shape))
            print("proposal_assigned_gt: \n{}".format(proposal_assigned_gt[0, :10]))
        proposal_assigned_gt = X.forward_debug(proposal_assigned_gt, callback=print_fun)

        rbbox_target = X.encode_rbbox(proposal, proposal_assigned_gt, im_info, rbbox_target_mean, rbbox_target_std)
        rbbox_target_zero = mx.sym.zeros_like(rbbox_target)
        rbbox_target = mx.sym.concat(rbbox_target_zero, rbbox_target, dim=-1)
        rbbox_target = X.reshape(rbbox_target, (-3, -2))

        """
        # debug rbbox_target
        def print_fun(gid, niter, rbbox_target):
            print("rbbox_target: \n{}".format(rbbox_target[:10]))
        rbbox_target = X.forward_debug(rbbox_target, callback=print_fun)
        """

        bbox_weight = bbox_weight.reshape((-1, 2, 4))
        rbbox_weight = X.concat([bbox_weight, mx.sym.slice_axis(bbox_weight, axis=-1, begin=0, end=1)], axis=-1, name='rbbox_weight').reshape((-1, 10))

        roi_feat = roi_extractor.get_roi_feature(rcnn_feat, proposal)
        bbox_loss = bbox_head.get_loss(roi_feat, bbox_cls, rbbox_target, rbbox_weight)

        return X.group(rpn_loss + bbox_loss)

    @classmethod
    def get_test_symbol(cls, backbone, neck, rpn_head, roi_extractor, bbox_head):
        rec_id, im_id, im_info, proposal, proposal_score = \
            FasterRcnn.get_rpn_test_symbol(backbone, neck, rpn_head)

        rcnn_feat = backbone.get_rcnn_feature()
        rcnn_feat = neck.get_rcnn_feature(rcnn_feat)

        roi_feat = roi_extractor.get_roi_feature_test(rcnn_feat, proposal)
        cls_score, bbox_xyxy = bbox_head.get_prediction(roi_feat, im_info, proposal)

        return X.group([rec_id, im_id, im_info, cls_score, bbox_xyxy])

    @classmethod
    def get_rpn_test_symbol(cls, backbone, neck, rpn_head):
        if cls._rpn_output is not None:
            return cls._rpn_output

        im_info = X.var("im_info")
        im_id = X.var("im_id")
        rec_id = X.var("rec_id")

        rpn_head.get_anchor()
        rpn_feat = backbone.get_rpn_feature()
        rpn_feat = neck.get_rpn_feature(rpn_feat)

        (proposal, proposal_score) = rpn_head.get_all_proposal(rpn_feat, im_info)

        cls._rpn_output = X.group([rec_id, im_id, im_info, proposal, proposal_score])
        return cls._rpn_output
